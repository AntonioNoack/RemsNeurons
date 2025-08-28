package me.anno.remsneurons.network

import me.anno.gpu.buffer.Attribute
import me.anno.gpu.buffer.CompactAttributeLayout.Companion.bind
import me.anno.gpu.buffer.OpenGLBuffer
import me.anno.gpu.buffer.StaticBuffer
import me.anno.gpu.shader.ComputeShader
import me.anno.gpu.shader.GLSLType
import me.anno.gpu.shader.GPUShader
import me.anno.gpu.shader.builder.Variable
import me.anno.maths.Maths.sq
import me.anno.remsneurons.LearningParams
import me.anno.remsneurons.NetworkLayout
import me.anno.remsneurons.layers.Layer
import me.anno.utils.assertions.assertFalse
import me.anno.utils.structures.maps.LazyMap
import org.joml.Vector3i
import org.lwjgl.opengl.GL46C.GL_SHADER_STORAGE_BARRIER_BIT
import org.lwjgl.opengl.GL46C.glMemoryBarrier
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.random.Random

class GPUNetwork private constructor(
    networkLayout: NetworkLayout, numInputs: Int, numOutputs: Int, batchSize: Int,
    numWeights: Int, numNodes: Int
) : Network<OpenGLBuffer>(
    networkLayout, batchSize,

    create("Inputs[$numInputs x $batchSize]", numInputs * batchSize),
    create("Weights[$numWeights]", numWeights),
    create("Activated[$numNodes x $batchSize]", numNodes * batchSize),
    create("Targets[$numOutputs x $batchSize]", numOutputs * batchSize),
    create("Deltas[$numNodes x $batchSize]", numNodes * batchSize)
) {

    constructor(networkLayout: NetworkLayout, batchSize: Int) :
            this(networkLayout, networkLayout.numInputs, networkLayout.numOutputs, batchSize, networkLayout.numWeights, networkLayout.numNodes)

    fun synchronize() {
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
    }

    override fun clearDeltas() {
        val shader = clearShader
        shader.use()
        shader.v1i("bufferSize", networkLayout.numNodes * batchSize)
        shader.bindBuffer(0, deltas)
        shader.runBySize(networkLayout.numNodes * batchSize)
        synchronize()
    }

    private fun bindOffsets(shader: GPUShader) {
        shader.v1i("currInputOffset", currInputOffset)
        shader.v1i("currOutputOffset", currOutputOffset)
        shader.v1i("currWeightOffset", currWeightOffset)
        shader.v1i("numInputs", currNumInputs)
        shader.v1i("numOutputs", currNumOutputs)
    }

    override fun forwardImpl(layer: Layer) {
        val shader = forwardShaders[ForwardKey(
            batchSize, layer.numOutputs,
            layer.forwardGLSL
        )]
        shader.use()
        bindOffsets(shader)
        shader.bindBuffer(0, weights)
        shader.bindBuffer(1, currInputs) // input - weights > values
        shader.bindBuffer(2, activated)
        shader.runBySize(batchSize, currNumOutputs)
        synchronize()

        runActivation(layer)
    }

    private fun runActivation(layer: Layer) {
        if (layer.activation.forwardGLSL.isEmpty()) return
        val shader = activationShaders[ActivationKey(
            batchSize, layer.numOutputs, layer.activation.forwardGLSL,
            layer.activation.hasInterdependencies
        )]
        shader.use()
        bindOffsets(shader)
        shader.bindBuffer(0, activated)
        runActivation(layer, shader)
    }

    private fun runActivationInv(layer: Layer) {
        if (layer.activation.backwardGLSL.isEmpty()) return // quick-path
        val shader = activationInvShaders[ActivationKey(
            batchSize, layer.numOutputs,
            layer.activation.backwardGLSL,
            layer.activation.hasInterdependencies
        )]
        shader.use()
        bindOffsets(shader)
        shader.bindBuffer(0, activated)
        shader.bindBuffer(1, deltas)
        runActivation(layer, shader)
    }

    private fun runActivation(layer: Layer, shader: ComputeShader) {
        if (layer.activation.hasInterdependencies) {
            shader.runBySize(batchSize, 1)
        } else {
            shader.runBySize(batchSize, layer.numOutputs)
        }
        synchronize()
    }

    private val deltasF = FloatArray(batchSize * networkLayout.numOutputs)

    override fun evalImpl(needsError: Boolean): Float {
        val shader = evalShaders[EvalKey(batchSize, networkLayout.numOutputs)]
        shader.use()
        bindOffsets(shader)
        shader.bindBuffer(0, targets) // expected
        shader.bindBuffer(1, activated) // actual
        shader.bindBuffer(2, deltas) // dst
        shader.runBySize(batchSize)
        synchronize()

        return if (needsError) {
            // this is a massive slowdown!
            val endOffset = batchSize.toLong() * (networkLayout.numNodes - networkLayout.numOutputs)
            deltas.readAsFloatArray(endOffset, deltasF)

            val errorSum = deltasF.sumOf { sq(it.toDouble()) }.toFloat()
            sqrt(errorSum / batchSize)
        } else Float.NaN
    }

    override fun backwardImpl(learningParams: LearningParams, layer: Layer) {
        runActivationInv(layer)
        runBackwardsConvolution(learningParams, layer)
        assertFalse(learningParams.normalize)
    }

    private fun runBackwardsConvolution(learningParams: LearningParams, layer: Layer) {
        val shader = backwardShaders[BackwardKey(batchSize, layer.numWeights, layer.backwardGLSL)]
        shader.use()
        bindOffsets(shader)
        shader.v1f("learningRate", learningParams.learningRate)
        shader.v1b("notFirstLayer", layer !== networkLayout.layers.first())
        shader.bindBuffer(0, weights)
        shader.bindBuffer(1, currInputs) // input - weights > values
        shader.bindBuffer(2, activated)
        shader.bindBuffer(3, deltas)
        shader.runBySize(layer.numWeights)
        synchronize()
    }

    override fun setInput(bi: Int, ni: Int, value: Float) {
        val nio = inputs.getOrCreateNioBuffer()
        nio.asFloatBuffer().put(bi * networkLayout.numInputs + ni, value)
    }

    override fun setTarget(bi: Int, no: Int, expected: Float) {
        val nio = targets.getOrCreateNioBuffer()
        nio.asFloatBuffer().put(bi * networkLayout.numOutputs + no, expected)
    }

    override fun prepareInputs() {
        prepareBuffer(inputs, networkLayout.numInputs)
    }

    override fun prepareTargets() {
        prepareBuffer(targets, networkLayout.numOutputs)
    }

    private fun prepareBuffer(buffer: OpenGLBuffer, baseSize: Int) {
        val numTargetBytes = 4 * batchSize * baseSize
        buffer.getOrCreateNioBuffer().position(numTargetBytes)
        buffer.cpuSideChanged()
        buffer.ensureBuffer()
    }

    override fun initializeWeights(rnd: Random) {
        val nio = weights.getOrCreateNioBuffer()
        val nioF = nio.asFloatBuffer()
        val layers = networkLayout.layers
        for (li in layers.indices) {
            val layer = layers[li]
            val factor = 2f / sqrt(layer.numInputsPerNode.toFloat())
            for (i in layer.weightOffset until layer.weightOffset + layer.numWeights) {
                val weight = (rnd.nextFloat() - 0.5f) * factor
                nioF.put(i, weight)
            }
        }
        nio.position(networkLayout.numWeights * 4)
        weights.cpuSideChanged()
    }

    override fun inspectWeights(): FloatArray {
        return weights.readAsFloatArray()
    }

    override fun inspectDeltas(): FloatArray {
        return deltas.readAsFloatArray()
    }

    override fun inspectActivated(): FloatArray {
        return activated.readAsFloatArray()
    }

    override fun setWeights(weights: FloatArray) {
        val dst = this.weights
        dst.destroy()

        val nio = dst.getOrCreateNioBuffer()
        nio.asFloatBuffer().put(weights)
        nio.position(weights.size * 4)
        dst.ensureBuffer()
    }

    companion object {

        val attr = bind(Attribute("value", 1))
        fun create(name: String, size: Int): OpenGLBuffer {
            val buffer = StaticBuffer(name, attr, size)
            buffer.uploadEmpty(size * 4L)
            return buffer
        }

        data class ForwardKey(val batchSize: Int, val numOutputs: Int, val forward: String)
        data class ActivationKey(val batchSize: Int, val numOutputs: Int, val activation: String, val hasInterdependencies: Boolean)

        data class EvalKey(val batchSize: Int, val numOutputs: Int)
        data class BackwardKey(val batchSize: Int, val numWeights: Int, val backward: String)

        private fun getSize2d(xi: Int, yi: Int): Vector3i {
            return Vector3i(min(xi, 256), min(1024 / xi, yi), 1)
        }

        private fun getSize1d(xi: Int): Vector3i {
            return Vector3i(min(xi, 256), 1, 1)
        }

        private val activationShaders = LazyMap { key: ActivationKey ->
            createActivationShader(key, "activation", false)
        }

        private val activationInvShaders = LazyMap { key: ActivationKey ->
            createActivationShader(key, "activationInv", true)
        }

        private fun createActivationShader(key: ActivationKey, name: String, inv: Boolean): ComputeShader {
            val (batchSize, numOutputs, activationInv, hasInterdependencies) = key
            val buffers = if (inv) {
                "" +
                        "layout(std430, binding=0) readonly buffer activatedBuffer1 { float activated[]; };\n" +
                        "layout(std430, binding=1)          buffer deltasBuffer1    { float deltas[]; };\n"
            } else {
                "" +
                        "layout(std430, binding=0) buffer activatedBuffer1 { float activated[]; };\n"
            }
            return if (hasInterdependencies) {
                ComputeShader(
                    name, getSize1d(batchSize), listOf(
                        Variable(GLSLType.V1I, "currOutputOffset"),
                    ), "" +
                            buffers +
                            "void main() {\n" +
                            "   int bi = int(gl_GlobalInvocationID.x);\n" +
                            "   if(bi>=$batchSize)return;\n" +
                            "   int i0 = (bi+0)*$numOutputs+currOutputOffset;\n" +
                            "   int i1 = (bi+1)*$numOutputs+currOutputOffset;\n" +
                            activationInv +
                            "}\n"
                )
            } else {
                ComputeShader(
                    name, getSize2d(batchSize, numOutputs), listOf(
                        Variable(GLSLType.V1I, "currOutputOffset"),
                    ), "" +
                            buffers +
                            "void main() {\n" +
                            "   int bi = int(gl_GlobalInvocationID.x);\n" +
                            "   int no = int(gl_GlobalInvocationID.y);\n" +
                            "   if(bi>=$batchSize||no>=$numOutputs)return;\n" +
                            "   int i = bi*$numOutputs+no+currOutputOffset;\n" +
                            activationInv +
                            "}\n"
                )
            }
        }

        private val evalShaders = LazyMap { key: EvalKey ->
            val (batchSize, numOutputs) = key
            ComputeShader(
                "eval", getSize2d(batchSize, numOutputs), listOf(
                    Variable(GLSLType.V1I, "currInputOffset"),
                    Variable(GLSLType.V1I, "currOutputOffset"),
                    Variable(GLSLType.V1I, "currWeightOffset")
                ), "" +
                        "layout(std430, binding=0) readonly  buffer expectedBuffer1 { float expected[]; };\n" +
                        "layout(std430, binding=1) readonly  buffer actualBuffer1   { float actual[]; };\n" +
                        "layout(std430, binding=2) writeonly buffer deltasBuffer1   { float deltas[]; };\n" +
                        "void main(){\n" +
                        "   int bi = int(gl_GlobalInvocationID.x);\n" +
                        "   int no = int(gl_GlobalInvocationID.y);\n" +
                        "   if(bi>=$batchSize||no>=$numOutputs)return;\n" +
                        "   int index = bi*$numOutputs+no;\n" +
                        "   deltas[index+currOutputOffset] = expected[index] - actual[index+currOutputOffset];\n" +
                        "}\n"
            )
        }

        private val clearShader = ComputeShader(
            "clear", Vector3i(512, 1, 1), listOf(
                Variable(GLSLType.V1I, "bufferSize")
            ), "" +
                    "layout(std430, binding=0) writeonly buffer valuesBuffer1 { float values[]; };\n" +
                    "void main(){\n" +
                    "   int index = int(gl_GlobalInvocationID.x);\n" +
                    "   if(index >= bufferSize) return;\n" +
                    "   values[index] = 0.0;\n" +
                    "}\n"
        )

        private val forwardShaders = LazyMap { key: ForwardKey ->
            val (batchSize, numOutputs, forward) = key
            ComputeShader(
                "forward", getSize2d(batchSize, numOutputs), listOf(
                    Variable(GLSLType.V1I, "currInputOffset"),
                    Variable(GLSLType.V1I, "currOutputOffset"),
                    Variable(GLSLType.V1I, "currWeightOffset"),
                    Variable(GLSLType.V1I, "numInputs"),
                    Variable(GLSLType.V1I, "numOutputs")
                ), "" +
                        "layout(std430, binding=0) readonly  buffer weightsBuffer1   { float weights[]; };\n" +
                        "layout(std430, binding=1) readonly  buffer inputsBuffer1    { float inputs[]; };\n" +
                        "layout(std430, binding=2) writeonly buffer activatedBuffer1 { float activated[]; };\n" +

                        getInIndex + getOutIndex +
                        getWeight + getInput + setOutSum +

                        "void main(){\n" +
                        "   int bi = int(gl_GlobalInvocationID.x);\n" +
                        "   int no = int(gl_GlobalInvocationID.y);\n" +
                        "   if(bi>=$batchSize||no>=numOutputs)return;\n" +
                        forward +
                        "}\n"
            )
        }

        private val backwardShaders = LazyMap { key: BackwardKey ->
            val (batchSize, numWeights, backward) = key
            ComputeShader(
                "backward", 460, getSize1d(numWeights), listOf(
                    Variable(GLSLType.V1I, "currInputOffset"),
                    Variable(GLSLType.V1I, "currOutputOffset"),
                    Variable(GLSLType.V1I, "currWeightOffset"),
                    Variable(GLSLType.V1I, "numInputs"),
                    Variable(GLSLType.V1I, "numOutputs"),
                    Variable(GLSLType.V1F, "learningRate"),
                    Variable(GLSLType.V1B, "gradient"),
                ), "" +
                        // "#extension GL_ARB_shader_atomic_float : require\n" +
                        "#extension GL_NV_shader_atomic_float : require\n" +

                        "layout(std430, binding=0)          buffer weightsBuffer1   { float weights[]; };\n" +
                        "layout(std430, binding=1) readonly buffer inputsBuffer1    { float inputs[]; };\n" +
                        "layout(std430, binding=3)          buffer deltasBuffer1    { float deltas[]; };\n" +

                        getInIndex + getOutIndex +
                        getWeight + setWeight +
                        getInput + getOutDelta + addInDelta +

                        "void main(){\n" +
                        "   int weightIndex = int(gl_GlobalInvocationID.x);\n" +
                        "   if(weightIndex>=$numWeights)return;\n" +
                        "   const int batchSize = $batchSize;\n" +
                        backward +
                        "}\n"
            )
        }

        val getWeight = "float getWeight(int weightIndex) {\n" +
                "   return weights[weightIndex+currWeightOffset];\n" +
                "}\n"
        val setWeight = "void setWeight(int weightIndex, float value) {\n" +
                "   weights[weightIndex+currWeightOffset] = value;\n" +
                "}\n"

        val getInIndex = "int getInIndex(int bi,int ni) { return bi*numInputs+ni+currInputOffset; }\n"
        val getOutIndex = "int getOutIndex(int bi,int no) { return bi*numOutputs+no+currOutputOffset; }\n"

        val getInput = "float getInput(int bi, int ni) {\n" +
                "   return inputs[getInIndex(bi,ni)];\n" +
                "}\n"
        val getOutDelta = "float getOutDelta(int bi, int no) {\n" +
                "   return deltas[getOutIndex(bi,no)];\n" +
                "}\n"
        val addInDelta = "void addInDelta(int bi, int ni, float delta) {\n" +
                "   atomicAdd(deltas[getInIndex(bi,ni)], delta);\n" +
                "}\n"
        val setOutSum = "void setOutSum(int bi, int no, float value) {\n" +
                "   activated[getOutIndex(bi,no)] = value;\n" +
                "}\n"

    }
}