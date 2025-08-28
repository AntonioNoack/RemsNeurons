package me.anno.remsneurons.network

import me.anno.remsneurons.LearningParams
import me.anno.remsneurons.NetworkLayout
import me.anno.remsneurons.activations.Activation
import me.anno.remsneurons.layers.Layer
import me.anno.utils.assertions.assertSame
import me.anno.utils.assertions.assertTrue
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.random.Random

class CPUNetwork(networkLayout: NetworkLayout, batchSize: Int) : Network<FloatArray>(
    networkLayout, batchSize,

    FloatArray(networkLayout.numInputs * batchSize),
    FloatArray(networkLayout.numWeights),
    FloatArray(networkLayout.numNodes * batchSize),
    FloatArray(networkLayout.numOutputs * batchSize),
    FloatArray(networkLayout.numNodes * batchSize)
) {

    fun getInput(bi: Int, ni: Int): Float {
        return currInputs[getInIndex(bi, ni)]
    }

    fun getOutput(bi: Int, no: Int): Float {
        return activated[getOutIndex(bi, no)]
    }

    fun getOutDelta(bi: Int, no: Int): Float {
        return deltas[getOutIndex(bi, no)]
    }

    fun setInDelta(bi: Int, ni: Int, value: Float) {
        assertTrue(currOutputOffset > 0)
        deltas[getInIndex(bi, ni)] = value
    }

    fun addInDelta(bi: Int, ni: Int, value: Float) {
        assertTrue(currOutputOffset > 0)
        deltas[getInIndex(bi, ni)] += value
    }

    fun setOutDelta(bi: Int, no: Int, delta: Float) {
        // println("DX[${getOutIndex(bi, no)}] = $delta")
        deltas[getOutIndex(bi, no)] = delta
    }

    fun setOutSum(bi: Int, no: Int, value: Float) {
        activated[getOutIndex(bi, no)] = value
    }

    fun getWeight(index: Int): Float {
        return weights[getCurrWeightIndex(index)]
    }

    fun setWeight(index: Int, value: Float) {
        weights[getCurrWeightIndex(index)] = value
    }

    override fun initializeWeights(rnd: Random) {
        val layers = networkLayout.layers
        for (li in layers.indices) {
            val layer = layers[li]
            val factor = 2f / sqrt(layer.numInputsPerNode.toFloat())
            for (i in layer.weightOffset until layer.weightOffset + layer.numWeights) {
                val weight = (rnd.nextFloat() - 0.5f) * factor
                weights[i] = weight
            }
        }
    }

    fun normalizeDeltas(params: LearningParams, i0: Int, i1: Int) {
        if (i1 - i0 < 2) return
        var absMax = 1e-30f
        for (i in i0 until i1) {
            absMax = max(absMax, abs(deltas[i]))
        }
        val multiplier = params.learningRate / absMax
        for (i in i0 until i1) {
            deltas[i] *= multiplier
        }
    }

    override fun setInput(bi: Int, ni: Int, value: Float) {
        assertTrue(ni in 0 until numInputs)
        inputs[bi * numInputs + ni] = value
    }

    override fun setTarget(bi: Int, no: Int, expected: Float) {
        targets[bi * networkLayout.numOutputs + no] = expected
    }

    override fun clearDeltas() {
        deltas.fill(0f)
    }

    override fun forwardImpl(layer: Layer) {
        for (bi in 0 until batchSize) {
            for (no in 0 until currNumOutputs) {
                layer.applyForward(this, bi, no)
            }
            applyActivation(bi, layer.activation)
        }
    }

    override fun evalImpl(needsError: Boolean): Float {
        val numOutputs = currNumOutputs
        var errorSum = 0.0
        for (bi in 0 until batchSize) {
            for (no in 0 until numOutputs) {
                val actual = getOutput(bi, no)
                val expected = targets[bi * numOutputs + no]
                val delta = expected - actual
                // println("Expected vs actual $expected vs $actual -> $delta")
                setOutDelta(bi, no, delta)
                errorSum += delta * delta
            }
        }
        return sqrt(errorSum / batchSize).toFloat()
    }

    override fun backwardImpl(learningParams: LearningParams, layer: Layer) {

        val gradient = layer !== networkLayout.layers.first()
        for (bi in 0 until batchSize) {
            applyInvActivation(bi, layer.activation)
        }

        // println("Deltas: ${deltas.toList()}")

        for (wi in 0 until layer.numWeights) {
            layer.applyBackward(this, wi, learningParams, gradient)
        }

        if (gradient && learningParams.normalize) {
            val i2 = currInputOffset
            normalizeDeltas(learningParams, i2, i2 + batchSize * numInputs)
        }
    }

    override fun prepareInputs() {
        // nothing to do
    }

    override fun prepareTargets() {
        // nothing to do
    }

    override fun inspectWeights(): FloatArray = weights
    override fun inspectDeltas(): FloatArray = deltas
    override fun inspectActivated(): FloatArray = activated

    private fun applyActivation(bi: Int, activation: Activation) {
        val i0 = getOutIndex(bi, 0)
        activation.forwardKotlin.justApply(activated, i0, i0 + currNumOutputs)
    }

    private fun applyInvActivation(bi: Int, activation: Activation) {
        // apply activation-network factor correction
        val i0 = currOutputOffset + bi * currNumOutputs
        activation.backwardKotlin.mulApply(activated, deltas, i0, i0 + currNumOutputs)
    }

    override fun setWeights(weights: FloatArray) {
        assertSame(this.weights, weights)
    }

}