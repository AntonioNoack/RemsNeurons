package me.anno.remsneurons.layers

import me.anno.maths.Maths.clamp
import me.anno.remsneurons.LearningParams
import me.anno.remsneurons.activations.Activation
import me.anno.remsneurons.network.CPUNetwork
import me.anno.utils.assertions.assertTrue
import me.anno.utils.types.Strings.iff
import org.joml.Vector2i

/**
 * when this works, we can implement Conv1d using Conv2d? -> yes, but we don't want to over-complicate things...
 * */
class ConvolutionalLayer2d private constructor(
    val inputSeriesSize: Vector2i,
    val numInputAttributes: Int,

    val convSize: Vector2i,
    val padEnds: Boolean,

    val numOutputAttributes: Int,

    activation: Activation,

    val numOutputsPerKernel: Vector2i,

    val offset: Vector2i,
    val numWeightsPerKernel: Int

) : Layer(
    inputSeriesSize.x * inputSeriesSize.y * numInputAttributes,
    convSize.x * convSize.y * numInputAttributes * numOutputAttributes,
    (
            if (padEnds) inputSeriesSize.x * inputSeriesSize.y
            else (inputSeriesSize.x - convSize.x + 1) * (inputSeriesSize.y - convSize.y + 1)
            ) * numOutputAttributes,
    numInputAttributes * convSize.x * convSize.y,
    activation,
    "" +
            "int seriesIndex = no % ${numOutputsPerKernel.x * numOutputsPerKernel.y};\n" +
            "int seriesIndexX = seriesIndex % ${numOutputsPerKernel.x};\n" +
            "int seriesIndexY = seriesIndex / ${numOutputsPerKernel.x};\n" +
            "int outAttrIndex = no / ${numOutputsPerKernel.x * numOutputsPerKernel.y};\n" +
            "float convSum = 0.0;\n" +
            "for(int ai=0;ai<$numInputAttributes;ai++){\n" +
            "   for(int ciy=0;ciy<${convSize.y};ciy++){\n" +
            "       for(int cix=0;cix<${convSize.x};cix++){\n" +
            "           int weightIndex = outAttrIndex * $numWeightsPerKernel + ai * ${convSize.x * convSize.y} + ciy * ${convSize.x} + cix;\n" +
            "           int nix = seriesIndexX + ${offset.x} + cix;\n" +
            "           int niy = seriesIndexY + ${offset.y} + ciy;\n" +
            ("           nix = clamp(nix,0,${inputSeriesSize.x - 1});\n" +
                    "    niy = clamp(niy,0,${inputSeriesSize.y - 1});\n").iff(padEnds) +
            "           int ni = ai * ${inputSeriesSize.x * inputSeriesSize.y} + niy * ${inputSeriesSize.x} + nix;\n" +
            "           convSum += getInput(bi,ni) * getWeight(weightIndex);\n" +
            "       }\n" +
            "   }\n" +
            "}\n" +
            "setOutSum(bi,no,convSum);\n",
    "" +
            "int localWeightXY = weightIndex % ${convSize.x * convSize.y};\n" +
            "int localWeightX = localWeightXY % ${convSize.x};\n" +
            "int localWeightY = localWeightXY / ${convSize.x};\n" +
            "int inAttrIndex = (weightIndex / ${convSize.x * convSize.y}) % $numInputAttributes;\n" +
            "int outAttrIndex = weightIndex / $numWeightsPerKernel;\n" +
            "float originalWeight = getWeight(weightIndex);\n" +
            "float deltaWeight = 0.0;\n" +
            "for(int bi=0;bi<batchSize;bi++){\n" +
            "   for(int noiy=0;noiy<${numOutputsPerKernel.y};noiy++){\n" +
            "       for(int noix=0;noix<${numOutputsPerKernel.x};noix++){\n" +
            "           int niix = clamp(noix + localWeightX + ${offset.x}, 0, ${inputSeriesSize.x - 1});\n" +
            "           int niiy = clamp(noiy + localWeightX + ${offset.y}, 0, ${inputSeriesSize.y - 1});\n" +
            "           int ni = inAttrIndex * ${inputSeriesSize.x * inputSeriesSize.y} + niiy * ${inputSeriesSize.x} + niix;\n" +
            "           int no = outAttrIndex * ${numOutputsPerKernel.x * numOutputsPerKernel.y} + noiy * ${numOutputsPerKernel.x} + noix;\n" +
            "           float inputI = getInput(bi,ni);\n" +
            "           float delta = getOutDelta(bi,no);\n" +
            "           deltaWeight += inputI * delta;" +
            "           if(gradient) {\n" +
            "               addInDelta(bi,ni,originalWeight*delta);\n" +
            "           }\n" +
            "       }\n" +
            "   }\n" +
            "}\n" +
            "setWeight(weightIndex, originalWeight + learningRate * deltaWeight);\n"
) {

    constructor(
        inputSeriesSize: Vector2i, numInputAttributes: Int, convSize: Vector2i,
        padEnds: Boolean, numOutputAttributes: Int, activation: Activation,
    ) : this(
        inputSeriesSize, numInputAttributes, convSize, padEnds, numOutputAttributes, activation,
        if (padEnds) inputSeriesSize else Vector2i(inputSeriesSize.x - convSize.x + 1, inputSeriesSize.y - convSize.y + 1),
        if (padEnds) Vector2i(-convSize.x.shr(1), -convSize.y.shr(1)) else Vector2i(0),
        convSize.x * convSize.y * numInputAttributes
    )

    override fun applyForward(network: CPUNetwork, bi: Int, no: Int) {

        val numOutputsPerKernel = (numOutputsPerKernel.x * numOutputsPerKernel.y)
        val seriesIndex = no % numOutputsPerKernel
        val seriesIndexX = seriesIndex % inputSeriesSize.x
        val seriesIndexY = seriesIndex / inputSeriesSize.x

        val outAttrIndex = no / numOutputsPerKernel

        var convSum = 0f
        for (ai in 0 until numInputAttributes) {
            for (ciy in 0 until convSize.y) {
                for (cix in 0 until convSize.x) {
                    val weightIndex = outAttrIndex * numWeightsPerKernel + ai * (convSize.x * convSize.y) + ciy * convSize.x + cix
                    var nix = seriesIndexX + offset.x + cix
                    var niy = seriesIndexY + offset.y + ciy
                    if (padEnds) nix = clamp(nix, 0, inputSeriesSize.x - 1)
                    if (padEnds) niy = clamp(niy, 0, inputSeriesSize.y - 1)

                    // else no clamping necessary
                    val ni = ai * (inputSeriesSize.x * inputSeriesSize.y) + niy * inputSeriesSize.x + nix
                    // println("$no/$niY/$ci -> $seriesIndex, $attributeIndex, $ni, $weightIndex (${network.getInput(bi, ni)} * ${network.getWeight(weightIndex)})")
                    convSum += network.getInput(bi, ni) * network.getWeight(weightIndex)
                    // println("convSum[$bi,$no] += [$niY,$ci]: ${network.getInput(bi, ni)}[$ni by $niY,$niX] * ${network.getWeight(weightIndex)}[$weightIndex]")
                }
            }
        }
        // println("[$bi,$no] = $convSum")
        network.setOutSum(bi, no, convSum)
    }

    override fun applyBackward(network: CPUNetwork, weightIndex: Int, params: LearningParams, gradient: Boolean) {

        // for all values that use this weight index,
        //  find input and output value, and add in their contribution

        // a weight is part of all "times" (1d series) and outputs,
        // but only for a single attributes

        val localWeightXY = weightIndex % (convSize.x * convSize.y)
        val localWeightX = localWeightXY % convSize.x
        val localWeightY = localWeightXY / convSize.x

        val inAttrIndex = (weightIndex / (convSize.x * convSize.y)) % numInputAttributes
        val outAttrIndex = weightIndex / numWeightsPerKernel

        val originalWeight = network.getWeight(weightIndex)
        var deltaWeight = 0f
        for (bi in 0 until network.batchSize) {
            for (noiy in 0 until numOutputsPerKernel.y) {
                for (noix in 0 until numOutputsPerKernel.x) {

                    val nix = clamp(noix + localWeightX + offset.x, 0, inputSeriesSize.x - 1)
                    val niy = clamp(noiy + localWeightY + offset.y, 0, inputSeriesSize.y - 1)
                    // if (nii !in 0 until inputSeriesSize) continue // edge case, quite literally

                    val ni = inAttrIndex * (inputSeriesSize.x * inputSeriesSize.y) + niy * inputSeriesSize.x + nix
                    val no = outAttrIndex * (numOutputsPerKernel.x * numOutputsPerKernel.y) + noiy * numOutputsPerKernel.x + noix

                    val input = network.getInput(bi, ni)
                    assertTrue(input.isFinite()) { "Input is NaN" }

                    val delta = network.getOutDelta(bi, no)
                    // println("  W[$weightIndex,$bi,$noi] from [$ni,$no] += ${params.learningRate} * $input * $delta")
                    deltaWeight += input * delta

                    if (gradient) {
                        val deltaSum = originalWeight * delta
                        network.addInDelta(bi, ni, deltaSum)
                    }
                }
            }
        }

        // println("W[$weightIndex -> $localWeightX,$inAttrIndex,$outAttrIndex, $originalWeight] += ${params.learningRate} * $deltaWeight -> ${originalWeight + params.learningRate * deltaWeight}")
        network.setWeight(weightIndex, originalWeight + params.learningRate * deltaWeight)
    }
}