package me.anno.remsneurons.layers

import me.anno.maths.Maths.clamp
import me.anno.remsneurons.LearningParams
import me.anno.remsneurons.activations.Activation
import me.anno.remsneurons.network.CPUNetwork
import me.anno.utils.assertions.assertTrue
import me.anno.utils.types.Strings.iff

class ConvolutionalLayer1d private constructor(
    val inputSeriesSize: Int,

    val numInputAttributes: Int,

    val convSize: Int,
    val padEnds: Boolean,

    val numOutputAttributes: Int,

    activation: Activation,

    val numOutputsPerKernel: Int,

    val offset: Int,
    val numWeightsPerKernel: Int


) : Layer(
    inputSeriesSize * numInputAttributes, convSize * numInputAttributes * numOutputAttributes,
    (if (padEnds) inputSeriesSize else inputSeriesSize - convSize + 1) * numOutputAttributes,
    numInputAttributes * convSize,
    activation,
    "" +
            "int seriesIndex = no % $numOutputsPerKernel;\n" +
            "int outAttrIndex = no / $numOutputsPerKernel;\n" +
            "float convSum = 0.0;\n" +
            "for(int niY=0;niY<$numInputAttributes;niY++){\n" +
            "   for(int ci=0;ci<$convSize;ci++){\n" +
            "       int weightIndex = outAttrIndex * $numWeightsPerKernel + niY * $convSize + ci;\n" +
            "       int niX = seriesIndex + $offset + ci;\n" +
            "       niX = clamp(niX,0,${inputSeriesSize - 1});\n".iff(padEnds) +
            "       int ni = niY * $inputSeriesSize + niX;\n" +
            "       convSum += getInput(bi,ni) * getWeight(weightIndex);\n" +
            "   }\n" +
            "}\n" +
            "setOutSum(bi,no,convSum);\n",
    "" +
            "int localWeightX = weightIndex % $convSize;\n" +
            "int inAttrIndex = (weightIndex / $convSize) % $numInputAttributes;\n" +
            "int outAttrIndex = weightIndex / $numWeightsPerKernel;\n" +
            "float originalWeight = getWeight(weightIndex);\n" +
            "float deltaWeight = 0.0;\n" +
            "for(int bi=0;bi<batchSize;bi++){\n" +
            "   for(int noi=0;noi<$numOutputsPerKernel;noi++){\n" +
            "       int nii = clamp(noi + localWeightX + $offset, 0, ${inputSeriesSize - 1});\n" +
            "       int ni = inAttrIndex * $inputSeriesSize + nii;\n" +
            "       int no = outAttrIndex * $numOutputsPerKernel + noi;\n" +
            "       float inputI = getInput(bi,ni);\n" +
            "       float delta = getOutDelta(bi,no);\n" +
            "       deltaWeight += inputI * delta;" +
            "       if(gradient) {\n" +
            "           addInDelta(bi,ni,originalWeight*delta);\n" +
            "       }\n" +
            "   }\n" +
            "}\n" +
            "setWeight(weightIndex, originalWeight + learningRate * deltaWeight);\n"
) {

    constructor(
        inputSeriesSize: Int, numInputAttributes: Int, convSize: Int,
        padEnds: Boolean, numOutputAttributes: Int, activation: Activation,
    ) : this(
        inputSeriesSize, numInputAttributes, convSize, padEnds, numOutputAttributes, activation,
        if (padEnds) inputSeriesSize else inputSeriesSize - convSize + 1,
        if (padEnds) -convSize.shr(1) else 0,
        convSize * numInputAttributes
    )

    override fun applyForward(network: CPUNetwork, bi: Int, no: Int) {

        val seriesIndex = no % numOutputsPerKernel
        val outAttrIndex = no / numOutputsPerKernel

        var convSum = 0f
        for (niY in 0 until numInputAttributes) {
            for (ci in 0 until convSize) {
                val weightIndex = outAttrIndex * numWeightsPerKernel + niY * convSize + ci
                var niX = seriesIndex + offset + ci
                if (padEnds) niX = clamp(niX, 0, inputSeriesSize - 1)
                // else no clamping necessary
                val ni = niY * inputSeriesSize + niX
                // println("$no/$niY/$ci -> $seriesIndex, $attributeIndex, $ni, $weightIndex (${network.getInput(bi, ni)} * ${network.getWeight(weightIndex)})")
                convSum += network.getInput(bi, ni) * network.getWeight(weightIndex)
                // println("convSum[$bi,$no] += [$niY,$ci]: ${network.getInput(bi, ni)}[$ni by $niY,$niX] * ${network.getWeight(weightIndex)}[$weightIndex]")
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

        val localWeightX = weightIndex % convSize
        val inAttrIndex = (weightIndex / convSize) % numInputAttributes
        val outAttrIndex = weightIndex / numWeightsPerKernel

        val originalWeight = network.getWeight(weightIndex)
        var deltaWeight = 0f
        for (bi in 0 until network.batchSize) {
            for (noi in 0 until numOutputsPerKernel) {
                val nii = clamp(noi + localWeightX + offset, 0, inputSeriesSize - 1) // todo is this offset correct here?
                // if (nii !in 0 until inputSeriesSize) continue // edge case, quite literally

                val ni = inAttrIndex * inputSeriesSize + nii
                val no = outAttrIndex * numOutputsPerKernel + noi

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

        // println("W[$weightIndex -> $localWeightX,$inAttrIndex,$outAttrIndex, $originalWeight] += ${params.learningRate} * $deltaWeight -> ${originalWeight + params.learningRate * deltaWeight}")
        network.setWeight(weightIndex, originalWeight + params.learningRate * deltaWeight)
    }
}