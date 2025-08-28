package me.anno.remsneurons.layers

import me.anno.remsneurons.LearningParams
import me.anno.remsneurons.activations.Activation
import me.anno.remsneurons.network.CPUNetwork
import me.anno.utils.assertions.assertTrue

// todo implement convolutional layer
//  and we might need multidimensional inputs...
//   = our layers just interpret them as such

class FullyConnectedLayer(
    numInputs: Int, numOutputs: Int,
    activation: Activation,
) : Layer(
    numInputs, numInputs * numOutputs, numOutputs,
    numInputs, activation,
    "" +
            "float sum = 0.0;\n" +
            "for(int ni=0;ni<numInputs;ni++){\n" +
            "   int weightIndex = no*$numInputs+ni;\n" +
            "   sum += getWeight(weightIndex) * getInput(bi,ni);\n" +
            "}\n" +
            "setOutSum(bi,no,sum);\n",
    "" +
            "int ni = weightIndex % $numInputs;\n" +
            "int no = weightIndex / $numInputs;\n" +
            "float originalWeight = getWeight(weightIndex);\n" +
            "float deltaWeight = 0.0;\n" +
            "for(int bi=0;bi<batchSize;bi++){\n" +
            "   float inputI = getInput(bi,ni);\n" +
            "   float delta = getOutDelta(bi,no);\n" +
            "   deltaWeight += inputI * delta;\n" +
            "   if (gradient) {\n" +
            "       float deltaSum = originalWeight * delta;\n" +
            "       addInDelta(bi,ni,deltaSum);\n" +
            "   }\n" +
            "}\n" +
            "setWeight(weightIndex,originalWeight + learningRate * deltaWeight);\n"
) {

    override fun applyForward(network: CPUNetwork, bi: Int, no: Int) {
        var sum = 0f
        for (ni in 0 until numInputs) {
            val weightIndex = no * numInputs + ni
            sum += network.getWeight(weightIndex) * network.getInput(bi, ni)
        }
        network.setOutSum(bi, no, sum)
    }

    override fun applyBackward(network: CPUNetwork, weightIndex: Int, params: LearningParams, gradient: Boolean) {

        val ni = weightIndex % numInputs
        val no = weightIndex / numInputs

        val weightIndex = no * numInputs + ni
        val originalWeight = network.getWeight(weightIndex)
        var deltaWeight = 0f
        // batch must be accumulated in the middle to not change weights!
        for (bi in 0 until network.batchSize) {

            val input = network.getInput(bi, ni)
            assertTrue(input.isFinite()) { "Input is NaN" }

            val delta = network.getOutDelta(bi, no)
            // println("W[$weightIndex] += ${params.learningRate} * $input * $delta")
            deltaWeight += input * delta

            if (gradient) {
                val deltaSum = originalWeight * delta
                network.addInDelta(bi, ni, deltaSum)
            }
        }
        network.setWeight(weightIndex, originalWeight + params.learningRate * deltaWeight)
    }
}