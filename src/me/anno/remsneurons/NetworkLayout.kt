package me.anno.remsneurons

import me.anno.remsneurons.layers.Layer
import me.anno.utils.assertions.assertEquals

class NetworkLayout {

    var numWeights = 0
    var numNodes = 0

    val layers = ArrayList<Layer>()

    val numInputs get() = layers.firstOrNull()?.numInputs ?: 0
    val numOutputs get() = layers.lastOrNull()?.numOutputs ?: 0

    fun addLayer(layer: Layer) {
        val prevSize = layers.lastOrNull()?.numOutputs
        if (prevSize != null) assertEquals(prevSize, layer.numInputs)

        layer.inputOffset = numNodes - layer.numInputs
        layer.outputOffset = numNodes
        layer.weightOffset = numWeights

        numNodes += layer.numOutputs
        numWeights += layer.numWeights

        layers.add(layer)
    }

    companion object {

        // todo who think about convolutional network, and whether
        //     forward always uses batchSize * numOutputs ( = accumulate inputs for each output)
        //     backward always uses numInputs * numOutputs ( = for each weight)
        //  as their well-parallelizable execution sizes

    }
}