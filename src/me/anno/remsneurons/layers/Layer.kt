package me.anno.remsneurons.layers

import me.anno.remsneurons.LearningParams
import me.anno.remsneurons.activations.Activation
import me.anno.remsneurons.network.CPUNetwork

// each layer has N inputs and M outputs
// todo layers may be splits for recurrent networks...
// todo inputs and outputs usually form a tree, but could form a circle for text-processing networks
// all computation should happen in compute/graphics shaders
abstract class Layer(
    val numInputs: Int,
    val numWeights: Int,
    val numOutputs: Int,
    val numInputsPerNode: Int,
    val activation: Activation,
    val forwardGLSL: String,
    val backwardGLSL: String,
) {

    var inputOffset = 0
    var outputOffset = 0
    var weightOffset = 0

    abstract fun applyForward(network: CPUNetwork, bi: Int, no: Int)
    abstract fun applyBackward(network: CPUNetwork, weightIndex: Int, params: LearningParams, gradient: Boolean)
}