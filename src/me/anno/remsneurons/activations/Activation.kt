package me.anno.remsneurons.activations

open class Activation(
    val forwardKotlin: ForwardFunc,
    val backwardKotlin: BackwardFunc,
    val forwardGLSL: String,
    val backwardGLSL: String,
    val hasInterdependencies: Boolean,
)