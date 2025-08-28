package me.anno.remsneurons.activations

fun interface ForwardFunc {
    fun justApply(values: FloatArray, i0: Int, i1: Int)
}