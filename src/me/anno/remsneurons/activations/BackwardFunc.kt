package me.anno.remsneurons.activations

fun interface BackwardFunc {
    fun mulApply(activated: FloatArray, deltas: FloatArray, i0: Int, i1: Int)
}