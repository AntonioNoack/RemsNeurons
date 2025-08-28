package me.anno.remsneurons.activations

import kotlin.math.exp

object Sigmoid : Activation(
    { values, i0, i1 ->
        for (i in i0 until i1) {
            values[i] = 1f / (1f + exp(-values[i]))
        }
    }, { activated, deltas, i0, i1 ->
        for (i in i0 until i1) {
            val sigmoid = activated[i]
            deltas[i] *= sigmoid * (1f - sigmoid)
        }
    }, "activated[i] = 1.0 / (1.0 + exp(-activated[i]));\n",
    "float value = activated[i]; deltas[i] *= value * (1.0 - value);\n", false
)