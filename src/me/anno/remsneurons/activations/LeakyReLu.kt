package me.anno.remsneurons.activations

open class LeakyReLu(val leak: Float = 0.05f) : Activation(
    { values, i0, i1 ->
        for (i in i0 until i1) {
            val value = values[i]
            values[i] = if (value > 0f) value else value * leak
        }
    }, { activated, deltas, i0, i1 ->
        for (i in i0 until i1) {
            val value = activated[i]
            if (value < 0f) deltas[i] *= leak
        }
    }, "float value = sums[i]; activated[i] = max(value, value * $leak);\n",
    "if(activated[i] < 0.0) deltas[i] *= $leak;\n", false
)