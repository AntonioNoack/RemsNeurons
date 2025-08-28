package me.anno.remsneurons.activations

import kotlin.math.exp
import kotlin.math.min

// correct only if used with cross-entropy loss
object SoftMax : Activation(
    { values, i0, i1 ->
        val maxValue = 1e300
        var sum = 0.0
        for (i in i0 until i1) {
            sum += min(exp(values[i].toDouble()), maxValue)
        }
        val factor = 1.0 / sum
        for (i in i0 until i1) {
            values[i] = (min(exp(values[i].toDouble()), maxValue) * factor).toFloat()
        }
    }, { activated, deltas, i0, i1 ->
        // is this correct to do nothing???
    }, "" +
            "float sum = 0.0;" +
            "for(int i=i0;i<i1;i++) {\n" +
            "   sum += min(exp(activated[i]),1e30);\n" +
            "}\n" +
            "float factor = 1.0 / sum;\n" +
            "for(int i=i0;i<i1;i++) {\n" +
            "   activated[i] = min(exp(activated[i]),1e30) * factor;\n" +
            "}\n",
    "", true
)