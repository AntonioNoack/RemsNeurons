package me.anno.remsneurons.activations

object Identity : Activation(
    { values, i0, i1 ->
        // nothing to do
    }, { activated, deltas, i0, i1 ->
        // nothing to do
    }, "", "", false
)