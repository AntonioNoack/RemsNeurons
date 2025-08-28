package me.anno.remsneurons.network

import me.anno.remsneurons.LearningParams
import me.anno.remsneurons.NetworkLayout
import me.anno.remsneurons.layers.Layer
import me.anno.utils.assertions.assertEquals
import me.anno.utils.assertions.assertTrue
import kotlin.math.max
import kotlin.random.Random

// tensor: just some array
abstract class Network<Tensor>(
    val networkLayout: NetworkLayout,
    val batchSize: Int,

    val inputs: Tensor,
    val weights: Tensor,
    val activated: Tensor,

    val targets: Tensor,
    val deltas: Tensor,
) {

    val numInputs = networkLayout.numInputs

    var currInputOffset = 0
    var currOutputOffset = 0
    var currWeightOffset = 0

    var currNumInputs = 0
    var currNumOutputs = 0
    var currNumWeights = 0

    var currInputs: Tensor = inputs

    fun getInIndex(bi: Int, ni: Int): Int {
        assertTrue(bi in 0 until batchSize)
        assertTrue(ni in 0 until currNumInputs)
        return bi * currNumInputs + ni + currInputOffset
    }

    fun getOutIndex(bi: Int, no: Int): Int {
        assertTrue(bi in 0 until batchSize)
        assertTrue(no in 0 until currNumOutputs)
        return bi * currNumOutputs + no + currOutputOffset
    }

    fun getCurrWeightIndex(index: Int): Int {
        assertTrue(index in 0 until currNumWeights)
        return index + currWeightOffset
    }

    abstract fun setInput(bi: Int, ni: Int, value: Float)
    abstract fun setTarget(bi: Int, no: Int, expected: Float)

    abstract fun forwardImpl(layer: Layer)
    abstract fun prepareInputs()
    abstract fun prepareTargets()
    abstract fun clearDeltas()
    abstract fun evalImpl(needsError: Boolean): Float
    abstract fun backwardImpl(learningParams: LearningParams, layer: Layer)
    abstract fun initializeWeights(rnd: Random)
    abstract fun inspectWeights(): FloatArray
    abstract fun inspectDeltas(): FloatArray
    abstract fun inspectActivated(): FloatArray
    abstract fun setWeights(weights: FloatArray)

    fun bindLayer(layer: Layer) {
        val batchSize = batchSize
        currInputs = if (layer.inputOffset < 0) inputs else activated
        currInputOffset = max(layer.inputOffset * batchSize, 0)
        currOutputOffset = layer.outputOffset * batchSize
        currNumInputs = layer.numInputs
        currNumOutputs = layer.numOutputs
        currWeightOffset = layer.weightOffset
        currNumWeights = layer.numWeights

        if (layer.inputOffset >= 0) {
            assertEquals(currOutputOffset, currInputOffset + batchSize * layer.numInputs)
        } else {
            assertEquals(currOutputOffset, 0)
        }
    }

    /**
     * aka runForward
     * */
    fun predict() {
        prepareInputs()
        val layers = networkLayout.layers
        for (i in layers.indices) {
            val layer = layers[i]
            bindLayer(layer)
            forwardImpl(layer)
        }
    }

    var ctr = 0

    /**
     * aka runForward + runBackward
     * */
    fun learn(learningParams: LearningParams, needsError: Boolean): Float {
        val print = false && ctr++ < 3
        predict()

        if (print) println("    Values[$ctr]: ${inspectActivated().toList()}")

        clearDeltas()
        prepareTargets()

        val error = runEvaluation(needsError)
        if (print) println("    Deltas0[$ctr]: ${inspectDeltas().toList()}")

        runBackward(learningParams)

        if (print && networkLayout.layers.size > 1) println("    Deltas1[$ctr]: ${inspectDeltas().toList()}")
        if (print) println("    Weights[$ctr]: ${inspectWeights().toList()}")

        return error
    }

    private fun runBackward(learningParams: LearningParams) {
        val layers = networkLayout.layers
        for (i in layers.lastIndex downTo 0) {
            val layer = layers[i]
            bindLayer(layer)
            backwardImpl(learningParams, layer)
        }
    }

    private fun runEvaluation(needsError: Boolean): Float {
        val lastLayer = networkLayout.layers.last()
        bindLayer(lastLayer)
        return evalImpl(needsError)
    }

}