package me.anno.remsneurons

import me.anno.remsneurons.NetworkGradientTest.Companion.forEachNetwork
import me.anno.remsneurons.NetworkGradientTest.Companion.setupGraphics
import me.anno.remsneurons.activations.Identity
import me.anno.remsneurons.layers.ConvolutionalLayer1d
import me.anno.remsneurons.network.CPUNetwork
import me.anno.utils.assertions.assertEquals
import org.junit.jupiter.api.Test
import kotlin.random.Random

class Conv1dTest {
    @Test
    fun testResponseByHandNoPadding() {
        val layout = NetworkLayout()
        layout.addLayer(ConvolutionalLayer1d(10, 1, 3, false, 1, Identity))

        assertEquals(10, layout.numInputs)
        assertEquals(8, layout.numOutputs)
        assertEquals(3, layout.numWeights)

        val batchSize = 1
        val network = CPUNetwork(layout, batchSize)
        network.weights[0] = -1f
        network.weights[1] = +2f
        network.weights[2] = -1f

        val signal = FloatArray(10)
        signal[5] = 3f

        val expectedOutput = FloatArray(8)
        expectedOutput[3] = -3f
        expectedOutput[4] = +6f
        expectedOutput[5] = -3f

        for (ni in 0 until 10) network.setInput(0, ni, signal[ni])
        network.predict()

        assertEquals(
            expectedOutput,
            network.activated
        )

    }

    @Test
    fun testResponseByHandWithPadding() {

        setupGraphics()

        val layout = NetworkLayout()
        layout.addLayer(ConvolutionalLayer1d(10, 2, 3, true, 2, Identity))

        assertEquals(20, layout.numInputs)
        assertEquals(20, layout.numOutputs)
        assertEquals(12, layout.numWeights)

        val batchSize = 1
        forEachNetwork(layout, batchSize) { network ->
            val weights = network.inspectWeights()
            weights.fill(0f)
            weights[0] = -1f
            weights[1] = +2f
            weights[2] = -1f

            weights[9] = +1f
            weights[10] = -1f
            network.setWeights(weights)

            println("Initial Weights: " + network.inspectWeights().toList())

            val signal = FloatArray(20)
            signal[5] = 3f
            signal[15] = 7f

            val expectedOutput = FloatArray(20)
            expectedOutput[4] = -3f
            expectedOutput[5] = +6f
            expectedOutput[6] = -3f
            expectedOutput[15] = -7f
            expectedOutput[16] = +7f

            for (ni in signal.indices) network.setInput(0, ni, signal[ni])
            network.predict()

            assertEquals(expectedOutput, network.inspectActivated())
        }
    }

    @Test
    fun testLearnEdgeDetector() {

        setupGraphics()

        val layout = NetworkLayout()
        layout.addLayer(ConvolutionalLayer1d(10, 2, 3, true, 2, Identity))

        assertEquals(20, layout.numInputs)
        assertEquals(20, layout.numOutputs)
        assertEquals(12, layout.numWeights)

        val batchSize = 1
        forEachNetwork(layout, batchSize) { network ->
            val rnd = Random(1543)
            network.initializeWeights(rnd)

            println("Initial weights: ${network.inspectWeights().toList()}")

            val params = LearningParams(1f / batchSize, false)
            repeat(10) { it ->

                val i0 = rnd.nextInt(1, 9)
                val i1 = rnd.nextInt(10, 19)
                val v0 = rnd.nextFloat()
                val v1 = rnd.nextFloat()

                network.setInput(0, i0, v0)
                network.setInput(0, i1, v1)

                network.setTarget(0, i0 - 1, -v0)
                network.setTarget(0, i0 + 0, 2f * v0)
                network.setTarget(0, i0 + 1, -v0)

                network.setTarget(0, i1 - 1, +v1)
                network.setTarget(0, i1 + 0, -v1)

                val error = network.learn(params, true)

                println("[$batchSize,$it] Deltas: ${network.inspectDeltas().toList()}")
                println("[$batchSize,$it] Error: $error, Weights: ${network.inspectWeights().toList()}")

                // clear state
                for (i in 0 until 20) {
                    network.setInput(0, i, 0f)
                    network.setTarget(0, i, 0f)
                }
            }

            val weights = network.inspectWeights()
            assertEquals(-1f, weights[0], 0.1f)
            assertEquals(+2f, weights[1], 0.1f)
            assertEquals(-1f, weights[2], 0.1f)
            assertEquals(+0f, weights[3], 0.1f)
            assertEquals(+0f, weights[4], 0.1f)
            assertEquals(+0f, weights[5], 0.1f)
            assertEquals(+0f, weights[6], 0.1f)
            assertEquals(+0f, weights[7], 0.1f)
            assertEquals(+0f, weights[8], 0.1f)
            assertEquals(+0f, weights[9], 0.1f)
            assertEquals(-1f, weights[10], 0.1f)
            assertEquals(+1f, weights[11], 0.1f)
        }
    }
}