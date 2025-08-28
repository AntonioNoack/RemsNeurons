package me.anno.remsneurons

import me.anno.maths.Maths.TAUf
import me.anno.remsneurons.NetworkGradientTest.Companion.forEachNetwork
import me.anno.remsneurons.NetworkGradientTest.Companion.setupGraphics
import me.anno.remsneurons.activations.Identity
import me.anno.remsneurons.activations.Sigmoid
import me.anno.remsneurons.layers.ConvolutionalLayer2d
import me.anno.remsneurons.layers.FullyConnectedLayer
import me.anno.remsneurons.network.CPUNetwork
import me.anno.utils.assertions.assertEquals
import org.joml.Vector2i
import org.junit.jupiter.api.Test
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

class Conv1dVia2dTest {
    @Test
    fun testResponseByHandNoPadding() {
        val layout = NetworkLayout()
        layout.addLayer(
            ConvolutionalLayer2d(
                Vector2i(10, 1), 1, Vector2i(3, 1),
                false, 1, Identity
            )
        )

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
        layout.addLayer(
            ConvolutionalLayer2d(
                Vector2i(10, 1), 2, Vector2i(3, 1),
                true, 2, Identity
            )
        )

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
        layout.addLayer(
            ConvolutionalLayer2d(
                Vector2i(10, 1), 2, Vector2i(3, 1),
                true, 2, Identity
            )
        )

        assertEquals(20, layout.numInputs)
        assertEquals(20, layout.numOutputs)
        assertEquals(12, layout.numWeights)

        val batchSize = 1
        forEachNetwork(layout, batchSize) { network ->
            val rnd = Random(1543)
            network.initializeWeights(rnd)

            println("Initial weights: ${network.inspectWeights().toList()}")

            val params = LearningParams(1f / batchSize, false)
            val n = 10
            repeat(n) { it ->

                // clear state
                for (i in 0 until 20) {
                    network.setInput(0, i, 0f)
                    network.setTarget(0, i, 0f)
                }

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

                if (true) {

                    val print = it % 2 == 0 || it == n - 1
                    val error = network.learn(params, print)
                    if (print) println("[$batchSize,$it] Error: $error, Weights: ${network.inspectWeights().toList()}")

                } else {

                    val error = network.learn(params, true)

                    println("[$batchSize,$it] Values: ${network.inspectActivated().toList()}")
                    println("[$batchSize,$it] Deltas: ${network.inspectDeltas().toList()}")
                    println("[$batchSize,$it] Error: $error, Weights: ${network.inspectWeights().toList()}")
                    println()
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

    @Test
    fun testLearnFrequencyDetectorDeep() {

        setupGraphics()

        var len = 64

        val layout = NetworkLayout()
        layout.addLayer(ConvolutionalLayer2d(Vector2i(len, 1), 1, Vector2i(7, 1), false, 5, Sigmoid)); len -= 6
        layout.addLayer(ConvolutionalLayer2d(Vector2i(len, 1), 5, Vector2i(7, 1), false, 5, Sigmoid)); len -= 6
        layout.addLayer(FullyConnectedLayer(len * 5, 10, Sigmoid))
        layout.addLayer(FullyConnectedLayer(10, 2, Identity))

        val batchSize = 8
        forEachNetwork(layout, batchSize) { network ->
            val rnd = Random(1543)
            network.initializeWeights(rnd)

            val params = LearningParams(1f / batchSize, false)
            val n = 500
            val bn = n / 10
            var lastError = Float.POSITIVE_INFINITY
            repeat(n) { it ->

                for (bi in 0 until batchSize) {

                    val frequency = 0.1f
                    val phase = rnd.nextFloat() * TAUf

                    network.setTarget(bi, 0, cos(phase))
                    network.setTarget(bi, 1, sin(phase))

                    for (ni in 0 until len) {
                        network.setInput(bi, ni, sin(phase + ni * frequency))
                    }
                }

                val print = it % bn == 0 || it == n - 1
                val error = network.learn(params, print)
                if (print) println("[$batchSize,$it] Error: $error")

                lastError = error

            }

            assertEquals(0f, lastError, 0.05f)
        }
    }
}