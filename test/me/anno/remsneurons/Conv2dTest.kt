package me.anno.remsneurons

import me.anno.maths.Maths.TAUf
import me.anno.remsneurons.NetworkGradientTest.Companion.forEachNetwork
import me.anno.remsneurons.NetworkGradientTest.Companion.setupGraphics
import me.anno.remsneurons.activations.Identity
import me.anno.remsneurons.activations.Sigmoid
import me.anno.remsneurons.layers.ConvolutionalLayer2d
import me.anno.remsneurons.layers.FullyConnectedLayer
import me.anno.remsneurons.network.CPUNetwork
import me.anno.remsneurons.network.GPUNetwork
import me.anno.utils.assertions.assertEquals
import org.joml.Vector2i
import org.junit.jupiter.api.Test
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

class Conv2dTest {

    @Test
    fun testResponseByHandNoPadding() {
        val layout = NetworkLayout()
        layout.addLayer(
            ConvolutionalLayer2d(
                Vector2i(7, 7), 1, Vector2i(3, 3),
                false, 1, Identity
            )
        )

        assertEquals(7 * 7, layout.numInputs)
        assertEquals(5 * 5, layout.numOutputs)
        assertEquals(3 * 3, layout.numWeights)

        val batchSize = 1
        val network = CPUNetwork(layout, batchSize)
        network.weights[0] = 1f
        network.weights[1] = 2f
        network.weights[2] = 3f
        network.weights[3] = 4f
        network.weights[4] = 5f
        network.weights[5] = 6f
        network.weights[6] = 7f
        network.weights[7] = 8f
        network.weights[8] = 9f

        val signal = FloatArray(7 * 7)
        signal[4 * 7 + 3] = 1f

        val expectedOutput = FloatArray(5 * 5)
        expectedOutput[2 * 5 + 1] = 9f
        expectedOutput[2 * 5 + 2] = 8f
        expectedOutput[2 * 5 + 3] = 7f

        expectedOutput[3 * 5 + 1] = 6f
        expectedOutput[3 * 5 + 2] = 5f
        expectedOutput[3 * 5 + 3] = 4f

        expectedOutput[4 * 5 + 1] = 3f
        expectedOutput[4 * 5 + 2] = 2f
        expectedOutput[4 * 5 + 3] = 1f

        for (ni in signal.indices) {
            network.setInput(0, ni, signal[ni])
        }

        network.predict()

        assertEquals(expectedOutput, network.activated)

    }

    @Test
    fun testLearnEdgeDetector() {

        setupGraphics()

        val layout = NetworkLayout()
        layout.addLayer(
            ConvolutionalLayer2d(
                Vector2i(10, 10), 2, Vector2i(3, 3),
                true, 2, Identity
            )
        )

        assertEquals(200, layout.numInputs)
        assertEquals(200, layout.numOutputs)
        assertEquals(36, layout.numWeights)

        val batchSize = 1
        forEachNetwork(layout, batchSize) { network ->
            val rnd = Random(1543)
            network.initializeWeights(rnd)

            val params = LearningParams(1f / batchSize, false)
            val n = 10
            repeat(n) { it ->

                // clear state
                for (i in 0 until 200) {
                    network.setInput(0, i, 0f)
                    network.setTarget(0, i, 0f)
                }

                val x0 = rnd.nextInt(1, 9)
                val x1 = rnd.nextInt(1, 9)
                val y0 = rnd.nextInt(1, 9)
                val y1 = rnd.nextInt(1, 9)
                val v0 = rnd.nextFloat()
                val v1 = rnd.nextFloat()

                network.setInput(0, y0 * 10 + x0, v0)
                network.setInput(0, 100 + y1 * 10 + x1, v1)

                network.setTarget(0, y0 * 10 + x0 - 10, -v0)
                network.setTarget(0, y0 * 10 + x0 - 1, -v0)
                network.setTarget(0, y0 * 10 + x0 + 0, 2f * v0)
                network.setTarget(0, y0 * 10 + x0 + 1, -v0)
                network.setTarget(0, y0 * 10 + x0 + 10, -v0)

                network.setTarget(0, 100 + y1 * 10 + x1 - 1, +v1)
                network.setTarget(0, 100 + y1 * 10 + x1 + 0, -v1)

                val print = it % 2 == 0 || it == n - 1
                val error = network.learn(params, print)
                if (print) println("[$batchSize,$it] Error: $error")

            }

            val weights = network.inspectWeights()
            println(weights.toList())
            assertEquals(+0f, weights[0], 0.1f)
            assertEquals(-1f, weights[1], 0.1f)
            assertEquals(+0f, weights[2], 0.1f)
            assertEquals(-1f, weights[3], 0.1f)
            assertEquals(+2f, weights[4], 0.1f)
            assertEquals(-1f, weights[5], 0.1f)
            assertEquals(+0f, weights[6], 0.1f)
            assertEquals(-1f, weights[7], 0.1f)
            assertEquals(+0f, weights[8], 0.1f)

            for (i in 9 * 1 until 9 * 3) {
                assertEquals(+0f, weights[i], 0.1f)
            }

            assertEquals(+0f, weights[27], 0.1f)
            assertEquals(+0f, weights[28], 0.1f)
            assertEquals(+0f, weights[29], 0.1f)
            assertEquals(+0f, weights[30], 0.1f)
            assertEquals(-1f, weights[31], 0.1f)
            assertEquals(+1f, weights[32], 0.1f)
            assertEquals(+0f, weights[33], 0.1f)
            assertEquals(+0f, weights[34], 0.1f)
            assertEquals(+0f, weights[35], 0.1f)
        }
    }

    @Test
    fun testLearnPhaseDetectorDeep() {

        setupGraphics()

        var lenX = 32
        var lenY = 32

        val layout = NetworkLayout()
        layout.addLayer(ConvolutionalLayer2d(Vector2i(lenX, lenY), 1, Vector2i(7, 7), false, 20, Sigmoid)); lenX -= 6; lenY -= 6
        layout.addLayer(ConvolutionalLayer2d(Vector2i(lenX, lenY), 20, Vector2i(7, 7), false, 20, Sigmoid)); lenX -= 6; lenY -= 6
        layout.addLayer(FullyConnectedLayer(lenX * lenY * 20, 50, Sigmoid))
        layout.addLayer(FullyConnectedLayer(50, 10, Sigmoid))
        layout.addLayer(FullyConnectedLayer(10, 4, Identity))

        val batchSize = 8
        val network = GPUNetwork(layout, batchSize)
        val rnd = Random(1543)
        network.initializeWeights(rnd)

        // todo bug: this is random even though it definitely should not be...
        //  is our atomic-add the culprit? or is it a synchronization issue?

        val params = LearningParams(0.1f / batchSize, false)
        val n = 2500
        val bn = 100
        var lastError = Float.POSITIVE_INFINITY
        repeat(n) { it ->

            for (bi in 0 until batchSize) {

                val frequency = TAUf / lenX
                val phaseX = rnd.nextFloat() * TAUf
                val phaseY = rnd.nextFloat() * TAUf

                network.setTarget(bi, 0, cos(phaseX))
                network.setTarget(bi, 1, sin(phaseX))
                network.setTarget(bi, 2, cos(phaseY))
                network.setTarget(bi, 3, sin(phaseY))

                for (ny in 0 until lenY) {
                    for (nx in 0 until lenX) {
                        network.setInput(
                            bi, ny * lenX + nx,
                            sin(phaseX + nx * frequency) +
                                    sin(phaseY + ny * frequency)
                        )
                    }
                }
            }

            val print = it % bn == 0 || it == n - 1
            val error = network.learn(params, print)
            if (print) println("[$batchSize,$it] Error: $error")

            lastError = error

        }

        assertEquals(0f, lastError, 0.20f)
    }
}