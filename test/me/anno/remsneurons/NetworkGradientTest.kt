package me.anno.remsneurons

import me.anno.jvm.HiddenOpenGLContext
import me.anno.maths.Maths.dtTo01
import me.anno.maths.Maths.pow
import me.anno.maths.Maths.sq
import me.anno.remsneurons.activations.Identity
import me.anno.remsneurons.activations.Sigmoid
import me.anno.remsneurons.layers.FullyConnectedLayer
import me.anno.remsneurons.network.CPUNetwork
import me.anno.remsneurons.network.GPUNetwork
import me.anno.remsneurons.network.Network
import me.anno.utils.assertions.assertEquals
import me.anno.utils.assertions.assertLessThan
import me.anno.utils.assertions.assertLessThanEquals
import me.anno.utils.assertions.assertTrue
import org.apache.logging.log4j.LogManager
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import kotlin.math.abs
import kotlin.math.sin
import kotlin.random.Random

class NetworkGradientTest {
    companion object {

        fun forEachNetwork(layout: NetworkLayout, batchSize: Int, callback: (network: Network<*>) -> Unit) {
            println("\n[CPU, x$batchSize], Layers: ${layout.layers.map { "@(${it.inputOffset}->${it.outputOffset} M${it.weightOffset})" }}")
            callback(CPUNetwork(layout, batchSize))
            println("\n[GPU, x$batchSize]")
            callback(GPUNetwork(layout, batchSize))
        }

        fun setupGraphics() {
            LogManager.disableInfoLogs("LWJGLDebugCallback,GFX,HiddenOpenGLContext,Saveable,DefaultConfig,ComputeShaderStats,Threads")
            HiddenOpenGLContext.createOpenGL()
        }

    }

    @BeforeEach
    fun init() {
        setupGraphics()
    }

    @Test
    fun testScalingNetwork() {

        val layout = NetworkLayout()
        layout.addLayer(FullyConnectedLayer(1, 1, Identity))

        forEachNetwork(layout, 1) { network ->
            val rnd = Random(1655)
            network.initializeWeights(rnd)

            val params = LearningParams(10f, false)

            val factor = 5f
            var lastError = Float.POSITIVE_INFINITY

            repeat(50) {
                val expected = rnd.nextFloat()
                network.setInput(0, 0, expected / factor)
                network.setTarget(0, 0, expected)

                val print = it == 0 || it % 25 == 24
                val error = network.learn(params, print)
                val weights = network.inspectWeights()
                if (print) println("[$it] Error($expected): |$error|, Weights: ${weights.toList()}")

                val theoreticalError = abs(factor - weights[0])
                assertLessThanEquals(theoreticalError, lastError)
                lastError = theoreticalError
            }

            val weights = network.inspectWeights()
            assertEquals(factor, weights[0], 0.05f)
        }
    }

    @Test
    fun testScalingNetworkBatching() {
        val layout = NetworkLayout()
        layout.addLayer(FullyConnectedLayer(1, 1, Identity))

        val batchSize = 16
        forEachNetwork(layout, batchSize) { network ->

            val rnd = Random(1655)
            val network = CPUNetwork(layout, batchSize)
            network.initializeWeights(rnd)

            assertEquals(1, layout.numInputs)
            assertEquals(1, layout.numWeights)
            assertEquals(1, layout.numNodes)
            assertEquals(1, layout.numOutputs)

            val params = LearningParams(20f / batchSize, false)

            val factor = 5f
            var lastError = Float.POSITIVE_INFINITY

            repeat(30) {
                for (bi in 0 until batchSize) {
                    val expected = rnd.nextFloat()
                    network.setInput(bi, 0, expected / factor)
                    network.setTarget(bi, 0, expected)
                }

                val print = it == 0 || it % 5 == 4
                val error = network.learn(params, print)
                val weights = network.inspectWeights()
                if (print) println("[$batchSize,$it] Error: $error, Weights: ${weights.toList()}")

                val theoreticalError = abs(factor - weights[0])
                assertLessThanEquals(theoreticalError, lastError)
                lastError = theoreticalError
            }

            val weights = network.inspectWeights()
            println(weights[0])
            assertEquals(factor, weights[0], 0.05f)
        }
    }

    @Test
    fun testFindCorrectInputNetwork() {
        val layout = NetworkLayout()
        layout.addLayer(FullyConnectedLayer(3, 1, Identity))

        assertEquals(3, layout.numInputs)
        assertEquals(3, layout.numWeights)
        assertEquals(1, layout.numNodes)
        assertEquals(1, layout.numOutputs)

        val batchSize = 16
        forEachNetwork(layout, batchSize) { network ->
            val rnd = Random(1655)
            network.initializeWeights(rnd)

            val params = LearningParams(0.5f / batchSize, false)

            var lastError = Float.POSITIVE_INFINITY
            repeat(50) {

                // define training
                for (bi in 0 until batchSize) {
                    val expected = rnd.nextFloat() * 2f - 1f
                    network.setInput(bi, 0, expected)
                    network.setInput(bi, 1, rnd.nextFloat() * 2f - 1f)
                    network.setInput(bi, 2, rnd.nextFloat() * 2f - 1f)
                    network.setTarget(bi, 0, expected)
                }

                val print = it == 0 || it % 25 == 24
                val error = network.learn(params, print)
                val weights = network.inspectWeights()
                if (print) println("[$batchSize,$it] Error: $error, Weights: ${weights.toList()}")

                val theoreticalError = sq(1f - weights[0]) + sq(weights[1]) + sq(weights[2])
                assertLessThan(theoreticalError, lastError)
                lastError = theoreticalError
            }

            val weights = network.inspectWeights()
            println(weights[0])
            assertEquals(1f, weights[0], 0.2f)
            assertEquals(0f, weights[1], 0.2f)
            assertEquals(0f, weights[2], 0.2f)
        }
    }

    @Test
    fun testDeepNetworkGradients() {
        val layout = NetworkLayout()
        layout.addLayer(FullyConnectedLayer(1, 1, Identity))
        layout.addLayer(FullyConnectedLayer(1, 1, Identity))
        layout.addLayer(FullyConnectedLayer(1, 1, Identity))
        layout.addLayer(FullyConnectedLayer(1, 1, Identity))

        for (batchSize in listOf(1, 2, 3, 16)) {
            forEachNetwork(layout, batchSize) { network ->
                network.initializeWeights(Random(1564))

                val params = LearningParams(1f / batchSize, false)
                repeat(10) {

                    // define training
                    for (bi in 0 until batchSize) {
                        network.setInput(bi, 0, 1f)
                        network.setTarget(bi, 0, 0.5f)
                    }

                    val weights = network.inspectWeights()
                    val error = network.learn(params, true)
                    println("[$batchSize,$it] Error: $error, Weights: ${weights.toList()}")
                    assertTrue(error > 0f, "Something is wrong :(")
                }

                val weights = network.inspectWeights()
                val product = weights.reduce { a, b -> a * b }
                assertEquals(0.5f, product, 0.1f)
            }
        }
    }

    @Test
    fun testDeepSigmoids() {
        val layout = NetworkLayout()
        layout.addLayer(FullyConnectedLayer(3, 3, Sigmoid))
        layout.addLayer(FullyConnectedLayer(3, 3, Sigmoid))
        layout.addLayer(FullyConnectedLayer(3, 3, Sigmoid))
        layout.addLayer(FullyConnectedLayer(3, 1, Identity))

        for (batchSize in listOf(100)) {
            forEachNetwork(layout, batchSize) { network ->
                val rnd = Random(1562)
                network.initializeWeights(rnd)

                val params = LearningParams(1f / batchSize, false)

                val n = 10000
                val dn = n / 20
                val falloff = pow(0.1f, 1f / n)
                var lastError = Float.POSITIVE_INFINITY
                repeat(n) {

                    // define training
                    for (bi in 0 until batchSize) {
                        val expected = rnd.nextFloat() * 2f - 1f
                        network.setInput(bi, 0, expected)
                        network.setInput(bi, 1, 1f)
                        network.setInput(bi, 2, rnd.nextFloat())
                        network.setTarget(bi, 0, expected)
                    }

                    val print = true || it == 0 || it % dn == dn - 1 || it == n - 1
                    val error = network.learn(params, print)
                    if (print) println("[$batchSize,$it] Error: $error")
                    lastError = error

                    params.learningRate *= falloff
                }

                assertEquals(0f, lastError, 0.03f)
            }
        }
    }

    @Test
    fun testDeepSigmoidsSine() {
        val layout = NetworkLayout()
        layout.addLayer(FullyConnectedLayer(3, 3, Sigmoid))
        layout.addLayer(FullyConnectedLayer(3, 3, Sigmoid))
        layout.addLayer(FullyConnectedLayer(3, 3, Sigmoid))
        layout.addLayer(FullyConnectedLayer(3, 1, Identity))

        for (batchSize in listOf(100)) {
            forEachNetwork(layout, batchSize) { network ->
                val rnd = Random(1562)
                network.initializeWeights(rnd)

                val params = LearningParams(1f / batchSize, false)

                val n = 1000
                val dn = n / 20
                val falloff = pow(0.1f, 1f / n)
                var lastError = Float.POSITIVE_INFINITY
                repeat(n) {

                    // define training
                    for (bi in 0 until batchSize) {
                        val expected = rnd.nextFloat() * 2f - 1f
                        network.setInput(bi, 0, sin(expected))
                        network.setInput(bi, 1, 1f)
                        network.setInput(bi, 2, rnd.nextFloat())
                        network.setTarget(bi, 0, expected)
                    }

                    val print = it == 0 || it % dn == dn - 1 || it == n - 1
                    val error = network.learn(params, print)
                    if (print) println("[$batchSize,$it] Error: $error")
                    lastError = error

                    params.learningRate *= falloff
                }

                assertEquals(0f, lastError, 0.1f)
            }
        }
    }

    @Test
    fun testDeepSigmoidsXor() {
        val layout = NetworkLayout()
        layout.addLayer(FullyConnectedLayer(4, 3, Sigmoid))
        layout.addLayer(FullyConnectedLayer(3, 3, Sigmoid))
        layout.addLayer(FullyConnectedLayer(3, 3, Sigmoid))
        layout.addLayer(FullyConnectedLayer(3, 1, Identity))

        for (batchSize in listOf(1, 4, 16, 64)) {
            forEachNetwork(layout, batchSize) { network ->
                val rnd = Random(1562)
                network.initializeWeights(rnd)

                val params = LearningParams(0.01f / batchSize, false)

                val n = 5000
                val dn = n / 20
                val falloff = pow(0.1f, 1f / n)
                var errSumSmooth = 1f
                val errSumDt = dtTo01(3f / dn)
                repeat(n) {

                    // define training
                    for (bi in 0 until batchSize) {
                        val a = rnd.nextFloat()
                        val b = rnd.nextFloat()
                        val expected = a * (1f - b) + b * (1f - a)
                        network.setInput(bi, 0, a)
                        network.setInput(bi, 1, b)
                        network.setInput(bi, 2, 1f)
                        network.setInput(bi, 3, rnd.nextFloat())
                        network.setTarget(bi, 0, expected)
                    }

                    val error = network.learn(params, true)
                    errSumSmooth += (error - errSumSmooth) * errSumDt
                    if (it == 0 || it % dn == dn - 1) println("[$batchSize,$it] Error: $error ($errSumSmooth)")

                    params.learningRate *= falloff
                }

                assertEquals(0f, errSumSmooth, 0.2f)
            }
        }
    }

    @Test
    fun testSwapNetwork() {
        val layout = NetworkLayout()
        layout.addLayer(FullyConnectedLayer(3, 2, Identity))

        assertEquals(3, layout.numInputs)
        assertEquals(6, layout.numWeights)
        assertEquals(2, layout.numNodes)
        assertEquals(2, layout.numOutputs)

        val batchSize = 4
        forEachNetwork(layout, batchSize) { network ->

            val rnd = Random(1655)
            network.initializeWeights(rnd)

            val params = LearningParams(1f / batchSize, false)
            println("[$batchSize,I] Weights: ${network.inspectWeights().toList()}")

            var lastError = Float.POSITIVE_INFINITY
            val n = 50
            repeat(n) {

                // define training
                for (bi in 0 until batchSize) {
                    val expected0 = rnd.nextFloat()
                    val expected1 = rnd.nextFloat()
                    network.setInput(bi, 0, expected1)
                    network.setInput(bi, 1, expected0)
                    network.setInput(bi, 2, rnd.nextFloat())
                    network.setTarget(bi, 0, expected0)
                    network.setTarget(bi, 1, expected1)
                }

                val print = it == 0 || it % 25 == 24 || it == n - 1
                val error = network.learn(params, print)
                val weights = network.inspectWeights()
                if (print) println("[$batchSize,$it] Error: $error, Weights: ${weights.toList()}")

                val theoreticalError = sq(weights[0]) + sq(1f - weights[1]) + sq(weights[2]) +
                        sq(1f - weights[3]) + sq(weights[4]) + sq(weights[5])
                if (batchSize > 2) assertLessThanEquals(theoreticalError, lastError)
                lastError = theoreticalError
            }

            val weights = network.inspectWeights()
            assertEquals(0f, weights[0], 0.1f)
            assertEquals(1f, weights[1], 0.1f)
            assertEquals(0f, weights[2], 0.1f)
            assertEquals(1f, weights[3], 0.1f)
            assertEquals(0f, weights[4], 0.1f)
            assertEquals(0f, weights[5], 0.1f)
        }
    }

}