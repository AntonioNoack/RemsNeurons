package me.anno.remsneurons

import me.anno.io.files.Reference.getReference
import me.anno.remsneurons.NetworkGradientTest.Companion.setupGraphics
import me.anno.remsneurons.activations.Sigmoid
import me.anno.remsneurons.activations.SoftMax
import me.anno.remsneurons.layers.FullyConnectedLayer
import me.anno.remsneurons.network.GPUNetwork
import me.anno.utils.types.Booleans.toFloat
import org.junit.jupiter.api.Test
import kotlin.math.min
import kotlin.random.Random

class MNISTDataSetTest {

    @Test
    fun testTrainingMNIST() {

        setupGraphics()

        val folder = getReference("E:/Documents/Datasets")
        val labels = folder.getChild("train-labels.bin").readBytesSync()
        val images = folder.getChild("train-images.bin").readBytesSync()

        val imgWidth = 28
        val imgHeight = 28
        val imgSize = imgWidth * imgHeight

        val numSamples = min(labels.size, images.size / imgSize)

        val n = 100_000
        val batchSize = 600 // why is 600 the minimum for good results???
        val layout = NetworkLayout()
        // todo bug: this is not getting better than ~0.27 :(, what do we do wrong?
        layout.addLayer(FullyConnectedLayer(imgWidth * imgHeight, 50, Sigmoid))
        layout.addLayer(FullyConnectedLayer(50, 10, Sigmoid))
        layout.addLayer(FullyConnectedLayer(10, 10, SoftMax))
        //layout.addLayer(FullyConnectedLayer(80, 30, Sigmoid))
        //layout.addLayer(FullyConnectedLayer(30, 30, Sigmoid))
        val network = GPUNetwork(layout, batchSize)
        val rnd = Random(15645)
        network.initializeWeights(rnd)

        println(network.inspectWeights().toList().subList(0, 100))

        // todo how is this able to learn things with a factor of 0 ????
        val params = LearningParams(0f, false)
        repeat(n) {

            for (bi in 0 until batchSize) {
                val sampleIndex = rnd.nextInt(numSamples)
                val offset = sampleIndex * imgSize
                for (ni in 0 until imgSize) {
                    val pixel = images[ni + offset].toInt().and(0xff) / 255f
                    network.setInput(bi, ni, pixel)
                }
                val label = labels[sampleIndex].toInt()
                for (no in 0 until 10) network.setTarget(bi, no, (no == label).toFloat())
            }

            val print = it % 100 == 0 || it - 1 == n
            val error = network.learn(params, print)
            if (print) println("[$batchSize,$it] Error: $error")

        }
    }
}