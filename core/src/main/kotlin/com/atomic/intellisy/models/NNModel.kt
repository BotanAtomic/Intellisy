package com.atomic.intellisy.models

import com.atomic.intellisy.classifier.ImageClassifier
import com.atomic.intellisy.image.Image
import org.deeplearning4j.nn.api.NeuralNetwork
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.util.ModelSerializer
import org.deeplearning4j.zoo.model.VGG16
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File

interface NNModel {

    fun getModel(classifier: ImageClassifier): NeuralNetwork

    fun getScaler(): DataNormalization = ImagePreProcessingScaler(0.0, 1.0)

    fun restore(file: File): NeuralNetwork?

    fun output(network: NeuralNetwork, image: Image): INDArray =
        (network as MultiLayerNetwork).output(image)
}


class VGG16Model : NNModel {

    override fun getModel(classifier: ImageClassifier): NeuralNetwork {
        classifier.configuration.apply {
            return VGG16.builder()
                .numClasses(classifier.getDataset().getClassCount())
                .inputShape(intArrayOf(format.channels.toInt(), width.toInt(), height.toInt()))
                .updater(Nesterovs())
                .cacheMode(cacheMode)
                .cudnnAlgoMode(cudnnAlgoMode)
                .workspaceMode(workspaceMode)
                .build()
                .init()
        }
    }

    override fun getScaler(): DataNormalization {
        return VGG16ImagePreProcessor()
    }

    override fun restore(file: File): NeuralNetwork? = ModelSerializer.restoreComputationGraph(file)

    override fun output(network: NeuralNetwork, image: Image): INDArray =
        (network as ComputationGraph).outputSingle(image)

}

class SimpleCNNModel : NNModel {

    override fun getModel(classifier: ImageClassifier): NeuralNetwork {
        val classCount = classifier.getDataset().getClassCount()
        classifier.configuration.apply {
            val inputShape = longArrayOf(format.channels, width, height)


            val conf = NeuralNetConfiguration.Builder().seed(seed)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Adam())
                .cacheMode(cacheMode)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Same)
                .list()
                .layer(ConvolutionLayer.Builder(7, 7).nIn(inputShape[0]).nOut(16).build())
                .layer(BatchNormalization())
                .layer(ConvolutionLayer.Builder(7, 7).nIn(16).nOut(16).build())
                .layer(BatchNormalization())
                .layer(maxPool())

                .layer(dropoutLayer()) // block 2
                .layer(convLayer(32))
                .layer(BatchNormalization())
                .layer(convLayer(32))
                .layer(BatchNormalization())
                .layer(maxPool())

                .layer(dropoutLayer()) // block 3
                .layer(convLayer(64))
                .layer(BatchNormalization())
                .layer(convLayer(64))
                .layer(BatchNormalization())
                .layer(maxPool())

                .layer(dropoutLayer()) // block 4
                .layer(convLayer(128))
                .layer(BatchNormalization())
                .layer(convLayer(128))
                .layer(BatchNormalization())
                .layer(maxPool())


                .layer(dropoutLayer()) // block 5
                .layer(convLayer(256))
                .layer(BatchNormalization())
                .layer(ConvolutionLayer.Builder(3, 3).nOut(classCount).build())
                .layer(GlobalPoolingLayer.Builder(PoolingType.MAX).build())
                .layer(OutputLayer.Builder().nOut(classCount).build())
                .setInputType(
                    InputType.convolutional(inputShape[2], inputShape[1], inputShape[0])
                ).build()

            return MultiLayerNetwork(conf).apply { init() }
        }
    }

    override fun restore(file: File): NeuralNetwork? = ModelSerializer.restoreMultiLayerNetwork(file)

}

class SmallCNNModel : NNModel {

    override fun getModel(classifier: ImageClassifier): NeuralNetwork {
        classifier.configuration.apply {
            val inputShape = intArrayOf(format.channels.toInt(), width.toInt(), height.toInt())


            val conf = NeuralNetConfiguration.Builder().seed(seed)
                .activation(Activation.RECTIFIEDTANH)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Adam())
                .cacheMode(cacheMode)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Same)
                .list()
                .layer(ConvolutionLayer.Builder().nIn(inputShape[0]).nOut(32).build())
                .layer(maxPool())
                .layer(convLayer(64, 0.0))
                .layer(convLayer(64, 1.0))
                .layer(maxPool())
                .layer(DenseLayer.Builder().nOut(512).dropOut(0.4).build())
                .layer(
                    OutputLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                        .nOut(classifier.getDataset().getClassCount())
                        .activation(Activation.SOFTMAX)
                        .build()
                )
                .setInputType(
                    InputType.convolutional(
                        inputShape[2].toLong(),
                        inputShape[1].toLong(),
                        inputShape[0].toLong()
                    )
                ).build()

            return MultiLayerNetwork(conf).apply { init() }
        }
    }

    override fun restore(file: File): NeuralNetwork? = ModelSerializer.restoreMultiLayerNetwork(file)
}

private fun dropoutLayer(value: Double = 0.5): DropoutLayer {
    return DropoutLayer.Builder(value).build()
}

private fun maxPool(kernel: IntArray = intArrayOf(2, 2), stride: IntArray = intArrayOf(2, 2)): SubsamplingLayer {
    return SubsamplingLayer.Builder(kernel, stride).build()
}

private fun convLayer(
    out: Int,
    bias: Double = 0.0,
    kernel: IntArray = intArrayOf(3, 3),
    stride: IntArray = intArrayOf(1, 1),
    padding: IntArray = intArrayOf(1, 1)
): ConvolutionLayer {
    return ConvolutionLayer.Builder(kernel, stride, padding)
        .nOut(out)
        .biasInit(bias)
        .build()
}