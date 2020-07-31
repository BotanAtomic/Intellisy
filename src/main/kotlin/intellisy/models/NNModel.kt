package intellisy.models

import intellisy.core.ImageClassifier
import org.deeplearning4j.nn.api.NeuralNetwork
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.zoo.model.VGG16
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions

interface NNModel {

    fun getModel(classifier: ImageClassifier): NeuralNetwork

    fun getScaler(): DataNormalization = ImagePreProcessingScaler(0.0, 1.0)

}


class VGG16Model : NNModel {

    override fun getModel(classifier: ImageClassifier): NeuralNetwork {
        classifier.configuration.apply {
            return VGG16.builder()
                .numClasses(classifier.dataset.classCount)
                .inputShape(intArrayOf(format.channel.toInt(), width.toInt(), height.toInt()))
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

}

class SimpleCNNModel : NNModel {

    override fun getModel(classifier: ImageClassifier): NeuralNetwork {
        classifier.configuration.apply {
            val inputShape = intArrayOf(format.channel.toInt(), width.toInt(), height.toInt())


            val conf = NeuralNetConfiguration.Builder().seed(seed)
                .activation(Activation.IDENTITY)
                .weightInit(WeightInit.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Adam())
                .cacheMode(cacheMode)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Same)
                .list()
                .layer(
                    ConvolutionLayer.Builder(7, 7)
                        .nIn(inputShape[0]).nOut(16)
                        .build()
                )
                .layer(BatchNormalization())
                .layer(
                    ConvolutionLayer.Builder(7, 7)
                        .nIn(16)
                        .nOut(16)
                        .build()
                )
                .layer(BatchNormalization())
                .layer(reluActivation())
                .layer(maxPool())

                .layer(dropoutLayer(0.5)) // block 2
                .layer(convLayer(32))
                .layer(BatchNormalization())
                .layer(convLayer(32))
                .layer(BatchNormalization())
                .layer(reluActivation())
                .layer(maxPool())

                .layer(dropoutLayer(0.5)) // block 3
                .layer(convLayer(64))
                .layer(BatchNormalization())
                .layer(convLayer(64))
                .layer(BatchNormalization())
                .layer(reluActivation())
                .layer(maxPool())

                .layer(dropoutLayer(0.5)) // block 4
                .layer(convLayer(128))
                .layer(BatchNormalization())
                .layer(convLayer(128))
                .layer(BatchNormalization())
                .layer(reluActivation())
                .layer(maxPool())


                .layer(dropoutLayer(0.5)) // block 5
                .layer(convLayer(256))
                .layer(BatchNormalization())
                .layer(
                    ConvolutionLayer.Builder(3, 3)
                        .nOut(classifier.dataset.classCount)
                        .build()
                )
                .layer(GlobalPoolingLayer.Builder(PoolingType.MAX).build())
                .layer(OutputLayer.Builder().nOut(classifier.dataset.classCount).build())
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
}

class SmallCNNModel : NNModel {

    override fun getModel(classifier: ImageClassifier): NeuralNetwork {
        classifier.configuration.apply {
            val inputShape = intArrayOf(format.channel.toInt(), width.toInt(), height.toInt())


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
                .layer(DenseLayer.Builder().nOut(512).dropOut(0.5).build())
                .layer(
                    OutputLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                        .nOut(classifier.dataset.classCount)
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
}

private fun reluActivation(): ActivationLayer? {
    return ActivationLayer.Builder().activation(Activation.RELU).build()
}

private fun dropoutLayer(value: Double): DropoutLayer {
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