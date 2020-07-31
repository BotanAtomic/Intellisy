package intellisy.models

import intellisy.core.ImageClassifier
import org.deeplearning4j.zoo.model.SimpleCNN
import org.deeplearning4j.zoo.model.VGG16
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor

interface NNModel {

    fun getModel(classifier: ImageClassifier): Any

    fun getScaler(): DataNormalization = ImagePreProcessingScaler(0.0, 1.0)

}

class VGG16Model : NNModel {

    override fun getModel(classifier: ImageClassifier): Any {
        classifier.configuration.apply {
            return VGG16.builder()
                    .numClasses(classifier.dataset.classCount)
                    .inputShape(intArrayOf(format.channel.toInt(), width.toInt(), height.toInt()))
                    .updater(updater)
                    .cacheMode(cacheMode)
                    .cudnnAlgoMode(cudnnAlgoMode)
                    .workspaceMode(workspaceMode)
                    .build()
        }
    }

    override fun getScaler(): DataNormalization {
        return VGG16ImagePreProcessor()
    }

}

class SimpleCNNModel : NNModel {

    override fun getModel(classifier: ImageClassifier): Any {
        classifier.configuration.apply {
            return SimpleCNN.builder()
                    .numClasses(classifier.dataset.classCount)
                    .inputShape(intArrayOf(format.channel.toInt(), width.toInt(), height.toInt()))
                    .updater(updater)
                    .cacheMode(cacheMode)
                    .cudnnAlgoMode(cudnnAlgoMode)
                    .workspaceMode(workspaceMode)
                    .build()
                    .init()
        }
    }

}