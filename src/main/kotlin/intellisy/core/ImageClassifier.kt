package intellisy.core

import intellisy.configuration.ClassifierConfiguration
import intellisy.dataset.Dataset
import intellisy.exception.NoDatasetException
import intellisy.image.Image
import intellisy.image.ImageLoader
import intellisy.models.NNModel
import intellisy.models.SimpleCNNModel
import intellisy.prediction.Prediction
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.api.NeuralNetwork
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.io.File

open class ImageClassifier
    (
    val configuration: ClassifierConfiguration = ClassifierConfiguration(),
    private val dataset: Dataset? = null,
    private val model: NNModel = SimpleCNNModel()
) {

    private lateinit var neuralNetwork: NeuralNetwork

    val imageLoader = ImageLoader(configuration)

    private fun eval(dataset: DataSetIterator?): Evaluation? {
        if (dataset == null) return null
        val evaluation = Evaluation()
        neuralNetwork.doEvaluation(dataset, evaluation)
        return evaluation
    }

    fun restore(file: File) {
        model.restore(file)?.let { neuralNetwork = it }
    }

    open fun train(callback: (Evaluation) -> Unit): Evaluation? {
        if (dataset == null) throw NoDatasetException()

        dataset.init(configuration)

        neuralNetwork = model.getModel(this)
        val dataNormalizer = model.getScaler()

        dataset.apply {
            listOf(getTrainingSet(), getValidationSet(), getTestSet()).forEach {
                if (it != null) {
                    dataNormalizer.fit(it)
                    it.preProcessor = dataNormalizer
                }
            }


            for (i in 0 until configuration.epochs) {
                neuralNetwork.fit(getTrainingSet())

                eval(getValidationSet())?.let(callback)
            }
            return eval(getTestSet())
        }
    }

    fun predict(image: Image): Prediction {
        model.getScaler().transform(image)
        val output = model.output(neuralNetwork, image)
        val index = output.argMax(1).getInt(0)
        return Prediction(index, output.getDouble(index))
    }

    fun save(file: File) = kotlin.runCatching {
        ModelSerializer.writeModel(neuralNetwork as Model, file, true)
    }.isSuccess

    fun getDataset(): Dataset {
        if (dataset == null) throw NoDatasetException()
        return dataset
    }


}