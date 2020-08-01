package intellisy.core

import intellisy.configuration.ClassifierConfiguration
import intellisy.dataset.Dataset
import intellisy.models.NNModel
import intellisy.models.SimpleCNNModel
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.api.NeuralNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

class ImageClassifier
    (
    val configuration: ClassifierConfiguration = ClassifierConfiguration(),
    val dataset: Dataset,
    private val model: NNModel = SimpleCNNModel()
) {

    lateinit var neuralNetwork: NeuralNetwork

    private fun eval(dataset: DataSetIterator?): Evaluation {
        val evaluation = Evaluation()
        neuralNetwork.doEvaluation(dataset, evaluation)
        return evaluation
    }

    fun train(callback: (Evaluation) -> Unit): Evaluation? {
        dataset.init(configuration)

        neuralNetwork = model.getModel(this)
        (neuralNetwork as Model).setListeners(ScoreIterationListener(50))
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

                if (getValidationSet() != null) {
                    callback(eval(getValidationSet()))
                }
            }
            if (getTestSet() != null) {
                return@train eval(getTestSet())
            } else return@train null
        }
    }

    fun predict() {

    }

}