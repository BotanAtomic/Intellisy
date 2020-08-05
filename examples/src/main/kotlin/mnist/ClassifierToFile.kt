package mnist

import intellisy.configuration.ClassifierConfiguration
import intellisy.core.ImageClassifier
import intellisy.dataset.Dataset
import intellisy.image.ImageFormat
import intellisy.models.SmallCNNModel
import java.io.File

fun main() {
    val classifier = ImageClassifier(
        dataset = Dataset.fromFolder(
            trainFolder = File("dataset/mnist/train"),
            testFolder = File("dataset/mnist/test")
        ),
        configuration = ClassifierConfiguration(
            width = 28,
            height = 28,
            format = ImageFormat.GRAYSCALE
        ),
        model = SmallCNNModel()
    )

    val testEval = classifier.train { validationEval -> println(validationEval.stats()) }

    classifier.save(File("models/MNIST_SMALL_CNN.zip"))
    println(testEval?.stats())
}