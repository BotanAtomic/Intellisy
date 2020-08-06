package mnist

import com.atomic.intellisy.classifier.ImageClassifier
import com.atomic.intellisy.configuration.ClassifierConfiguration
import com.atomic.intellisy.dataset.Dataset
import com.atomic.intellisy.image.ImageFormat
import com.atomic.intellisy.models.SmallCNNModel
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