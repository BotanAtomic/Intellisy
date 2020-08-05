package mnist

import com.atomic.intellisy.configuration.ClassifierConfiguration
import com.atomic.intellisy.core.ImageClassifier
import com.atomic.intellisy.dataset.Dataset
import com.atomic.intellisy.image.ImageFormat
import com.atomic.intellisy.models.SmallCNNModel
import java.io.File

fun main() {
    val classifier = ImageClassifier(
        dataset = Dataset.fromFolder(
            trainFolder = File("examples/dataset/mnist/train"),
            testFolder = File("examples/dataset/mnist/test")
        ),
        configuration = ClassifierConfiguration(
            width = 28,
            height = 28,
            format = ImageFormat.GRAYSCALE,
            validationSplit = 0.1
        ),
        model = SmallCNNModel()
    )

    val testEval = classifier.train { validationEval -> println(validationEval.stats()) }

    println(testEval?.stats()) // > 99.2%
}