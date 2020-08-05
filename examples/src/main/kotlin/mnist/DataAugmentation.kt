package mnist

import com.atomic.intellisy.configuration.ClassifierConfiguration
import com.atomic.intellisy.core.ImageClassifier
import com.atomic.intellisy.dataset.Dataset
import com.atomic.intellisy.image.ImageFormat
import com.atomic.intellisy.image.ImageTransformation
import com.atomic.intellisy.models.SmallCNNModel
import org.datavec.image.transform.CropImageTransform
import org.datavec.image.transform.RotateImageTransform
import java.io.File
import java.util.*

fun main() {
    val random = Random()

    val classifier = ImageClassifier(
        dataset = Dataset.fromFolder(
            trainFolder = File("dataset/mnist/train"),
            testFolder = File("dataset/mnist/test")
        ),
        configuration = ClassifierConfiguration(
            width = 28,
            height = 28,
            format = ImageFormat.GRAYSCALE,
            dataAugmentation = ImageTransformation().apply {
                add(CropImageTransform(random, 5), probability = .05)
                add(RotateImageTransform(random, 5.0f, 5.0f, 30.0f, 0.2f), probability = .05)
            },
            epochs = 15
        ),
        model = SmallCNNModel()
    )

    val testEval = classifier.train { validationEval -> println(validationEval.stats()) }
    println(testEval?.stats()) // > 99%

    classifier.save(File("models/MNIST_SMALL_CNN_DA.zip"))
}