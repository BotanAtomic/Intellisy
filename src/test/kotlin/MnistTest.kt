import intellisy.configuration.ClassifierConfiguration
import intellisy.core.ImageClassifier
import intellisy.dataset.Dataset
import intellisy.image.ImageFormat
import java.io.File

fun main() {
    val classifier = ImageClassifier(
        dataset = Dataset.fromFolder(
            trainFolder = File("dataset/mnist")
        ),
        configuration = ClassifierConfiguration(
            width = 28,
            height = 28,
            format = ImageFormat.GRAYSCALE
        )
    )


}