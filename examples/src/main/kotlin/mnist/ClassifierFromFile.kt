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

    classifier.apply {
        restore(File("models/MNIST_SMALL_CNN.zip"))
        val prediction =
            predict(imageLoader.fromUrl("https://i.gyazo.com/e2e5e2a969b4de11e55c0ea4b389bae9.png")) //must be five

        assert(prediction.index == 5)
    }
}