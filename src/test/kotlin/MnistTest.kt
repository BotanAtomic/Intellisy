import intellisy.configuration.ClassifierConfiguration
import intellisy.core.ImageClassifier
import intellisy.dataset.Dataset
import intellisy.image.ImageFormat
import intellisy.models.SimpleCNNModel
import intellisy.models.SmallCNNModel
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.test.assertNotNull

class MnistTest {

    @Test
    fun mnistTestSmallCNN() {
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

        classifier.save(File("models/smallCNN.zip"))
        assertNotNull(testEval)
        assertTrue(testEval.accuracy() > 0.992)
    }

    @Test
    fun mnistTestSimpleCNN() {
        val classifier = ImageClassifier(
            dataset = Dataset.fromFolder(
                trainFolder = File("dataset/mnist/train"),
                testFolder = File("dataset/mnist/test")
            ),
            configuration = ClassifierConfiguration(
                width = 34,
                height = 34,
                format = ImageFormat.GRAYSCALE,
                epochs = 20,
                batchSize = 16
            ),
            model = SimpleCNNModel()
        )

        val testEval = classifier.train { validationEval -> println(validationEval.stats()) }

        assertNotNull(testEval)
        assertTrue(testEval.accuracy() >= 0.99)
    }

    @Test
    fun mnistTestSmallCNNFromFile() {
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
            restore(File("models/smallCnn.zip"))
            val prediction =
                predict(imageLoader.fromUrl("https://i.gyazo.com/e2e5e2a969b4de11e55c0ea4b389bae9.png")) //must be five

            assertTrue(prediction.index == 5)
        }

    }

}