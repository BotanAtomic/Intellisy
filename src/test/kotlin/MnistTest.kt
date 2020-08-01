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
                width = 48,
                height = 48,
                format = ImageFormat.GRAYSCALE,
                epochs = 15
            ),
            model = SimpleCNNModel()
        )

        val testEval = classifier.train { validationEval -> println(validationEval.stats()) }

        assertNotNull(testEval)
        assertTrue(testEval.accuracy() >= 0.99)
    }

}