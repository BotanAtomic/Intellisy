import intellisy.configuration.ClassifierConfiguration
import intellisy.core.ImageClassifier
import intellisy.dataset.Dataset
import intellisy.image.ImageFormat
import intellisy.image.ImageTransformation
import intellisy.models.SimpleCNNModel
import intellisy.models.SmallCNNModel
import org.datavec.image.transform.CropImageTransform
import org.datavec.image.transform.RotateImageTransform
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.util.*
import kotlin.test.assertNotNull

class MnistTest {

    @Test
    fun mnistSmallCNN() {
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
    fun mnistSimpleCNN() {
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
    fun mnistSmallCNNFromFile() {
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

    @Test
    fun mnistSmallCNNDataAugmentation() {
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
                    add(CropImageTransform(random, 5), probability = 0.2)
                    add(RotateImageTransform(random, 10.0f, 10.0f, 60.0f, 1.2f), probability = .1)
                },
                epochs = 30
            ),
            model = SmallCNNModel()
        )

        val testEval = classifier.train { validationEval -> println(validationEval.stats()) }

        classifier.save(File("models/smallCNN_augmented.zip"))
        assertNotNull(testEval)
        assertTrue(testEval.accuracy() > 0.992)
    }

}