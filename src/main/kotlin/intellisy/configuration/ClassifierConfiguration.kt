package intellisy.configuration

import intellisy.image.ImageFormat
import org.datavec.image.loader.NativeImageLoader
import java.security.SecureRandom

data class ClassifierConfiguration(
        val width: Long = 28,
        val height: Long = 28,
        val format: ImageFormat = ImageFormat.GRAYSCALE,
        val allowedFormats: List<String> = NativeImageLoader.ALLOWED_FORMATS.toList(),
        val batchSize: Int = 32,
        val epochs: Int = -1,
        val seed: Long = SecureRandom().nextLong(),
        val validationSplit: Double = 0.2
)