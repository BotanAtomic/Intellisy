package intellisy.image

import intellisy.configuration.ClassifierConfiguration
import org.datavec.image.loader.NativeImageLoader
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.opencv.core.Mat
import java.io.File
import java.net.URL
import javax.imageio.ImageIO

typealias Image = INDArray

class ImageLoader(configuration: ClassifierConfiguration) {

    private val loader = NativeImageLoader(
        configuration.width,
        configuration.height,
        configuration.format.channels
    )

    fun fromBytes(array: ByteArray): Image = loader.asMatrix(Nd4j.fromByteArray(array))

    fun fromMat(mat: Mat): Image = loader.asMatrix(mat)

    fun fromNumpy(mat: Mat): Image = loader.asMatrix(mat)

    fun fromFile(file: File): Image = loader.asMatrix(file)

    fun fromUrl(url: String): Image = loader.asMatrix(ImageIO.read(URL(url)))

}
