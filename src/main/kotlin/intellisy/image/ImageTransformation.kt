package intellisy.image

import org.datavec.image.transform.ImageTransform
import org.datavec.image.transform.PipelineImageTransform
import org.nd4j.common.primitives.Pair

class ImageTransformation(private val shuffle: Boolean = true) {
    private val pipeline = ArrayList<Pair<ImageTransform, Double>>()

    fun add(transformation: ImageTransform, probability: Double): ImageTransformation {
        pipeline.add(Pair(transformation, probability))
        return this
    }

    fun buildPipeline() = PipelineImageTransform(pipeline, shuffle)

}