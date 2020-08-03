package intellisy.dataset

import intellisy.configuration.ClassifierConfiguration
import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.api.io.filters.RandomPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.io.File
import java.util.*


interface Dataset {

    fun getClassCount(): Int

    fun getTrainingSet(): DataSetIterator?

    fun getValidationSet(): DataSetIterator?

    fun getTestSet(): DataSetIterator?

    fun init(configuration: ClassifierConfiguration)

    companion object {

        fun fromFolder(trainFolder: File, testFolder: File? = null, balanced: Boolean = false) =
            FolderDataset(trainFolder, testFolder, balanced)

    }
}

class FolderDataset(
    private val trainFolder: File,
    private val testFolder: File?,
    private val balanced: Boolean
) : Dataset {

    private var classCount = 0
    private var trainSet: DataSetIterator? = null
    private var validationSet: DataSetIterator? = null
    private var testSet: DataSetIterator? = null

    private fun getRecordReaderIterator(
        file: File,
        configuration: ClassifierConfiguration,
        vararg weights: Double = doubleArrayOf(1.0)
    ): List<DataSetIterator> {
        val random = Random(configuration.seed)
        val labelMaker = ParentPathLabelGenerator()
        val fileSplit = FileSplit(file, configuration.allowedFormats.toTypedArray(), random)
        val imageFilter = when (balanced) {
            true -> BalancedPathFilter(random, configuration.allowedFormats.toTypedArray(), labelMaker, 0)
            else -> RandomPathFilter(random, configuration.allowedFormats.toTypedArray(), 0)
        }
        val split = fileSplit.sample(imageFilter, *weights)

        return split.map {
            val recordReader = ImageRecordReader(
                configuration.height,
                configuration.width,
                configuration.format.channels, labelMaker
            )

            recordReader.initialize(it, configuration.dataAugmentation.buildPipeline())

            if (classCount == 0)
                classCount = recordReader.labels.size

            RecordReaderDataSetIterator(recordReader, configuration.batchSize, 1, recordReader.labels.size)
        }
    }


    override fun getClassCount() = classCount

    override fun getTrainingSet() = trainSet

    override fun getValidationSet() = validationSet

    override fun getTestSet() = testSet

    override fun init(configuration: ClassifierConfiguration) {
        configuration.apply {
            val weights: DoubleArray = when {
                validationSplit > 0 -> doubleArrayOf(1 - validationSplit, validationSplit)
                else -> doubleArrayOf(1.0)
            }

            val datasetList = getRecordReaderIterator(
                trainFolder,
                this,
                *weights
            )

            trainSet = datasetList.first()

            if (datasetList.size == 2)
                validationSet = datasetList.last()

            if (testFolder != null) {
                testSet = getRecordReaderIterator(testFolder, configuration).first()
            }
        }
    }

}