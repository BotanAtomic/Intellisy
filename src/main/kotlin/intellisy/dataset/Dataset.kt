package intellisy.dataset

import intellisy.configuration.ClassifierConfiguration
import org.datavec.api.io.filters.RandomPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.io.File
import java.util.*


abstract class Dataset {

    var classCount: Int = 0

    var trainSet: DataSetIterator? = null
    var validationSet: DataSetIterator? = null
    var testSet: DataSetIterator? = null

    abstract fun loadDataset(configuration: ClassifierConfiguration)

    companion object {

        fun fromFolder(trainFolder: File, testFolder: File? = null) = FolderDataset(trainFolder, testFolder)

    }
}

class FolderDataset(private val trainFolder: File, private val testFolder: File?) : Dataset() {

    private fun getRecordReaderIterator(
        file: File,
        configuration: ClassifierConfiguration,
        vararg weights: Double = doubleArrayOf(1.0)
    ): List<DataSetIterator> {
        val random = Random(configuration.seed)
        val labelMaker = ParentPathLabelGenerator()
        val fileSplit = FileSplit(file, configuration.allowedFormats.toTypedArray(), random)
        val randomFilter = RandomPathFilter(random, configuration.allowedFormats.toTypedArray(), 0)
        val split = fileSplit.sample(randomFilter, *weights)

        return split.map {
            val recordReader = ImageRecordReader(
                configuration.height,
                configuration.width,
                configuration.format.channel, labelMaker
            )

            recordReader.initialize(it, configuration.imageTransformation.buildPipeline())

            if (super.classCount == 0)
                super.classCount = recordReader.labels.size

            RecordReaderDataSetIterator(recordReader, configuration.batchSize, 1, recordReader.labels.size)
        }
    }

    override fun loadDataset(configuration: ClassifierConfiguration) {
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

            super.trainSet = datasetList[0]

            if (datasetList.size > 1)
                super.validationSet = datasetList[1]

            if (testFolder != null) {
                super.testSet = getRecordReaderIterator(testFolder, configuration)[0]
            }
        }
    }

}