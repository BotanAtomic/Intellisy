package intellisy.dataset

import intellisy.configuration.ClassifierConfiguration
import intellisy.utils.toPair
import org.datavec.api.io.filters.RandomPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.io.File
import java.util.*


abstract class Dataset {

    abstract fun getDatasetIterators(configuration: ClassifierConfiguration): Pair<DataSetIterator, DataSetIterator>

    companion object {

        fun fromCsv(file: File, labelIndex: Int) = CSVDataset(file, labelIndex)

        fun fromFolder(folder: File) = FolderDataset(folder)

    }
}

class FolderDataset(private val folder: File) : Dataset() {

    override fun getDatasetIterators(configuration: ClassifierConfiguration): Pair<DataSetIterator, DataSetIterator> {
        val random = Random(configuration.seed)
        val labelMaker = ParentPathLabelGenerator()
        val fileSplit = FileSplit(folder, configuration.allowedFormats.toTypedArray(), random)
        val randomFilter = RandomPathFilter(random, configuration.allowedFormats.toTypedArray(), 0)
        val split = fileSplit.sample(randomFilter, 1 - configuration.validationSplit, configuration.validationSplit)

        return split.map {
            val recordReader = ImageRecordReader(
                    configuration.height,
                    configuration.width,
                    configuration.format.channel, labelMaker).apply { initialize(it) }
            RecordReaderDataSetIterator(recordReader, configuration.batchSize, 1, recordReader.labels.size)
        }.toPair()
    }

}

class CSVDataset(private val file: File, private val labelIndex: Int) : Dataset() {

    override fun getDatasetIterators(configuration: ClassifierConfiguration): Pair<DataSetIterator, DataSetIterator> {
        TODO("Not yet implemented")
    }

}