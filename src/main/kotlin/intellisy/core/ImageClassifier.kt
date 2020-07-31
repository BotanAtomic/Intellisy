package intellisy.core

import intellisy.configuration.ClassifierConfiguration
import intellisy.dataset.Dataset
import intellisy.exception.NoDatasetException

class ImageClassifier
(
        val configuration: ClassifierConfiguration = ClassifierConfiguration()
) {

    lateinit var dataset: Dataset


    fun train() {
        if (::dataset.isInitialized.not())
            throw NoDatasetException()
    }

    fun predict() {

    }

}