package intellisy.core

import intellisy.configuration.ClassifierConfiguration
import intellisy.dataset.Dataset
import intellisy.exception.NoDatasetException

class ImageClassifier
(
        var dataset: Dataset? = null,
        val configuration: ClassifierConfiguration = ClassifierConfiguration()
) {


    fun train() {
        if (dataset == null)
            throw NoDatasetException()
    }

    fun predict() {

    }

}