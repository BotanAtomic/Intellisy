package intellisy.core

import intellisy.dataset.Dataset
import intellisy.exception.NoDatasetException

class ImageClassifier
(
        var dataset: Dataset? = null
) {


    fun train() {
        if (dataset == null)
            throw NoDatasetException()
    }

    fun predict() {

    }

}