package com.atomic.intellisy.listeners

import com.atomic.intellisy.classifier.ImageClassifier
import org.deeplearning4j.nn.api.Model
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator


class EpochListener(
    private val callback: ((ImageClassifier, DataSetIterator?) -> Unit)
) : Listener {

    override fun onEpochDone(classifier: ImageClassifier, model: Model, validationSet: DataSetIterator?) {
        callback(classifier, validationSet)
    }

}