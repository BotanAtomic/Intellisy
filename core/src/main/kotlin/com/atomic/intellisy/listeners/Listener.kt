package com.atomic.intellisy.listeners

import com.atomic.intellisy.classifier.ImageClassifier
import org.deeplearning4j.nn.api.Model
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

interface Listener {

    fun onEpochDone(
        classifier: ImageClassifier,
        model: Model,
        validationSet: DataSetIterator?
    ) {
    }

    fun onIterationDone(
        classifier: ImageClassifier,
        model: Model,
        validationSet: DataSetIterator?,
        iteration: Int
    ) {
    }

}