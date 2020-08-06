package com.atomic.intellisy.listeners

import com.atomic.intellisy.classifier.ImageClassifier
import org.deeplearning4j.nn.api.Model
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator


class IterationListener(
    private val frequency: Int = 10,
    private val callback: ((ImageClassifier, DataSetIterator?, Int) -> Unit)
) : Listener {

    override fun onIterationDone(
        classifier: ImageClassifier,
        model: Model,
        validationSet: DataSetIterator?,
        iteration: Int
    ) {
        if (iteration % frequency == 0) {
            callback(classifier, validationSet, iteration)
        }
    }

}