package com.atomic.intellisy.listeners

import com.atomic.intellisy.classifier.ImageClassifier
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.optimize.api.TrainingListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

class ListenerWrapper(
    private val classifier: ImageClassifier,
    private val listeners: List<Listener>,
    private val validationSet: DataSetIterator?
) : TrainingListener {
    override fun iterationDone(model: Model?, iteration: Int, epoch: Int) {
        model?.let {
            listeners.forEach { listener -> listener.onIterationDone(classifier, it, validationSet, iteration) }
        }
    }

    override fun onEpochEnd(model: Model?) {
        model?.let {
            listeners.forEach { listener -> listener.onEpochDone(classifier, it, validationSet) }
        }
    }


    override fun onGradientCalculation(model: Model?) {
    }

    override fun onBackwardPass(model: Model?) {
    }

    override fun onForwardPass(model: Model?, activations: MutableList<INDArray>?) {
    }

    override fun onForwardPass(model: Model?, activations: MutableMap<String, INDArray>?) {
    }

    override fun onEpochStart(model: Model?) {
    }

}