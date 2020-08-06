package com.atomic.intellisy.configuration

import com.atomic.intellisy.image.ImageFormat
import com.atomic.intellisy.image.ImageTransformation
import com.atomic.intellisy.listeners.Listener
import org.datavec.image.loader.NativeImageLoader
import org.deeplearning4j.nn.conf.CacheMode
import org.deeplearning4j.nn.conf.WorkspaceMode
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import java.security.SecureRandom

data class ClassifierConfiguration(
    val width: Long = 28,
    val height: Long = 28,
    val format: ImageFormat = ImageFormat.GRAYSCALE,
    val allowedFormats: List<String> = NativeImageLoader.ALLOWED_FORMATS.toList(),
    val dataAugmentation: ImageTransformation = ImageTransformation(),


    val batchSize: Int = 32,
    val epochs: Int = 10,
    val seed: Long = SecureRandom().nextLong(),
    val validationSplit: Double = 0.1,
    val listeners: List<Listener>? = null,

    val cacheMode: CacheMode = CacheMode.DEVICE,
    val cudnnAlgoMode: ConvolutionLayer.AlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST,
    val workspaceMode: WorkspaceMode = WorkspaceMode.ENABLED
)