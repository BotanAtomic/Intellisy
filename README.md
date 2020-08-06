# Intellisy

![Logo](https://github.com/BotanAtomic/Intellisy/raw/master/logo.png =100x)

More documentation comming soon...


```Kotlin
 val classifier = ImageClassifier(
    dataset = Dataset.fromFolder(
        trainFolder = File("dataset/mnist/train"),
        testFolder = File("dataset/mnist/test")
    ),
    configuration = ClassifierConfiguration(
        width = 28,
        height = 28,
        format = ImageFormat.GRAYSCALE
    )
 )

 val testEval = classifier.train { validationEval -> println(validationEval.stats()) }

 println("Model accuracy: ${testEval.accuracy()}) // 99.2%
  ```

