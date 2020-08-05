plugins {
    kotlin("jvm")
}

group = "com.atomic"
version = "1.0"

val dl4jVersion = "1.0.0-beta7"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))

    println("GPU ${project.findProperty("dl4j_gpu")}")

    if (project.findProperty("dl4j_gpu") == "true") {
        val cudaVersion = when (project.hasProperty("cuda-version")) {
            true -> project.property("cuda-version")
            else -> "10.1"
        }
        api("org.nd4j:nd4j-cuda-$cudaVersion-platform:$dl4jVersion")
        api("org.deeplearning4j:deeplearning4j-cuda-$cudaVersion:$dl4jVersion")
    } else {
        implementation("org.nd4j:nd4j-native-platform:$dl4jVersion")
        implementation("org.deeplearning4j:deeplearning4j-core:$dl4jVersion")
    }

    api("org.nd4j:nd4j-api:$dl4jVersion")
    api("org.datavec:datavec-data-image:$dl4jVersion")
    api("org.deeplearning4j:deeplearning4j-zoo:$dl4jVersion")

    runtimeOnly("org.slf4j:slf4j-simple:1.7.30")

    testImplementation(kotlin("test"))
    testImplementation(kotlin("test-junit"))
}
