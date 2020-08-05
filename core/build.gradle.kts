plugins {
    kotlin("jvm")
}

group = "com.atomic"
version = "1.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))

    if (project.findProperty("gpu") == true) {
        val cudaVersion = when (project.hasProperty("cuda-version")) {
            true -> project.property("cuda-version")
            else -> "10.1"
        }
        api("org.nd4j:nd4j-cuda-$cudaVersion-platform:1.0.0-beta7")
        api("org.deeplearning4j:deeplearning4j-cuda-$cudaVersion:1.0.0-beta7")
    } else {
        implementation("org.nd4j:nd4j-native:1.0.0-beta7")
        implementation("org.deeplearning4j:deeplearning4j-core:1.0.0-beta7")
    }

    api("org.nd4j:nd4j-api:1.0.0-beta7")
    api("org.datavec:datavec-data-image:1.0.0-beta7")
    api("org.deeplearning4j:deeplearning4j-zoo:1.0.0-beta7")
    api("org.knowm.xchart:xchart:3.6.4")

    runtimeOnly("org.slf4j:slf4j-simple:1.7.30")

    testImplementation(kotlin("test"))
    testImplementation(kotlin("test-junit"))
}
