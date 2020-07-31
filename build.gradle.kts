plugins {
    kotlin("jvm") version "1.3.72"
}

group = "com.botan"
version = "1.0"


repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("org.nd4j:nd4j-cuda-10.1-platform:1.0.0-beta7")
    implementation("org.nd4j:nd4j-api:1.0.0-beta7")
    implementation("org.datavec:datavec-data-image:1.0.0-beta7")
    implementation("org.deeplearning4j:deeplearning4j-cuda-10.1:1.0.0-beta7")
    implementation("org.deeplearning4j:deeplearning4j-zoo:1.0.0-beta7")
    implementation("org.slf4j:slf4j-simple:1.7.30")
    implementation("org.knowm.xchart:xchart:3.6.4")
}