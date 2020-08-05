plugins {
    kotlin("jvm") version "1.3.72"
}


group = "com.botan"
version = "1.0"


repositories {
    mavenCentral()
}

tasks.withType<Test>().all {
    jvmArgs("-Xmx1G", "-Xmx8G")
}

dependencies {
    implementation(kotlin("stdlib"))
    api("org.nd4j:nd4j-cuda-10.1-platform:1.0.0-beta7")
    //implementation("org.nd4j:nd4j-native:1.0.0-beta7")
    //implementation("org.nd4j:nd4j-native:1.0.0-beta7:windows-x86_64-avx2")
    api("org.nd4j:nd4j-api:1.0.0-beta7")
    api("org.datavec:datavec-data-image:1.0.0-beta7")
    api("org.deeplearning4j:deeplearning4j-cuda-10.1:1.0.0-beta7")
    //implementation("org.deeplearning4j:deeplearning4j-core:1.0.0-beta7")
    api("org.deeplearning4j:deeplearning4j-zoo:1.0.0-beta7")
    api("org.slf4j:slf4j-simple:1.7.30")
    api("org.knowm.xchart:xchart:3.6.4")

    testImplementation(kotlin("test"))
    testImplementation(kotlin("test-junit"))
}

subprojects {
    repositories {
        mavenCentral()
    }
}