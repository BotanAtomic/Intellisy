plugins {
    kotlin("jvm") version "1.3.72"
}


group = "com.atomic"
version = "1.0"


repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
}

subprojects {
    repositories {
        mavenCentral()
    }
}