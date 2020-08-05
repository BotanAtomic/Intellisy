plugins {
    kotlin("jvm")
}

group = "com.intellisy"
version = "1.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation(project(":core"))
}
