// Root build. AGP 9.x has built-in Kotlin: do NOT apply org.jetbrains.kotlin.android here or in
// app/. The Compose compiler plugin is still applied; its version must equal the Kotlin version.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.compose) apply false
}
