package org.rakshak.edge

import android.content.Context
import java.io.FileNotFoundException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.rakshak.edge.llm.LlmExplainer
import org.rakshak.edge.llm.ModelFiles
import org.rakshak.edge.ml.StudentDetector

/**
 * Holds the detector and the explainer for the app's lifetime. Both load on first use, on a
 * background dispatcher. Missing files give a readable error, not a crash.
 */
class Engine(private val context: Context) : AutoCloseable {
    private var detector: StudentDetector? = null
    private var explainer: LlmExplainer? = null

    /** Load error for the detector, if any (shown on screen). */
    @Volatile
    var detectorError: String? = null
        private set

    private fun loadDetector(): StudentDetector? {
        detector?.let { return it }
        return try {
            val bytes = context.assets.open(STUDENT_ASSET).use { it.readBytes() }
            StudentDetector.fromBytes(bytes).also { detector = it }
        } catch (e: FileNotFoundException) {
            detectorError = "The detector model is not in the app assets (assets/$STUDENT_ASSET). " +
                "See docs/ANDROID_APP.md for where to copy it from."
            null
        } catch (e: Exception) {
            detectorError = "The detector failed to load: ${e.message}"
            null
        }
    }

    suspend fun check(text: String): Result<Analysis> = withContext(Dispatchers.Default) {
        val d = loadDetector() ?: return@withContext Result.failure(
            IllegalStateException(detectorError ?: "detector unavailable"),
        )
        runCatching { Analyzer.analyse(text, d) }
    }

    fun explainerInstalled(): Boolean = ModelFiles.installed(context)

    /** Runs the Llama explainer on top of [analysis]. Falls back to the template on any error. */
    suspend fun explain(analysis: Analysis): Result<Analysis> = withContext(Dispatchers.Default) {
        runCatching {
            val llm = explainer ?: run {
                val model = ModelFiles.modelFile(context) ?: error("no external files directory")
                val tok = ModelFiles.tokenizerFile(context) ?: error("no external files directory")
                LlmExplainer(model.absolutePath, tok.absolutePath).also { it.load(); explainer = it }
            }
            Analyzer.explain(analysis, llm)
        }
    }

    override fun close() {
        detector?.close()
        explainer?.close()
        detector = null
        explainer = null
    }

    companion object {
        const val STUDENT_ASSET = "sms_student.onnx"
    }
}
