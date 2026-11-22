package org.rakshak.edge

import android.content.Context
import java.io.FileNotFoundException
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.rakshak.edge.analysis.Risk
import org.rakshak.edge.llm.LlmExplainer
import org.rakshak.edge.llm.ModelFiles
import org.rakshak.edge.ml.StudentDetector
import org.rakshak.edge.perf.InferenceTiming
import org.rakshak.edge.perf.measure

/**
 * Holds the detector and the explainer for the app's lifetime. Both load on first use, on a
 * background dispatcher. Missing files give a readable error, not a crash.
 */
class Engine(private val context: Context) : AutoCloseable {
    private var detector: StudentDetector? = null
    private var explainer: LlmExplainer? = null
    private val detectorLock = Mutex()
    private val explainerLock = Mutex()

    @Volatile
    private var closed = false

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

    suspend fun check(text: String, timing: InferenceTiming? = null): Result<Analysis> = withContext(Dispatchers.Default) {
        detectorLock.withLock {
            runCatching {
                ensureOpen()
                val d = detector ?: timing.measure("detector_load") { loadDetector() }
                Analyzer.analyse(text, d?.scorer(timing) ?: error(detectorError ?: "Detector unavailable"), timing = timing)
            }
        }
    }

    fun explainerInstalled(): Boolean = ModelFiles.installed(context)

    /** Runs Llama without discarding the advisory already on screen if generation fails. */
    suspend fun explain(analysis: Analysis, timing: InferenceTiming? = null): Result<Analysis> =
        withContext(Dispatchers.Default) {
            explainerLock.withLock {
                try {
                    ensureOpen()
                    if (analysis.detectorOnly || analysis.lang == "hinglish" || analysis.pkg.risk == Risk.LOW) {
                        return@withLock Result.success(analysis)
                    }
                    val llm = explainer ?: timing.measure("llm_load") { loadExplainer() }
                    val onVerified = timing?.let { it::pointVerified }
                    Result.success(Analyzer.explain(analysis, llm.generator(timing), onVerified))
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (error: Exception) {
                    Result.failure(error)
                }
            }
        }

    private fun loadExplainer(): LlmExplainer {
        val model = ModelFiles.modelFile(context) ?: error("No external files directory")
        val tokenizer = ModelFiles.tokenizerFile(context) ?: error("No external files directory")
        val llm = LlmExplainer(model.absolutePath, tokenizer.absolutePath)
        try {
            llm.load()
        } catch (error: Exception) {
            llm.close()
            throw error
        }
        explainer = llm
        return llm
    }

    private fun ensureOpen() {
        kotlin.check(!closed) { "Inference engine is closed" }
    }

    override fun close() {
        closed = true
        CoroutineScope(Dispatchers.Default).launch {
            detectorLock.withLock {
                detector?.close()
                detector = null
            }
            explainerLock.withLock {
                explainer?.close()
                explainer = null
            }
        }
    }

    companion object {
        const val STUDENT_ASSET = "sms_student.onnx"
    }
}
