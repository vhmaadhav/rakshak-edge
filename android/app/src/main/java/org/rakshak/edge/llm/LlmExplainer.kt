package org.rakshak.edge.llm

import android.content.Context
import java.io.File
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.isActive
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import org.pytorch.executorch.extension.llm.LlmModule
import org.rakshak.edge.perf.InferenceTiming
import org.rakshak.edge.perf.measure

/**
 * Where the explainer files live. Push them with adb (docs/ANDROID_APP.md):
 *   adb push <Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte> <dir>/
 *   adb push <tokenizer.model> <dir>/
 * with <dir> = /sdcard/Android/data/org.rakshak.edge/files/models (app-specific external storage).
 */
object ModelFiles {
    const val MODEL_NAME = "Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte"
    const val TOKENIZER_NAME = "tokenizer.model"

    fun dir(context: Context): File? = context.getExternalFilesDir("models")

    fun modelFile(context: Context): File? = dir(context)?.let { File(it, MODEL_NAME) }

    fun tokenizerFile(context: Context): File? = dir(context)?.let { File(it, TOKENIZER_NAME) }

    fun installed(context: Context): Boolean =
        modelFile(context)?.isFile == true && tokenizerFile(context)?.isFile == true
}

/**
 * ExecuTorch Llama explainer. Greedy decoding (temperature 0), no BOS or EOS added by the
 * runner (the prompt already carries the Llama 3 special tokens), KV cache reset before each
 * prompt, as the Pi runner does in rbench/runners/pte_llm.py.
 */
class LlmExplainer(modelPath: String, tokenizerPath: String) : AutoCloseable {
    private val module = LlmModule(modelPath, tokenizerPath, 0.0f)

    /** Force the native load so its latency is counted separately from generation. */
    fun load() {
        module.load()
    }

    fun generator(timing: InferenceTiming?): TokenGenerator =
        TokenGenerator { prompt, maxNewTokens, stopWhen ->
            val context = currentCoroutineContext()
            context.ensureActive()
            val config = LlmGenerationConfig.create()
                .maxNewTokens(maxNewTokens)
                .echo(false)
                .temperature(0.0f)
                .numBos(0)
                .numEos(0)
                .build()
            val pieces = StringBuilder()
            var stopRequested = false
            var failureCode: Int? = null
            timing.measure("llm_generate") {
                module.resetContext()
                module.generate(prompt, config, object : LlmCallback {
                    override fun onResult(result: String) {
                        if (result.isNotEmpty()) timing?.tokenReceived()
                        pieces.append(result)
                        if (!stopRequested && (!context.isActive || stopWhen(pieces.toString()))) {
                            stopRequested = true
                            module.stop()
                        }
                    }

                    override fun onStats(stats: String) = Unit

                    override fun onError(errorCode: Int, message: String) {
                        failureCode = errorCode
                    }
                })
            }
            context.ensureActive()
            failureCode?.let { error("ExecuTorch generation failed (code $it)") }
            pieces.toString()
        }

    override fun close() {
        module.close()
    }
}
