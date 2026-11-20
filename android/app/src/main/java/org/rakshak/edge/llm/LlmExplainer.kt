package org.rakshak.edge.llm

import android.content.Context
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmModule

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
class LlmExplainer(private val modelPath: String, private val tokenizerPath: String) :
    TokenGenerator, AutoCloseable {

    private var module: LlmModule? = null

    /** Loads the .pte on first use. Blocking, so it runs on Dispatchers.Default. */
    suspend fun load() {
        withContext(Dispatchers.Default) { loadBlocking() }
    }

    private fun loadBlocking() {
        if (module != null) return
        val m = LlmModule(modelPath, tokenizerPath, 0.0f)
        m.load()
        module = m
    }

    override suspend fun generate(prompt: String, maxNewTokens: Int, stopWhen: (String) -> Boolean): String =
        withContext(Dispatchers.Default) {
            loadBlocking()
            val m = module ?: error("explainer model is not loaded")
            val pieces = StringBuilder()
            var stopRequested = false
            m.resetContext()
            m.generate(
                prompt,
                maxNewTokens,
                object : LlmCallback {
                    override fun onResult(result: String) {
                        // Tokens after the stop request are kept: the caller cuts at the sentence end.
                        pieces.append(result)
                        if (!stopRequested && stopWhen(pieces.toString())) {
                            stopRequested = true
                            m.stop()
                        }
                    }

                    override fun onStats(stats: String) {
                        // Timing stats are not shown in the app.
                    }
                },
                false,
                0.0f,
                0,
                0,
            )
            pieces.toString()
        }

    override fun close() {
        module?.close()
        module = null
    }
}
