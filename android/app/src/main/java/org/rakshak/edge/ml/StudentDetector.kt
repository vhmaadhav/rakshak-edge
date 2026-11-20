package org.rakshak.edge.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.exp
import org.rakshak.edge.StudentScorer
import org.rakshak.edge.analysis.Canon
import org.rakshak.edge.analysis.MAX_LEN
import org.rakshak.edge.analysis.encodeBytes
import org.rakshak.edge.analysis.ruleFlags

/**
 * The SMS student (dual-view char-CNN) on ONNX Runtime. Inputs, as exported by
 * src/rakshak/models/student.py export_onnx:
 *   raw    int64 [1, 256]  UTF-8 bytes of the message, padded with 256
 *   canon  int64 [1, 256]  UTF-8 bytes of the canonical message, padded with 256
 *   side   float32 [1, 16] rule flags over the canonical message (EVIDENCE_TYPES order)
 * Output "logit" float [1]; the scam probability is sigmoid(logit).
 */
class StudentDetector private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
) : StudentScorer, AutoCloseable {

    override fun score(text: String, canon: Canon): Double {
        val raw = encodeBytes(text)
        val canonIds = encodeBytes(canon.normalized)
        val side = ruleFlags(canon)
        val seqShape = longArrayOf(1, MAX_LEN.toLong())
        val sideShape = longArrayOf(1, side.size.toLong())
        return OnnxTensor.createTensor(env, LongBuffer.wrap(raw), seqShape).use { rawT ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(canonIds), seqShape).use { canonT ->
                OnnxTensor.createTensor(env, FloatBuffer.wrap(side), sideShape).use { sideT ->
                    val inputs = mapOf<String, OnnxTensor>("raw" to rawT, "canon" to canonT, "side" to sideT)
                    session.run(inputs).use { result ->
                        val logit = when (val value = result.get(0).value) {
                            is FloatArray -> value[0]
                            else -> error("unexpected logit type: ${value?.javaClass}")
                        }
                        1.0 / (1.0 + exp(-logit.toDouble()))
                    }
                }
            }
        }
    }

    override fun close() {
        session.close()
    }

    companion object {
        /** Loads the model from its bytes (the app reads assets/sms_student.onnx). */
        fun fromBytes(modelBytes: ByteArray): StudentDetector {
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
            val session = env.createSession(modelBytes, opts)
            val inputs = session.inputNames
            check(inputs.containsAll(listOf("raw", "canon", "side"))) { "unexpected student inputs: $inputs" }
            return StudentDetector(env, session)
        }
    }
}
