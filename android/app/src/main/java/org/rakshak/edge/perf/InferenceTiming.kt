package org.rakshak.edge.perf

import android.os.Process
import android.os.SystemClock
import android.os.Trace
import android.util.Log
import java.io.File
import org.json.JSONObject
import org.rakshak.edge.BuildConfig

/** One UI request. Only the benchmark APK records timing, without logging the message or output. */
class InferenceTiming private constructor(
    private val path: String,
    private val id: Long?,
    private val startNs: Long,
) {
    private val stagesNs = linkedMapOf<String, Long>()
    private var firstTokenNs: Long? = null
    private var firstVerifiedPointNs: Long? = null

    internal fun recordStage(name: String, elapsedNs: Long) {
        stagesNs[name] = (stagesNs[name] ?: 0L) + elapsedNs
    }

    fun tokenReceived() {
        if (firstTokenNs == null) firstTokenNs = SystemClock.elapsedRealtimeNanos()
    }

    fun pointVerified() {
        if (firstVerifiedPointNs == null) firstVerifiedPointNs = SystemClock.elapsedRealtimeNanos()
    }

    fun finish(outcome: String, frameSubmitted: Boolean, modelPoints: Int?, totalPoints: Int?) {
        val endNs = SystemClock.elapsedRealtimeNanos()
        val stages = JSONObject()
        stagesNs.forEach { (name, nanos) -> stages.put(name, millis(nanos)) }
        val sample = JSONObject()
            .put("id", id ?: JSONObject.NULL)
            .put("path", path)
            .put("outcome", outcome)
            .put("frame_submitted", frameSubmitted)
            .put("elapsed_ms", millis(endNs - startNs))
            .put("first_token_ms", firstTokenNs?.let { millis(it - startNs) } ?: JSONObject.NULL)
            .put("first_verified_point_ms", firstVerifiedPointNs?.let { millis(it - startNs) } ?: JSONObject.NULL)
            .put("llm_points", modelPoints ?: JSONObject.NULL)
            .put("total_points", totalPoints ?: JSONObject.NULL)
            .put("process_peak_rss_kib", peakRssKiB() ?: JSONObject.NULL)
            .put("stages_ms", stages)
        Log.i("RakshakPerf", sample.toString())
    }

    companion object {
        const val EXTRA_SAMPLE_ID = "org.rakshak.edge.perf.SAMPLE_ID"

        fun start(path: String, id: Long? = null, fromProcessStart: Boolean = false): InferenceTiming? {
            if (!BuildConfig.INFERENCE_BENCHMARK) return null
            val start = if (fromProcessStart) {
                Process.getStartElapsedRealtime() * 1_000_000L
            } else {
                SystemClock.elapsedRealtimeNanos()
            }
            return InferenceTiming(path, id, start)
        }

        private fun millis(nanos: Long): Double = nanos / 1_000_000.0

        private fun peakRssKiB(): Long? = runCatching {
            File("/proc/self/status").useLines { lines ->
                lines.firstOrNull { it.startsWith("VmHWM:") }
                    ?.removePrefix("VmHWM:")?.trim()?.substringBefore(' ')?.toLongOrNull()
            }
        }.getOrNull()
    }
}

internal inline fun <T> InferenceTiming?.measure(name: String, work: () -> T): T {
    if (this == null) return work()
    val start = SystemClock.elapsedRealtimeNanos()
    Trace.beginSection("Rakshak/$name")
    try {
        return work()
    } finally {
        Trace.endSection()
        recordStage(name, SystemClock.elapsedRealtimeNanos() - start)
    }
}
