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
    internal val parallelSetup: Boolean,
) {
    private val stagesNs = linkedMapOf<String, Long>()
    private var firstTokenNs: Long? = null
    private var firstVerifiedPointNs: Long? = null

    @Synchronized
    internal fun recordStage(name: String, detail: String?, elapsedNs: Long) {
        stagesNs[name] = (stagesNs[name] ?: 0L) + elapsedNs
        if (detail != null) stagesNs[detail] = (stagesNs[detail] ?: 0L) + elapsedNs
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
        synchronized(this) {
            stagesNs.forEach { (name, nanos) -> stages.put(name, millis(nanos)) }
        }
        val sample = JSONObject()
            .put("id", id ?: JSONObject.NULL)
            .put("path", path)
            .put("parallel_setup", parallelSetup)
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
        const val EXTRA_PARALLEL_SETUP = "org.rakshak.edge.perf.PARALLEL_SETUP"

        fun start(
            path: String,
            id: Long? = null,
            fromProcessStart: Boolean = false,
            parallelSetup: Boolean = false,
        ): InferenceTiming? {
            if (!BuildConfig.INFERENCE_BENCHMARK) return null
            val start = if (fromProcessStart) {
                Process.getStartElapsedRealtime() * 1_000_000L
            } else {
                SystemClock.elapsedRealtimeNanos()
            }
            return InferenceTiming(path, id, start, parallelSetup)
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

private val ruleRegexTiming = ThreadLocal<InferenceTiming>()

internal fun activeRuleRegexTiming(): InferenceTiming? =
    if (BuildConfig.INFERENCE_BENCHMARK) ruleRegexTiming.get() else null

// The first run initializes Rules.kt and compiles its patterns on the calling thread.
// Scope the timing around that call so class initialization is included in the breakdown.
internal inline fun <T> InferenceTiming?.measureRules(work: () -> T): T = measure("rules") {
    if (this == null) return@measure work()
    ruleRegexTiming.set(this)
    try {
        work()
    } finally {
        ruleRegexTiming.remove()
    }
}

internal inline fun <T> InferenceTiming?.measure(name: String, detail: String? = null, work: () -> T): T {
    if (this == null) return work()
    val start = SystemClock.elapsedRealtimeNanos()
    Trace.beginSection("Rakshak/${detail ?: name}")
    try {
        return work()
    } finally {
        Trace.endSection()
        recordStage(name, detail, SystemClock.elapsedRealtimeNanos() - start)
    }
}
