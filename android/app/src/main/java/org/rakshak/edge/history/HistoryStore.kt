package org.rakshak.edge.history

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.UUID
import org.json.JSONArray
import org.json.JSONObject
import org.rakshak.edge.analysis.Tier

/**
 * Retention and storage policy. Defaults keep 30 days and at most 200 records, and store no
 * message content at all.
 */
data class HistoryPolicy(
    val retentionDays: Int = 30,
    val maxRecords: Int = 200,
    /** When true, [NewEvent.content] is reduced by [Redaction] and kept. When false, it is dropped. */
    val storeRedactedContent: Boolean = false,
) {
    init {
        require(retentionDays >= 1) { "retentionDays must be at least 1" }
        require(maxRecords >= 1) { "maxRecords must be at least 1" }
    }
}

/** What the caller passes after a check. [content] is the raw SMS text or URL. */
data class NewEvent(
    val eventType: EventType,
    val tier: Tier,
    val detectorId: String,
    val evidenceIds: List<String>,
    val explanationSource: ExplanationSource,
    val content: String? = null,
)

/**
 * Local, on-device check history, stored as JSON lines in app-private storage.
 *
 * What is kept: timestamp, event type, risk tier, detector id, evidence ids, explanation source,
 * an optional user annotation, and an optional correction. What is never kept: raw message text
 * or full URLs. Redacted content is kept only when [HistoryPolicy.storeRedactedContent] is true.
 *
 * Corrections and annotations are stored for offline evaluation only. Nothing in the detector,
 * the rules, the decision code or the explainer reads this store, so a correction cannot change
 * detector policy. [correct] only rewrites the one record it names.
 *
 * Retention runs on every read and write: records older than the policy's retention window are
 * removed, then the oldest records beyond [HistoryPolicy.maxRecords] are removed.
 *
 * Calls are synchronised on one lock and do file I/O. Call them from a background dispatcher.
 */
class HistoryStore(
    private val dir: File,
    private val policy: HistoryPolicy = HistoryPolicy(),
    private val clock: () -> Long = System::currentTimeMillis,
) {
    private val file = File(dir, FILE_NAME)
    private val lock = Any()

    /** Appends one record. Returns the stored record, which has the id the UI can correct later. */
    fun record(event: NewEvent): HistoryRecord = synchronized(lock) {
        val now = clock()
        val redacted = if (policy.storeRedactedContent) {
            event.content?.let { raw ->
                if (event.eventType == EventType.LINK) Redaction.redactUrl(raw) else Redaction.redactText(raw)
            }
        } else {
            null
        }
        val record = HistoryRecord(
            id = UUID.randomUUID().toString(),
            timestampMs = now,
            eventType = event.eventType,
            tier = event.tier,
            detectorId = event.detectorId,
            evidenceIds = event.evidenceIds,
            explanationSource = event.explanationSource,
            redactedContent = redacted,
        )
        writeAll(prune(readAll() + record, now))
        record
    }

    /** Current records after retention, oldest first. */
    fun list(): List<HistoryRecord> = synchronized(lock) {
        val now = clock()
        val stored = readAll()
        val kept = prune(stored, now)
        if (kept.size != stored.size) writeAll(kept)
        kept
    }

    /** Sets or clears the user's annotation on one record. Returns false if the id is unknown. */
    fun annotate(id: String, text: String?): Boolean = updateOne(id) { it.copy(annotation = clean(text, MAX_ANNOTATION_LENGTH)) }

    /**
     * Sets or clears the user's correction on one record. Only the named record changes. Pass null
     * to clear it. Returns false if the id is unknown.
     */
    fun correct(id: String, correction: Correction?): Boolean = updateOne(id) {
        it.copy(correction = correction?.let { c -> Correction(c.wrongResult, clean(c.suspectedKind, MAX_KIND_LENGTH)) })
    }

    /** Deletes every record and the history file. */
    fun clearAll() = synchronized(lock) {
        file.delete()
        File(dir, TMP_NAME).delete()
    }

    /** The whole history as one JSON document: a format tag, the export time, and the records. */
    fun exportJson(): String = synchronized(lock) {
        val records = list()
        JSONObject().apply {
            put("format", EXPORT_FORMAT)
            put("exported_at_ms", clock())
            put("record_count", records.size)
            put("records", JSONArray().apply { records.forEach { put(it.toJson()) } })
        }.toString(2)
    }

    /** Writes [exportJson] to [out] as UTF-8. The caller owns and closes the stream. */
    fun exportTo(out: OutputStream) {
        out.write(exportJson().toByteArray(Charsets.UTF_8))
        out.flush()
    }

    private fun updateOne(id: String, change: (HistoryRecord) -> HistoryRecord): Boolean = synchronized(lock) {
        val now = clock()
        val all = prune(readAll(), now)
        val index = all.indexOfFirst { it.id == id }
        if (index < 0) return@synchronized false
        val updated = all.toMutableList().also { it[index] = change(all[index]) }
        writeAll(updated)
        true
    }

    private fun prune(records: List<HistoryRecord>, now: Long): List<HistoryRecord> {
        val cutoff = now - policy.retentionDays * DAY_MS
        val fresh = records.filter { it.timestampMs >= cutoff }.sortedBy { it.timestampMs }
        return if (fresh.size > policy.maxRecords) fresh.takeLast(policy.maxRecords) else fresh
    }

    private fun readAll(): List<HistoryRecord> {
        if (!file.isFile) return emptyList()
        return file.readLines(Charsets.UTF_8).mapNotNull { line ->
            if (line.isBlank()) return@mapNotNull null
            runCatching { HistoryRecord.fromJson(JSONObject(line)) }.getOrNull()
        }
    }

    /** Rewrites the whole file through a temp file and an atomic rename, so a crash never leaves half a file. */
    private fun writeAll(records: List<HistoryRecord>) {
        if (records.isEmpty()) {
            file.delete()
            return
        }
        dir.mkdirs()
        val tmp = File(dir, TMP_NAME)
        try {
            FileOutputStream(tmp).use { out ->
                val bytes = records.joinToString(separator = "\n", postfix = "\n") { it.toJson().toString() }
                    .toByteArray(Charsets.UTF_8)
                out.write(bytes)
                out.fd.sync()
            }
            Files.move(tmp.toPath(), file.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        } finally {
            tmp.delete()
        }
    }

    companion object {
        const val FILE_NAME = "history.jsonl"
        const val TMP_NAME = "history.jsonl.tmp"
        const val EXPORT_FORMAT = "rakshak-history-v1"
        const val MAX_ANNOTATION_LENGTH = 500
        const val MAX_KIND_LENGTH = 100
        private const val DAY_MS = 24L * 60 * 60 * 1000

        /** The app's history store, in app-private storage under filesDir/history. */
        fun forContext(context: Context, policy: HistoryPolicy = HistoryPolicy()): HistoryStore =
            HistoryStore(File(context.filesDir, "history"), policy)

        /** Trims, blanks to null, and caps the length. */
        private fun clean(text: String?, max: Int): String? =
            text?.trim()?.takeIf { it.isNotEmpty() }?.take(max)
    }
}
