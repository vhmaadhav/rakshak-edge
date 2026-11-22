package org.rakshak.edge.history

import org.json.JSONArray
import org.json.JSONObject
import org.rakshak.edge.analysis.Tier

/** What the user checked. The wire value is what is written to the history file. */
enum class EventType(val wire: String) {
    SMS("sms"),
    LINK("link");

    companion object {
        fun fromWire(value: String): EventType? = entries.firstOrNull { it.wire == value }
    }
}

/** Whether the explanation on screen came from the Llama explainer or from the template. */
enum class ExplanationSource(val wire: String) {
    GENERATED("generated"),
    TEMPLATE("template");

    companion object {
        fun fromWire(value: String): ExplanationSource? = entries.firstOrNull { it.wire == value }
    }
}

/**
 * The user's correction of one result. Stored for offline evaluation only. It is never read by
 * the detector, the rules, the decision code or the explainer, so it cannot change detector
 * policy. [suspectedKind] is the user's own free-text guess, capped at [HistoryStore.MAX_KIND_LENGTH].
 */
data class Correction(
    val wrongResult: Boolean,
    val suspectedKind: String? = null,
)

/**
 * One local history entry. It holds no message text and no full URL. [redactedContent] is only
 * ever set when the store's policy allows it (see [HistoryPolicy.storeRedactedContent]).
 *
 * [detectorId] names the model that produced the decision (for example "sms_student.onnx").
 * [evidenceIds] are the rule ids that fired (for example "E1", "E4"), not the matched text.
 */
data class HistoryRecord(
    val id: String,
    val timestampMs: Long,
    val eventType: EventType,
    val tier: Tier,
    val detectorId: String,
    val evidenceIds: List<String>,
    val explanationSource: ExplanationSource,
    val annotation: String? = null,
    val correction: Correction? = null,
    val redactedContent: String? = null,
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id)
        put("ts", timestampMs)
        put("event", eventType.wire)
        put("tier", tier.name)
        put("detector", detectorId)
        put("evidence", JSONArray(evidenceIds))
        put("explanation", explanationSource.wire)
        annotation?.let { put("annotation", it) }
        correction?.let { c ->
            put("correction", JSONObject().apply {
                put("wrong_result", c.wrongResult)
                c.suspectedKind?.let { put("suspected_kind", it) }
            })
        }
        redactedContent?.let { put("redacted", it) }
    }

    companion object {
        private val ID_RE = Regex("[A-Za-z0-9_-]{1,64}")
        private val EVIDENCE_RE = Regex("[A-Za-z0-9_-]{1,16}")
        private val DETECTOR_RE = Regex("[A-Za-z0-9_.-]{1,64}")

        /** Returns null for a line that does not match the schema, so a damaged line is skipped. */
        fun fromJson(json: JSONObject): HistoryRecord? {
            val id = json.optString("id", "")
            val eventType = EventType.fromWire(json.optString("event", "")) ?: return null
            val tier = runCatching { Tier.valueOf(json.optString("tier", "")) }.getOrNull() ?: return null
            val explanation = ExplanationSource.fromWire(json.optString("explanation", "")) ?: return null
            val detector = json.optString("detector", "")
            if (!ID_RE.matches(id) || !DETECTOR_RE.matches(detector) || !json.has("ts")) return null
            val evidence = json.optJSONArray("evidence")?.let { arr ->
                (0 until arr.length()).mapNotNull { i -> arr.optString(i, "").takeIf { EVIDENCE_RE.matches(it) } }
            }.orEmpty()
            val correction = json.optJSONObject("correction")?.let { c ->
                Correction(
                    wrongResult = c.optBoolean("wrong_result", false),
                    suspectedKind = c.optString("suspected_kind", "").takeIf { it.isNotEmpty() },
                )
            }
            return HistoryRecord(
                id = id,
                timestampMs = json.optLong("ts"),
                eventType = eventType,
                tier = tier,
                detectorId = detector,
                evidenceIds = evidence,
                explanationSource = explanation,
                annotation = json.optString("annotation", "").takeIf { it.isNotEmpty() },
                correction = correction,
                redactedContent = json.optString("redacted", "").takeIf { it.isNotEmpty() },
            )
        }
    }
}
