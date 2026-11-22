package org.rakshak.edge.history

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.rakshak.edge.TestData
import org.rakshak.edge.analysis.Tier

class HistoryStoreTest {
    @get:Rule
    val tmp = TemporaryFolder()

    private var now = 1_800_000_000_000L
    private val day = 24L * 60 * 60 * 1000

    private fun store(policy: HistoryPolicy = HistoryPolicy()) =
        HistoryStore(tmp.root, policy, clock = { now })

    private fun sms(content: String? = null, tier: Tier = Tier.SCAM) = NewEvent(
        eventType = EventType.SMS,
        tier = tier,
        detectorId = "sms_student.onnx",
        evidenceIds = listOf("E1", "E4"),
        explanationSource = ExplanationSource.TEMPLATE,
        content = content,
    )

    companion object {
        @JvmStatic
        @BeforeClass
        fun setUp() = TestData.install()
    }

    @Test
    fun redaction_keeps_only_registrable_domain_and_masks_digits() {
        assertEquals("sbi-kyc.in", Redaction.redactUrl("https://login.sbi-kyc.in/verify?otp=123456#x"))
        assertEquals("[link]", Redaction.redactUrl("http://192.168.1.1/login"))
        assertEquals(
            "Your OTP ###### for a/c ########## at sbi-kyc.in and mail [email]",
            Redaction.redactText("Your OTP 482913 for a/c 1234567890 at http://sbi-kyc.in/x and mail a.b@x.com"),
        )
    }

    @Test
    fun redaction_is_capped_in_length() {
        val long = "word ".repeat(100)
        assertTrue(Redaction.redactText(long).length <= Redaction.MAX_LEN)
    }

    @Test
    fun default_policy_stores_no_message_content() {
        val s = store()
        s.record(sms(content = "Pay 500 at http://sbi-kyc.in/x?otp=123 now"))
        val onDisk = java.io.File(tmp.root, HistoryStore.FILE_NAME).readText()
        assertFalse(onDisk.contains("otp"))
        assertFalse(onDisk.contains("sbi-kyc"))
        assertNull(s.list().single().redactedContent)
    }

    @Test
    fun redacted_content_is_kept_only_when_policy_allows() {
        val s = store(HistoryPolicy(storeRedactedContent = true))
        s.record(sms(content = "Pay 500 at http://sbi-kyc.in/x?otp=123 now"))
        val onDisk = java.io.File(tmp.root, HistoryStore.FILE_NAME).readText()
        assertFalse(onDisk.contains("otp=123"))
        assertEquals("Pay ### at sbi-kyc.in now", s.list().single().redactedContent)
    }

    @Test
    fun retention_removes_records_older_than_the_window() {
        val s = store(HistoryPolicy(retentionDays = 30))
        s.record(sms())
        now += 10 * day
        s.record(sms())
        now += 25 * day
        // First record is 35 days old and goes; the second is 25 days old and stays.
        assertEquals(1, s.list().size)
        now += 31 * day
        assertTrue(s.list().isEmpty())
    }

    @Test
    fun retention_caps_the_record_count_keeping_the_newest() {
        val s = store(HistoryPolicy(maxRecords = 3))
        val ids = (1..5).map { i ->
            now += 1000
            s.record(sms(tier = Tier.entries[i % 3])).id
        }
        assertEquals(ids.takeLast(3), s.list().map { it.id })
    }

    @Test
    fun clear_all_deletes_everything() {
        val s = store()
        s.record(sms())
        s.record(sms())
        s.clearAll()
        assertTrue(s.list().isEmpty())
        assertFalse(java.io.File(tmp.root, HistoryStore.FILE_NAME).exists())
    }

    @Test
    fun export_contains_records_and_no_raw_content() {
        val s = store()
        val a = s.record(sms(content = "secret body 999"))
        s.record(NewEvent(EventType.LINK, Tier.VERIFY, "sms_student.onnx", emptyList(), ExplanationSource.GENERATED))
        s.annotate(a.id, "from my bank")
        val json = JSONObject(s.exportJson())
        assertEquals(HistoryStore.EXPORT_FORMAT, json.getString("format"))
        assertEquals(2, json.getInt("record_count"))
        val records = json.getJSONArray("records")
        assertEquals(2, records.length())
        assertEquals("from my bank", records.getJSONObject(0).getString("annotation"))
        assertFalse(json.toString().contains("secret body"))
    }

    @Test
    fun correction_changes_only_the_named_record() {
        val s = store()
        val a = s.record(sms(tier = Tier.SCAM))
        val b = s.record(sms(tier = Tier.SAFE))
        s.annotate(b.id, "note on b")
        val before = s.list().associateBy { it.id }

        assertTrue(s.correct(a.id, Correction(wrongResult = true, suspectedKind = "lottery")))

        val after = s.list().associateBy { it.id }
        assertEquals(Correction(true, "lottery"), after.getValue(a.id).correction)
        // Everything except the correction field is unchanged on the corrected record...
        assertEquals(before.getValue(a.id), after.getValue(a.id).copy(correction = null))
        // ...and the other record is untouched entirely.
        assertEquals(before.getValue(b.id), after.getValue(b.id))
        assertEquals(Tier.SCAM, after.getValue(a.id).tier)
    }

    @Test
    fun correction_can_be_cleared_and_unknown_ids_are_refused() {
        val s = store()
        val a = s.record(sms())
        s.correct(a.id, Correction(wrongResult = true))
        assertTrue(s.correct(a.id, null))
        assertNull(s.list().single().correction)
        assertFalse(s.correct("no-such-id", Correction(wrongResult = true)))
        assertFalse(s.annotate("no-such-id", "x"))
    }

    @Test
    fun annotation_and_suspected_kind_are_capped_and_blank_is_null() {
        val s = store()
        val a = s.record(sms())
        s.annotate(a.id, "x".repeat(600))
        assertEquals(HistoryStore.MAX_ANNOTATION_LENGTH, s.list().single().annotation!!.length)
        s.correct(a.id, Correction(true, "  ${"k".repeat(300)}  "))
        assertEquals(HistoryStore.MAX_KIND_LENGTH, s.list().single().correction!!.suspectedKind!!.length)
        s.annotate(a.id, "   ")
        assertNull(s.list().single().annotation)
    }

    @Test
    fun damaged_lines_are_skipped() {
        val s = store()
        s.record(sms())
        java.io.File(tmp.root, HistoryStore.FILE_NAME).appendText("{not json\n{\"id\":\"../x\"}\n")
        assertEquals(1, s.list().size)
        assertTrue(s.list().all { it.evidenceIds == listOf("E1", "E4") })
    }

    @Test
    fun export_to_stream_writes_the_same_json() {
        val s = store()
        s.record(sms())
        val out = java.io.ByteArrayOutputStream()
        s.exportTo(out)
        assertEquals(HistoryStore.EXPORT_FORMAT, JSONObject(out.toString("UTF-8")).getString("format"))
    }
}
