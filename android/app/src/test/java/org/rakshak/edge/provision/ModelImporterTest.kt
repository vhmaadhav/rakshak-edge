package org.rakshak.edge.provision

import java.io.ByteArrayInputStream
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.security.MessageDigest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.rakshak.edge.llm.ModelFiles

class ModelImporterTest {
    @get:Rule
    val tmp = TemporaryFolder()

    private val payload = "pretend this is a model file".repeat(50).toByteArray()
    private val payloadSha = sha(payload)

    private fun sha(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    private fun importer(sha: String? = payloadSha, maxBytes: Long = 1_000_000) = ModelImporter(
        targetDir = File(tmp.root, "models"),
        manifest = listOf(ArtifactSpec("model", "model.bin", sha, maxBytes)),
    )

    private fun partials(dir: File): List<String> =
        dir.listFiles()?.map { it.name }?.filter { it.endsWith(ModelImporter.PARTIAL_SUFFIX) }.orEmpty()

    /** Yields [head] bytes, then fails like a dropped SAF stream would. */
    private class FailingStream(private val head: ByteArray) : InputStream() {
        private var pos = 0
        override fun read(): Int {
            if (pos < head.size) return head[pos++].toInt() and 0xff
            throw IOException("connection dropped")
        }
    }

    @Test
    fun matching_hash_is_verified_and_installed() {
        val imp = importer()
        val result = imp.importStream("model", ByteArrayInputStream(payload))
        assertTrue(result is ImportResult.Verified)
        val target = (result as ImportResult.Verified).file
        assertArrayEquals(payload, target.readBytes())
        assertEquals(ArtifactState.VERIFIED, imp.status("model"))
        assertTrue(partials(target.parentFile).isEmpty())
    }

    @Test
    fun wrong_hash_is_rejected_and_nothing_is_installed() {
        val imp = importer(sha = sha("something else".toByteArray()))
        val result = imp.importStream("model", ByteArrayInputStream(payload))
        assertTrue("expected HashMismatch, got $result", result is ImportResult.HashMismatch)
        assertEquals(payloadSha, (result as ImportResult.HashMismatch).actual)
        assertEquals(ArtifactState.MISSING, imp.status("model"))
        assertFalse(File(tmp.root, "models/model.bin").exists())
        assertTrue(partials(File(tmp.root, "models")).isEmpty())
    }

    @Test
    fun atomic_replace_leaves_old_file_when_new_copy_fails_verification() {
        val imp = importer()
        imp.importStream("model", ByteArrayInputStream(payload))
        val target = File(tmp.root, "models/model.bin")

        // A copy whose bytes do not match the pinned digest must not replace the verified file.
        val result = imp.importStream("model", ByteArrayInputStream("corrupted bytes".toByteArray()))
        assertTrue(result is ImportResult.HashMismatch)
        assertArrayEquals(payload, target.readBytes())
        assertEquals(ArtifactState.VERIFIED, imp.status("model"))
    }

    @Test
    fun atomic_replace_swaps_in_a_new_verified_file() {
        val first = importer(sha = sha("v1".toByteArray()))
        first.importStream("model", ByteArrayInputStream("v1".toByteArray()))
        val second = importer(sha = sha("v2".toByteArray()))
        val result = second.importStream("model", ByteArrayInputStream("v2".toByteArray()))
        assertTrue(result is ImportResult.Verified)
        assertEquals("v2", File(tmp.root, "models/model.bin").readText())
        assertTrue(partials(File(tmp.root, "models")).isEmpty())
    }

    @Test
    fun partial_write_is_cleaned_up_and_previous_file_kept() {
        val imp = importer()
        imp.importStream("model", ByteArrayInputStream(payload))
        val target = File(tmp.root, "models/model.bin")

        val result = imp.importStream("model", FailingStream(payload.copyOf(100)))
        assertTrue("expected Rejected, got $result", result is ImportResult.Rejected)
        assertArrayEquals(payload, target.readBytes())
        assertTrue(partials(File(tmp.root, "models")).isEmpty())
    }

    @Test
    fun stale_partial_from_an_earlier_crash_is_removed_on_next_import() {
        val dir = File(tmp.root, "models").apply { mkdirs() }
        File(dir, "model.bin.123.partial").writeText("half a model")
        val result = importer().importStream("model", ByteArrayInputStream(payload))
        assertTrue(result is ImportResult.Verified)
        assertTrue(partials(dir).isEmpty())
    }

    @Test
    fun oversized_file_is_rejected_without_leaving_a_partial() {
        val imp = importer(maxBytes = 10)
        val result = imp.importStream("model", ByteArrayInputStream(payload))
        assertTrue(result is ImportResult.Rejected)
        assertFalse(File(tmp.root, "models/model.bin").exists())
        assertTrue(partials(File(tmp.root, "models")).isEmpty())
    }

    @Test
    fun unpinned_artifact_is_refused_and_reported_as_not_pinned() {
        val imp = importer(sha = null)
        assertTrue(imp.importStream("model", ByteArrayInputStream(payload)) is ImportResult.Rejected)
        File(tmp.root, "models").mkdirs()
        File(tmp.root, "models/model.bin").writeBytes(payload)
        assertEquals(ArtifactState.NOT_PINNED, imp.status("model"))
    }

    @Test
    fun unknown_artifact_is_refused() {
        val result = importer().importStream("nope", ByteArrayInputStream(payload))
        assertTrue(result is ImportResult.Rejected)
    }

    @Test
    fun status_reports_mismatch_for_a_file_that_does_not_hash_to_the_pin() {
        val dir = File(tmp.root, "models").apply { mkdirs() }
        File(dir, "model.bin").writeText("tampered after install")
        assertEquals(ArtifactState.MISMATCH, importer().status("model"))
    }

    @Test
    fun shipped_manifest_pins_the_recorded_digests_for_the_runtime_filenames() {
        val byId = ModelManifest.ARTIFACTS.associateBy { it.id }
        assertEquals(ModelFiles.MODEL_NAME, byId.getValue("llama_explainer").fileName)
        assertEquals(
            "8715cdba9e91f6bede00cc5f2d6b12397b95225da5630c4972a8da03001cda3b",
            byId.getValue("llama_explainer").sha256,
        )
        assertEquals(ModelFiles.TOKENIZER_NAME, byId.getValue("llama_tokenizer").fileName)
        assertEquals(
            "82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55",
            byId.getValue("llama_tokenizer").sha256,
        )
    }

    @Test
    fun shipped_manifest_refuses_a_file_that_has_the_right_name_but_wrong_bytes() {
        val imp = ModelImporter(File(tmp.root, "models"))
        val result = imp.importStream("llama_tokenizer", ByteArrayInputStream(payload))
        assertTrue("expected HashMismatch, got $result", result is ImportResult.HashMismatch)
        assertFalse(File(tmp.root, "models/${ModelFiles.TOKENIZER_NAME}").exists())
    }

    @Test
    fun invalid_digest_in_a_spec_is_rejected_at_construction() {
        try {
            ArtifactSpec("x", "x.bin", "NOT-HEX", 10)
            fail("expected IllegalArgumentException")
        } catch (expected: IllegalArgumentException) {
            // expected
        }
    }
}
