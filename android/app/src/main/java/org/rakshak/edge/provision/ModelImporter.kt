package org.rakshak.edge.provision

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.DigestInputStream
import java.security.MessageDigest
import org.rakshak.edge.llm.ModelFiles

/**
 * One model artifact the app can use. [sha256] is the lowercase hex digest of the exact file that
 * is expected. Null means no digest is pinned yet: such an artifact can be reported but never
 * imported, so nothing unverified can reach the app.
 */
class ArtifactSpec(
    val id: String,
    val fileName: String,
    val sha256: String?,
    val maxBytes: Long,
) {
    init {
        require(Regex("[A-Za-z0-9_.-]{1,64}").matches(id)) { "bad artifact id: $id" }
        require(Regex("[A-Za-z0-9_.-]{1,128}").matches(fileName)) { "bad file name: $fileName" }
        require(sha256 == null || Regex("[0-9a-f]{64}").matches(sha256)) { "sha256 must be 64 lowercase hex digits" }
        require(maxBytes > 0) { "maxBytes must be positive" }
    }
}

/**
 * The model identity manifest. The names come from the same constants the runtime loads
 * (ModelFiles in llm/LlmExplainer.kt), and the directory is ModelFiles.dir(context), so the importer
 * writes exactly where the runtime reads.
 *
 * The digests are the ones recorded in the repo for these exact files: docs/BASELINE.md and
 * deploy/pi/bench/registry.toml (the Pi copies were verified 2026-11-06), and android/README.md.
 * The .pte and tokenizer sizes match those records. Each file's digest is only trusted against its
 * own filename, so a renamed file is refused.
 */
object ModelManifest {
    private const val GIB = 1024L * 1024L * 1024L
    private const val MIB = 1024L * 1024L

    val ARTIFACTS: List<ArtifactSpec> = listOf(
        ArtifactSpec(
            id = "llama_explainer",
            fileName = ModelFiles.MODEL_NAME,
            sha256 = "8715cdba9e91f6bede00cc5f2d6b12397b95225da5630c4972a8da03001cda3b",
            maxBytes = 4 * GIB,
        ),
        ArtifactSpec(
            id = "llama_tokenizer",
            fileName = ModelFiles.TOKENIZER_NAME,
            sha256 = "82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55",
            maxBytes = 16 * MIB,
        ),
    )
}

/** Status of one artifact on disk, for the artifact status screen. */
enum class ArtifactState {
    /** No file under that name in the model directory. */
    MISSING,

    /** The file is present and its SHA-256 equals the pinned digest. */
    VERIFIED,

    /** The file is present but its SHA-256 differs from the pinned digest. */
    MISMATCH,

    /** The file is present, but no digest is pinned, so it cannot be verified. */
    NOT_PINNED,
}

/** Outcome of one import attempt. */
sealed interface ImportResult {
    /** The file was verified and atomically moved into place. */
    data class Verified(val file: File) : ImportResult

    /** The copied bytes do not hash to the pinned digest. The temp copy was deleted and nothing was replaced. */
    data class HashMismatch(val expected: String, val actual: String) : ImportResult

    /** The import was refused or failed. Any previous file under that name is unchanged. */
    data class Rejected(val reason: String) : ImportResult
}

/**
 * Imports model files offline from a user-chosen file, verifies them, and installs them into the
 * runtime's model directory, ModelFiles.dir(context) (the app's external files directory
 * `models/`, the same place Engine reads from).
 *
 * Each import streams the bytes into a temp file in the same directory and hashes them as they
 * are written. Only a verified copy is renamed over the target, with ATOMIC_MOVE. A mismatch, an
 * oversized file, or any I/O error deletes the temp file and leaves the previous file in place.
 * Stale temp files from an earlier crash are removed at the start of each import.
 *
 * How the UI should launch the picker (not wired yet; the UI calls this later):
 * ```
 * val picker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
 *     if (uri != null) {
 *         lifecycleScope.launch(Dispatchers.IO) {
 *             val result = importer.importUri("llama_explainer", contentResolver, uri)
 *             // show result on the artifact status screen, then refresh status(...)
 *         }
 *     }
 * }
 * picker.launch(arrayOf("application/octet-stream"))
 * ```
 * The picker returns a content:// Uri with a read grant for this session only. Do not persist
 * the grant: a copy is made in the same call, and nothing refers back to the original Uri.
 *
 * Call every method off the main thread. Hashing a multi-gigabyte file takes a while.
 */
class ModelImporter(
    private val targetDir: File,
    private val manifest: List<ArtifactSpec> = ModelManifest.ARTIFACTS,
) {
    init {
        require(manifest.map { it.id }.toSet().size == manifest.size) { "duplicate artifact ids" }
    }

    /** The spec for [id], or null when the manifest does not list it. */
    fun artifact(id: String): ArtifactSpec? = manifest.firstOrNull { it.id == id }

    /** The file that holds the verified artifact [id], whether or not it exists. */
    fun fileFor(id: String): File? = artifact(id)?.let { File(targetDir, it.fileName) }

    /** Status of artifact [id] on disk. Hashes the file when it exists and a digest is pinned. */
    fun status(id: String): ArtifactState {
        val spec = artifact(id) ?: return ArtifactState.MISSING
        val file = File(targetDir, spec.fileName)
        if (!file.isFile) return ArtifactState.MISSING
        val expected = spec.sha256 ?: return ArtifactState.NOT_PINNED
        return if (sha256Of(file) == expected) ArtifactState.VERIFIED else ArtifactState.MISMATCH
    }

    /** Imports from a SAF [uri] (see the class KDoc). Closes the stream it opens. */
    fun importUri(id: String, resolver: ContentResolver, uri: Uri): ImportResult {
        val input: InputStream? = try {
            resolver.openInputStream(uri)
        } catch (e: IOException) {
            return ImportResult.Rejected("could not open the chosen file: ${e.message}")
        }
        if (input == null) return ImportResult.Rejected("the chosen file is not readable")
        return input.use { importStream(id, it) }
    }

    /**
     * Imports from [input]. The caller owns [input] and closes it. Reads until EOF.
     */
    fun importStream(id: String, input: InputStream): ImportResult {
        val spec = artifact(id) ?: return ImportResult.Rejected("unknown artifact '$id'")
        val expected = spec.sha256
            ?: return ImportResult.Rejected("no digest is pinned for '$id', so the file cannot be verified")
        if (!targetDir.isDirectory && !targetDir.mkdirs()) {
            return ImportResult.Rejected("cannot create the model directory")
        }
        removeStaleTemps()
        val tmp = File.createTempFile("${spec.fileName}.", PARTIAL_SUFFIX, targetDir)
        try {
            val digest = MessageDigest.getInstance("SHA-256")
            var written = 0L
            FileOutputStream(tmp).use { out ->
                val buffer = ByteArray(BUFFER_SIZE)
                val hashed = DigestInputStream(input, digest)
                while (true) {
                    val n = hashed.read(buffer)
                    if (n < 0) break
                    written += n
                    if (written > spec.maxBytes) {
                        return ImportResult.Rejected("the file is larger than ${spec.maxBytes} bytes")
                    }
                    out.write(buffer, 0, n)
                }
                out.flush()
                out.fd.sync()
            }
            val actual = digest.digest().toHex()
            if (actual != expected) {
                return ImportResult.HashMismatch(expected, actual)
            }
            val target = File(targetDir, spec.fileName)
            Files.move(tmp.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
            return ImportResult.Verified(target)
        } catch (e: IOException) {
            return ImportResult.Rejected("the copy failed: ${e.message}")
        } finally {
            // Removes the temp file on every path that did not move it into place.
            if (tmp.exists()) tmp.delete()
        }
    }

    private fun removeStaleTemps() {
        targetDir.listFiles()?.filter { it.name.endsWith(PARTIAL_SUFFIX) }?.forEach { it.delete() }
    }

    companion object {
        const val PARTIAL_SUFFIX = ".partial"
        private const val BUFFER_SIZE = 64 * 1024

        /**
         * The runtime's model directory (ModelFiles.dir). Throws if the external files directory is
         * unavailable, which matches what the runtime reports in that case.
         */
        fun forContext(context: Context): ModelImporter =
            ModelImporter(checkNotNull(ModelFiles.dir(context)) { "No external files directory" })

        /** SHA-256 of the file's contents, as lowercase hex. */
        fun sha256Of(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input ->
                val buffer = ByteArray(BUFFER_SIZE)
                while (true) {
                    val n = input.read(buffer)
                    if (n < 0) break
                    digest.update(buffer, 0, n)
                }
            }
            return digest.digest().toHex()
        }

        private fun ByteArray.toHex(): String = joinToString("") { (it.toInt() and 0xff).toString(16).padStart(2, '0') }
    }
}
