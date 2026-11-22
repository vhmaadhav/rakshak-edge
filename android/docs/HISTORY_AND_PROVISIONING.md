# Local history and model provisioning

This note covers two data-layer pieces of the Android app that have no UI yet: the local check
history (`org.rakshak.edge.history`) and the offline model importer (`org.rakshak.edge.provision`).
Both keep everything on the device.

## Local history

### What is stored

Each check writes one JSON line to `filesDir/history/history.jsonl`:

| Field | Meaning |
|---|---|
| `id` | Random UUID, used to attach a correction or annotation later |
| `ts` | Epoch milliseconds when the check happened |
| `event` | `sms` or `link` |
| `tier` | `SCAM`, `VERIFY` or `SAFE` (the `Tier` enum name) |
| `detector` | Model id, for example `sms_student.onnx` |
| `evidence` | Rule ids that fired, for example `E1`, `E4` (never the matched text) |
| `explanation` | `generated` (Llama) or `template` |
| `annotation` | Optional user note, up to 500 characters |
| `correction` | Optional `{wrong_result, suspected_kind}`, `suspected_kind` up to 100 characters |
| `redacted` | Present only when the policy allows it (see below) |

The raw SMS text and the full URL are never written. The default policy stores no content at all.

### Redaction (opt-in)

With `HistoryPolicy(storeRedactedContent = true)`, the text is reduced before it is written:
emails become `[email]`, URLs and hosts become their registrable domain (or `[link]`), every
digit becomes `#`, and the result is capped at 120 characters. For a link, only the registrable
domain is kept. This is a reduction, not a guarantee: names and free words can remain, so the
default stays off.

### Retention

On every read and write the store:
1. drops records older than `retentionDays` (default **30**), then
2. keeps only the newest `maxRecords` (default **200**).

The file is rewritten through a temp file and an atomic rename.

### Corrections and annotations

A correction ("wrong result" plus what the user thinks it is) and an annotation are stored for
offline evaluation only. Nothing in the detector, the rules, the decision code or the explainer
reads the history store. A correction cannot change detector policy. `correct()` rewrites only
the record it names.

### Clear and export

- `clearAll()` deletes the history file.
- `exportJson()` returns one JSON document: `format` (`rakshak-history-v1`), `exported_at_ms`,
  `record_count` and `records`. `exportTo(OutputStream)` writes the same text as UTF-8. Use this
  with the Storage Access Framework (`CreateDocument`) for a user-chosen destination.

### API

```kotlin
val history = HistoryStore.forContext(context)          // or HistoryStore(dir, policy, clock)

history.record(NewEvent(
    eventType = EventType.SMS,                           // or LINK
    tier = Tier.SCAM,                                    // from the analysis result
    detectorId = "sms_student.onnx",
    evidenceIds = listOf("E1", "E4"),
    explanationSource = ExplanationSource.TEMPLATE,
    content = rawText,                                   // dropped unless the policy allows redaction
)): HistoryRecord
history.list(): List<HistoryRecord>
history.annotate(id: String, text: String?): Boolean
history.correct(id: String, correction: Correction?): Boolean   // null clears it
history.clearAll()
history.exportJson(): String
history.exportTo(out: OutputStream)
```

Call these off the main thread. They do file I/O and hold one lock.

## Model provisioning (offline import)

### Manifest

`ModelManifest.ARTIFACTS` lists the artifacts the app can import. Their names come from the
runtime constants in `llm/LlmExplainer.kt` (`ModelFiles.MODEL_NAME`, `ModelFiles.TOKENIZER_NAME`).
The bundled detector `sms_student.onnx` ships inside the APK and is not imported.

Each entry pins a SHA-256 digest of the exact file. The digests are taken from the repo records
for these exact filenames:

| Artifact | File | SHA-256 | Size | Recorded in |
|---|---|---|---|---|
| `llama_explainer` | `Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte` | `8715cdba9e91f6bede00cc5f2d6b12397b95225da5630c4972a8da03001cda3b` | 1,135,951,488 bytes | `docs/BASELINE.md`, `deploy/pi/bench/registry.toml`, `android/README.md` |
| `llama_tokenizer` | `tokenizer.model` | `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55` | 2,183,982 bytes | same |

The Pi copies were verified on 2026-11-06. The `.pte` was not re-hashed on 2026-11-12 (see the
table note in `android/README.md`). A file under the right name with different bytes is refused
(`HashMismatch`). A file with a different name is not an artifact the importer knows about.

### Storage

Files go to `ModelFiles.dir(context)`, the app's external files directory `models/`. This is the
same directory `Engine` and `ModelFiles` read from, so an imported file is used by the runtime
directly. Each import:
1. streams the chosen file into a temp file in the same directory, hashing it as it is written;
2. rejects the copy if it exceeds the artifact's `maxBytes`, or if the digest does not match;
3. renames the verified copy over the target with an atomic move.

A failed or mismatched import leaves any previous file in place. Stale `*.partial` files from an
earlier crash are removed at the start of the next import. The temp file is in the same directory
as the target, so the final rename never crosses a filesystem.

### API

```kotlin
val importer = ModelImporter.forContext(context)     // or ModelImporter(dir, manifest)

importer.importUri(id: String, resolver: ContentResolver, uri: Uri): ImportResult
importer.importStream(id: String, input: InputStream): ImportResult   // caller closes input
importer.status(id: String): ArtifactState           // MISSING | VERIFIED | MISMATCH | NOT_PINNED
importer.fileFor(id: String): File?
```

`ImportResult` is one of `Verified(file)`, `HashMismatch(expected, actual)` or `Rejected(reason)`.
`status()` hashes the file on disk, so call it from a background dispatcher.

### How the UI should call it

1. Register the picker with `ActivityResultContracts.OpenDocument()`. Launch it with a MIME
   filter of `application/octet-stream` (model files have no standard MIME type).
2. In the result callback, run `importUri(id, contentResolver, uri)` on `Dispatchers.IO`.
3. Show the `ImportResult` on the artifact status screen, then refresh `status(id)` for each row.
4. Do not persist the content URI permission. The copy is made within the same call.

Target ids are `llama_explainer` and `llama_tokenizer`.

`ModelFiles.dir(context)` returns null when the external files directory is unavailable.
`ModelImporter.forContext` then throws, with the same message the runtime reports.
