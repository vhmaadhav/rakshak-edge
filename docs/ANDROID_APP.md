# Android app (Rakshak Edge): architecture, parity and open items

Status (2026-11-20): **compiles, and the 27 JVM unit tests that existed then pass, including the 8 parity
tests over the 318-row Python fixture.** (`app/src/test` now has 53; HistoryStoreTest and ModelImporterTest
were added on 2026-11-22.) Built headless on the InstaBox VM (Debian x86_64, JDK 21, Gradle 9.8.0
wrapper, AGP 9.5.1, platform android-37.2, build-tools 37.0.0) with
`./gradlew :app:assembleDebug :app:testDebugUnitTest`. Debug APK: 78,719,078 bytes (arm64-v8a only,
ExecuTorch and ONNX Runtime native libraries included, no student ONNX and no Llama files inside).
**Device run (added 2026-11-23):** a detector-only benchmark on a OnePlus 13R (Android 16) is reported in
[benchmarks.md](benchmarks.md) section 1; its raw output is not committed, and Llama was not run on the
phone. No screenshots are committed. The parity tests run on the JVM's
`java.util.regex`, not Android's regex engine.

## 1. What the app does

Offline by construction. The manifest declares no INTERNET, no SMS and no network permission. A
message arrives by paste or by the system share sheet (`ACTION_SEND`, `text/plain`).

For each message:

1. **Canonicalise** (`analysis/Canonicalizer.kt`): NFKC, zero-width and bidi strip, homoglyph
   fold, `hxxp`/`[.]` de-obfuscation, URL extraction, spaced and dotted letter collapse, leetspeak
   fold.
2. **Rules** (`analysis/Rules.kt`): the 16 evidence types, in the same order as
   `src/rakshak/rules.py`, so the ids E1..En match. Domain rules use `analysis/UrlFeatures.kt`.
3. **Student** (`ml/StudentDetector.kt`): ONNX Runtime over `sms_student.onnx`. Inputs are the
   UTF-8 bytes of the raw text (`raw`), the bytes of the canonical text (`canon`), and the 16 rule
   flags over the canonical text (`side`). Output is a logit; the probability is its sigmoid.
4. **Decision** (`analysis/Decision.kt`, `Evidence.kt`): scam when the score is at or above
   `0.35127517580986023`; otherwise uncertain when a rule fired; otherwise benign. Quiet mode is the
   Pi default: a detector-only flag (scam with no rule evidence) is LOW, IGNORE, with the neutral
   Fix B text. Risk and action follow `build_package` and `choose_action`.
5. **Tier** shown on screen: HIGH or MEDIUM is SCAM, UNCERTAIN is VERIFY, LOW is SAFE. SAFE means
   "no known pattern", and the screen always says so. The Pi's own wording is kept.
6. **Remediation** (`analysis/Remediation.kt`): English and Hinglish catalogue.
7. **Explanation** (optional, `llm/`): Llama 3.2 1B SpinQuant on ExecuTorch writes one sentence per
   evidence type (Fix A `describe_kw` prompt, Fix D one call per type, Fix C echo strip and
   duplicate blanking). Each sentence passes the verifier (`analysis/Verifier.kt`): grounding,
   safe claims at HIGH and MEDIUM, positive instructions, injection phrasing, the lexicon keyword,
   and guard G2. A failing point falls back to its template. G1: LOW risk makes no model call.
   Hinglish uses the template only, as the Pi default does.

The summary text is always the template, as on the Pi. The model never changes the risk, the
action or the evidence.

## 2. Parity with the Python code

| Area | Kotlin | How it is checked |
|---|---|---|
| Canonical text, URLs | `Canonicalizer.kt` | `ParityTest.canonicalForm_and_urls_match` (318 rows) |
| 16 rule evidence (id, type, weight, matched text) | `Rules.kt` | `rule_evidence_matches` |
| 16 side flags, canonical and raw views | `Rules.kt` `ruleFlags` | `side_flags_match_for_both_views` |
| Student byte ids, raw and canonical | `Encoding.kt` | `byte_ids_match_the_student_input` |
| URL features and registered domain | `UrlFeatures.kt` | `url_features_match` |
| `detect_lang` | `LangDetect.kt` | `language_detection_matches` |
| Fact text and Llama describe_kw prompt | `llm/SlotExplainer.kt` | `facts_and_llama_prompts_match` |
| Risk, allowed actions, action, template summary, for each scam state | `Evidence.kt`, `Verifier.kt` | `decision_packages_match_for_each_state` |
| Public-suffix lookup (ICANN rules) | `UrlFeatures.kt` | `tools/export_assets.py` checks its reference against tldextract on 28 hosts (`suffix_check.json`, no mismatches); `UnitTest` checks a few hosts |

The fixture (`app/src/test/resources/parity_fixture.json`, 318 rows, 1.12 MB) comes from
`tools/make_parity_fixture.py` on the build VM. Rows (exact duplicate texts dropped): 180 from the SMS
campaign test split (`splits/campaign.json` partition `test`, matched on the `id` column; sampled 75 per
label, plus 15 Hinglish and 15 Devanagari rows; 95 benign and 85 scam after de-duplication), 49 GroundLM,
50 SmishX (text present), and 39 hand-made obfuscation cases (zero-width, homoglyphs, fullwidth,
mathematical letters, bidi, emoji, URLs with punycode, hxxp, IP literals, a 300-character message, a
100-character Devanagari run). 55 rows have rule evidence. Two rows are cut at 400 characters.

Known differences that the fixture may expose:

- **Offsets.** Spans are UTF-16 offsets in Kotlin and code-point offsets in Python. The fixture does
  not compare spans. The 30-character negation look-behind of the credential rule could differ
  after a supplementary-plane character (the emoji row exercises this).
- **Regex engine.** Python's `regex` and `re` are emulated with explicit classes (`PyRegex.kt`) and
  not with the Unicode flags, so the engine on Android is the ICU-based one. Unverified until the
  fixture runs.
- **SequenceMatcher autojunk.** `difflib` removes popular characters for strings of 200 or more
  characters. The Kotlin ratio does not. It matters only for two points of 200+ characters that
  are near-identical.
- **Brand order.** `brands.json` is read with `JSONObject`, which does not keep key order. The
  rules only test membership, so the decision does not change. `mentioned_brand_domains` order
  (used only in the Python prompt JSON) can differ.
- **Non-ASCII hosts.** Hosts are converted with `java.net.IDN.toASCII`. tldextract uses its own
  IDNA handling, so a rare IDN host may differ.
- **urlsplit.** The host is read by an approximation of `urllib.parse.urlsplit`. The fallback
  matches the Python helper for the cases in the fixture; other malformed URLs are not covered.

## 3. Versions

Checked on 2026-11-20 (rule: at least 14 days old).

| Component | Version | Source and date |
|---|---|---|
| ExecuTorch Android (`org.pytorch:executorch-android`) | 1.5.1 | Maven Central directory: 2026-10-01 (`repo1.maven.org/maven2/org/pytorch/executorch-android/`). Also the version the Pi runs (`deploy/pi/bench/rbench/runners/pte_llm.py`). |
| ONNX Runtime Android (`com.microsoft.onnxruntime:onnxruntime-android`) | 1.31.1 | Maven Central directory: 2026-10-26. 1.32.0 exists but is dated 2026-11-13 (research brief), so excluded. |
| Android Gradle Plugin | 9.5.1 | Google Maven metadata lists it as the newest stable (9.6.0-alpha04 is newer but alpha). Its release date was **not** confirmed. Release notes: min Gradle 9.6.0, JDK 17, build-tools 36.0.0, max API 37. |
| Gradle wrapper | 9.8.0 | Research brief (services.gradle.org, 2026-09-24). Not re-fetched in this session. |
| Kotlin (`kotlin-gradle-plugin`, for the Compose compiler plugin) | 2.4.21 | Maven Central directory: 2026-10-08. 2.5.0-RC is excluded. |
| Jetpack Compose BOM | 2026.11.01 | Google Maven metadata (latest); dated 2026-11-01 in the research brief. |
| `androidx.activity:activity-compose` | 1.13.0 | Google Maven metadata, latest stable. Date not checked. |
| `androidx.lifecycle:lifecycle-runtime-ktx` | 2.11.1 | Google Maven metadata, latest stable. Date not checked. |
| `kotlinx-coroutines-android` | 1.10.2 | Maven Central metadata; 1.11.0 exists, date not checked, so the older stable is used. |
| JUnit | 4.13.2 | Maven Central metadata (test only). |
| `org.json:json` (test only) | 20251224 | Maven Central metadata. Android's `org.json` is a stub on the JVM. |
| compileSdk / targetSdk / minSdk | 37.2 / 36 / 28 | The first build failed on 36: Compose BOM 2026.11.01 (`ui-android` 1.13.0) needs compileSdk 37.1 or later, and AndroidX core 1.19.0 and lifecycle 2.11.1 need 37. compileSdk is now `release(37) { minorApiLevel = 2 }`. targetSdk stays 36. |

The AGP 9.x built-in Kotlin support is used: `org.jetbrains.kotlin.android` is **not** applied.
The Compose compiler plugin is applied at Kotlin 2.4.21. Whether AGP 9.5.1's built-in Kotlin uses
that KGP version without extra configuration is unverified (see section 5).

## 4. Explainer files and the Llama notice

- Files: `Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte` and `tokenizer.model`, in
  `/sdcard/Android/data/org.rakshak.edge/files/models/` (app external files directory). Not in
  the APK. Steps are in `android/README.md`.
- Greedy decoding (temperature 0), `num_bos=0`, `num_eos=0`, KV cache reset before each prompt,
  20 new tokens per point, stop on a sentence end or on `"`, `{`, `}`, newline (the Pi's
  `slot_stop`).
- **Llama licence notice.** The About dialog shows "Built with Llama" and the licence text from
  `assets/rakshak/llama3_2_LICENSE.txt`.

## 5. Not done yet, and risks

- **Build history (first compile, 2026-11-20).** These three problems were found and fixed:
  1. `tools/build_headless.sh` exited with rc 141. Under `pipefail`, `yes` dies of SIGPIPE in
     `yes | sdkmanager --licenses`.
  2. compileSdk had to move from 36 to 37.2 (section 3).
  3. `PyRegex.kt` rejected `\S`, `\D` and `\B`, and the verifier's URL pattern uses `\S`. That
     broke static initialisation of `VerifierKt` and `SlotExplainerKt` and failed 8 tests. Each is
     now translated: `\S` to "not Python whitespace", `\D` to `\P{Nd}`, and `\B` to "not a
     word boundary" (outside character classes only). The new test
     `negated_python_classes_translate` covers them.

  The ExecuTorch and ONNX Runtime calls below compile against the real AARs (1.5.1, 1.31.1), so
  their signatures are right. Their runtime behaviour is still unchecked. The VM build used
  `org.gradle.jvmargs=-Xmx1536m` and `org.gradle.workers.max=3` (not committed) to fit the 7 GB VM.
- **Device run: detector only.** See [benchmarks.md](benchmarks.md) section 1 (OnePlus 13R; raw output not
  committed). No device test of the Llama explainer and no screenshots.
- **API calls checked only at compile time:**
  - ExecuTorch `LlmModule(String, String, Float)`, `load()`, `resetContext()`, `stop()`,
    `close()` and the 7-argument `generate(prompt, seqLen, callback, echo, temperature, numBos,
    numEos)` are taken from the research brief's javadoc summary. The Java `LlmCallback` shape
    (`onResult`, `onStats`) is also from the brief.
  - ONNX Runtime `OnnxTensor.createTensor(env, LongBuffer, long[])`, `session.run(map)` and
    `result.get(0).value`, and `env.createSession(byte[], opts)`, are from the ORT Java API as the
    brief describes it. Not checked against javadoc.
- **AGP built-in Kotlin with KGP 2.4.21.** Works as configured: the build compiles with only the
  Compose plugin applied at 2.4.21.
- **Android regex engine.** The app avoids `UNICODE_CHARACTER_CLASS` and `Character.getName`, but
  the fixture has not been run on Android's regex.
- **Model file.** The ONNX student is not in git. Without it, the detector reports a missing asset.
- **SAFE label.** LOW risk shows "SAFE (no known pattern)". The Pi's own text says LOW does not mean
  safe. The screen repeats that, but a one-word tier name is still a usability risk that a
  reviewer should look at.
- **Hinglish wording.** The Hinglish remediation text is a first draft and needs a native-speaker
  review before release (as `remediation.py` says).
- **Not ported:** the URL model (`url_model.py`, verify mode only), the network (NIDS) path, the
  rank arm, the ELA scoring and the JSON contract parsing (the app assembles points itself), the
  `Flags` counters, and the `detector_only` verify mode.
