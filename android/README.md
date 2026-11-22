# Rakshak Edge: Android app

Offline scam-SMS and link checker for Android (package `org.rakshak.edge`, minSdk 28, arm64-v8a).
Design, parity method, versions and open items: [`docs/ANDROID_APP.md`](../docs/ANDROID_APP.md).

Status: **builds, and all 27 JVM unit tests pass** (headless, InstaBox VM, 2026-11-20). The 27
include 8 parity tests over the 318-row Python fixture. **Not run on a device yet.** Details are
in `docs/ANDROID_APP.md`.

## 1. Build box (Debian 13, x86_64)

Use `tools/build_headless.sh` after the one-time steps below. Disk needed: about 2 GB.

```bash
# JDK 17 and unzip
sudo apt-get install -y openjdk-17-jdk-headless unzip
# Command-line tools: download commandlinetools-linux-*_latest.zip from
# https://developer.android.com/studio#command-line-tools-only (the page gives the current build
# number), then:
export CMDLINE_TOOLS_ZIP=$HOME/Downloads/commandlinetools-linux-XXXXXXXX_latest.zip
```

### Gradle wrapper (once)

The Gradle 9.8.0 wrapper is committed: `gradlew`, `gradlew.bat` and
`gradle/wrapper/gradle-wrapper.jar`. The jar's sha256 `238e777f…21abd5` matches Gradle's published
checksum, and `distributionSha256Sum` pins the distribution zip. Nothing to do here.

### Build and unit tests

```bash
cd android
./tools/build_headless.sh          # sdkmanager packages, then:
# ./gradlew :app:assembleDebug :app:testDebugUnitTest --no-daemon
```

The APK lands in `app/build/outputs/apk/debug/app-debug.apk`.

## 2. Model files (not in git)

**Detector (APK asset, needed to check messages).** Copy the SMS student ONNX into the assets:

```bash
cp <path to best_student.onnx> app/src/main/assets/sms_student.onnx
sha256sum app/src/main/assets/sms_student.onnx
# expected 5341e14ca02a523bd746a71dd67903439e9e6c6f6fc3193fa1c6f786953f2466, 388127 bytes
```

The file is the `dual-base/dual/s1` campaign run, listed in `docs/SYSTEM_V1.md` section 1a. On the
Pi it is `models/sms_student/best_student.onnx`. The Kaggle output is
`kaggle/out/sms_student/output/best_student.onnx`, which is not in this repository.

**Explainer (external files, needed only for "Explain with Llama").** Push the files with adb:

```bash
adb shell mkdir -p /sdcard/Android/data/org.rakshak.edge/files/models
adb push Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte /sdcard/Android/data/org.rakshak.edge/files/models/
adb push tokenizer.model /sdcard/Android/data/org.rakshak.edge/files/models/
```

Expected sha256 (from `docs/SYSTEM_V1.md` section 1a): `.pte`
`8715cdba9e91f6bede00cc5f2d6b12397b95225da5630c4972a8da03001cda3b` (1,135,951,488 bytes),
tokenizer `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55` (2,183,982 bytes).
Recheck: the `.pte` was not re-hashed on 2026-11-12 (see the table note).

On Android 11 and later, `adb push` into `Android/data/<package>` can be refused for release
builds. If so, push through `adb shell run-as org.rakshak.edge` (debug builds) or check the path the
app reports. This has not been tried on a device.

## 3. Benchmark on a physical phone

With the pinned `sms_student.onnx` asset in place, build the profileable benchmark APK. It has
its own package, `org.rakshak.edge.benchmark`, and can be installed alongside the regular app.

```bash
./gradlew :app:testDebugUnitTest :app:assembleBenchmark --no-daemon
adb devices -l
adb install -r app/build/outputs/apk/benchmark/app-benchmark.apk
python3 tools/benchmark_android.py run --runs 30 --out verdict-timings.json
```

Use one unlocked USB-debugging device, or set `ANDROID_SERIAL` to choose it. For an offline run,
enable airplane mode and keep USB connected. Keep the display refresh setting, battery mode and
thermal conditions comparable across builds. Run the instrumented baseline commit and then the
single-rule-pass commit with the same inputs and run counts. Save each run to its own JSON file.

- **Cold Share** force-stops the benchmark package before each synthetic `ACTION_SEND`. The clock
  starts at Android's app process start timestamp. **Warm Share** keeps the app and ONNX session
  running and starts timing when the app receives each intent. Both end when Android submits
  the result frame. The harness excludes the chooser and ADB transport from these timings.
- To sample **Check**, paste a message or keep the one from the Share run and tap **Check message**.
  To sample **Explain**, create the app's model directory by completing a check, then use the ADB
  steps in section 2 with `org.rakshak.edge.benchmark` as the package. Check an English message with
  visible scam evidence and tap **Explain with Llama**. The first request loads the model. Check
  again before the next Explain tap to retain the loaded model and get a new advisory.
- Use `python3 tools/benchmark_android.py report --out tap-timings.json` after the manual taps.
  This reads the current `RakshakPerf` logcat history, which can include earlier runs. The `run`
  command isolates its own Share samples. The report separates real Llama results, partial results,
  templates and errors. First verified point is measured in the verifier. The app renders all
  verified points together when the advisory is ready.
- `stages_ms` breaks out detector load, canonicalization, rules, ONNX and decision time. Within
  `rules`, the trace records Python-pattern translation (`rules_regex_translate`), regex compilation
  through `rxOf` (`rules_regex_compile`), text-rule matching (`rules_text_match`) and URL-rule
  evaluation (`rules_domain`). Lazy regex work inside URL evaluation can appear in both breakdowns.
  `rules_compile/…` entries attribute compilation time to named text-rule patterns.
  `llm_load` and `llm_generate` separate Llama setup and generation. `process_peak_rss_kib` is the
  resident high-water mark over the app process lifetime when Android exposes it. Use the `Rakshak/…`
  Perfetto trace sections to diagnose stage contention. Timing records contain no SMS text or
  generated sentences and are exclusive to the benchmark build.

To compare loading the detector before preprocessing against running both concurrently, use the
same profiling APK for both arms. The command below sends each input to both arms and alternates
their order across fixture cycles. `--runs` gives the sample count per arm. Setup stage timings
may overlap in the parallel arm, so compare request-to-verdict frame times directly.

```bash
python3 tools/benchmark_android.py run --mode cold --compare-setup --runs 16 --out cold-setup.json
```

## 4. Regenerating the data assets and the parity fixture

These run on a machine with the Python package (`src/rakshak`) and its data, not in this app build.

```bash
# assets: brands, lexicons, licence, public-suffix rules, language lexicon, suffix check
PYTHONPATH=<repo>/src python tools/export_assets.py <repo> <out_dir>
# parity fixture (318 rows from the SMS campaign test split, GroundLM, SmishX, hand-made edge cases)
PYTHONPATH=<repo>/src python tools/make_parity_fixture.py <data_dir> <out.json>
```

Copy the outputs to `app/src/main/assets/rakshak/` and `app/src/test/resources/parity_fixture.json`.
The assets already in git came from these scripts, run on 2026-11-20.

## 5. Layout

```
app/src/main/java/org/rakshak/edge/
  analysis/   Canonicalizer, Rules, UrlFeatures (public suffix), LangDetect, Evidence,
              Decision, Encoding, Remediation, Verifier, PyRegex
  llm/        SlotExplainer (prompt, cleanup, guards), LlmExplainer (ExecuTorch), ModelFiles
  ml/         StudentDetector (ONNX Runtime)
  data/       RuleData (asset loader)
  ui/         RakshakScreen (Compose), About dialog with the Llama notice
  Analyzer.kt, Engine.kt, MainActivity.kt
app/src/test/java/org/rakshak/edge/   ParityTest (fixture), UnitTest
app/src/test/resources/               parity_fixture.json, suffix_check.json
tools/                                export_assets.py, make_parity_fixture.py, build_headless.sh
```
