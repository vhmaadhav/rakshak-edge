# Rakshak Edge: Android app

Offline scam-SMS and link checker for Android (package `org.rakshak.edge`, minSdk 28, arm64-v8a).
Design, parity method, versions and open items: [`docs/ANDROID_APP.md`](../docs/ANDROID_APP.md).

Status: source and tests written, **not compiled and not run**. Nothing here has been built on a
device or on a build machine yet.

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

The repository has no `gradle-wrapper.jar`, because it cannot be generated without Gradle. Create
the wrapper with a Gradle 9.8.0 install (9.6.0 or newer is required by AGP 9.5):

```bash
cd android
gradle wrapper --gradle-version 9.8.0
```

This writes `gradlew`, `gradlew.bat` and `gradle/wrapper/gradle-wrapper.jar`. Commit them if you
want the wrapper versioned.

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

## 3. Regenerating the data assets and the parity fixture

These run on a machine with the Python package (`src/rakshak`) and its data, not in this app build.

```bash
# assets: brands, lexicons, licence, public-suffix rules, language lexicon, suffix check
PYTHONPATH=<repo>/src python tools/export_assets.py <repo> <out_dir>
# parity fixture (318 rows from the SMS campaign test split, GroundLM, SmishX, hand-made edge cases)
PYTHONPATH=<repo>/src python tools/make_parity_fixture.py <data_dir> <out.json>
```

Copy the outputs to `app/src/main/assets/rakshak/` and `app/src/test/resources/parity_fixture.json`.
The assets already in git came from these scripts, run on 2026-11-20.

## 4. Layout

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
