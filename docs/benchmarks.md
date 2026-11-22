# Benchmarks

Status as of 2026-11-22: **the Android on-device benchmark has not been run.** No device numbers are recorded in this
repository yet. The Raspberry Pi 5 numbers below come from earlier committed documents, and were not re-measured for this page.

## Android on-device benchmark (not run)

Two things were missing on the build machine on 2026-11-22:

1. **No phone was visible.** `adb devices -l` returned an empty list, and Windows listed no phone among its USB devices.
   Likely causes: the phone is not plugged into this PC, the cable is charge-only, or USB debugging is not enabled or authorised.
2. **The detector model is not available.** The app needs `sms_student.onnx` in `android/app/src/main/assets/`. The file is
   not in git (see `.gitignore`) and was not found on the machine. Pinned checksum, from `android/README.md`:
   `5341e14ca02a523bd746a71dd67903439e9e6c6f6fc3193fa1c6f786953f2466` (388,127 bytes). Without it the app cannot score a message,
   so any latency figure would not describe the real model.

What did work on 2026-11-22 (commit `42378d0`, after merging PRs #20 to #23): the app compiles and builds an APK,
`./gradlew :app:testDebugUnitTest :app:assembleDebug` passes **53 / 53** JVM unit tests (this includes the Python-parity tests).
These tests run on the JVM, not on a phone, so they say nothing about latency, memory or Android's regex engine.

### How to produce the numbers (about 10 minutes once a phone and the model are present)

```bash
cd android
cp <path to best_student.onnx> app/src/main/assets/sms_student.onnx
sha256sum app/src/main/assets/sms_student.onnx      # must match the checksum above
./gradlew :app:assembleBenchmark --no-daemon
adb devices -l                                       # the phone must show as "device"
adb install -r app/build/outputs/apk/benchmark/app-benchmark.apk
python3 tools/benchmark_android.py run --mode cold --runs 30 --out cold.json
python3 tools/benchmark_android.py run --mode warm --runs 30 --out warm.json
```

The harness records detector-stage timings (`stages_ms`), request-to-verdict time, and peak resident memory
(`process_peak_rss_kib`). It does not record SMS text. Run each mode at least three times and report the mean. Accuracy is
not measured on the phone. Parity with the Python detector is covered by the JVM parity tests over the 318-row fixture. Full method:
`android/README.md` section 3.

Once a run exists, add a table here (device model, Android version, build commit, mode, runs, mean and p50/p95 latency, peak RSS)
and update the "Android has not run on a device" rows in `docs/START_HERE.md` and `android/README.md`.

## Raspberry Pi 5 (from committed documents)

Measured on the Pi 5 (4 GB) at 2.4 GHz with the thermal governor stopped. Peak temperature 76.3 °C. Source:
[PERF_FAST_INFERENCE.md](PERF_FAST_INFERENCE.md) (methodology and limits are there), baseline in [BASELINE_PI.md](BASELINE_PI.md).

| Metric | Synchronous baseline | Verdict-first |
|---|---:|---:|
| HTTP p50 (old full response vs. time to verdict) | 794 ms | 5.5 ms |
| Sequential throughput | 1.04 msg/s | 250 msg/s |
| Peak throughput (4 workers, 16 clients, batching) | n/a | 1,050 msg/s |
| Verdict flips vs. the single-message path, 5,080 messages (FP32 ONNX) | n/a | 0 |

- When an explanation needs the LLM, it took 1.2 s in isolation and 11.8 s p50 in a 20-message burst.
- Baseline run (BASELINE_PI.md, table B1/B2): request total p50 was 1,393.5 ms for 40 English messages with the explainer,
  and 4.5 ms for 40 Hinglish messages on the template path (no Llama call). Llama time to first token p50 was 1,101 ms.
- Energy is a PMIC estimate, not a calibrated meter. Earlier runs had an under-voltage flag set (see START_HERE "Known limits").

Detector accuracy figures (F1, recall, false-positive rates) are in the README and [RESULTS_INDEX.md](RESULTS_INDEX.md).
