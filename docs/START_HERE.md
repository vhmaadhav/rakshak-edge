# Rakshak-Edge: start here

Rakshak-Edge is an offline scam checker for SMS and links, running on an Android phone and on a Raspberry Pi 5 (Arm).
A small detector (SMS and URL character CNNs plus deterministic rules) decides SCAM, UNCERTAIN or SAFE, and abstains when it is not sure.
A Llama 3.2 1B explainer (ExecuTorch) may write sentences about the evidence; a verifier drops any sentence that cites evidence the detector did not produce, and a template always stands behind it.

Status key: **available** = run on the target hardware and documented; **experimental** = code and tests exist, but it was run only on a laptop, on synthetic data, or not on a device; **not yet** = not in the code path.

## Supported inputs

| Deployment | Input | Status | Basis |
|---|---|---|---|
| Android | Message pasted into the app, or text shared from another app (`ACTION_SEND`, `text/plain`) | experimental: builds; 53 JVM unit tests in `android/app/src/test`; one on-device detector run reported on a OnePlus 13R (Android 16) in [benchmarks.md](benchmarks.md) section 1 (raw output not committed); Llama not yet run on the phone | [android/README.md](../android/README.md), [ANDROID_APP.md](ANDROID_APP.md) |
| Android | Links inside a message (canonicaliser, domain rules) | experimental: same as above | [ANDROID_APP.md](ANDROID_APP.md) section 1 |
| Android | "Explain with Llama" (ExecuTorch, on the phone) | experimental: wired in code; without the model files the app shows the template | [ANDROID_APP.md](ANDROID_APP.md) sections 1, 4 |
| Android | Standalone URL check with the URL model | not yet (not ported) | [ANDROID_APP.md](ANDROID_APP.md) section 5 |
| Android | Network flows | not yet | [ANDROID_APP.md](ANDROID_APP.md) section 5 |
| Pi 5 | SMS check, `POST /analyze` (quiet mode is the default) | available: score parity and latency measured on the Pi | [SYSTEM_V1.md](SYSTEM_V1.md) sections 2b, 2c |
| Pi 5 | Explanation of SMS evidence (Llama on the Pi, template fallback) | available; the verifier rejects most model sentences, the template covers them | [EDGE_PIPELINE.md](EDGE_PIPELINE.md) section 3c; [BASELINE_PI.md](BASELINE_PI.md) section C |
| Pi 5 | Hinglish SMS | available, template text only (first draft) | [EDGE_PIPELINE.md](EDGE_PIPELINE.md) section 5 |
| Pi 5 | Link check, `POST /analyze_url` (URL model and domain rules, every mode) | available: parity and latency measured on the Pi | [URL_SERVICE_RESULTS.md](URL_SERVICE_RESULTS.md) U4, U5 |
| Pi 5 | URL evidence inside the SMS check (`/analyze`) | experimental: verify mode only; off in quiet mode | [URL_SERVICE_RESULTS.md](URL_SERVICE_RESULTS.md) gate U2 |
| Pi 5 | Network flows: pcap body or `{"flows": [...]}`, `POST /analyze_flows` | experimental: synthetic captures only; rules first | [NIDS_EDGE.md](NIDS_EDGE.md) |
| Pi 5 | Live interface monitor (`python -m rakshak.nids.pi_monitor --iface ...`) | experimental: code exists, alert advice only, never blocks; field false-alarm rate not yet measured; not wired into the service | [NIDS_LIVE.md](NIDS_LIVE.md); `src/rakshak/nids/pi_monitor.py`; [NIDS_EDGE.md](NIDS_EDGE.md) "Flow extraction" |
| Pi 5 | Offline proof (no egress) | available as a check: `scripts/offline_check.sh` runs tests in a no-network namespace | [OFFLINE_ASSURANCE.md](OFFLINE_ASSURANCE.md) |
| Pi 5 | Explanation of network alerts by Llama | not yet (template only) | [NIDS_EDGE.md](NIDS_EDGE.md) "Not done or untested" |

## Install and first run

### Android

Build box: JDK 17, Android command-line tools, about 2 GB disk. Full steps are in [android/README.md](../android/README.md).

```bash
cd android
# 1. The detector is not in git. Copy the pinned SMS student into the assets before building.
cp <path to best_student.onnx> app/src/main/assets/sms_student.onnx
sha256sum app/src/main/assets/sms_student.onnx
#    expected 5341e14ca02a523bd746a71dd67903439e9e6c6f6fc3193fa1c6f786953f2466, 388127 bytes
# 2. Build and run the JVM tests
./gradlew :app:assembleDebug :app:testDebugUnitTest --no-daemon
# 3. Install on one USB-debugging phone (arm64)
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Explainer files (optional; needed only for "Explain with Llama"). They are not in the APK. Push them with adb:

```bash
adb shell mkdir -p /sdcard/Android/data/org.rakshak.edge/files/models
adb push Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte /sdcard/Android/data/org.rakshak.edge/files/models/
adb push tokenizer.model /sdcard/Android/data/org.rakshak.edge/files/models/
```

Expected sha256: `.pte` `8715cdba9e91f6bede00cc5f2d6b12397b95225da5630c4972a8da03001cda3b`, tokenizer `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55`.
Android 11 and later may refuse this path for release builds; that has not been tried on a device.

**Model provisioning.** The offline importer (file picker) and local check history are written in the data layer (`org.rakshak.edge.provision`, `org.rakshak.edge.history`), but **no UI is wired to them yet**, so use the adb steps above. See [android/docs/HISTORY_AND_PROVISIONING.md](../android/docs/HISTORY_AND_PROVISIONING.md).

First run: open Rakshak Edge, paste a message and tap **Check message**. Or share text from another app to Rakshak Edge. After a check, tap **Explain with Llama** to request the sentences. The app declares no network permission.

### Raspberry Pi 5

The Pi runs the service from `~/rakshak-bench`. Model files are not in git: they come from `kaggle/out/` on the laptop (see [SYSTEM_V1.md](SYSTEM_V1.md) section 1a). Details: [deploy/pi/bench/README.md](../deploy/pi/bench/README.md).

```bash
# Laptop: copy code and pinned files to the Pi (set PI_HOST and PI_KEY if not the defaults)
deploy/pi/bench/sync_bench.sh --dry-run
deploy/pi/bench/sync_bench.sh

# Pi, once: virtualenv. ExecuTorch (and CPU torch) are needed for the Llama explainer; versions from docs/LLAMA_PI_RESULTS.md
python3 -m venv ~/rakshak-bench/.venv
~/rakshak-bench/.venv/bin/pip install onnxruntime numpy pandas pyarrow
~/rakshak-bench/.venv/bin/pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu
~/rakshak-bench/.venv/bin/pip install "executorch==1.5.1"

# Pi: start the service (it blocks until it is listening; it runs with no network route)
~/rakshak-bench/deploy/pi/service/start_service.sh
curl -s http://127.0.0.1:8770/health
curl -s -X POST http://127.0.0.1:8770/analyze -H 'Content-Type: application/json' \
  -d '{"sms": "Your account is blocked. Verify KYC now at http://example.test/kyc"}'
curl -s -X POST http://127.0.0.1:8770/analyze_url -H 'Content-Type: application/json' \
  -d '{"url": "http://example.test/kyc"}'
curl -s --data-binary @capture.pcap http://127.0.0.1:8770/analyze_flows   # synthetic captures only

# Pi: stop
~/rakshak-bench/deploy/pi/service/stop_service.sh
```

A minimal page is served at `GET /` on the same port.

## Check that the service stays offline

`scripts/offline_check.sh` runs the offline test selection inside a network namespace with no interface but loopback, after a negative control that must fail to connect. It exits 3 if `unshare -rn` is unavailable; `OFFLINE_CHECK_ALLOW_NO_NAMESPACE=1` runs without isolation and labels the result NOT OFFLINE-VERIFIED. Details: [OFFLINE_ASSURANCE.md](OFFLINE_ASSURANCE.md).

## Verify the artifacts

`scripts/release_manifest.py` writes `release_manifest.json` and `SHA256SUMS`: the git commit (full SHA), branch, dirty flag, UTC time, and size and sha256 of each file, with the device and modality each one serves. Files come from the Pi registry, the Android assets, and any `--artifact` you name.

```bash
# Write the manifest (add --bench-root ~/rakshak-bench for Pi files that are not in the repo)
python3 scripts/release_manifest.py --out-dir release
# Add the APK
python3 scripts/release_manifest.py --artifact android/app/build/outputs/apk/debug/app-debug.apk@android:app --out-dir release
# Re-hash everything in a manifest; exit 1 on any missing or changed file
python3 scripts/release_manifest.py --verify release/release_manifest.json
# From the repo root
sha256sum -c release/SHA256SUMS
```

A missing file makes the write step exit 1 (use `--allow-missing` to keep going). On a clone without `kaggle/out/` and the Llama files, those entries are reported as missing; this is expected.

## Release placeholders (fill at release time)

The Phase 1 checklist, with the verified deadline and the status of each deliverable, is [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md).

- APK: release v1.0.0-phase1 (draft; APK pending the SMS model file)
- Demo video: pending, see issue #29
- Technical report: `report/main.tex` (LaTeX source in this repo; PDF on release v1.0.0-phase1)
- TODO(release): frozen commit SHAs (repository, Pi code bundle, Android build) and the release manifest hash. Filled when the release is published.

## Known limits

Each number is copied from the committed document named in its row. Where two documents differ, both are named.

| Limit | Number or fact | Source |
|---|---|---|
| The student flags legitimate policy messages on its own | 471 / 1,133 = 41.57 % of GroundLM policy negatives (detector level, before rules and quiet mode). The rule and threshold choices were made on the same rows, so this set is not an independent test | [GROUNDLM_EDGE_FPR.md](GROUNDLM_EDGE_FPR.md); [BASELINE.md](BASELINE.md) |
| Quiet mode (the default) hides most of those flags | 2 / 1,133 = 0.18 % visible warnings on the same set | [GROUNDLM_EDGE_FPR.md](GROUNDLM_EDGE_FPR.md), table row (f) |
| Quiet mode costs recall on smishing | SmishX visible recall 53 / 226 = 23.45 % in quiet mode, 173 / 226 = 76.55 % in verify mode. Campaign-test visible recall 16.20 % in quiet mode | [GROUNDLM_EDGE_FPR.md](GROUNDLM_EDGE_FPR.md) |
| Smishing recall of the student alone | 157 / 226 = 69.47 % [63.18, 75.11] on SmishX; 5.78 % text overlap with training | [BASELINE.md](BASELINE.md) "SmishX, smishing recall" |
| Temporal split precision is low | precision 0.276 on 77 scam messages, FPR 1.8 % at the shipped threshold | [BASELINE.md](BASELINE.md) "Temporal" |
| The URL model does not transfer to other benign URL sources | FPR 69.4 % on the `url_domain` test at the shipped threshold; gate U2 failed (5 / 1,133 visible, limit 4), so URL evidence stays out of quiet mode | [URL_SERVICE_RESULTS.md](URL_SERVICE_RESULTS.md) |
| Network results are on synthetic captures | 26 files, 2017 flows, built with scapy from RFC 1918 addresses. Rules first: 11 / 11 attack instances detected, 1 / 6 benign files alerted. No real traffic was tested | [NIDS_EDGE.md](NIDS_EDGE.md) "Holdout results", "Not done or untested" |
| The learned network model does not generalise across datasets | Leave-one-dataset-out: the best arm passes its mean-MCC gate but fails the FPR limit (1 %) on all four targets; the verdict says the pass does not support deployment | [NIDS_XDATA_RESULTS.md](NIDS_XDATA_RESULTS.md) "Verdict" |
| Explainer on the Pi: model sentences are often rejected | Raw verifier pass 27 of 33 points (26 of 33 in [EDGE_PIPELINE.md](EDGE_PIPELINE.md) section 3c; small n); 4 of 31 calls unterminated; the template fills the rest | [BASELINE_PI.md](BASELINE_PI.md) section C; [EDGE_PIPELINE.md](EDGE_PIPELINE.md) section 3 |
| Explainer prompt-injection result is structural | The injected payload never reaches the explainer prompt; the slot pass on the LLM path is weak and small-n | [PTE_INJECTION_RESULTS.md](PTE_INJECTION_RESULTS.md) "Verdict" |
| Pi numbers were taken on USB power, without a cooler | Clean-power runs: evidence 30, direct runner, total p50 989 ms, p95 1,887 ms. Energy is a PMIC estimate, not a calibrated meter. Earlier runs had an under-voltage flag set | [EDGE_PIPELINE.md](EDGE_PIPELINE.md) section 4 and clean-power table; [SYSTEM_V1.md](SYSTEM_V1.md) section 4 |
| One URL-latency run breached the 65 °C start rule | Start temperatures reached 71.4 °C (verify batch) and 65.9 °C (quiet batch); the quiet-versus-verify comparison is therefore confounded by temperature. The 50 ms added-latency gate is met either way | [URL_SERVICE_RESULTS.md](URL_SERVICE_RESULTS.md) "Thermal note on U5" |
| Android device evidence is thin | One detector-only benchmark run on a OnePlus 13R is reported in [benchmarks.md](benchmarks.md) section 1, but its raw output is not committed. No on-device Llama output and no screenshots are in this tree. Parity tests run on the JVM regex engine, not Android's | [ANDROID_APP.md](ANDROID_APP.md) sections 2, 5; [android/README.md](../android/README.md) |
| Hinglish output is a first draft | Template text only for the Pi default; the native-speaker check is not done | [EDGE_PIPELINE.md](EDGE_PIPELINE.md) section 5, "Untested" |
| One campaign F1 value is unresolved | 0.9813 (ONNX file) versus 0.9749 (stored run log) | [BASELINE.md](BASELINE.md) note 1 |

Measured results by component, with their splits, are indexed in [RESULTS_INDEX.md](RESULTS_INDEX.md). The frozen system description, with hashes, is [SYSTEM_V1.md](SYSTEM_V1.md).
