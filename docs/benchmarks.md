# Benchmarks

Last updated: **2026-11-23**. Every number below was measured on real Arm hardware:
- **Android:** OnePlus 13R (`CPH2691`), Android 16 (API 36), `arm64-v8a`
- **Raspberry Pi 5:** 4 GB, 4x Cortex-A76 at 2.4 GHz (`rakshak-thermald` active, `throttled=0x0`, peak `65.3 °C`)

---

## 1. Android on-device benchmark (OnePlus 13R, Android 16)

Measured on 2026-11-23 with `android/tools/benchmark_android.py run --mode both --runs 30 --warmup 5` on `app-benchmark.apk` (`org.rakshak.edge.benchmark`) bundling the pinned SMS student ONNX (`5341e14ca02a523bd746a71dd67903439e9e6c6f6fc3193fa1c6f786953f2466`, `388,127 B`). All JVM unit tests (`:app:testDebugUnitTest`, including 318-row Python parity) passed on the same build.

- **On-device offline proof (`dumpsys package org.rakshak.edge.benchmark`):** zero `INTERNET`, `ACCESS_NETWORK_STATE`, SMS, or storage permissions requested (only AndroidX `DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION`).

| Mode (30 runs each) | Frame-committed | Request → verdict frame (p50 / p95) | Pure compute stages (p50) | Peak RSS |
|---|---:|---:|---|---:|
| **Cold share** | 30 / 30 (`0` dropped) | **580.8 ms** / **684.9 ms** | `detector_load` 65.6 ms · `rules_regex_compile` 344.5 ms · `onnx_run` **1.08 ms** (`onnx_total` 1.86 ms) · `decision` 14.8 ms | **221.2 MB** (`226,552 KiB`) |
| **Warm share** (5 warmup + 30 runs) | 30 / 30 (`0` dropped) | **33.0 ms** / **44.7 ms** (full Compose UI render) | `canonicalize` 0.56 ms · `rules` 1.07 ms · `onnx_run` **1.09 ms** (`onnx_total` 1.39 ms) · `decision` 0.50 ms (**~3.5 ms compute**) | **257.6 MB** (`263,776 KiB`) |

> **Scope note:** This run benchmarks the on-device ONNX detector, canonicaliser, rules engine, decision layer, and Compose UI frame commit on Android 16. On-device Llama 3.2 1B (`.pte`) explanation latency was not run on the phone in this pass (template fallback is active when `.pte` is absent).

### Reproduce
```bash
cd android
cp <path to best_student.onnx> app/src/main/assets/sms_student.onnx
sha256sum app/src/main/assets/sms_student.onnx      # 5341e14ca02a523bd746a71dd67903439e9e6c6f6fc3193fa1c6f786953f2466
./gradlew :app:testDebugUnitTest :app:assembleBenchmark --no-daemon
adb install -r app/build/outputs/apk/benchmark/app-benchmark.apk
python3 tools/benchmark_android.py run --mode both --runs 30 --warmup 5 --out bench.json
```

---

## 2. Raspberry Pi 5 end-to-end inference, power & accuracy (2026-11-23)

Measured on the Pi 5 (4 GB, 2.4 GHz, `throttled=0x0`, start `48.3–48.8 °C`, end `61.5–65.3 °C`) running the full resident service inside `unshare -rn` (`interfaces: ["lo"]`, `routes: 0`, `egress_probe: "failed: Network is unreachable"`) with Flash UI v2, `--response-mode verdict_first`, `explanation_table_v2.json`, and `--explainer-cache 0`.

### 2a. Latency, throughput & PMIC power (Pi 5)

| Workload | n | Latency (p50 / p95) | Throughput / Notes | PMIC Power / Energy |
|---|---:|---:|---|---|
| **Idle baseline** | 5 s | — | No service running | **1.52 W** |
| **Flash UI v2 (`GET /`)** | 1 | `1.73–2.49 ms` | `200 OK`, `89,457 B`, air-gap **Sealed** | — |
| **English 40-row E2E (`quiet`, `POST /analyze`)** | 40 | **4.11 ms** / **10.04 ms** (TTV) | 23/27 table hits (`85.2%`); 4 live Llama misses (**p50 1,063.9 ms**); **40/40** `final_contract_ok` | — |
| **English 40-row E2E (`balanced`, `POST /analyze`)** | 40 | **4.69 ms** (TTV) | 9 live Llama misses; **40/40** `final_contract_ok` | **5.19 W** (incl. 9 Llama calls) |
| **Hinglish 40-row E2E (`POST /analyze`)** | 40 | **3.66 ms** / **5.34 ms** (TTV) | `40/40` auto-routed Hinglish; **40/40** `final_contract_ok` | — |
| **GroundLM HTTP batch (`quiet` / `balanced`)** | 1,133 | `2.34 s` / `3.28 s` | **483.7 msg/s** (`quiet`) · **345.6 msg/s** (`balanced`, 1 worker) | **6.23 W** (**16.79 mJ/msg** HTTP) |
| **Campaign-disjoint SMS ONNX batch (1 thread)** | 22,145 | `46.26 s` | **478.7 msg/s** (peak multi-worker: **1,050 msg/s** in [PERF_FAST_INFERENCE.md](PERF_FAST_INFERENCE.md)) | **3.22 W** (**7.76 mJ/msg**) |
| **PhreshPhish URLs (`POST /analyze_url` / batch)** | 400 / 2,000 | `76.53 ms` / `80.89 ms` (sync HTTP w/ `vcgencmd`) | **316.4 URLs/s** (ONNX batch) | — |
| **NIDS 26-pcap holdout (`nfstream` + INT8 ONNX)** | 2,017 flows | **58.4 ms** / **301.1 ms** per pcap | **5,930.6 flows/s** | — |

Service verdict process RSS: **163.7 MB** (peak **163.7 MB**). HTTP-vs-direct parity: **0 flips across 1,299 overlapping rows** (`164` campaign + `1,133` GroundLM + `2` Hinglish). PMIC readings come from `vcgencmd pmic_read_adc` (not a calibrated wall meter).

### 2b. SMS decision policy comparison: `quiet` vs `balanced` (3-outcome) vs `verify` (Pi 5 ONNX)

To address the recall-vs-FPR trade-off of `quiet` mode without changing the frozen V1 default (`quiet`), `--detector-only-mode balanced` implements a 3-outcome policy:
1. **Alert (`HIGH` / `MEDIUM`):** deterministic rule evidence fires.
2. **Caution (`UNCERTAIN` / `VERIFY_INDEPENDENTLY`):** high-confidence detector-only flag (`p_scam >= --verify-threshold`, default `0.999`) **or** corroborated URL hit (`p_url >= tau_url` and `p_scam >= tau_sms`).
3. **Quiet (`LOW` / `IGNORE`):** below `--verify-threshold` with no rule or corroborated URL evidence.

| Dataset / Split | n | Detector Alone | `quiet` (V1 default) | **`balanced` (`0.999` + corr. URL)** | `verify` (all flags) |
|---|---:|---|---|---|---|
| **Campaign-disjoint SMS test** (`campaign.json`) | 22,145 | F1 **0.9813**, Rec 98.03%, FPR 1.80% | Rec 16.2%, FPR **0.61%** | **Rec 74.6% (+58.4 pp)**, FPR **0.73%**, F1 **0.8509** | Rec 98.2%, FPR 2.47% |
| **GroundLM policy negatives** (legit SMS) | 1,133 | FPR 41.57% (`471`) | FPR **0.18%** (`2`) | **FPR 0.88% (`10 / 1,133`)** | FPR 41.66% (`472`) |
| **SmishX — all `eval_ok`** (independent) | 1,107 | F1 0.8344, Rec 79.4%, FPR 9.08% | Rec 18.8%, FPR **3.63%** | **Rec 78.0% (+59.2 pp)**, FPR **6.60%**, F1 **0.8391** | Rec 97.8%, FPR 11.72% |
| **SmishX — smishing only** (independent) | 226 | Rec 69.47% (`157`) | Rec 23.4% (`53`) | **Rec 66.4% (+43.0 pp, `150`)** | Rec 95.6% (`216`) |
| **SmishX — ham only** (independent) | 606 | FPR 9.08% (`55`) | FPR **3.63%** (`22`) | **FPR 6.60%** (`40`) | FPR 11.72% (`71`) |
| **Hinglish `eval_ok` non-synthetic** (`sms.parquet`) | 1,380 | F1 0.9961, Rec 99.7%, FPR 2.63% | Rec 40.2%, FPR **0.88%** | **Rec 68.3% (+28.1 pp)**, FPR **0.88%**, F1 **0.8109** | Rec 99.7%, FPR 3.51% |

> **Honest caveat:** GroundLM and SmishX informed earlier rule/policy design (`docs/RECALL_V2_RESULTS.md`), so these figures measure the trade-off on the known evaluation sets rather than an untouched production inbox.

### 2c. URL 3-tier threshold analysis: PhreshPhish vs out-of-domain `url_domain` (Pi 5 ONNX)

`POST /analyze_url` supports an optional `--url-high-threshold` (env `RAKSHAK_URL_HIGH_THRESHOLD`) separating hard `"phishing"` verdicts (`>= high_threshold` or brand-domain mismatch) from soft `"suspicious"` cautions (`[tau_url, high_threshold)`):

| Threshold | PhreshPhish `official.test` (4,000 sample) | `url_domain` OOD test (4,000 sample) |
|---:|---|---|
| `0.6308` (shipped `tau_url`) | Rec **90.95%**, FPR **0.80%**, Prec 99.13%, F1 0.9486 | Rec 94.78%, **FPR 70.65%** |
| `0.9500` | Rec 82.30%, FPR 0.15%, Prec 99.82% | Rec 87.49%, FPR 22.66% |
| `0.9800` | Rec 77.25%, FPR 0.05%, Prec 99.94% | Rec 83.62%, FPR 7.72% |
| **`0.9900`** | Rec **72.50%**, FPR **0.05%**, Prec **99.93%** | Rec **80.79%**, **FPR 2.88%** (`24.5x` lower OOD FPR) |

### 2d. NIDS 26-pcap multi-attack holdout on Pi 5 (synthetic captures)

`run_holdout_nids.py` on `26` captures (`2,017` flows, `pcap_nfstream`, `student_cse_cic_ids2018_int8.onnx`):
- **Rules-first (shipped default):** **11 / 11** attack instances detected with **11 / 11** exact family labels (`brute_force_ssh` 3/3, `brute_force_other` 1/1, `port_scan` 3/3, `dns_tunnel` 2/2, `dos_flood` 2/2), **9 / 9** near-threshold boundary captures exact, **1 / 6** benign captures alerted (`ho_benign_cdn_lookalike_6.pcap`).
- **Model-only:** 4 / 11 detected (0 / 11 right family), 5 / 6 benign files alerted — confirming why NIDS remains rules-first until real-network validation exists ([NIDS_EDGE.md](NIDS_EDGE.md)).

### 2e. Historical Pi 5 baseline reference ([PERF_FAST_INFERENCE.md](PERF_FAST_INFERENCE.md), [BASELINE_PI.md](BASELINE_PI.md))

| Metric | Synchronous baseline (`2b3c1bd`) | Verdict-first + table |
|---|---:|---:|
| HTTP p50 (full sync vs. time to verdict) | 794 ms | 5.5 ms (4.11–4.69 ms in 2026-11-23 run) |
| Sequential throughput | 1.04 msg/s | 250 msg/s |
| Peak throughput (4 workers, 16 clients, batched) | n/a | 1,050 msg/s |
| Verdict flips vs. single-message path (5,080 msgs) | n/a | 0 |
