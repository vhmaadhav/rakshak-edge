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
| **Cold share** | 30 / 30 (`0` dropped) | **580.8 ms** / **684.9 ms** | `detector_load` 65.6 ms · `rules_regex_compile` 344.5 ms · `onnx_run` **1.08 ms** (`onnx_total` 1.86 ms) · `decision` 14.8 ms | **221.2 MiB** (`226,552 KiB`) |
| **Warm share** (5 warmup + 30 runs) | 30 / 30 (`0` dropped) | **33.0 ms** / **44.7 ms** (full Compose UI render) | `canonicalize` 0.56 ms · `rules` 1.07 ms · `onnx_run` **1.09 ms** (`onnx_total` 1.39 ms) · `decision` 0.50 ms (**~3.5 ms compute**) | **257.6 MiB** (`263,776 KiB`) |

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

### 2b. SMS decision policy comparison: `quiet` vs transactional-aware `balanced` vs `verify` (Pi 5 ONNX)

To resolve the recall-vs-FPR trade-off of `quiet` mode without changing the frozen V1 default (`quiet`), `--detector-only-mode balanced` implements a **transactional-aware 3-outcome policy**:
1. **Alert (`HIGH` / `MEDIUM`):** deterministic rule evidence fires.
2. **Caution (`UNCERTAIN` / `VERIFY_INDEPENDENTLY`):**
   - Corroborated URL hit (`p_url >= tau_url` and `p_scam >= tau_sms`), **or**
   - Detector-only flag (`p_scam >= tau_sms`, no rule/URL hit) exceeding a two-tier threshold:
     - **General non-transactional messages:** `--verify-threshold` (default `0.9566`, calibrated on `campaign` `cal` non-transactional no-rule negatives at FPR ≤ 0.1%, `n=5,334`; zero test/GroundLM leakage).
     - **Transactional-looking notices** (bank/UPI/OTP/delivery/utility/balance vocabulary): `--tx-verify-threshold` (default `0.999`), preventing legitimate transactional false alerts.
3. **Quiet (`LOW` / `IGNORE`):** below the applicable threshold with no rule or corroborated URL evidence.

> **Superseded:** the shipped `balanced` mode is the passive-transaction-aware `balanced_v3` logic (`deploy/pi/service/edge_decision.py`, `is_passive_transactional_notice`). Its numbers are in [VALIDATION_REPORT.md](VALIDATION_REPORT.md) section 1a (campaign Rec 88.60%, FPR 0.83%, F1 0.9355; GroundLM FPR 1.50%, 17 / 1,133). The v2 columns below are kept as history. SmishX `eval_ok` positives here are 501 = 226 smishing + 275 spam, so its recall is not comparable with the 226-smishing recall in the README.

| Dataset / Split | n | `quiet` (V1 default) | `balanced` v1 (flat `0.999`) | **`balanced` v2 (`0.9566` / `0.999` tx-aware + hybrid URL; superseded, see note)** | **`balanced` v2 cal-0.5% (`0.7064` / `0.999` + hybrid URL)** | `verify` (all flags) |
|---|---:|---|---|---|---|---|
| **Campaign-disjoint SMS test** (`campaign.json`) | 22,145 | Rec 16.20%, FPR **0.61%**, F1 0.2774 | Rec 74.57%, FPR 0.73%, F1 0.8509 | **Rec 85.91% (+69.7 pp)**, **FPR 0.83%**, **F1 0.9202** | **Rec 89.78%**, **FPR 1.19%**, **F1 0.9404** | Rec 98.18%, FPR 2.47% |
| **GroundLM policy negatives** (legit SMS) | 1,133 | FPR **0.18%** (`2`) | FPR 0.88% (`10`) | **FPR 0.97% (`11 / 1,133`)** | **FPR 1.24% (`14 / 1,133`)** | FPR 41.66% (`472`) |
| **SmishX — all `eval_ok`** (independent) | 1,107 | Rec 18.76%, FPR **3.63%**, F1 0.3047 | Rec 78.04%, FPR 6.60%, F1 0.8391 | **Rec 81.04%** (ONNX) / **78.64%** (hybrid), FPR **6.60%**, F1 **0.8574** / 0.8428 | **Rec 82.24%**, FPR 6.93%, **F1 0.8628** | Rec 97.80%, FPR 11.72% |
| **Hinglish campaign test** (`campaign.json`) | 398 | Rec 27.81%, FPR **2.56%**, F1 0.4331 | Rec 67.19%, FPR 2.56%, F1 0.8007 | **Rec 84.38% (+56.6 pp)**, **FPR 2.56%**, **F1 0.9122** | **Rec 89.06%**, **FPR 2.56%**, **F1 0.9390** | Rec 99.38%, FPR 6.41% |

> **Honest caveat:** GroundLM and SmishX informed earlier rule/policy design (`docs/RECALL_V2_RESULTS.md`), so these figures measure the trade-off on the known evaluation sets rather than an untouched production inbox. Calibration of `0.9566` / `0.7064` used only `campaign` `cal` negatives.

### 2c. Offline Hybrid URL Intelligence: PhreshPhish vs out-of-domain `url_domain` (Pi 5)

To eliminate the domain-shift false positives of the frozen ONNX `url_student` (which memorized PhreshPhish's deep-path benign distribution and spiked on root-homepage benign domains in `url_domain`), `--url-hybrid` (`src/rakshak/url_domain_intel.py`) combines two independent offline layers via calibrated logit fusion (`0.5 * logit(p_onnx) + 0.5 * logit(p_domain)`), with zero DNS or network access:
1. **Lightweight ONNX URL model:** frozen `best_url_student.onnx`.
2. **Domain Intelligence (`DomainIntelScorer`, `src/rakshak/data/url_domain_intel.npz`, 63 KB):** scheme-agnostic lexical/structural features (homograph/IDN, leetspeak brand spoofing, brand-in-host/path mismatch, free-hosting platforms, risky TLDs, shorteners) + 32,768 CRC32-hashed char(3–5) n-grams trained strictly **domain-disjoint** across multi-source URLs (zero registered-domain overlap with PhreshPhish test, `url_domain` cal, or `url_domain` test).
3. **Three-outcome `/analyze_url` verdict (`--url-high-threshold`):** `"phishing"` (`>= high_threshold`, brand mismatch, or homograph brand spoof), `"suspicious"` (`[tau_url, high_threshold)` or structural risk signal), `"no_signal"` (safe).

Measured on Raspberry Pi 5 (`5.64 ms/URL` / **177.2 URLs/s** single-URL full `HybridUrlScorer.assess`, `throttled=0x0`):

| Model & Operating Point | PhreshPhish `official.test` (10,000) | `url_domain` OOD test (30,432 — full) |
|---|---|---|
| ONNX `url_student` alone @ `0.6308` (shipped) | AUC 0.9926 · Rec **92.53%** · FPR 0.82% · F1 0.9564 | AUC 0.9110 · Rec 94.35% · **FPR 69.42%** · F1 0.8698 |
| ONNX `url_student` alone @ `0.9900` | Rec 74.95% · FPR 0.15% · F1 0.8560 | Rec 80.46% · FPR 3.09% · F1 0.8868 |
| **Hybrid (`--url-hybrid`) @ `0.6308` (`suspicious`)** | **AUC 0.9915** · **Rec 90.75%** · **FPR 1.09%** · **F1 0.9451** | **AUC 0.9769** · **Rec 94.00%** · **FPR 5.97%** (`11.6x` lower) · **F1 0.9595** |
| **Hybrid (`--url-hybrid`) @ `0.7500` (`phishing`)** | **Rec 88.42%** · **FPR 0.74%** · **F1 0.9342** | **Rec 92.08% (+11.6 pp vs ONNX@0.99)** · **FPR 2.22%** (`31.3x` lower) · **F1 0.9552** |

### 2d. NIDS 26-pcap multi-attack holdout on Pi 5 (synthetic captures)

Evaluated on all `26` holdout captures (`2,017` flows, `pcap_nfstream`, `student_cse_cic_ids2018_int8.onnx`, Pi 5 `throttled=0x0`):
- **Rules-first pre-registered baseline (`--nids-dns-min-distinct 1`, default):** **11 / 11** attack instances detected with **11 / 11** exact family labels (`brute_force_ssh` 3/3, `brute_force_other` 1/1, `port_scan` 3/3, `dns_tunnel` 2/2, `dos_flood` 2/2), **9 / 9** near-threshold boundary captures exact, **1 / 6** benign captures alerted (`ho_benign_cdn_lookalike_6.pcap`, 6 repeated lookups of a single long CDN hostname), **5,154.3 flows/s**.
- **Rules-first with distinct-qname DNS guard (`--nids-dns-min-distinct 3`):** **0 / 6 benign captures alerted (`0.0%` session-level FPR)** while retaining **11 / 11** attack detections with **11 / 11** exact family labels and **9 / 9** boundary captures exact (**5,132.0 flows/s**).
- **Model-only:** 4 / 11 detected (0 / 11 right family), 5 / 6 benign files alerted — confirming why NIDS remains rules-first until real-network validation exists ([NIDS_EDGE.md](NIDS_EDGE.md)).

### 2e. Historical Pi 5 baseline reference ([PERF_FAST_INFERENCE.md](PERF_FAST_INFERENCE.md), [BASELINE_PI.md](BASELINE_PI.md))

| Metric | Synchronous baseline (`2b3c1bd`) | Verdict-first + table |
|---|---:|---:|
| HTTP p50 (full sync vs. time to verdict) | 794 ms | 5.5 ms (4.11–4.69 ms in 2026-11-23 run) |
| Sequential throughput | 1.04 msg/s | 250 msg/s |
| Peak throughput (4 workers, 16 clients, batched) | n/a | 1,050 msg/s |
| Verdict flips vs. single-message path (5,080 msgs) | n/a | 0 |
