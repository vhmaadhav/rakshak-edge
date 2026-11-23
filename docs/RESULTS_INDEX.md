# Results index

Every committed results document and results JSON in `docs/` and `reports/`. Each line says what it measures, on which data, for which component, and links the file. Plans and pre-registrations are not indexed here; each results document links its plan.

Three kinds of number are kept apart:

- **Detector** metrics score a model or a rule set on a labelled split, before the user sees anything.
- **Visible-warning policy** metrics count what the user sees (quiet or verify mode, abstention, alerts).
- **Latency** is split into detector-only time, share-to-render, first explanation and full explanation. Where a document does not use one of these terms, the line says what it does measure. Share-to-render on Android (cold and warm share, request to verdict frame) is reported only in [benchmarks.md](benchmarks.md) section 1, without committed raw output.

Exposure: many SMS rows were used to design rules and thresholds. Those documents say "in-sample". SmishX is the one set the docs describe as independent of training. The 41.57 % and similar figures are not independent tests.

## 1. Detector metrics

| Document | What it measures | Data / split | Component | Link |
|---|---|---|---|---|
| Frozen v0 benchmark | F1, PR-AUC, recall and FPR for SMS, URL and NIDS models with campaign-cluster CIs; contamination check | Campaign-disjoint test (22,145 rows); Hinglish `lang_shift`; temporal; PhreshPhish; `url_domain`; CSE-CIC-IDS2018 | SMS student, URL student, NIDS student | [BASELINE.md](BASELINE.md) |
| Frozen v0, Pi half | Score parity Pi vs laptop (300 rows, 0 flips); explainer on Pi; NIDS device row | Same 300 SMS rows; Pi runs | SMS student, explainer | [BASELINE_PI.md](BASELINE_PI.md) |
| GroundLM false-positive test (in-sample) | Student and rule false-positive counts on legitimate policy messages | GroundLM policy negatives (1,133 scoreable), in-sample for rules | Rules, student | [GROUNDLM_FPR.md](GROUNDLM_FPR.md) |
| Frozen-rule before/after tables | Rule-only change in false positives and recall, with paired intervals | GroundLM (in-sample), ScamShield (in-sample), SmishX (independent) | Rules | [REVIEW_AUDIT.md](REVIEW_AUDIT.md) |
| Rule coverage | Share of fraud, promotion and benign SMS that fire no rule | ScamShield, sampled with a fixed seed; three slices | Rules | [RULE_COVERAGE.md](RULE_COVERAGE.md) |
| Rank-arm measurement correction | Fixes a zero-time, zero-token report in the explainer-evaluation harness; no detector result | Kaggle run, 2026-11-01 | Explainer evaluation harness | [RANK_ARM_AUDIT.md](RANK_ARM_AUDIT.md) |
| Campaign-clustered SMS retrain arms A, B, C | F1 and SmishX TP against gates; all killed | Campaign test; SmishX; GroundLM | SMS student | [SMS_RETRAIN_RESULTS.md](SMS_RETRAIN_RESULTS.md) |
| Consented-inbox retrain R0–R2 | Gate results (G1–G5); all failed; placeholder tags act as a shortcut | Consented inbox, scrubbed text; phone 1 held out | SMS student | [INBOX_RETRAIN_RESULTS.md](INBOX_RETRAIN_RESULTS.md) |
| Inbox filled-text arms S0, S1 | Gates H4–H6; SmishX TP; test-donor legitimate FPR | Consented inbox, filled text; test donor | SMS student | [INBOX_FILLED_RESULTS.md](INBOX_FILLED_RESULTS.md) |
| Inbox threshold calibration (W1f) | Threshold at ≤1 % FPR on the train donor; killed (near-zero recall) | Consented inbox legitimate rows (phones 2+3) | Encoder threshold | [INBOX_CALIBRATION_RESULTS.md](INBOX_CALIBRATION_RESULTS.md) |
| Surrogate diagnostic D0 | Shipped v0 student scored on scrubbed vs tag-filled inbox sets; no training | Consented inbox (private), `maadhavvh/rakshak-data` | SMS student v0 | [SURROGATE_D0_RESULTS.md](SURROGATE_D0_RESULTS.md) |
| Teacher fine-tune T0, T1 | Campaign F1 (filled), SmishX TP, legitimate FPR; T1 stop rule fires | Campaign; SmishX; inbox donor | mmBERT teacher | [TEACHER_FINETUNE_RESULTS.md](TEACHER_FINETUNE_RESULTS.md) |
| Teacher full-suite scoring (W1a) | Gated arms, GroundLM false positives, threshold transfer | Fold `holdout_scamshield`; GroundLM; SmishX | mmBERT teacher | [TEACHER_SUITE_RESULTS.md](TEACHER_SUITE_RESULTS.md) |
| Teacher cross-source ranking (D1) | AUC on the held-out source (fails 0.85 gate) | Fold `holdout_scamshield` | mmBERT teacher | [TEACHER_XSOURCE_RESULTS.md](TEACHER_XSOURCE_RESULTS.md) |
| Encoder with licensed exposure (W1e) | Ranking and calibration; same-source threshold too low; killed | GroundLM, SmishX, campaign; fold `holdout_scamshield` | mmBERT encoder | [ENCODER_EXPOSURE_RESULTS.md](ENCODER_EXPOSURE_RESULTS.md) |
| Source-adversary training | ScamShield delta (paired campaign bootstrap); kept for further validation | ScamShield held-out source; SmishX | SMS student | [SOURCE_ADVERSARY_RESULTS.md](SOURCE_ADVERSARY_RESULTS.md) |
| Source balance and group-DRO | ScamShield F1 under paired gate; both killed | ScamShield | SMS student | [SOURCE_BALANCE_RESULTS.md](SOURCE_BALANCE_RESULTS.md) |
| Clean KD versus no KD | Lang-shift F1 and fold loss; killed | Per-fold frozen train IDs; `lang_shift` | SMS student | [CLEAN_KD_RESULTS.md](CLEAN_KD_RESULTS.md) |
| Student false-alarm diagnosis | Transactional false alarms on lexical proxies (not human labels) | Transactional proxy sets | SMS student | [TRANSACTIONAL_DIAGNOSTIC.md](TRANSACTIONAL_DIAGNOSTIC.md) |
| Hard-negative student diagnostics | Student-only statistics for the transactional arm | Synthetic transactional negatives plus originals | SMS student | [TRANSACTIONAL_HARDNEG_STUDENT_RESULTS.md](TRANSACTIONAL_HARDNEG_STUDENT_RESULTS.md) |
| Multilingual compact-input experiment | Legacy spam-inclusive mean F1 on a `lang_shift` sample; candidate gate GO, in-sample | `lang_shift` sample; fresh byte control | SMS model candidate | [MULTILINGUAL_RESULTS.md](MULTILINGUAL_RESULTS.md) |
| Frozen fusion v1 | Detector safety gate KILL; exact-checkpoint artifact gate GO | Frozen fusion cal and test splits; detector endpoints | Fusion (SMS + URL + rules) | [FUSION_V1_RESULTS.md](FUSION_V1_RESULTS.md) |
| Cal-only fusion and transactional candidate | Candidate killed; original decision unmeasurable (no fraud labels in UCI) | UCI (no fraud positives); cal-only models | Fusion | [FUSION_TRANSACTIONAL_RESULTS.md](FUSION_TRANSACTIONAL_RESULTS.md) |
| Hard-negative arm, fused | Fused endpoints; killed on caps | GroundLM, SmishX, transactional negatives | Fusion | [TRANSACTIONAL_HARDNEG_FUSED_RESULTS.md](TRANSACTIONAL_HARDNEG_FUSED_RESULTS.md) |
| URL student, PhreshPhish plus `url_domain` (W2) | Average precision at 1 % FPR; worse on domain-novel URLs; killed | PhreshPhish official test (168,060); `url_domain` (30,432); 3 seeds | URL student | [URL_GENERAL_RESULTS.md](URL_GENERAL_RESULTS.md) |
| URL three-seed controls (E0) | Controls only; fixed teacher stopped on nonfinite loss (unmeasurable) | URL calibration outputs; PhreshPhish test not used for the decision | URL student | [URL_THREE_SEED_RESULTS.md](URL_THREE_SEED_RESULTS.md) |
| URL controls, calibration outputs | Saved calibration predictions for the three-seed URL controls | Calibration split only | URL student | [URL_CONTROLS_CAL.md](URL_CONTROLS_CAL.md) |
| URL threshold and service evaluation | τ_url at ≤1 % benign FPR on a 5 % PhreshPhish carve; PhreshPhish official test FPR 0.92 %, recall 92.0 %; `url_domain` FPR 69.4 % at τ_url | PhreshPhish carve (cal), official test, `url_domain` | URL student | [URL_SERVICE_RESULTS.md](URL_SERVICE_RESULTS.md) |
| Network holdout, synthetic captures | Attack instances detected (rules first 11 / 11; model only 4 / 11), benign files alerting (1 / 6 rules first) | 26 synthetic pcaps, 2017 flows, scapy RFC 1918 | NIDS (rules, model) | [NIDS_EDGE.md](NIDS_EDGE.md) |
| NIDS leave-one-dataset-out (NetFlow-v2) | Mean MCC and FPR per target; best arm passes the mean-MCC gate and fails the FPR limit on all four targets | UNSW-NB15, CSE-CIC-IDS2018, BoT-IoT, ToN-IoT; leave one out | NIDS models (LightGBM arms, student) | [NIDS_XDATA_RESULTS.md](NIDS_XDATA_RESULTS.md); JSON: [results/nids_xdata/metrics.json](results/nids_xdata/metrics.json) |
| NIDS provenance check | Whether host pairs can be recovered for every frozen row; no model scored | Frozen NIDS sample | NIDS data | [NIDS_PROVENANCE_RESULTS.md](NIDS_PROVENANCE_RESULTS.md) |
| Overview of the SOTA push | One-line outcome per workstream (teacher, encoder, inbox, URL) | All workstreams listed above | All | [SOTA_PUSH_RESULTS.md](SOTA_PUSH_RESULTS.md) |

## 2. Visible-warning policy metrics

What the user sees in quiet mode (default) and verify mode. Denominators are given as the document gives them.

| Document | What it measures | Data / split | Component | Link |
|---|---|---|---|---|
| GroundLM and SmishX, full edge path | Visible warning on 1,133 GroundLM negatives: quiet 2 (0.18 %), verify 472 (41.66 %); SmishX visible recall quiet 53 / 226 (23.45 %) vs verify 173 / 226 (76.55 %); campaign visible recall quiet 16.20 %; SmishX ham FPR quiet 22 / 606 | GroundLM policy negatives (in-sample for rules); SmishX (independent); campaign test | Edge decision path, no LLM | [GROUNDLM_EDGE_FPR.md](GROUNDLM_EDGE_FPR.md) |
| Fused-system measurement (URL, rules, student) | Fused SAFE / UNCERTAIN / SCAM endpoints, FPR and recall with campaign-cluster CIs | Frozen fusion test splits | Fusion | [FUSION_V1_RESULTS.md](FUSION_V1_RESULTS.md) |
| Evidence-required alert | Alert caps on GroundLM (SCAM 0 / 1,133) and rules-only recall gates; killed on rules recall | GroundLM, SmishX, campaign | Alert policy | [EVIDENCE_REQUIRED_ALERT_RESULTS.md](EVIDENCE_REQUIRED_ALERT_RESULTS.md) |
| Template-family calibration | Family-specific maps reduce to the global fallback; nine external checks fail | Campaign, GroundLM, SmishX | Alert policy | [TEMPLATE_CALIBRATION_RESULTS.md](TEMPLATE_CALIBRATION_RESULTS.md) |
| Cross-fitted GroundLM calibration, seed 3 (W1g) | GroundLM false positives 10 / 1,133 pass; recall and campaign F1 fail | GroundLM, SmishX, campaign; folds `sms_groupneg` | Encoder + threshold | [CROSSFIT_CONFIRM_RESULTS.md](CROSSFIT_CONFIRM_RESULTS.md) |
| Three-seed ensemble (W1h) | GroundLM false positives 22 / 1,133; recall and campaign F1 fail | Same protocol as W1g, seeds 4–6 | Encoder ensemble | [ENSEMBLE_CONFIRM_RESULTS.md](ENSEMBLE_CONFIRM_RESULTS.md) |
| Recall V2 (reputation-guarded URL evidence) | Gates G1–G5: GroundLM visible FP, SmishX TP and ham FP, campaign FPR, PhreshPhish recall; R3 fails G1, G3, G5; default unchanged | GroundLM, SmishX, campaign, PhreshPhish, `url_domain` | URL evidence in quiet mode | [RECALL_V2_RESULTS.md](RECALL_V2_RESULTS.md); JSON: [results/recall_v2/metrics.json](results/recall_v2/metrics.json) |
| URL evidence in the Pi service | Gate U2 fails (5 / 1,133 visible, limit 4); U3 SmishX visible recall 53 → 181 in verify, 53 → 143 with corroboration | GroundLM; SmishX | URL evidence on `/analyze` | [URL_SERVICE_RESULTS.md](URL_SERVICE_RESULTS.md) |
| Hard-negative arm, fused (policy view) | Visible FP caps and SmishX recall against control; killed | GroundLM, SmishX | Fusion policy | [TRANSACTIONAL_HARDNEG_FUSED_RESULTS.md](TRANSACTIONAL_HARDNEG_FUSED_RESULTS.md) |
| Adversarial, boundary & Indian-context validation (P0/P2/P3/P4) | Passive-tx balanced SMS (campaign 88.6% rec / 0.83% FPR, paired Indian bank 20/20 rec), DNS low-and-slow multi-horizon (0/6 benign PCAPs, 4/6 slow beacons), URL hybrid adversarial (29/29 attacks, 1.37% OOD FPR on a 4,000-row `url_domain` sample; 2.22% on the full set, [benchmarks.md](benchmarks.md) 2c), Indian SMS strata & field harness | Campaign test (22,145), GroundLM (1,133 neg / 91 pos), SmishX (1,107), paired Indian bank (40), 18 DNS boundaries + 26 PCAPs, 48 adversarial URLs + PhreshPhish/`url_domain` | SMS balanced v3, Hybrid URL, NIDS rules | [VALIDATION_REPORT.md](VALIDATION_REPORT.md) |
| Real-hardware benchmarks (Android + Pi 5, 2026-11-23) | Android cold/warm share latency and RSS (detector only); Pi 5 verdict-first latency, throughput, PMIC power; SMS `balanced` and hybrid-URL accuracy on the Pi | OnePlus 13R (Android 16); Pi 5 4 GB; campaign test, GroundLM, SmishX, PhreshPhish, `url_domain`, 26 PCAPs | Android app, Pi service | [benchmarks.md](benchmarks.md) (raw outputs not committed) |

## 3. Explainer quality

| Document | What it measures | Data / split | Component | Link |
|---|---|---|---|---|
| ELA QLoRA frozen comparison | Slot evidence-point pass for base and selected adapter, campaign-clustered CI; GO for verifier-gated advisory | Frozen 300-message sample; synthetic dev selects checkpoint | Explainer (Llama 3.2 1B) | [ELA_QLORA_RESULTS.md](ELA_QLORA_RESULTS.md) |
| Explainer speed and termination (x86) | Unterminated calls, raw verifier pass, final contract, tokens and calls per advisory; no candidate passes all five gates; shipped default kept | 30 evidence rows, 3 repeats, real `.pte`, x86 VM | Explainer (SpinQuant INT4 `.pte`) | [EXPLAINER_SPEED_RESULTS.md](EXPLAINER_SPEED_RESULTS.md); JSON: [results/explainer_speed/metrics.json](results/explainer_speed/metrics.json) |
| Injection and slot pass on the Pi | Injection pairs: 0 harmful flips; the injected payload never reaches the prompt; slot pass on the LLM path weak (small n) | 150 pairs (300 rows) on the Pi | Explainer `.pte` | [PTE_INJECTION_RESULTS.md](PTE_INJECTION_RESULTS.md) |
| Llama on the Pi | Model loads and generates; sentence quality and raw pass on the device | 40-row eval, Pi 5 | Explainer on Pi | [LLAMA_PI_RESULTS.md](LLAMA_PI_RESULTS.md) |
| Edge pipeline results log | Guards G1 and G2, explainer fixes A to D, raw point pass 26–27 of 33, Hinglish template, network endpoint | 40-row eval, Pi 5 and laptop | Edge pipeline | [EDGE_PIPELINE.md](EDGE_PIPELINE.md) |

## 4. Latency and device measurements

Definitions as the documents use them. "Decision time" excludes the explainer. "Total" is the whole request including explanation. TTFT is time to first token.

| Document | Detector-only time | Share-to-render | First explanation | Full explanation | Device and conditions |
|---|---|---|---|---|---|
| [EDGE_PIPELINE.md](EDGE_PIPELINE.md) | Not separated | Not measured | TTFT 422 ms (direct, one reuse run, same file) | Total p50 / p95 989 / 1,887 ms (evidence 30, direct, clean power); soak 100 requests p50 956 ms, p95 2,152 ms | Pi 5, USB supply, no cooler, throttle 0x0 at start and end |
| [BASELINE_PI.md](BASELINE_PI.md) | Not separated | Not measured | TTFT p50 1,101 ms, p95 1,218 ms (runner path, 31 calls) | Request p50 1,393.5 ms, p95 2,931.6 ms (runner, pre-default, 40 rows) | Pi 5, 3 threads, 2026-11-07; the runner path is no longer the default |
| [SYSTEM_V1.md](SYSTEM_V1.md) (section 2c, quoting the sources above) | URL decision time added p95 2.35 ms (verify minus quiet) | Not measured | Not given | As above | Pi 5, USB, no cooler |
| [URL_SERVICE_RESULTS.md](URL_SERVICE_RESULTS.md) | URL server total p50 4.79 ms, p95 5.75 ms (n = 295); decision time verify p95 10.50 ms vs quiet 8.16 ms (n = 95 each) | Not measured | Not applicable | Not applicable | Pi 5, 3 threads, 2026-11-12; thermal rule breached on SMS batches |
| [NIDS_EDGE.md](NIDS_EDGE.md) | Per-flow scoring p50 / p95 reported per pcap; Pi throughput 6,188 flows/s (scoring only) | Not measured | Not applicable (template only) | Per-request p50 56.4 ms, p95 125.8 ms | Pi 5, 3 threads; laptop 27,816 flows/s for comparison |
| [EXPLAINER_SPEED_RESULTS.md](EXPLAINER_SPEED_RESULTS.md) | Not applicable | Not measured | Not measured | x86 LLM p50 615 ms (baseline); Pi figure is a derived projection (989 ms times the ratio), not a measurement | x86 VM, 4 vCPU shared, 3 threads |
| [ENCODER_PI_RESULTS.md](ENCODER_PI_RESULTS.md) | Partial only (run stopped at the 64 °C bench limit) | Not measured | Not applicable | Not applicable | Pi 5; the gate was not evaluated |
| [PTE_INJECTION_RESULTS.md](PTE_INJECTION_RESULTS.md) | Not applicable | Not measured | Not measured | Not measured as latency | Pi 5 |
| [ANDROID_APP.md](ANDROID_APP.md) | Not measured | Not measured | Not measured | Not measured | Device numbers are in [benchmarks.md](benchmarks.md) section 1 (detector only, raw output not committed); 53 JVM unit tests |

## 5. Android

| Document | What it measures | Data / split | Component | Link |
|---|---|---|---|---|
| Android app parity and build | Kotlin canonical text, rules, side flags, byte ids, decisions match the Python code on 318 rows; 27 JVM unit tests passed on 2026-11-20 (53 exist now) | 318-row parity fixture (SMS campaign test, GroundLM, SmishX, hand-made cases) | Android app | [ANDROID_APP.md](ANDROID_APP.md) |
| Local history and model import | Design and data layer only; no UI wiring yet | Not applicable | Android app | [../android/docs/HISTORY_AND_PROVISIONING.md](../android/docs/HISTORY_AND_PROVISIONING.md) |

## 6. Assurance checks

| Document | What it measures | Data / split | Component | Link |
|---|---|---|---|---|
| Offline assurance | Tests run in a no-network namespace with a negative control; Python-level socket patch | Default test selection | All on-device code paths | [OFFLINE_ASSURANCE.md](OFFLINE_ASSURANCE.md) |

## 7. JSON results

| File | Contents | Linked from |
|---|---|---|
| [results/nids_xdata/metrics.json](results/nids_xdata/metrics.json) | NIDS leave-one-dataset-out tables, gates, verdict, per-fold | NIDS_XDATA_RESULTS.md |
| [results/recall_v2/metrics.json](results/recall_v2/metrics.json) | Recall V2 arms, thresholds, gate limits, R3 gates | RECALL_V2_RESULTS.md |
| [results/explainer_speed/metrics.json](results/explainer_speed/metrics.json) | Explainer arms and baseline, token and wall-time counts | EXPLAINER_SPEED_RESULTS.md |

## 8. Not results documents

- [reports/Rakshak%20Edge%20improvement%20plan.md](../reports/Rakshak%20Edge%20improvement%20plan.md): a planning note that quotes numbers from the documents above; it is not a measurement.
