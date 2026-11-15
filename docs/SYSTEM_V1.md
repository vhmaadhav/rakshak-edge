# Rakshak-Edge SYSTEM_V1: what Phase 1 ships

Frozen description, written 2026-11-12, for the Phase 1 deadline (2026-11-23). The technical report and the README quote
this file. Every number carries its source (repo-relative path and section, or line). Nothing here is rounded beyond what
the source shows. Where sources disagree, section 6 gives both. A later change is a new version (`SYSTEM_V2`), not an edit.

Hashes are `sha256sum` of the working tree on 2026-11-12. Sizes are bytes. A "v0" column means the value at the frozen
v0 commit `ac45aea` (`docs/BASELINE.md`, section "Frozen system v0").

---

## 0. Repo state

| Item | Value | Source |
|---|---|---|
| Git HEAD | `90809f2` ("NIDS rules-first path, baseline v0, SMS retrain experiments, faster Pi explainer") | `git log`; `docs/STATUS.md` 2026-11-12 entry |
| v0 freeze commit | `ac45aea` (on-device edge pipeline on the Pi 5) | `docs/BASELINE.md` "Code" |
| Uncommitted, decision-relevant | `deploy/pi/service/edge_pipeline.py`, `edge_decision.py`, `rakshak_edge_service.py`, `src/rakshak/evidence.py`, `verifier.py`, `remediation.py`, `deploy/pi/bench/registry.toml` (modified); `src/rakshak/url_model.py` (new) | `git status`, 2026-11-12 |
| Uncommitted, not decision-relevant | NIDS holdout scripts (`make_holdout_pcaps.py`, `run_holdout_nids.py`), `url_parity.py`, `url_latency.py`, URL plan and results docs, tests | `git status`, 2026-11-12 |
| Uncommitted URL path | Adds the URL model to `/analyze` (verify mode only) and `POST /analyze_url`. Quiet mode does not call the URL scorer | `deploy/pi/service/edge_decision.py` `URL_EVIDENCE_MODE = "verify"` (line 31); `docs/URL_SERVICE_RESULTS.md` |

### 0a. Code hashes

| File | Working tree 2026-11-12 (sha256) | HEAD `90809f2` | v0 `ac45aea` | Status |
|---|---|---|---|---|
| `src/rakshak/rules.py` | `9bde4ae25a44e5721daa4a1bd1c97304376d75765e3bed5d952dc5e845078e5a` | `9bde4ae25a44e5721daa4a1bd1c97304376d75765e3bed5d952dc5e845078e5a` | `9bde4ae25a44e5721daa4a1bd1c97304376d75765e3bed5d952dc5e845078e5a` | identical to HEAD and v0; unchanged since v0 |
| `src/rakshak/canonicalize.py` | `892a89ad7619181e9d41f609bf46df52dc76622441b3c789ffb2c95228e783cb` | `892a89ad7619181e9d41f609bf46df52dc76622441b3c789ffb2c95228e783cb` | `892a89ad7619181e9d41f609bf46df52dc76622441b3c789ffb2c95228e783cb` | identical to HEAD and v0; unchanged since v0 |
| `src/rakshak/evidence.py` | `2aeec432badc48f6f302b22e43ff86a23c9405d116d8af14e8ce0cd95878cbdc` | `df8300c6f9ee1d0392640df3cfcc674d60fdf2e4bca95bf1334f9b90d85302f2` | `df8300c6f9ee1d0392640df3cfcc674d60fdf2e4bca95bf1334f9b90d85302f2` | WT adds the URL-model item and DO_NOT_OPEN_LINK (+12 lines), uncommitted; HEAD = v0 |
| `src/rakshak/verifier.py` | `8c01f9009cfd5d7ae2f8c791e454cf19f82d4ba27a976d0e526bd773f3f8fd40` | `f10890b50d981a5b660d48620400bb6b8075d0b224a1559d1a3f2843e1c4558a` | `f10890b50d981a5b660d48620400bb6b8075d0b224a1559d1a3f2843e1c4558a` | WT adds the URL type phrase (+3 lines), uncommitted; HEAD = v0 |
| `src/rakshak/remediation.py` | `2051b644d4ff0a6d9e2c9d54c5795966ce11b72d60e81e889c2e4d50df0f97d0` | `7a65bc6a2b4094b5bff29886f98e448b8e3fb9e0fc99bf04da79027b4ae3a4ff` | `not in commit` | WT adds DO_NOT_OPEN_LINK text (+17 lines vs HEAD), uncommitted; not in v0 |
| `src/rakshak/url_model.py` | `199101f40e29640693a31b93fa9b0c1e8996680a02a034d1bb80109fd0e922a6` | `not in commit` | `not in commit` | new, untracked; not in v0 |
| `src/rakshak/url_features.py` | `3849783a2fd62621553b2e200ccba305dce7ca629c3b5e753efe1ecbdba4f91e` | `3849783a2fd62621553b2e200ccba305dce7ca629c3b5e753efe1ecbdba4f91e` | `3849783a2fd62621553b2e200ccba305dce7ca629c3b5e753efe1ecbdba4f91e` | unchanged; same in v0 and HEAD |
| `src/rakshak/nids/host_rules.py` | `b0a42e9f5565ebf2e9cfb178a05e96d09402539edd4fa0ff3f0bb34e8f18b015` | `b0a42e9f5565ebf2e9cfb178a05e96d09402539edd4fa0ff3f0bb34e8f18b015` | `not in commit` | unchanged since HEAD; not in v0 |
| `deploy/pi/service/edge_decision.py` | `e03665d3ac80079eea14702f443ef091cf55250d166b341a3ef569a1ec903375` | `1e3d985bf6923f45454054a9cf85481b02a44dc6afe881edd42c4ed68fc2fe19` | `1e3d985bf6923f45454054a9cf85481b02a44dc6afe881edd42c4ed68fc2fe19` | HEAD = v0 (quiet-mode decision code); WT adds URL branch, uncommitted |
| `deploy/pi/service/edge_pipeline.py` | `d2005a7b51c4ad718d75976004924b4e4efb62ee3a53a92b8b7eaac5679ac2b1` | `3b6eb9ad36c736d7f2af1df1317bbfd503be2e495a641dcae40b157d0927f149` | `4ff46f0334d102eb90115849327070356ad3184c9930735f7ce734827dc2403a` | HEAD adds remediation fields (differs from v0); WT adds URL path, uncommitted |
| `deploy/pi/service/rakshak_edge_service.py` | `ac30745da45d36e2e2ac5bd572790b03e25abc098ac31d37efdd51aefc9982e4` | `3529668e52616eddef9761e71b2bf9c18995af159c129f9961e85e150a04167f` | `e2452aaa08eba05a429b0d349717ac112592e4bbea04915e0cb8523e7766a790` | HEAD adds network endpoints (differs from v0); WT uncommitted |
| `deploy/pi/service/nids_pipeline.py` | `1fa880bbace8111a51d77f08c8ab7cb0c25884a5a2034659673ec840439b218b` | `1fa880bbace8111a51d77f08c8ab7cb0c25884a5a2034659673ec840439b218b` | `not in commit` | unchanged since HEAD; not in v0 |
| `deploy/pi/bench/registry.toml` | `8715ef85ab835cf810c13284e82bf4b122fcc9353e393444aaa09b41a7499b8a` | `fbf290517a60a99f1c1ad3ab1776d539600bec3a39037f7518242e4ae7d6099c` | `fbf290517a60a99f1c1ad3ab1776d539600bec3a39037f7518242e4ae7d6099c` | WT adds url_student threshold (tau_url), uncommitted; HEAD = v0 |

Hashes are full 64-character `sha256sum` values. `not in commit` means the file did not exist at that revision.

---

## 1. What ships

### 1a. Models and pinned artifacts

| Component | File (repo path or Pi path) | sha256 | Bytes | Threshold | Role | Source |
|---|---|---|---|---|---|---|
| SMS student | `kaggle/out/sms_student/output/best_student.onnx`; Pi `models/sms_student/best_student.onnx` | `5341e14ca02a523bd746a71dd67903439e9e6c6f6fc3193fa1c6f786953f2466` | 388127 | **0.35127517580986023** (campaign cal max-F1) | Detector for `/analyze`. Dual-view char-CNN (raw and canonical bytes, 16 rule flags). Identity: campaign-run `dual-base/dual/s1` | `docs/BASELINE.md` "Shipped artifacts" and "Contamination check"; `deploy/pi/bench/registry.toml` `sms_student` |
| SMS student, not shipped | `kaggle/out/sms_student-kd/output/best_student.onnx` | `684e5fee…d26bd58fcd` (full: `684e5feedc584512bc7226f4a4bfc59413a007e1e24b35e9d7728ed26bd58fcd`) | not measured | 0.1628570556640625 | KD variant, registered, not the default | `registry.toml` `sms_student_kd`; `docs/BASELINE.md` "Shipped artifacts" |
| URL student | `kaggle/out/url_student/output/best_url_student.onnx`; Pi `models/url_student/best_url_student.onnx` | `1bbd2c57cb541a7cbc287887ef51976d54240a353655827a511c63ec56213fc7` | 2741232 | **τ_url = 0.6308487409530461** (smallest cal threshold with benign FPR ≤ 1% on the 5% PhreshPhish carve) | `/analyze`: verify mode only. `/analyze_url`: every mode. Char-CNN wide, raw view, max_len 200 | `docs/URL_SERVICE_RESULTS.md` "Threshold"; `registry.toml` `url_student` |
| URL side scaler | `kaggle/out/url_student/output/url_side_scaler.json` | `c6b3fc968b03339a33052fe814a9bb851d3035cb302b06458955c2a5e5b6501e` | 2083 | n/a | 29 side features, mean and std, clip 10, NaN as 0 | `docs/BASELINE.md` "Shipped artifacts" |
| NIDS INT8 student | `kaggle/out/nids_baselines/output/student_cse_cic_ids2018_int8.onnx`; Pi `models/nids/student_cse_cic_ids2018_int8.onnx` | `d3dfdcb16dcaa2780cd39c56d4c156c193dc2c67caeb4e443fc0447e2f9b8d14` | 6729 | alert **0.0015007257461547852**; uncertain **0.0005502104759216309** | Network model. Model-only flags are quiet UNCERTAIN, never an alert on their own | `docs/BASELINE.md` "Shipped artifacts"; `deploy/pi/service/nids_pipeline.py` lines 39-41; `docs/NIDS_EDGE.md` "Model choice" |
| Explainer LLM | `models/llama32_1b_spinquant/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte` (Pi only) | `8715cdba9e91f6bede00cc5f2d6b12397b95225da5630c4972a8da03001cda3b` | 1135951488 | n/a | Llama 3.2 1B, SpinQuant INT4, ExecuTorch 1.5.1. Writes sentences for evidence advisories only; no call at LOW (G1). Never changes risk or action | `registry.toml` `llama32_1b_spinquant`; `docs/BASELINE.md` "Shipped artifacts". **Not on the laptop: not re-hashed on 2026-11-12.** The Pi copy was verified 2026-11-06 (`docs/BASELINE.md`, registry note) |
| Explainer tokenizer | `models/llama32_1b_spinquant/tokenizer.model` (Pi) | `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55` | 2183982 | n/a | Tokenizer for the `.pte` | `registry.toml` `extras` |

### 1b. Components and roles

| Component | File | Role | Source |
|---|---|---|---|
| Rules (SMS) | `src/rakshak/rules.py`, `canonicalize.py` | Evidence types (urgency, prize_lottery, digital_arrest, shortened_url, suspicious_url, brand_domain_mismatch, apk_install, parcel_customs, account_suspension, kyc_block_urgency, …). Risk and action from evidence | `docs/BASELINE.md` "Code"; `docs/EDGE_PIPELINE.md` section 2 |
| Decision (SMS) | `src/rakshak/evidence.py`, `deploy/pi/service/edge_decision.py`, `edge_pipeline.py` | Scam state = score ≥ 0.35127…; else rule evidence = uncertain; else benign. Risk and action from `build_package` and `choose_action` | `docs/EDGE_PIPELINE.md` section 2 "Decision and the shipped path it uses" |
| Detector-only mode | `edge_decision.py` `DEFAULT_DETECTOR_ONLY_MODE = "quiet"` | Student flag with no rule evidence: risk LOW, action IGNORE, neutral summary, no LLM call | `docs/EDGE_PIPELINE.md` section 3, "Detector-only mode" |
| URL model (SMS path) | `src/rakshak/url_model.py`, `edge_decision.py` (`URL_EVIDENCE_MODE = "verify"`) | Adds `phishing_url_model` evidence on `/analyze` in verify mode only. Quiet mode ignores it (gate U2 failed) | `docs/URL_SERVICE_RESULTS.md` "Gates", U2 |
| URL check | `POST /analyze_url`, `src/rakshak/url_features.py` | URL score plus domain rules (`brand_domain_mismatch`, `shortened_url`, `suspicious_url`). Verdict and remediation in every mode | `docs/URL_SERVICE_PLAN.md` "Service behaviour" |
| Explainer guards | `src/rakshak/verifier.py` G1, G2 | G1: no LLM for LOW risk. G2: attribution guard (LOW rejects any attribution; MEDIUM and UNCERTAIN allow it only in a hedge; HIGH not checked) | `docs/EDGE_PIPELINE.md` section 1 |
| Explainer fixes | `deploy/pi/bench/rbench/slot.py`, `edge_pipeline.py` | Fix A describe prompt (`describe_kw`); Fix B detector-only path; Fix C echo strip and duplicate blanking; Fix D one call per evidence type | `docs/EDGE_PIPELINE.md` sections 3, 3b, 3c |
| Hinglish routing | `src/rakshak/datasets/lang.py`, `edge_pipeline.py` | `detect_lang` routing. Default `--hinglish-mode template` (verifier templates, no LLM) | `docs/EDGE_PIPELINE.md` section 5 |
| Network path | `POST /analyze_flows`; `src/rakshak/nids/host_rules.py`; `host_alerts.py`; `pcap_flows.py`; `deploy/pi/service/nids_pipeline.py` | Rules-first per-host alerts (port_scan, brute_force_ssh, brute_force_other, dns_tunnel, dos_flood). Model flags used as evidence only | `docs/NIDS_EDGE.md` "Rules-first policy" |
| Remediation | `src/rakshak/remediation.py` | Deterministic `{code, title, steps}` per action. SMS: 8 codes; network: 11 codes | `docs/NIDS_EDGE.md` "Remediation" |

### 1c. Service defaults (`deploy/pi/service/rakshak_edge_service.py`)

| Setting | Default | Flag or env | Source |
|---|---|---|---|
| Detector-only mode | `quiet` | `--detector-only-mode`, `RAKSHAK_DETECTOR_ONLY_MODE` | `edge_decision.py` line 27; `rakshak_edge_service.py` `add_argument` |
| URL evidence on `/analyze` | off in quiet (verify only) | none | `edge_decision.py` line 31 |
| Threads | **3** | `--threads` | `rakshak_edge_service.py` `add_argument("--threads", default=3)`; `docs/EDGE_PIPELINE.md` section 4 |
| Explainer runner | `direct` (prefix reuse on the system turn) | `--llm-runner`, `RAKSHAK_LLM_RUNNER` | `rakshak_edge_service.py` `_env_llm_runner_default`; `docs/EDGE_PIPELINE.md` "Prefix reuse" |
| Explainer cache | **256** entries (LRU, exact by prompt) | `--explainer-cache`, `RAKSHAK_EXPLAINER_CACHE` | `rakshak_edge_service.py` `_env_cache_default`; `docs/EDGE_PIPELINE.md` "Explainer cache" |
| Thermal refusal | 503 `cooling` at SoC ≥ **75.0 °C** | `--temp-limit` | `rakshak_edge_service.py` line 81 `DEFAULT_TEMP_LIMIT = 75.0`; `docs/BASELINE.md` "Service settings" |
| Hinglish mode | `template` | `--hinglish-mode`, `RAKSHAK_HINGLISH_MODE` | `edge_pipeline.py` line 57 |
| Network model mode | `quiet` (rules decide) | `--net-model-mode`, `RAKSHAK_NET_MODEL_MODE` | `nids_pipeline.py` line 41; `docs/NIDS_EDGE.md` "Model roles" |
| Flow source | `auto` (nfstream when importable, else stdlib) | `--nids-flow-source` | `rakshak_edge_service.py`; `docs/NIDS_EDGE.md` "Flow extraction" |
| Network isolation | own network namespace, `lo` only, no route, Unix socket relay | `start_service.sh` | `docs/EDGE_PIPELINE.md` section 2 "Offline proof" |

---

## 2. Headline numbers

### 2a. Detector level (the student alone, before rules, explainer or quiet mode)

| Component | Split or protocol | Metric (CI or denominator) | Value | Valid for v0? | Source |
|---|---|---|---|---|---|
| SMS student | Campaign-disjoint test, `campaign` (22,145 rows, 11,193 scam) | F1 | **0.9813** [0.9794, 0.9830] | **VALID** (contamination check) | `docs/BASELINE.md` "Campaign-disjoint test"; contamination table row "campaign test" |
| SMS student | same | PR-AUC; precision; recall; FPR | 0.9977 [0.9973, 0.9981]; 0.9824; 0.9803; 0.0180 | VALID | same row |
| SMS student | Campaign test, perturbation harness (9 families × 3 seeds) | clean recall; mean ASR; worst family | 0.9803; 0.0023; char_spacing 0.9770 | VALID (canonicaliser on) | `docs/BASELINE.md` "SMS robustness" |
| SMS student | Hinglish, `lang_shift` (1,677 rows, 357 scam) | F1; PR-AUC; FPR | 0.9860 [0.976, 0.994]; 0.9984; 0.0045 | VALID per contamination table, **with caveat** (section 4) | `docs/BASELINE.md` "Hinglish (lang_shift)"; contamination table |
| SMS student | `temporal` (11,029 rows, **77 scam**) | F1; precision; recall; FPR | 0.4298 [0.394, 0.473]; **0.276**; 0.974; 0.0180 | PARTIAL (0.55% overlap) | `docs/BASELINE.md` "Temporal" row |
| SMS student | GroundLM policy negatives, scoreable (1,133) | FPR = FP / N (Wilson 95%) | **471 / 1,133 = 41.57%** [38.73, 44.46] | **PARTIAL**: 5.65% text overlap with training (64 of 1,133) | `docs/BASELINE.md` "GroundLM policy negatives, student"; contamination table |
| SMS student | SmishX smishing (226 eval_ok) | recall (Wilson 95%) | **157 / 226 = 69.47%** [63.18, 75.11] | **PARTIAL**: 5.78% text overlap (64 of 1,107) | `docs/BASELINE.md` "SmishX, smishing recall"; contamination table |
| SMS student | SmishX legitimate (606 eval_ok) | ham FPR (Wilson 95%) | 55 / 606 = 9.08% [7.04, 11.63] | PARTIAL | `docs/BASELINE.md` "SmishX, legitimate FPR" |
| SMS rules only | GroundLM policy negatives, rule evidence | FPR | 12 / 1,133 = 1.06% [0.61, 1.84] | PARTIAL | `docs/GROUNDLM_EDGE_FPR.md` table (b) |
| SMS student, not in v0 | SMS source holdouts (ScamShield, UCI, Mishra), legacy UCI and Mishra splits | F1 | 0.9805; 0.9544; 0.9833; 0.9685; 0.9820 | **IN-SAMPLE** (70 to 78% row overlap). Do not quote as v0 | `docs/BASELINE.md` contamination table |
| URL student | PhreshPhish official test, URL-only | AP and P@R=0.90 at 0.05%, 0.1%, 0.5%, 1%, 5% FPR | 0.4843 / 0.5991 / 0.8090 / 0.8647 / 0.9413 (AP) | Overlap with training not audited in the sources; no URL-only published target | `docs/BASELINE.md` "URL (char-CNN student ONNX)" |
| URL student | PhreshPhish full test (168,060 rows) | AP; P@R=0.90 | 0.9914; 0.9928 | ours only | `docs/BASELINE.md` "PhreshPhish full test" |
| URL student | `url_domain` test (30,432 rows, out of PhreshPhish training) | PR-AUC; ROC-AUC; recall at its own cal FPR 1% | 0.9746; 0.911; 0.779 (test FPR 1.12%) | ours only; **its own threshold, not τ_url** | `docs/BASELINE.md` "url_domain test" |
| URL student at τ_url | PhreshPhish official.test (91,260 benign) | FPR (Wilson 95%); recall; precision | 0.92% [0.86, 0.99]; 92.0%; 98.8% | U1 pass | `docs/URL_SERVICE_RESULTS.md` U1 |
| URL student at τ_url | `url_domain` test | FPR; recall | 69.4%; 94.4% | reported, not gated | `docs/URL_SERVICE_RESULTS.md` "Reported, not gated" |
| NIDS student (INT8, cse_cic_ids2018) | Within-dataset held-out random split | F1 at max-F1 threshold (bootstrap CI); MCC; ROC-AUC; PR-AUC | 0.796 [0.777, 0.812]; 0.777; 0.877; 0.794 | Kaggle report; **v0 artifact, older evaluation code** (`docs/BASELINE.md` NIDS provenance) | `docs/BASELINE.md` "NIDS" T0 row; `kaggle/out/nids_baselines/output/report.md` "Student" |
| NIDS student | same | recall at FPR 0.1%; at FPR 1% | 0.622; 0.679 | same | `docs/BASELINE.md` NIDS T0 row; `docs/NIDS_EDGE.md` "Student" table |
| NIDS student | Cross-dataset (trained cse, tested elsewhere) | F1 to unsw / bot_iot / ton_iot | 0.297 / 0.332 / 0.718 | Near chance on two pairs | `docs/BASELINE.md` NIDS row; `docs/NIDS_EDGE.md` cross-dataset table |

### 2b. System level (what a user sees in quiet mode, the default)

| Measure | Value | Denominator | Source |
|---|---|---|---|
| GroundLM, visible warning (any action other than IGNORE, quiet) | **2 / 1,133 = 0.18%** [0.05, 0.64] (Wilson) | 1,133 scoreable policy negatives. PARTIAL data | `docs/GROUNDLM_EDGE_FPR.md` "User-visible warning" table; `docs/BASELINE.md` "GroundLM, system-level visible warning" |
| GroundLM, visible warning in verify mode (not shipped) | 472 / 1,133 = 41.66% | same | `docs/GROUNDLM_EDGE_FPR.md` |
| SmishX, visible smishing recall (quiet) | **53 / 226 = 23.45%** [18.40, 29.39] | 226 smishing eval_ok. PARTIAL | `docs/GROUNDLM_EDGE_FPR.md` SmishX table, row (f) quiet |
| SmishX, visible ham FPR (quiet) | 22 / 606 = 3.63% [2.41, 5.44] | 606 legitimate | same |
| Campaign test, visible recall (quiet) | 16.20% | 11,193 positives. Not the same as 2a recall, which is the score, not the visible action | `docs/GROUNDLM_EDGE_FPR.md` table |
| Campaign test, visible FPR (quiet) | 0.61% | 10,952 negatives | same |
| Cost of quiet mode (for reference) | SmishX visible recall 76.55% (verify) to 23.45% (quiet); campaign 98.12% to 16.20% | | `docs/EDGE_PIPELINE.md` "Detector-only mode"; `docs/HANDOFF.md` open decision (a) |
| Visible-warning bank and UPI false alarms | Not shown in quiet mode (161 / 161 bank, 146 / 146 UPI in verify) | | `docs/GROUNDLM_EDGE_FPR.md` "Where the FPs are" |
| URL on `/analyze` | off in quiet (gate U2 failed: 2 → 5 of 1,133) | | `docs/URL_SERVICE_RESULTS.md` U2 |
| URL on `/analyze_url` | verdict in every mode | | `docs/URL_SERVICE_PLAN.md` |
| Hinglish SMS (30 scam, 10 benign, template mode) | detector flags 30 / 30 scam, 0 / 10 benign; user alert (MEDIUM+) 24 / 30 scam, 1 / 10 benign (row 34, a rule false positive, `account_suspension`) | 40 rows | `docs/EDGE_PIPELINE.md` section 5, arm B |
| Explainer contract (every final output passes) | 30 / 30 evidence, 40 / 40 all (template fallback by construction) | 40 rows | `docs/EDGE_PIPELINE.md` section 3c; `docs/BASELINE_PI.md` section C |
| Network, rules-first (quiet), holdout, 26 files, 2017 flows | attacks detected **11 / 11** (main and mixed sets), family right 11 / 11 | 11 attack instances | `docs/NIDS_EDGE.md` "Holdout results" |
| Network, rules-first, benign files alerting | **1 / 6** (`ho_benign_cdn_lookalike_6`, which meets the length rule on purpose) | 6 benign files | same |
| Network, model-only (not shipped) | 4 / 11 detected; family right 0 / 11; benign files alerting 5 / 6 | same | same |
| Network, combined (`--net-model-mode alert`, not default) | 11 / 11 detected; benign files alerting 6 / 6 | same | same |
| Network, near-threshold cases (rules-first) | fired: 10 SSH at 5 s, 20 scan ports, 5 DNS queries, 200 SYN. Did not fire: 9 SSH, 10 SSH at 7 s, 40 slow scan ports, 4 DNS, 150 SYN. Misses predicted: SSH at 7 s, scan at 1 s per port | 10 near-threshold files | `docs/NIDS_EDGE.md` "Near-threshold, rules-first" |

**Holdout caveat (NIDS).** The holdout was fixed before the run (seed label `holdout-2026-11-12`), but the rule parameters
were chosen by reading the rule constants so that each case sits at a stated edge. The files are synthetic, from scapy,
RFC 1918 only, with nothing sent. This is a functional check, not a measure of accuracy on real traffic (`docs/NIDS_EDGE.md`
"What this says").

### 2c. Pi (device) measurements, with conditions

Condition key: **USB** = USB PSU (`usb_max_current_enable=1`), **no cooler**. **0x0** = throttle flag at every sample. **0x50000** = under-voltage flag, set since boot (`docs/EDGE_PIPELINE.md` section 4 and "Prefix reuse").

| Measure | Value | Run and conditions | Source |
|---|---|---|---|
| Idle power (PMIC, not calibrated) | 1.558 W (sd 0.221) | 2026-11-07, 60 s, service loaded, USB, 0x0 | `docs/BASELINE_PI.md` section B "Idle power" |
| Score parity, SMS (300 rows) | max abs prob diff 0.0; **0 flips** at 0.35127… | Pi vs laptop, 2026-11-07, 1 intra-op thread, 0x0 | `docs/BASELINE_PI.md` section A |
| Score parity, URL (300 rows) | service path 0 / 300 mismatches; **0 flips** at τ_url; laptop vs Pi bit-identical | 2026-11-12, Pi SoC 46.9 to 50.7 °C, 0x0 for the batch | `docs/URL_SERVICE_RESULTS.md` "U4 Pi half" |
| Latency, English 40 (all), **runner path** (pre-default) | request p50 1,393.5 [1,252, 1,573] ms; p95 2,931.6 ms | 2026-11-07, 3 threads, USB, 0x0. The `runner` path was the default until 2026-11-11 | `docs/BASELINE_PI.md` section B1 |
| Latency, evidence 30, **direct** (shipped default), clean power | total p50 **989 ms**; p95 **1,887 ms** (runner, same day: 1,722 / 3,078) | 2026-11-11, 5.15 V input rail, throttle 0x0 at every start and end, cache off | `docs/EDGE_PIPELINE.md` "Clean-power runs" table |
| Latency, evidence 30, direct, under-voltage | total p50 957 ms; runner 1,683 ms | 2026-11-11, first run, flag **0x50000** at start and end. Not the quotable numbers | `docs/EDGE_PIPELINE.md` 40-row table; `docs/STATUS.md` 2026-11-11 |
| Latency, Hinglish 40 (template) | request p50 4.5 ms; p95 7.9 ms | 2026-11-07, 0x0 | `docs/BASELINE_PI.md` section B2 |
| LLM call (runner, pre-default) | p50 1,558.1 ms; p95 1,967.1 ms (n = 31) | 2026-11-07, 0x0 | `docs/BASELINE_PI.md` section B1 |
| TTFT, runner | p50 1,101 ms | 2026-11-07, 0x0 | same |
| Decode speed | 13.4 tok/s (runner, B1); 11.9 (soak); 14.64 (direct, clean soak) | 2026-11-07 and 2026-11-11 | `docs/BASELINE_PI.md`; `docs/EDGE_PIPELINE.md` clean-power table |
| Soak, 200 requests, 3 threads, runner | request p50 1,817.7; p95 3,824.9; p99 6,119.3 ms; 0 aborts, 0 503 | 2026-11-07, SoC max 70.8 °C, 0x0 | `docs/BASELINE_PI.md` section B soak |
| Soak, 100 requests, 3 threads, direct, clean power | request p50 **956 ms**; p95 **2,152 ms**; 0 aborts | 2026-11-11, 0x0 at start and end | `docs/EDGE_PIPELINE.md` clean-power table |
| Soak, 100 requests, threads 2 / 3 / 4 | request p50 2,400 / 1,939 / 1,820 ms; p95 9,265 / 7,152 / 7,524 ms; SoC max 69.7 / 71.9 / 73.0 °C | 2026-11-06, one run each, 0x0 | `docs/EDGE_PIPELINE.md` section 4 |
| Energy per request, evidence 30, direct, clean | p50 **3.52 J** (runner 7.12 J) | 2026-11-11, PMIC sum, not a calibrated meter | `docs/EDGE_PIPELINE.md` clean-power table |
| Energy per request, B1 English 40 | p50 6.70 J raw; 4.30 J idle-subtracted | 2026-11-07, PMIC | `docs/BASELINE_PI.md` section B1 |
| Memory (service) | RSS steady 1,655.8 to 1,679.9 MB; peak 1,839.3 MB | 2026-11-07, 4 GB board | `docs/BASELINE_PI.md` "Llama load, RSS, SoC" |
| Llama load time | 15.8 s wall at that start (method 2.6 s); 7.6 s at an earlier start | 2026-11-07 | `docs/BASELINE_PI.md` "Provenance" and section B |
| Explainer, raw point pass (model sentences that pass the verifier) | 27 / 33 (82%) | 2026-11-07 (runner, 40 rows) | `docs/BASELINE_PI.md` section C |
| Explainer, raw point pass, Fix D run | 26 / 33 (79%) | 2026-11-06 (3 threads, 0x0) | `docs/EDGE_PIPELINE.md` section 3c; `docs/BASELINE.md` explainer row |
| Explainer, unterminated calls | 4 / 31 (12.9%) | 2026-11-07 | `docs/BASELINE_PI.md` section C |
| Explainer, mode llm / partial / template (evidence 30) | 21 / 1 / 8 (2026-11-07, runner); 22 / 1 / 7 (Fix D) | | `docs/BASELINE_PI.md`; `docs/EDGE_PIPELINE.md` |
| Explainer, G2 rejections in the 40-row eval | 1 (after Fix A and B); 30 before (Fix A/B not applied) | | `docs/EDGE_PIPELINE.md` section 3 table |
| URL, added latency per `/analyze` (verify minus quiet, decision time) | p95 **2.35 ms** (mean 1.48 ms), gate ≤ 50 ms | 2026-11-12, n = 95 each, 3 threads. Thermal breach, see section 4 | `docs/URL_SERVICE_RESULTS.md` "U5" |
| URL, `/analyze_url` server total | p50 4.79 ms; p95 5.75 ms | 2026-11-12, n = 295 | same |
| Network, holdout on Pi (rules-first, 26 files) | 11 / 11 attacks; 1 / 6 benign alerting; identical to the laptop file for file | 2026-11-12, SoC start 54 °C, peak 59.0 °C, 0 pauses, throttle 0x0 | `docs/NIDS_EDGE.md` "Holdout on the Pi 5" |
| Network, Pi throughput (scoring only, rules mode) | 6,188 flows/s; per request p50 56.4 ms; p95 125.8 ms | 2026-11-12, same run | same |
| Network, laptop throughput (for comparison) | 27,816 flows/s; per request p50 1.2 ms; p95 15.8 ms | laptop, 2026-11-12 | `docs/NIDS_EDGE.md` "Holdout results" |
| Thermal refusal | 0 refusals (503) in every Pi run; the 75 °C gate was never reached. Max SoC: 70.8 (soak, 2026-11-07), 71.9 (soak, 2026-11-06, 3 threads), 73.0 (4 threads), 71.4 (URL U5 verify batch, start temperature) | all runs | `docs/BASELINE_PI.md`; `docs/EDGE_PIPELINE.md` section 4; `docs/URL_SERVICE_RESULTS.md` |
| Service restart and offline | offline proof: `lo` only, `routes: 0`, egress probe failed (2026-11-06) | | `docs/EDGE_PIPELINE.md` section 2 |

---

## 3. What does NOT ship

| Item | Result (the number that decided it) | Source |
|---|---|---|
| SMS retrain arms A, B, C (FinEE synthetic legit; minimal pairs; CloveAI) | KILL. SmishX TP 16 to 31 of 226 (gate ≥ 157); campaign F1 0.839 to 0.907 (gate ≥ 0.9713) | `docs/SMS_RETRAIN_RESULTS.md` "Verdict" |
| Inbox retrain arms R0, R1, R2 (scrubbed text plus one donor) | Fail G1 to G5. R0 seed 2 collapsed to τ ≈ 1.0. Placeholder tags act as a shortcut | `docs/INBOX_RETRAIN_RESULTS.md` "Gates" and "Confound" |
| Surrogate diagnostic D0 (tags filled, v0 scored) | H1 fails. Phone-1 legit FPR 98.6% scrubbed, 91.0% filled (shipped threshold). Stop rule 1 cancels S0 to S2 | `docs/SURROGATE_D0_RESULTS.md` table |
| Inbox filled arms S0, S1 | H4, H5, H6 fail. S1 SmishX TP 29 / 23 / 28 (gate ≥ 157); test-donor legit FPR 4.70% median (gate ≤ 1%) | `docs/INBOX_FILLED_RESULTS.md` "Verdict" |
| Teacher cross-source check D1 (mmBERT-base, fold `holdout_scamshield`) | "Does not transfer": A2 AUC 0.7225 < 0.85 (A1 passes at 0.9947). Teacher is not shipped | `docs/TEACHER_XSOURCE_RESULTS.md` "Verdict" |
| Teacher fine-tune T0 (campaign only) | Reported only. SmishX TP 4 / 226; campaign F1 (filled) 0.182 | `docs/TEACHER_FINETUNE_RESULTS.md` table |
| Teacher fine-tune T1 (+ inbox train donor) | No provisional pass. G5 1.14% (gate ≤ 1%), SmishX TP 57 / 226, campaign F1 (filled) 0.908. Stop rule 3 fires | `docs/TEACHER_FINETUNE_RESULTS.md` "Hypotheses" |
| INT8 teacher | INT8 dynamic quantisation not usable: argmax agreement 0.906, Pearson 0.912 against fp32. FP32 ONNX export works (`kaggle/out/sms_teacher_export/`) | `docs/TEACHER_FINETUNE_RESULTS.md`, last paragraph |
| `sms_student_kd` (KD variant) | Registered, not the service default, not v0 | `deploy/pi/bench/registry.toml`; `docs/BASELINE.md` "Shipped artifacts" |
| URL evidence in quiet mode on `/analyze` | Gate U2 fails: 5 / 1,133 visible (0.44%), limit ≤ 4. Shipped quiet keeps 2 / 1,133 | `docs/URL_SERVICE_RESULTS.md` U2; `docs/URL_SERVICE_PLAN.md` "Gates" |
| Verify mode as default SMS mode | Kept behind `--detector-only-mode verify`. Visible GroundLM FPR 41.66% | `docs/EDGE_PIPELINE.md` "Detector-only mode"; `docs/HANDOFF.md` open decision (a) |
| NIDS model-only alerting | 4 / 11 detected, family right 0 / 11, 5 / 6 benign alerting | `docs/NIDS_EDGE.md` "Holdout results" |
| NIDS combined mode as default (`--net-model-mode alert`) | Kept for comparison. 6 / 6 benign alerting | same |
| NIDS Llama explainer for network evidence | Not built. Network advisories are template only | `docs/NIDS_EDGE.md` "Not done or untested" |
| NIDS packet-rate DoS rule | Not built (flows carry first and last time only) | `docs/NIDS_EDGE.md` "Rule thresholds" |
| Hinglish LLM explainer (`--hinglish-mode llm`) | Raw point pass 8 / 37 (22%); 29 / 37 calls unterminated | `docs/EDGE_PIPELINE.md` section 5, arm C |
| `slot_kw` prompt | Worse (raw advisory 12 / 30); opt-in only | `docs/HANDOFF.md` "Llama 3.2 1B on ExecuTorch" |
| Inbox-trained models on the Pi | Nothing exported to the Pi | `docs/INBOX_RETRAIN_RESULTS.md`; `docs/INBOX_FILLED_RESULTS.md` |

---


### Added 2026-11-12 (SOTA push, all killed on pre-registered gates)

| Candidate | Doc | Why it does not ship |
|---|---|---|
| mmBERT-base teacher, cal-split threshold (W1a) | `docs/TEACHER_SUITE_RESULTS.md` | 679-725 GroundLM FP; the threshold does not transfer |
| mmBERT-small fold (W1b) | `docs/TEACHER_SUITE_RESULTS.md` | gated arms fail; weaker ranking than base |
| Encoder + licensed exposure (W1e) | `docs/ENCODER_EXPOSURE_RESULTS.md` | same-source calibration threshold far too low (620-773 FP) |
| Inbox-calibrated threshold (W1f) | `docs/INBOX_CALIBRATION_RESULTS.md` | scrubbed inbox pushes the threshold too high; near-zero recall |
| Cross-fitted GroundLM calibration, seed 3 (W1g) | `docs/CROSSFIT_CONFIRM_RESULTS.md` | FP 10/1,133 passes; recall and campaign F1 fail |
| 3-seed ensemble, same protocol (W1h) | `docs/ENSEMBLE_CONFIRM_RESULTS.md` | FP 22/1,133; recall and campaign F1 fail |
| URL student on PhreshPhish + url_domain (W2) | `docs/URL_GENERAL_RESULTS.md` | PP AP at 1 % fails; worse on domain-novel URLs |
| Dynamic INT8 encoder export | `docs/ENCODER_PI_RESULTS.md` | near-constant output (agreement 0.171 / 0.829) |

## 4. Known limits

| Limit | Number | Source |
|---|---|---|
| Student FPR on GroundLM at the detector level | **41.57%** (471 / 1,133). Student-only flags are most of the problem. 403 of 471 have score ≥ 0.8 | `docs/GROUNDLM_EDGE_FPR.md` "Where the FPs are"; `docs/BASELINE.md` |
| GroundLM rules were designed on the same rows | In-sample. The GroundLM number is not an independent test | `docs/GROUNDLM_EDGE_FPR.md` (caveat at top) |
| Bank and UPI legitimate messages | Flagged by the student 161 / 161 and 146 / 146 (verify mode). Hidden in quiet mode, so the user does not see them | `docs/GROUNDLM_EDGE_FPR.md` |
| Temporal split | Precision **0.276** on 77 scam. FPR 1.8% at the shipped threshold, so 2 of every 3 flags are benign | `docs/BASELINE.md` "Temporal" and "Known defect" |
| Quiet-mode recall cost | SmishX visible recall 23.45% (verify 76.55%). Campaign visible recall 16.20% (verify 98.12%) | `docs/GROUNDLM_EDGE_FPR.md`; `docs/EDGE_PIPELINE.md` |
| URL model out of domain | **FPR 69.4%** on `url_domain` test at τ_url (recall 94.4%). The threshold transfers within PhreshPhish, not to other benign sources. This is why quiet mode stays off | `docs/URL_SERVICE_RESULTS.md` "Reported, not gated" |
| URL model, PhreshPhish-trained, out of the URL-only target | AP 0.4843 at 0.05% FPR vs GTE URL+HTML 0.6050. Different input (URL-only) | `docs/BASELINE.md` URL table |
| Missing URL measurement | URLTran TPR at FPR 0.01%: **not measured** | `docs/BASELINE.md` URL row |
| NIDS evaluation | Synthetic pcaps only. Rule parameters chosen after reading rule constants. **Real traffic not tested.** The rules detected all 11 attack instances. The two predicted misses are outside those 11: SSH at 7 s spacing and a SYN scan at 1 s per port (both expected 'none' in the holdout set, `docs/NIDS_EDGE.md` near-threshold table) | `docs/NIDS_EDGE.md` "What this says" |
| NIDS cross-dataset | Near chance on unsw and bot_iot pairs. ROC-AUC below 0.5 on several rows. "A shipped alert is a signal to check, not a verdict" | `docs/NIDS_EDGE.md` "Cross-dataset" and "Conclusion for the device" |
| NIDS student leave-one-dataset-out | **Missing** (T5 student row). Needs NIDS data not in the repo | `docs/BASELINE.md` NIDS row |
| NIDS synthetic benign capture | Benign HTTPS shape alerted in the 2026-11-06 model-alert run | `docs/NIDS_EDGE.md` "Synthetic pcaps" |
| Test donor | Phone 1 (test) and phones 2+3 (train) are the **same person**. 14 scam rows in the test donor | `docs/INBOX_RETRAIN_RESULTS.md` "Data"; `docs/SURROGATE_D0_RESULTS.md` "Limits" |
| Inbox labels | Codex labels on scrubbed text, not human labels | `docs/INBOX_RETRAIN_RESULTS.md` |
| Campaign F1 identity | 0.9813 (ONNX, v0 table) vs 0.9749 (stored run log). Unresolved | `docs/BASELINE.md` note 1 |
| Shipped model identity | Campaign-run `dual-base/dual/s1`, trained on campaign train. About 1,019 trained rows not recoverable from the repo split (`run.log` says 77,609; repo split gives 76,591 plus 11,072 cal). Weight match not run (export parity 3.8e-06) | `docs/BASELINE.md` "Contamination check" |
| `lang_shift` result | Valid for the shipped model. Not a language-shift result: the campaign train includes other Hinglish campaigns. Row count 1,677 (split file) vs 1,600 (stored run) | `docs/BASELINE.md` contamination "Consequences" and note 2 |
| Hinglish SMS remediation and advisory text | First draft. Native-speaker review not done | `docs/NIDS_EDGE.md` "Remediation"; `docs/EDGE_PIPELINE.md` section 5 "Untested" |
| Explainer on the Pi | Raw point pass 26 to 27 of 33 (small n). 4 of 31 calls unterminated (12.9%). Echoes prompt words. The guards reject most model sentences (G2) | `docs/BASELINE_PI.md` section C; `docs/EDGE_PIPELINE.md` section 3 |
| Explainer prompt injection on the `.pte` | **Not measured** (FP16 numbers only). Baseline row MISSING | `docs/BASELINE.md` explainer row |
| Canonicaliser-off robustness | **Missing** (no raw-only ONNX pinned) | `docs/BASELINE.md` "Canonicaliser off" |
| Pi power and cooling | No cooler. USB PSU (`usb_max_current_enable=1`). Energy from PMIC, not a calibrated meter. Throttle 0x0 in the 2026-11-07 runs, the URL runs and the NIDS holdout run. Throttle **0x50000** in the first 2026-11-11 runs (under-voltage since boot) | `docs/BASELINE_PI.md` caveats; `docs/EDGE_PIPELINE.md` section 4 and "Prefix reuse"; `docs/NIDS_EDGE.md` |
| Thermal rule breach in the URL U5 run | SMS batches started at **71.4 °C** (verify) and **65.9 °C** (quiet), above the 65 °C start rule. Throttle 0x0. The quiet and verify comparison is confounded by temperature. The URL-only batch stayed at 57.6 °C | `docs/URL_SERVICE_RESULTS.md` "Thermal caveat" and "Thermal note on U5" |
| Thermal behaviour above 73 °C | Not measured. Service refusal is at 75 °C and was never reached | `docs/EDGE_PIPELINE.md` "Untested" |
| Soak | 100 or 200 requests, one run per setting. The thread choice (3 over 4) is a judgement | `docs/EDGE_PIPELINE.md` section 4 |
| Live 503 refusal | Tested with a mock only | `docs/EDGE_PIPELINE.md` "Untested"; `docs/NIDS_EDGE.md` |
| Relay under concurrent connections | Not tested (the service is serial) | `docs/EDGE_PIPELINE.md` "Untested" |
| Restart after a crash | Not tested | `docs/EDGE_PIPELINE.md` "Untested" |

---

## 5. Open before submission (Phase 1, due 2026-11-23)

| Item | Status | Source |
|---|---|---|
| Final Pi numbers with an Active Cooler and a 27 W (5 A) PSU | Not done. Every Pi number above is USB and no cooler | `docs/HANDOFF.md` open decision (b); `docs/EDGE_PIPELINE.md` section 2 |
| Phase 1 report | Not written. This file is its source | `docs/HANDOFF.md` 2026-11-12 entry |
| Demo video | Not made | `docs/HANDOFF.md` "Ranked next tasks" 5 |
| Android app (ExecuTorch LlamaDemo, AAR 1.5.1) | Not built. The device is not confirmed | `docs/HANDOFF.md` ranked task 4; open decision (c) |
| Repo visibility and judge-readable state | Not done | `docs/STATUS.md` 2026-11-12 ("judge-readable repo") |
| Commit of the uncommitted URL path, NIDS holdout and registry change | Not committed. HEAD is `90809f2`. Commit decision belongs to Maadhav | `git status`; `docs/STATUS.md` 2026-11-12 |
| Second donor for the inbox data | Needed by the teacher fine-tune stop rule 3 (Sachin) | `docs/TEACHER_FINETUNE_RESULTS.md`; `docs/HANDOFF.md` |
| Native-speaker review of Hinglish | Not done | `docs/EDGE_PIPELINE.md` section 5 |
| Decision on the HIGH "is a scam" wording (unhedged at HIGH) | Open | `docs/HANDOFF.md` ranked task 6 |
| Re-run of the URL U5 batches under `thermal_run.py --kill 65` | Not done. The 2026-11-12 batches breached the 65 °C start rule | `docs/URL_SERVICE_RESULTS.md` "Thermal note on U5" |
| Filling the U2 decision (keep URL evidence off in quiet mode) | Decided by the gate. A new pre-registration is needed to change it | `docs/URL_SERVICE_RESULTS.md` "What a different choice would take" |

---

## 6. Source conflicts and gaps

Where two sources give different values or labels, both are cited. The number this file uses is marked.

| # | Topic | Source A | Source B | Used here |
|---|---|---|---|---|
| 1 | Campaign F1 of the shipped SMS model | 0.9813 (ONNX on the shipped file, `docs/BASELINE.md` "Campaign-disjoint test") | 0.9749 (stored `dual-base/dual/s1`, `kaggle/out/sms_student/output/run.log` line 64; `docs/BASELINE.md` note 1) | 0.9813 as the v0 measurement. The gap is unresolved (BASELINE says so) |
| 2 | Explainer raw point pass, same 40 rows | 26 / 33 (`docs/EDGE_PIPELINE.md` section 3c, Fix D, 2026-11-06) | 27 / 33 (`docs/EDGE_PIPELINE.md` 2026-11-11 table; `docs/BASELINE_PI.md` section C, 2026-11-07) | Both shown. Cause not identified in the sources |
| 3 | NIDS model quality claims | `deploy/pi/bench/registry.toml` `nids_cse_cic_ids2018_int8`: "Latency only (synthetic inputs)", no quality reported | `docs/BASELINE.md` NIDS row and `docs/NIDS_EDGE.md` cite the Kaggle report (F1 0.796) | Quality numbers from the Kaggle report, cited as "v0 artifact, older evaluation code" (`docs/BASELINE.md` NIDS provenance). The registry note is out of date |
| 4 | Synthetic benign and SSH on the Pi | 2026-11-06 model-alert run: benign ALERT HIGH dos_ddos; SSH family dos_ddos (`docs/NIDS_EDGE.md` "Synthetic pcaps") | 2026-11-07 rules-first WIP: benign UNCERTAIN, no alert; port scan HIGH (`docs/BASELINE_PI.md` section D) | Different code modes. The shipped rules-first result is the holdout in 2b. Both cited |
| 5 | URL FPR on `url_domain` | 1.12% at the url_domain unit's own cal threshold, recall 0.779 (`docs/BASELINE.md` "url_domain test") | 69.4% at τ_url (`docs/URL_SERVICE_RESULTS.md` "Reported") | Different thresholds. The shipped threshold is τ_url, so 69.4% is the one that applies |
| 6 | URL threshold | "No registered threshold for url_student" (`docs/BASELINE.md` v0, frozen 2026-11-07) | `registry.toml` now holds `threshold = 0.6308487409530461` (τ_url, 2026-11-12) | τ_url. The v0 freeze predates it |
| 7 | Freeze and tree state | `docs/BASELINE.md`: code `ac45aea`, working tree uncommitted, the four SMS source files byte-identical to `ac45aea` | Now HEAD `90809f2`. HEAD matches v0 for `rules.py`, `canonicalize.py`, `evidence.py`, `verifier.py`, `edge_decision.py`. The working tree differs in `evidence.py` (+12 lines), `verifier.py` (+3) and `edge_pipeline.py` (URL path) | Section 0a shows all three values. The default quiet decision logic is the HEAD code (`edge_decision.py` 1e3d985b) |
| 8 | Latency: which explainer runner | `docs/BASELINE_PI.md` (2026-11-07) measured the `runner` path | `docs/EDGE_PIPELINE.md` "Decision": `direct` is the default from 2026-11-11 | The shipped default is `direct`. Its clean-power numbers (2c) are the ones to quote. The BASELINE_PI latency is the earlier runner |
| 9 | Thermal limits in the job guards | Service refusal 75 °C (`rakshak_edge_service.py` line 81) | Job guards differ by run: `docs/HANDOFF.md` thermal_run pause 75, resume 65, kill 80; `docs/EDGE_PIPELINE.md` soak abort 79; `docs/URL_SERVICE_PLAN.md` start below 55, abort 65; `docs/NIDS_EDGE.md` holdout pause 63, resume 58, kill 65 | Only the 75 °C service refusal is a shipped rule. The job guards are per run and reported as recorded |
| 10 | Pi NIDS throughput | `docs/NIDS_EDGE.md` 2026-11-06: flows/s median 5,173 to 6,557 (service wall, per file) | `docs/BASELINE_PI.md` section D: flows/s p50 185 to 1,017 (per request, with HTTP) | Different definitions. Both cited; the 2026-11-12 holdout figure (6,188 flows/s, scoring only) is the one in 2c |
| 11 | Pi soak at 3 threads | `docs/BASELINE_PI.md`: 200 requests, p50 1,817.7 ms, SoC max 70.8 °C, 2026-11-07, runner | `docs/EDGE_PIPELINE.md` section 4: 100 requests, p50 1,939 ms, SoC max 71.9 °C, 2026-11-06 | Two runs on different days. Both cited |
| 12 | GroundLM text-present count | `docs/GROUNDLM_FPR.md` section 3: "text present 1,233" | `docs/GROUNDLM_EDGE_FPR.md` counts check: 1,393 minus 169 = 1,224 | 1,224 (the later doc says the 1,233 figure is wrong) |
| 13 | Column label in the inbox table | `docs/INBOX_RETRAIN_RESULTS.md` header "v0 / R0" | Column values 449 / 0 / 469 are the R0 seeds, not v0. v0 = 471 (`docs/SMS_RETRAIN_RESULTS.md`) | v0 = 471. The label is ambiguous in the source |
| 14 | `lang_shift` row count | 1,677 rows, 357 scam (split file, `docs/BASELINE.md` note 2) | 1,600 rows (stored run, same note) | 1,677 for the v0 table. Unresolved in the source |
| 15 | Throttle state on the Pi | `docs/BASELINE_PI.md`: 0x0 at every sample, 2026-11-07 | `docs/EDGE_PIPELINE.md`: 0x50000 at start and end of the first 2026-11-11 runs (under-voltage since boot) | Recorded per run. Both shown |
| 16 | Unterminated calls | 4 of 45 calls (`docs/BASELINE.md` explainer row, Fix C run) | 4 of 31 calls (`docs/BASELINE_PI.md` section C, 2026-11-07) | Different denominators. Both shown |
| 17 | Llama `.pte` hash availability | `registry.toml` pins the `.pte` sha256 and size | The `.pte` is not on the laptop, so it was not re-hashed on 2026-11-12 | Registry value used. The Pi copy was verified 2026-11-06 per `docs/BASELINE.md` |
