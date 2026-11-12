# Rakshak-Edge baseline v0

Frozen 2026-11-07. Every future improvement is compared against this file. A number counts for v0 only when it was
produced by the v0 artifact (same sha256), so every v0 cell cites its results file and job.

Conventions: F1, PR-AUC, ROC-AUC and rates are fractions unless marked %. "CI" is a campaign-clustered bootstrap
95% interval (`rakshak.eval.metrics.bootstrap_intervals`) unless marked Wilson 95%. "not v0" means the number exists
but was produced by a different model or configuration, so it cannot be cited as the v0 result.

## Frozen system v0

### Code

- Git commit: **`ac45aeaa2f10...`** (`ac45aea`, "On-device edge pipeline on the Pi 5: rbench, Llama explainer,
  guards, Hinglish"), on `main`.
- Working tree at freeze: uncommitted. The network path (`POST /analyze_flows`, `src/rakshak/nids/` host rules and
  pcap flows, `src/rakshak/remediation.py`, `deploy/pi/service/nids_pipeline.py`, `docs/NIDS_EDGE.md`, `research_notes/`,
  `reports/`) is **not part of v0**. The uncommitted SMS-path edits in `deploy/pi/service/edge_pipeline.py` only add
  remediation output fields; the decision logic in that file is unchanged in the diff. The uncommitted service edits
  (`rakshak_edge_service.py`, `index.html`) add the network endpoints and the page section; the SMS decision and
  explainer path in v0 is the committed one.
- The NIDS INT8 artifact is committed (`deploy/pi/bench/registry.toml`) and is part of v0 as an artifact. Its network
  serving path is not.

Source files that decide SMS risk and explainer checks, at `ac45aea` (`git show ac45aea:<path> | sha256sum`):

| File | sha256 | bytes |
|---|---|---|
| `src/rakshak/rules.py` | `9bde4ae25a44e5721daa4a1bd1c97304376d75765e3bed5d952dc5e845078e5a` | 20605 |
| `src/rakshak/canonicalize.py` | `892a89ad7619181e9d41f609bf46df52dc76622441b3c789ffb2c95228e783cb` | 10345 |
| `src/rakshak/evidence.py` | `df8300c6f9ee1d0392640df3cfcc674d60fdf2e4bca95bf1334f9b90d85302f2` | 8886 |
| `src/rakshak/verifier.py` | `f10890b50d981a5b660d48620400bb6b8075d0b224a1559d1a3f2843e1c4558a` | 33450 |

The four files are byte-identical in the working tree (`git diff ac45aea -- <path>` is empty). `src/rakshak/nids/evidence.py`
is a different file and is modified (uncommitted), so it is not in this table.

### Shipped artifacts

| Role | File (as used) | sha256 | bytes | Notes |
|---|---|---|---|---|
| SMS detector (shipped) | `models/sms_student/best_student.onnx` on the Pi; source `kaggle/out/sms_student/output/best_student.onnx` | `5341e14ca02a523bd746a71dd67903439e9e6c6f6fc3193fa1c6f786953f2466` | 388127 | Dual-view char-CNN, base size, Kaggle job `sms_student`, seed 1, `dual-base/dual/s1`. Inputs `raw` int64[256], `canon` int64[256], `side` float32[16]; output `logit`; probability = sigmoid(logit). **Threshold 0.35127517580986023** (campaign calibration max-F1, `kaggle/out/sms_student/output/metrics.json`, `splits.campaign['dual-base/dual/s1']`). |
| SMS detector, not shipped | `kaggle/out/sms_student-kd/output/best_student.onnx` | `684e5feedc584512bc7226f4a4bfc59413a007e1e24b35e9d7728ed26bd58fcd` | not measured | KD variant, registered but not the service default. Not v0. |
| URL detector (not in the service) | `kaggle/out/url_student/output/best_url_student.onnx` | `1bbd2c57cb541a7cbc287887ef51976d54240a353655827a511c63ec56213fc7` | 2741232 | Char-CNN `charcnn-wide`, trained on the PhreshPhish train split, seed 0 (`kaggle/jobs/url_student.py` export block). Inputs `raw` int64[200], `side` float32[29]; output `logit`. No threshold is registered for it. |
| URL side scaler | `kaggle/out/url_student/output/url_side_scaler.json` | `c6b3fc968b03339a33052fe814a9bb851d3035cb302b06458955c2a5e5b6501e` | 2083 | Mean and std over 29 columns, clip 10, NaN as 0. |
| NIDS INT8 student (shipped to the network path) | `models/nids/student_cse_cic_ids2018_int8.onnx`; `kaggle/out/nids_baselines/output/student_cse_cic_ids2018_int8.onnx` | `d3dfdcb16dcaa2780cd39c56d4c156c193dc2c67caeb4e443fc0447e2f9b8d14` | 6729 | MLP, 19 live features, 8 families, onnxruntime dynamic INT8. FP32 twin `0cb680ad904e2c37855dc58cf110e916a4169a35cf1fefaab70cbef2dc5b995c`. Thresholds in `kaggle/out/nids_baselines/output/results.json`. |
| Explainer LLM | `models/llama32_1b_spinquant/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte` | `8715cdba9e91f6bede00cc5f2d6b12397b95225da5630c4972a8da03001cda3b` | 1135951488 | Llama 3.2 1B, SpinQuant INT4, ExecuTorch 1.5.1. HF `executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET`, revision `911221e0f49d3354cbd4697a9c09feee25cadec7`. Tokenizer `tokenizer.model` `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55` (2183982 bytes). |

The `.pte` sha is the value in `deploy/pi/bench/registry.toml` (`llama32_1b_spinquant`), the same as the brief. The Pi
copies were verified 2026-11-06 (registry note).

Data tables the v0 job reads (`data/processed/`, the Kaggle dataset `maadhavvh/rakshak-data`): `sms.parquet`
`430ded382b22ea1c3c14469c80ff9b34a3214ab82257e89be20d67439053e135`; `urls.parquet`
`ff389178dd752a0240268e550e7c892164392d6c180372d537f6fc299e16efa2`; `urls_phreshphish.parquet`
`af9441871e9b542aad03e0c94dd702485a0ab41a786d5f0b0547836c5e322844`; `external/external_groundlm.parquet`
`979aaca4c6a922912d223515ec6a3373276a7f5308aaa8d72e17b9f092850f73`; `external/external_smishx.parquet`
`872d08e8885e101b33ac83732ec776e36d8cf183a669733baaf15165d3ec7780`.

### Service settings at `ac45aea`

| Setting | v0 value | Where |
|---|---|---|
| Detector-only mode | `quiet` (no explainer call; flags with no rule evidence are LOW / IGNORE) | `deploy/pi/service/edge_decision.py` `DEFAULT_DETECTOR_ONLY_MODE`; `--detector-only-mode` |
| Hinglish mode | `template` (verifier Hinglish templates, no LLM) | `deploy/pi/service/edge_pipeline.py` `DEFAULT_HINGLISH_MODE`; `--hinglish-mode` |
| Threads | 3 | `deploy/pi/service/rakshak_edge_service.py` `--threads` default |
| Explainer gate | G1 (no LLM at LOW), G2 (attribution guard), Fix A (describe prompt), Fix B (detector-only path), Fix C (cleanup), Fix D (one call per evidence type) | `src/rakshak/verifier.py`, `deploy/pi/service/edge_pipeline.py` |
| Thermal gate | explainer refused (503 `cooling`) at SoC >= 75 C; `rakshak_thermald` frequency cap | `DEFAULT_TEMP_LIMIT` |
| Offline | service in its own network namespace, `lo` only, Unix socket relay | `deploy/pi/service/start_service.sh` |

Verified against `git show ac45aea:<path>`.

## Benchmark suite v0

Every v0 cell below is measured by the Kaggle job `baseline_v0` on the shipped ONNX artifacts (sha256 checked in the job).
Job: `maadhavvh/rakshak-baseline-v0`, version 2, commit `ac45aea`, tree `2f209d389731+dirty` (the bundled working tree; the
pinned source files are byte-identical to `ac45aea`), status ok, 25 units, 0 failures, 1 533 s. Results:
`kaggle/out/baseline_v0/output/metrics.json` (copied from the kernel output). Thresholds are the shipped
0.35127517580986023 for SMS; URL has no registered threshold, so threshold-free metrics and the calibration 1 percent
FPR threshold are used. CIs: campaign-clustered bootstrap (1000 resamples) for SMS and url_domain; Wilson for counts;
the PhreshPhish table reports mean and SD over benchmark subsets, as the protocol does.

Protocol key: **same** = our protocol equals the target's; **partial** = same metric and family, split or denominator
differs; **different** = different metric, model, or protocol; **ours-only** = no published target.

### SMS (dual-view student, shipped ONNX)

| Benchmark | Split / protocol | v0 metric (CI, denominator) | Published target (source) | Protocol match | Provenance | Not v0 (stored, different model) |
|---|---|---|---|---|---|---|
| UCI SMS Spam, legacy random | holdout (random stratified, `legacy_uci_sms_spam`), 1 032 rows, 128 scam | F1 0.9685 [0.944, 0.988]; PR-AUC 0.9887 [0.973, 0.999]; acc 0.9922 | F1 0.9928 (BERT, 90/10 random, arXiv 2206.02443, UNVERIFIED primary); acc 0.9984 F1 0.9984 (RoBERTa, secondary) | partial (random split; target split not described, no dedup) | `metrics.json` `sms.legacy_uci_sms_spam:holdout` | holdout F1 at its own cal threshold 0.9688 (`kaggle/out/sms_student/output/metrics.json`) |
| UCI, 5 random folds | folds 0-4, 1 031 to 1 032 rows each | F1 mean 0.9544 (SD 0.0028): 0.9569, 0.9575, 0.9506, 0.9535, 0.9535 | as above | partial | `metrics.json` `legacy_uci_sms_spam:fold_0..4` | not stored at shipped threshold |
| Mishra & Soni (Mendeley) | holdout (`legacy_mishra_soni`), 189 rows, 84 scam | F1 0.9820 [0.958, 1.000]; acc 0.9841; PR-AUC 0.9978 | acc 97.9% (DSmishSMS, NCA 2021, secondary); 97.40% (SN CS 2022) | different (target is accuracy on an undescribed split; ours F1 and accuracy on a random holdout) | `metrics.json` `legacy_mishra_soni:holdout` | holdout F1 0.9701 at own threshold |
| Mishra & Soni, 5 folds | folds 0-4, 188 to 189 rows | F1 mean 0.9833 (SD 0.0065) | as above | different | `metrics.json` `legacy_mishra_soni:fold_0..4` | n/a |
| Campaign-disjoint test | `campaign` (70/10/20 by campaign_id), 22 145 rows, 11 193 scam | F1 0.9813 [0.9794, 0.9830]; PR-AUC 0.9977 [0.9973, 0.9981]; P 0.9824, R 0.9803, FPR 0.0180 | none (teacher mmBERT F1 0.987 [0.985, 0.989], same family, internal) | ours-only | `metrics.json` `sms.campaign` | F1 0.9749 at stored threshold (campaign-run model, `sms_student` metrics.json) |
| ScamShield source holdout | `source_holdout:holdout_scamshield`, 75 973 rows, 29 029 scam | F1 0.9805 [0.9791, 0.9819]; PR-AUC 0.9981; FPR 0.0180 | none | ours-only | `metrics.json` `sms.source_holdout:holdout_scamshield` | PR-AUC 0.6089 (stored model) |
| UCI source holdout | `source_holdout:holdout_uci_sms_spam`, 5 159 rows, 642 scam | F1 0.9544; PR-AUC 0.9884; recall 0.9611, FPR 0.0075 | none | ours-only | `metrics.json` `sms.source_holdout:holdout_uci_sms_spam` | PR-AUC 0.9075 (stored) |
| Mishra source holdout | `source_holdout:holdout_mishra_soni_smsphishing`, 941 rows, 417 scam | F1 0.9833; PR-AUC 0.9980; FPR 0.0191 | none | ours-only | `metrics.json` `sms.source_holdout:holdout_mishra_soni_smsphishing` | PR-AUC 0.9751 (stored) |
| Temporal | `temporal`, 11 029 rows, **77 scam** | F1 0.4298 [0.394, 0.473]; **precision 0.276**, recall 0.974, FPR 0.0180; PR-AUC 0.9321 [0.879, 0.974] | none | ours-only | `metrics.json` `sms.temporal` | F1 0.903 at its own threshold (stored, PR-AUC 0.890). The shipped threshold gives 1.8% FPR on this split: 2 of every 3 flags are benign. Known defect. |
| Hinglish (lang_shift) | `lang_shift`, 1 677 rows, 357 scam | F1 0.9860 [0.976, 0.994]; PR-AUC 0.9984; FPR 0.0045 | HinSpam F1 0.949 (HF card, self-reported, UNVERIFIED) | different (self-reported, no protocol) | `metrics.json` `sms.lang_shift` | stored lang_shift F1 0.356 (PR-AUC 0.497, n 1 600): a different model. See note 1 |
| SMS, ScamShield official | `legacy_scamshield:official`, 7 637 rows, 2 900 scam | F1 0.9785 [0.975, 0.982]; PR-AUC 0.9983 | none | ours-only | `metrics.json` `sms.legacy_scamshield:official` | n/a |
| GroundLM policy negatives, student | 1 294 policy negatives; **1 133 scoreable** (161 hash-only, not scoreable); student score >= threshold | **471 / 1 133 = 41.57%** [38.73, 44.46] Wilson | 0.4% global FPR (270M XLM-R, 1 294 negatives, ACL 2026.groundlm-1.79; abstract verified) | partial (same negatives, scoreable subset only; different model) | `groundlm_smishx.groundlm` (ONNX) | the laptop run gives the same 471 (`docs/GROUNDLM_EDGE_FPR.md`) |
| GroundLM policy negatives, rules | rule evidence (`rakshak.rules`, canonical text) | 12 / 1 133 = 1.06% [0.61, 1.84] | as above | partial | `groundlm_smishx.groundlm.rule_evidence_any` | same 12 in `docs/GROUNDLM_EDGE_FPR.md` |
| GroundLM, system-level visible warning (quiet) | full edge decision, quiet mode, MEDIUM and above | 2 / 1 133 = 0.18% [0.05, 0.64] (quiet visible); 1 / 1 133 = 0.09% at MEDIUM+ | as above | partial | **not in the job** (needs `deploy/pi/service` in the bundle). From `docs/GROUNDLM_EDGE_FPR.md` (laptop, run on the working tree; SMS decision lines unchanged, see Code) | verify mode 41.66% (not the shipped setting) |
| SmishX, smishing recall | eval_ok smishing, 226 rows | student 157 / 226 = **69.47%** [63.18, 75.11]; rules-only 58 / 226 = 25.66% | none for SmishX itself (its paper reports 98.8% accuracy, 3-class, different protocol) | ours-only | `groundlm_smishx.smishx` | quiet visible recall 23.45% (`docs/GROUNDLM_EDGE_FPR.md`, laptop) |
| SmishX, legitimate FPR | eval_ok legitimate, 606 rows | student 55 / 606 = **9.08%** [7.04, 11.63]; rules-only 30 / 606 = 4.95% | none | ours-only | `groundlm_smishx.smishx` | quiet visible FPR not in job |

Note 1 (stored vs v0). The stored `sms_student` metrics (`kaggle/out/sms_student/output/metrics.json`) hold one model per
split file (each split run trains its own model), and the stored campaign F1 at the shipped threshold is 0.9749 (PR-AUC
0.9967). The v0 ONNX gives 0.9813 on the same campaign test rows, so the shipped file is not the campaign-run model. The
export log does not name the source run. `docs/HANDOFF.md` says the shipped checkpoint is the ScamShield-holdout run for the
teacher. Our ONNX-only lang_shift result does not depend on the rule features (PR-AUC 0.9984 with rules, 0.9982 with zeroed
side features; `scratchpad` check). So the per-split stored numbers are not v0 and must not be cited as v0. Open item:
identify the training run of the shipped ONNX. The v0 numbers above are measured on the shipped file, not on the stored models.

Note 2. The stored lang_shift test has 1 600 rows; the current split file gives 1 677 (357 scam). Data pins match on Kaggle,
so the difference is in the stored run's split handling, not the data. Recorded, not resolved.

## Contamination check (2026-11-07)

**Status: done (2026-11-07).** Run identified from code and logs; row overlap measured on instabox (table below). Weight match not run (export parity 3.8e-6 in `run.log` line 1112 stands in). Caveat: the repo campaign split gives 76 591 train / 11 072 cal, the run logged 77 609 / 11 073, so about 1 019 trained rows are unrecoverable and unchecked.

**Identified run: `sms_student` campaign-run `dual-base/dual/s1`, trained on the campaign split's train rows only.** The export keeps one model, from the primary split.

Code, `kaggle/jobs/sms_student.py` (the same export logic appears in commits 9d1ddec, 6334b12 and cd89e46):
- `PRIMARY_SPLIT = "campaign"` (line 111 at cd89e46).
- `keep = is_primary and seed == args.seeds[0]` (line 340). Only the primary run's first seed keeps its model in memory.
- `fitted[(spec.label, spec.variant)] = {...}` runs only under `keep` (line 353). Other splits are scored and discarded.
- Export at line 391 onward: `entry = fitted[(label, variant)]`, then `export_onnx(entry["model"], ...)`, then `save_checkpoint(..., extra={"run": primary_name, ...})`. `best_key` is chosen by mean cal PR-AUC across runs (dual-base 0.9875, charcnn-base 0.9873).

Logs, `kaggle/out/sms_student/output/`:
- `run.log` line 9: `run campaign: sizes {'train': 77609, 'cal': 11073, 'test': 22145}`. Campaign is the first run.
- `run.log` line 64: `dual-base/dual/s1: F1 0.9749 PR-AUC 0.9967`. This matches the stored campaign value in the table above.
- `run.log` line 1112: `exported dual-base/dual: {... 'bytes': 388127, 'opset': 17} parity {'n': 64, 'max_abs_logit_diff': 3.8e-06}`. Parity is checked against the in-memory campaign model, not a per-split model.
- `run.log` line 1062: the `holdout_scamshield` run gives `dual-base/dual/s1: F1 0.7511 PR-AUC 0.6089`. That run is not kept, so it cannot be the shipped model.
- File hashes: `best_student.onnx` = `5341e14ca02a523bd746a71dd67903439e9e6c6f6fc3193fa1c6f786953f2466` (matches the shipped hash). `best_student.pt` = `1c0948e4548352f86fe670479795658e59619fd5084a116e766d8375dd3f7cfe`.
- `results.json`: `git_sha` 8005d189, `dirty: true`, `tree_sha256` 3645d5e2. Commit 8005d18 does not contain `kaggle/jobs/sms_student.py`, so the exact code that ran is not reproducible from git.

Consequences:
- `docs/HANDOFF.md` line 440 ("the shipped checkpoint is the scamshield-holdout run (last kept by cal PR-AUC)") is contradicted by the code and logs. Only the primary (campaign) run is kept.
- The shipped model's split-level numbers (lang_shift F1 0.986, ScamShield-holdout PR-AUC 0.998) are scores of a model trained on campaign train only. The per-split models score 0.356 and 0.609 on the same test sets (`run.log` lines 120 and 1062). Overlap (table below) confirms ScamShield/UCI/Mishra source-holdout and legacy splits are 70-78% in campaign train (IN-SAMPLE). lang_shift has 0 overlap: its 0.986 is valid for the shipped model but is not a language-shift result, because campaign train includes other Hinglish campaigns while the lang_shift protocol model trains on English only.
- The 0.9813 campaign figure in the table does not match the logged 0.9749 for the same run. The cause is not resolved here (threshold or mask differs).

Measured 2026-11-07 on instabox (`overlap_check.py`). Reference = campaign TRAIN + CAL rows rebuilt from the repo split file (76 591 + 11 072). Overlap by row id, and by normalised-text sha1 (lowercase, whitespace collapsed, digits to 0). Verdict on the larger of the two: VALID < 0.5 %, PARTIAL 0.5 to 20 %, IN-SAMPLE > 20 %.

| split | test rows | overlap by id | overlap by text | verdict |
|---|---|---|---|---|
| campaign test | 22 145 | 0 (0.00 %) | 1 (0.00 %) | VALID |
| lang_shift | 1 677 | 0 (0.00 %) | 0 (0.00 %) | VALID |
| temporal | 11 029 | 61 (0.55 %) | 62 (0.56 %) | PARTIAL |
| source_holdout scamshield | 75 973 | 59 500 (78.32 %) | 59 501 (78.32 %) | IN-SAMPLE |
| source_holdout uci | 5 159 | 3 743 (72.55 %) | 3 743 (72.55 %) | IN-SAMPLE |
| source_holdout mishra | 941 | 692 (73.54 %) | 692 (73.54 %) | IN-SAMPLE |
| legacy_scamshield official | 7 637 | 5 972 (78.20 %) | 5 972 (78.20 %) | IN-SAMPLE |
| legacy_uci holdout | 1 032 | 744 (72.09 %) | 744 (72.09 %) | IN-SAMPLE |
| legacy_uci fold_0 to fold_4 | 1 032 / 1 032 / 1 032 / 1 032 / 1 031 | 736 / 762 / 766 / 730 / 749 (70.7 to 74.2 %) | same as id | IN-SAMPLE |
| legacy_mishra holdout | 189 | 133 (70.37 %) | 133 (70.37 %) | IN-SAMPLE |
| legacy_mishra fold_0 to fold_4 | 189 / 188 / 188 / 188 / 188 | 138 / 137 / 135 / 146 / 136 (71.8 to 77.7 %) | same as id | IN-SAMPLE |
| GroundLM policy negatives (scoreable) | 1 133 | n/a (ids not shared) | 64 (5.65 %) | PARTIAL |
| SmishX eval_ok (text available) | 1 107 | n/a (ids not shared) | 64 (5.78 %) | PARTIAL |

Quote as v0 only the splits marked VALID.

**"About 4.7 generated tokens per request" (`docs/BASELINE_PI.md` lines 99, 121, 204).** Not a miscount; a denominator choice. B1 English (40 requests) has 188 generated tokens in total. The 40 includes 10 template-mode rows that make no LLM call. 188/40 = 4.70 (per request). 188/31 = 6.06 (per LLM call, 31 calls). 188/30 = 6.27 (evidence rows). The soak figure is 940/200 = 4.7, with 105 LLM rows out of 200. The "per request" column divides by all requests, so template-skipped rows dilute it. The 6.27 evidence figure reuses the same 188 total over 30 rows, which assumes all tokens fall in evidence rows. Not verified: whether the 4 unterminated calls contribute tokens. Source: `deploy/pi/bench/rbench/runners/pte_llm.py:180` (`num_generated_tokens`) and `deploy/pi/bench/rbench/worker.py:492-493` (aggregation). Numbers were not changed.

### SMS robustness (shipped ONNX, shipped threshold, campaign test, canonicaliser on)

| Benchmark | Protocol | v0 metric (denominator) | Published target (source) | Protocol match | Provenance | Not v0 |
|---|---|---|---|---|---|---|
| Perturbation harness, 9 families x 3 seeds | `rakshak.eval.robustness`, 11 193 scam, 10 952 benign (benign insert also on benign) | clean recall 0.9803 (clean FPR 0.0180); mean recall under attack 0.9803; mean ASR 0.0023; worst: char_spacing 0.9770 | Johnny (arXiv 2609.01171, preprint): BERT under homoglyph 41.3% accuracy (no AT), 99.5% (full AT), 79.7% (minimal AT); accuracy, not ASR | different (accuracy vs recall at a fixed threshold; their families and data) | `metrics.json` `robustness.sms_campaign_canon_on` | `.pt` campaign-run model: clean recall 0.9736, mean ASR 0.0024 |
| Homoglyph (`homoglyph_swap`) | same | recall under attack 0.9808; ASR 0.0030 | 41.3% (Johnny, BERT no AT) | different | same | stored 0.9748 |
| Invisible (`zero_width_insert`) | same | recall 0.9836; ASR 0.0011 | 94.9% (Johnny, BERT no AT) | different | same | stored 0.9784 |
| Canonicaliser off | raw-only ONNX | **MISSING**: no raw-only ONNX is pinned | - | - | - | - |

### URL (char-CNN student ONNX, 29 side features; no registered threshold)

| Benchmark | Protocol | v0 metric (CI or SD, denominator) | Published target (source) | Protocol match | Provenance | Not v0 |
|---|---|---|---|---|---|---|
| PhreshPhish URL-only, 0.05% | official split, test; mean (SD) over **586** subsets | AP 0.4843 (0.0649); P@R=0.90 0.0874 (0.0670) | GTE URL+HTML: AP 0.6050, P@R 0.0891 (arXiv 2507.10854 v2, Table 2) | different (no URL-only target exists) | `metrics.json` `url.phreshphish_protocol.per_rate` | same numbers as stored charcnn-wide/s0 (`url_student` report) |
| PhreshPhish URL-only, 0.1% | 296 subsets | AP 0.5991 (0.0435); P@R 0.1556 (0.0802) | GTE URL+HTML: 0.6879 / 0.1563 | different | as above | stored 0.5991 |
| PhreshPhish URL-only, 0.5% | 59 subsets | AP 0.8090 (0.0148); P@R 0.4245 (0.0858) | GTE URL+HTML: 0.8354 / 0.3736 | different | as above | stored 0.8090 |
| PhreshPhish URL-only, 1% | 29 subsets | AP 0.8647 (0.0085); P@R 0.5971 (0.0566) | GTE URL+HTML: 0.8762 / 0.5426 | different | as above | stored 0.8647 |
| PhreshPhish URL-only, 5% | 5 subsets | AP 0.9413 (0.0007); P@R 0.8798 (0.0027) | GTE URL+HTML: 0.9382 / 0.8576 | different | as above | stored 0.9413 |
| PhreshPhish 24% full benchmark | 118 738 rows (28 476 phish) | AP 0.9804; P@R 0.9785 | GTE URL+HTML: 0.9779 / 0.9746 | different | `url.phreshphish_protocol.full_benchmark_24pct` | stored 0.9804 |
| PhreshPhish full test | 168 060 rows (76 800 phish) | AP 0.9914; P@R 0.9928 | n/a | ours-only | `full_test_split` | n/a |
| url_domain test (out of PhreshPhish training) | 30 432 rows (22 957 scam); PhreshPhish-trained student | PR-AUC 0.9746; ROC-AUC 0.911; P@R=0.90 0.874; recall at cal FPR 1% = **0.779** (test FPR 1.12%), bootstrap [0.480, 0.900] | none | ours-only | `metrics.json` `url_domain.test` | url_domain-trained model: F1 0.999, recall 1.000 at 1% FPR (`url_student` report) - a different model |
| URL FPR 0.01% (URLTran) | not measured | **MISSING** | URLTran TPR 86.80% at FPR 0.01% (MILCOM 2021, self-reported, not replicated) | - | - | - |

### Explainer, device, NIDS

The explainer, device and NIDS rows are in the sections below (explainer), in `docs/BASELINE_PI.md` (device, measured
there), and in the NIDS section. NIDS: the shipped INT8 network model's T0 and LODO rows come from the NIDS job, not from
`baseline_v0`; the LODO student row is MISSING.



### Device (Raspberry Pi 5, no cooler, weak PSU)

Measured on the Pi: **measured there, see `docs/BASELINE_PI.md`** (score parity Pi vs laptop, per-request and per-LLM-call
latency, Llama load, decode tok/s, TTFT, RSS, J per request, max temperature, throttle). The Pi has no active cooler and
a USB-limited PSU, so those numbers are for this setup only. The device rows are not repeated here.

### Explainer (Llama 3.2 1B `.pte`, the v0 explainer)

| Row | v0 value | Denominator | Source | Notes |
|---|---|---|---|---|
| Raw point pass (model sentence passes the verifier) | **26 of 33 (79%)** | 33 points, 30 evidence rows (29 called) | `docs/EDGE_PIPELINE.md` section 3c (Fix D, Pi, 3 threads) | Fix C run: 30 of 45 (67%), superseded |
| Slot verifier point pass, evidence-bearing rows | 14 of 40 (35%) FP16 slot; 16 of 58 (27.6%) points | 40 rows; 58 points | `docs/LLAMA_PI_RESULTS.md`, `kaggle/out/llama_ela_eval` | **not v0**: FP16 `unsloth` model, not the `.pte`. The 0.91 in `metrics.json` is not used (260 of 300 clean rows carry no evidence). Use 35%. |
| LLM share of final output | 22 of 30 advisories from the LLM (73%); 23 of 30 with at least one LLM point (77%) | 30 evidence rows | section 3c | template 7, partial 1 |
| Unterminated rate | 4 of 45 calls (8.9%) | 45 calls | `docs/HANDOFF.md` residual defects (Fix A/B/C run) | Fix D run not reported; the template fallback catches these |
| G2 rejections | 1 distinct sentence (row 25, "is likely an investment scam") | 40 rows | `docs/EDGE_PIPELINE.md` section 3b | Fix C run |
| Final output passes the contract | 30 of 30 (evidence), 40 of 40 (all) | by construction (template fallback) | section 3c | |
| Prompt-injection harmful flips, slot arm | **MISSING for v0** (FP16 slot 0 of 150; freeform 35 of 150) | 150 attacks | `docs/HANDOFF.md` (ELA v2, unsloth FP16) | **not v0**: no injection run on the `.pte`; the measurement needs the Pi |
| Hinglish explainer | template by default; the LLM Hinglish prompt passes 8 of 37 raw points (22%) | 37 points | `docs/EDGE_PIPELINE.md` section 5 | not the default; the LLM mode is measured only |

### NIDS (INT8 cse_cic_ids2018 student, the shipped network model)

| Row | v0 value | Source | Notes |
|---|---|---|---|
| T0 within dataset (cse_cic_ids2018, held-out random split), F1 at max-F1 cal threshold | 0.796 [0.777, 0.812] | `kaggle/out/nids_baselines/output/report.md`, section "Student" | MCC 0.777; ROC-AUC 0.877; PR-AUC 0.794; FPR at max-F1 0.005; recall at cal FPR 0.1%: 0.622; at 1%: 0.679 |
| T5 leave-one-dataset-out, student | **MISSING** | the NIDS job has no student LODO row; the LODO table covers LightGBM, isolation forest, autoencoder and hybrids only | Needs the NIDS data (not in `rakshak-data`); out of this job's scope |
| Cross-dataset (T1 pairs, student) | cse to unsw F1 0.297; cse to bot_iot 0.332; cse to ton_iot 0.718 | same section | Near chance on the unsw and bot_iot pairs (see `docs/NIDS_EDGE.md`); not a claim |

Provenance for NIDS: the job ran at git `9d1ddeca` (dirty; `kaggle/out/nids_baselines/output/results.json`). That commit
differs from `ac45aea` in 117 files, including `rules.py` and `verifier.py`. The INT8 file is sha256 `d3dfdc...` (as above),
so these are the artifact's numbers, but the evaluation code is not the v0 code. Not re-run at `ac45aea`: the NIDS
Kaggle job needs its data, which `rakshak-data` does not hold. Treat as "v0 artifact, older evaluation code".



## Gaps vs published targets

No row is **same** protocol. The closest is UCI (partial: random split, the target's split is not described). Gap =
ours minus target, in absolute terms (points of F1, AP or rate; for FPR a negative gap is better).

**Protocol-matched rows:** none.

**Partial (same metric and family, different split or denominator), by size of gap:**

| Benchmark | v0 | Target | Gap | Why partial |
|---|---|---|---|---|
| GroundLM policy negatives, FPR (student) | 41.57% (471/1 133) | 0.4% (270M XLM-R, 1 294) | +41.2 pts (worse) | target is a different, larger system; ours is the shipped student on 1 133 scoreable rows |
| UCI SMS Spam holdout, F1 | 0.9685 | 0.9928 | -0.024 | random split, target split and dedup not described |
| UCI SMS Spam, 5 random folds, F1 mean | 0.9544 | 0.9928 | -0.038 | as above |

**Different (metric, model or protocol differs; informational):**

| Benchmark | v0 | Target | Gap | Note |
|---|---|---|---|---|
| PhreshPhish AP at 0.05% | 0.4843 | 0.6050 (GTE, URL+HTML) | -0.121 | URL-only vs URL+HTML; no URL-only target exists |
| PhreshPhish AP at 0.1% | 0.5991 | 0.6879 | -0.089 | as above |
| PhreshPhish AP at 0.5% | 0.8090 | 0.8354 | -0.026 | as above |
| PhreshPhish AP at 1% | 0.8647 | 0.8762 | -0.012 | as above |
| PhreshPhish AP at 5% | 0.9413 | 0.9382 | +0.003 | as above |
| PhreshPhish P@R=0.90 at 0.05% | 0.0874 | 0.0891 | -0.002 | as above |
| PhreshPhish P@R=0.90 at 0.1% | 0.1556 | 0.1563 | -0.001 | as above |
| PhreshPhish P@R=0.90 at 0.5% | 0.4245 | 0.3736 | +0.051 | as above |
| PhreshPhish P@R=0.90 at 1% | 0.5971 | 0.5426 | +0.055 | as above |
| PhreshPhish P@R=0.90 at 5% | 0.8798 | 0.8576 | +0.022 | as above |
| PhreshPhish 24% full benchmark, AP / P@R | 0.9804 / 0.9785 | 0.9779 / 0.9746 | +0.003 / +0.004 | as above |
| Hinglish, F1 | 0.9860 | 0.949 (HinSpam, self-reported) | +0.037 | target UNVERIFIED, no protocol |
| Mishra & Soni holdout, accuracy | 0.9841 | 0.979 (DSmishSMS, secondary) | +0.005 | target is accuracy on an undescribed split |
| Johnny, homoglyph | recall at fixed threshold 0.9808 (ASR 0.0030) | accuracy 41.3% (BERT, no AT) | not comparable | different metric (accuracy vs recall) and different data |
| URLTran, TPR at FPR 0.01% | **MISSING** (not measured) | 86.80% | n/a | the job does not reach FPR 0.01% |

**Open or missing for the gap table:** canonicaliser-off robustness (no raw-only ONNX); the explainer on the `.pte`
(prompt-injection flips, slot verifier pass: only FP16 numbers exist); network T5 leave-one-dataset-out for the student
(needs NIDS data); GroundLM and SmishX system-level visible-warning rates (need the decision code in the bundle; laptop
numbers only); GroundLM student-level on all 1 294 (161 texts are not available).

## Status and open items

- **Frozen:** code `ac45aea`, the artifacts and pins above. Future work changes these numbers only by a new version.
- **Job:** `baseline_v0` ran twice. Run 1 (version 1) failed on an SMS split-pattern bug (the URL split files were loaded
  into the SMS runs); run 2 (version 2) completed. Both launches are in the Kaggle history.
- **Checks passed inside the job:** artifact sha256 (ONNX SMS, ONNX URL, scaler), data sha256 (all five tables), source
  sha256 (four frozen files), side-feature parity against the runtime `url_features` (max diff 0.0 on 200 rows), and the
  GroundLM and SmishX counts reproduce the laptop numbers exactly (471/1 133, 12/1 133, 157/226, 55/606).
- **Discordances, not resolved:** (1) the v0 ONNX per-split numbers disagree with the stored `sms_student` metrics on
  lang_shift, ScamShield holdout, temporal and the campaign run (note 1); the stored numbers are per-split models. (2) The
  temporal split collapses at the shipped threshold (precision 0.28 on 77 scam; FPR 1.8%). (3) lang_shift size differs
  (note 2).
- **Tests:** `tests/test_baseline_v0.py` 14 passed; with `tests/test_kaggle_build_kernel.py`, `test_kaggle_resilience.py`,
  `test_kaggle_run_name.py`, `test_phreshphish_extract.py`, `test_colab_launcher.py`, `test_fusion_v1_job.py`: 116 passed.
  `.venv/bin/ruff check .` clean.
- **Device:** measured on the Pi in `docs/BASELINE_PI.md` by a separate agent.
- **Repo:** nothing committed, pushed, stashed, reset or checked out. No post outside our repos.

