# SMS student retrain: Indian legit exposure (pre-registered plan)

Written 2026-11-07, before any training for this plan. Kaggle only (T4); no Modal. Nothing here is tuned on
GroundLM, SmishX, SMS-Shield or any test split. Once Maadhav approves this file, the gates, arms, threshold rule
and data rules below are frozen. A change after that is logged as a dated amendment, before any result it could affect.

## 1. Why, and what is already known

- Baseline v0 (shipped ONNX `5341e14c…`, threshold 0.35128) flags **471 / 1 133 = 41.6 %** of GroundLM scoreable
  policy negatives and catches **157 / 226 = 69.5 %** of SmishX smishing (`docs/BASELINE.md`).
- The contamination check (`docs/BASELINE.md`, "Contamination check (2026-11-07)") identifies v0 as the
  `sms_student` **campaign run** `dual-base/dual/s1`. Valid v0 comparisons: **campaign test** (F1 0.9813) and
  **lang_shift** (0 overlap, but not a language-shift test for this model). GroundLM and SmishX have about 5.7 % text
  overlap with v0 training (PARTIAL). Temporal is PARTIAL (0.56 %). Source-holdout and legacy splits are IN-SAMPLE
  and are not used as v0 comparisons.
- **Prior evidence (`sms_groupneg`, Kaggle 2026-11-06, fetched today):** the same recipe plus the GroundLM negatives
  as label-0 rows, cross-fitted over 2 template-disjoint folds (653 clusters), at its campaign-cal max-F1 threshold
  0.2177: **11 / 1 133 = 0.97 %** held-out-fold FPR (control 459 / 1 133 = 40.5 %). The full model gets SmishX
  smishing 199 / 226 and campaign F1 0.974. This is an **in-distribution** result: same sources, template-held-out.
  It shows that exposure fixes the false-positive problem. It cannot ship, because the GroundLM pool includes
  vinit9638/flatmars text with no licence, and it is not a cross-source result. We report it as a reference row only.
- **Killed before (do not repeat):** synthetic deterministic transactional negatives, label-0 only
  (`docs/TRANSACTIONAL_HARDNEG*.md`). GroundLM student FPR fell to 9-17 %, but SmishX recall dropped on all seeds,
  lang_shift fell by 0.058, and the fused gates failed. Template-family calibration was too sparse. Source balance,
  group-DRO, clean KD and multilingual inputs did not move GroundLM FPR (35-42 %).
- GroundLM (ACL 2026.groundlm-1.79) reports 5 / 1 294 = 0.39 % at τ = 0.5. Its split is random with structural
  template overlap, its test pool is drawn partly from CloveAI, and its τ comes from a test-set sweep
  (OpenResearch review, 2026-11-07).

## 2. Data

### 2.1 Licence register (train use needs MIT / Apache-2.0 / CC BY)

| Source | Licence | Rows | Use in this plan |
|---|---|---|---|
| Our corpus `sms.parquet` | as in `docs/DATACARD.md` | 137 153 | train/cal/test as today (campaign split) |
| FinEE `Ranjit0034/finee-dataset`, synthetic | Apache-2.0 declared by the uploader; rights and consent for the underlying real messages unverified | 136 357 without an ICICI token; 64 670 unique; ≥2 533 near-dup clusters | **train** (legit, label 0), cluster-capped. The release has no real/synthetic flag (OpenResearch audit, 2026-11-07), so the ICICI-token filter is a best-effort proxy. Residual risk is recorded in §8 |
| FinEE, rows with an ICICI token | as above | 16 162 (heuristic; card says 2 419 are real) | **excluded**. The real rows cannot be isolated, so every ICICI-token row is dropped |
| junioralive Indian telecom CSV | MIT declared | 2 267 (1 522 ham / 745 spam) | **excluded**. 2 266 / 2 267 rows also occur in vinit9638 (no licence, and a GroundLM test source). It is not independent and would leak GroundLM-source text |
| IEEE DataPort "Spam SMS in Dravidian Languages" (10.21227/dcym-pd69) and "Spam SMS in Hindi Language" (10.21227/5y8x-n678) | CC BY 4.0 (verified on the record; donor consent not documented) | ham: about 7 322 Dravidian/English + 2 490 Hindi (paper counts) | **test-only, ham rows only**: the only real (crowd-collected) Indian negatives available. Spam labels are not scam gold (a Jio recharge reminder is labelled spam), so spam rows are unused. Reported, not gated. **Not available (amendment 2026-11-07, before launch): DataPort requires a paid subscription, so this set is skipped.** The job reports it only if the files appear |
| CloveAI `india-spam-sms` | MIT | 20 010; 435 near-dup clusters; 4 260 exact and 6 755 near dups of GroundLM negatives | **arm C only**, after removing every cluster that touches GroundLM |
| skyaara Indian Payment SMS | CC BY 4.0 | 172 (58 seeds + 114 variants) | **threshold validation** (legit) |
| Ridham115 audited synthetic | CC BY 4.0 | 1 580 (783 safe / 797 scam) | **threshold validation** (safe + scam) |
| SMS-Shield `eval_set.jsonl` | Apache-2.0 (repo) | 1 390 (488 safe), sender + body | **test-only** (new Indian synthetic test; sender not used) |
| GroundLM policy negatives / positives | per-row (paper Table 12) | 1 133 scoreable negatives | **test-only** in every new arm |
| SmishX | as today | 226 smishing / 606 legit eval_ok | **test-only** |
| IIITD, vinit9638, flatmars, Gandharv OTP | none / unresolved | – | **excluded** from every artifact |

Not yet on disk: junioralive, skyaara, Ridham115 data CSV, SMS-Shield eval. The Kaggle data-prep step fetches them,
dedups them and logs their counts (§2.3). Our corpus, the training sources and GroundLM have **no sender or DLT
field**; only 6 rows carry a DLT-like header in the text. The sender-feature arm is therefore **dropped**: nothing to
train it on. The SMS-Shield senders stay test-side metadata. A DLT-suffix ablation stays open future work and needs
data with sender IDs.

### 2.2 Dedup and leakage rules (applied before any split)

1. Normalise (lowercase, URL→token, digits→0, collapse whitespace). Drop exact and near (char-5-gram MinHash,
   Jaccard ≥ 0.8) duplicates of **any test-only row** (GroundLM 1 294 incl. hash-only by hash, SmishX, SMS-Shield,
   campaign test, lang_shift test, temporal test) from every train and validation pool. Remove the whole near-dup
   cluster, not only the matching row.
2. Cluster all training-side rows (ours + new) with the existing clusterer (`rakshak.data.dedup.assign_campaigns`,
   Jaccard ≥ 0.7 on `template_key`). Splits are group-disjoint by cluster (`assign_group_splits`).
3. Template flooding cap: from each new legit source, at most **5 rows per cluster**, and in total at most
   **min(12 000, 0.25 × our campaign-train negatives)** new legit rows. This is the same budget as the killed
   hard-negative arm, so the comparison is fair.
4. Validation and test pools never share a cluster with train.

**Amendment 2026-11-07 (approved by Maadhav before launch): two-level clustering of new sources.** At Jaccard 0.7
on normalised text, FinEE chains into 57 connected components (largest 5 105 rows). With those, the 5-per-cluster
cap would leave about 240 training rows. So two levels are used:
(a) **split groups** = 0.7 components. The FinEE hold-out for legit validation is drawn by whole split groups (15 %
of groups, seeded), and no split group appears in both train and validation;
(b) **cap groups** = 0.9 components, nested inside split groups. The 5-rows-per-group cap and the round-robin
sampling work on cap groups.
Test-only dedup (rule 1) is unchanged. The SMS-Shield eval is unavailable ("repository not found") and skipped.

### 2.3 Splits

- **Campaign split rebuilt** with the new rows added to train only. Campaign cal is unchanged, so early stopping
  and the cal side of the threshold rule match v0 (amendment 2026-11-07, before any code or launch). The campaign test rows stay the same
  22 145 rows (ids pinned), so the v0 comparison is like-for-like. lang_shift and temporal test rows also stay pinned.
- **Indian legit validation set (threshold only):** 15 % of FinEE clusters (held out of train), plus skyaara and
  Ridham115 safe. Ridham115 scam rows go into validation recall. Target ≥ 1 500 legit rows.
- Every count (rows in, dropped per rule, clusters, final sizes) goes to `prep_report.json` in the job output and
  into §7 of this file before results are read.

## 3. Arms (3 arms × 3 seeds = 9 runs)

Same model and recipe as v0 for every arm: dual char-CNN `dual-base`, raw and canon int64[256], side float32[16]
from the current `rules.py`, class-weighted BCE, AdamW 2e-3, weight decay 1e-4, batch 128, ≤ 15 epochs,
patience 3 on cal PR-AUC. Seeds 1, 2, 3. KD off.

- **A: v0 recipe + licensed legit.** Our campaign train + FinEE synthetic (capped). No CloveAI and no junioralive. This is
  the **cross-source** arm: no training row shares a source or a cluster with GroundLM.
- **B: A + minimal pairs.** For about 2 000 legit training templates, add a paired **scam** edit of the same
  template (label 1). Edit families: credential/OTP request, KYC-block urgency, fee/prize claim, a suspicious or
  shortened link, remote-access app, collect request. Edits come from `rules.py` evidence families. They are
  generated deterministically and stored with their template id. How B differs from the killed hard-negative arm:
  B pairs every legit template with a scam twin, so the model learns the edit rather than "bank-like means safe".
  The killed arm added label-0 notices only, and SmishX recall dropped.
- **C: A + CloveAI (GroundLM-stripped).** It tests whether same-family data adds more than A. GroundLM results
  for C are labelled **same-source (in-distribution adjacent)**, never cross-source, because GroundLM's test pool
  draws from CloveAI even after cluster stripping.

Reference rows, no new compute: v0, the `sms_groupneg` cross-fit (in-distribution, template-held-out), and
GroundLM's published XLM-R (in-distribution, random split).

## 4. Threshold rule (one per model, fixed before test scoring)

τ = the **smallest** threshold with FPR ≤ 1 % on the Indian legit validation set **and** FPR ≤ 2 % on campaign cal.
If no threshold meets both, use the campaign-cal max-F1 threshold and flag the run. GroundLM, SmishX, SMS-Shield and
every test split are scored only after τ is written to `thresholds.json`.

## 5. Pre-registered gates (an arm passes if every gate holds on the seed median and on at least 2 of 3 seeds)

| # | Gate | v0 reference | Pass condition |
|---|---|---|---|
| G1 | GroundLM FPR, 1 133 scoreable negatives | 471 (41.6 %) | ≤ 11 FP (≤ 0.97 %); Wilson CI reported |
| G2 | SmishX smishing recall, 226 | 157 (69.5 %) | ≥ 157 |
| G3 | Campaign-test F1, 22 145 | 0.9813 | ≥ 0.9713 |
| G4 | ONNX size | 388 KB | ≤ 10 MB |
| G5 | Pi latency (service eval, 3 threads) | p50 1.39 s / p95 2.93 s English 40-row | p50 and p95 within +10 % (step 4 only, winner only) |

Reported, not gated: SmishX legit FPR (v0 9.1 %), lang_shift F1 (v0 0.986; the killed arm lost 0.058 here, so a
drop > 0.03 is called out), temporal precision, SMS-Shield safe FPR and scam recall, per-template GroundLM FPR
(Bank/UPI/EMI/URL/Other buckets), and the fused / quiet-mode system FPR via the existing edge decision.

**Winner:** the passing arm with the fewest GroundLM FP (seed median). On a tie, the simpler arm wins (A < B < C).
If C passes and A or B also passes, a cross-source arm (A or B) is preferred for shipping. If no arm passes, the
result is a **kill** and is reported as such. No re-tuning on the same gates.

## 6. Evaluation and GroundLM row-by-row comparison

- A new job (`sms_retrain_eval`) reuses `baseline_v0`'s `sms_run_table`, `groundlm_table`, `smishx_table` and
  `wilson_interval`, with the same CIs and denominators. It pins each arm's ONNX by sha. The `baseline_v0` pins are
  not touched.
- Per row, on the 1 133 scoreable GroundLM negatives (join key `idx`): our decision vs GroundLM XLM-R's released
  decision (`xlmr_cleaned_test_probs.csv`, Zenodo 20480151, τ = 0.5). Report the 2 × 2 table, an exact McNemar test,
  and per-bucket counts. Positives: GroundLM's 99 policy positives (text-available subset) for paired recall.
  Every table labels in-distribution vs cross-source.

## 7. Compute and procedure

1. Data-prep step (Kaggle CPU, inside the job): fetch sources, dedup, cluster, split, write `prep_report.json`.
   New sources are pushed to `maadhavvh/rakshak-data` as a new dataset version via `push_data.sh`; DATA_PINS are
   updated in the new job only.
2. Training: `kaggle/jobs/sms_retrain.py`, one kernel per arm (3 kernels × 3 seeds), Kaggle T4.
   Estimate ≈ 1 100-1 500 s per run, so about 1.5 h per arm and well inside the weekly quota.
   One blocking `wait_event.sh` per launch; no polling.
3. Unit tests (`tests/test_sms_retrain.py`): dedup and cluster-removal rules, the cap, group-disjointness,
   pinned test ids unchanged, a threshold rule that never reads test data, and minimal-pair generation determinism.
   pytest and ruff run locally only when Tctl < 85 °C; otherwise on an instabox box (stopped afterwards).
4. Winner → ONNX export, Pi parity (bit-identical on 300 rows), Pi latency via the service eval (start < 55 °C,
   never ≥ 65 °C, no sudo).
5. `docs/SMS_RETRAIN_RESULTS.md`: tables vs v0, vs `sms_groupneg`, vs GroundLM, with every kill reported.

## 8. Known limits

- Every new legit training source is synthetic (FinEE; CloveAI in arm C). The real Indian negatives are test-side
  only: GroundLM (gated) and the IEEE DataPort ham rows (reported). No verified clean, real-only, licensed Indian
  transactional corpus exists (OpenResearch audit, 2026-11-07).
- FinEE's real ICICI rows are not flagged in the release. The ICICI-token filter may miss real rows that do not name
  the bank. If FinEE's provenance is later disputed, arm A is retrained without FinEE.
- GroundLM and VERITYGATE are by Sachin Gupta, who is not affiliated with this team (confirmed by Maadhav,
  2026-11-07). There is no conflict of interest to declare. Comparisons use only released predictions.
- About 1 019 v0 training rows are unrecoverable (split-file drift), so v0's own GroundLM/SmishX overlap is a lower bound.
- No sender/DLT data exists, so no sender ablation is possible in this plan.
