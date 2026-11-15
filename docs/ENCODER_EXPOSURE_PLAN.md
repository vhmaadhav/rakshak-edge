# Encoder exposure W1e: same-family CloveAI and FinEE training, calibration-only logit threshold (pre-registration)

Written 2026-11-12, before `kaggle/jobs/sms_encoder_exposure.py` exists and before any model is trained for this
plan or any test set is scored. Follows `docs/TEACHER_SUITE_RESULTS.md` ("Next (W1e)"). Kaggle only (T4); no Modal.
Frozen once the job is built. Any later change is a dated amendment, written before the result it could affect.

## 1. Question

The W1a teacher (mmBERT-base, `source_holdout:holdout_scamshield`) ranks SmishX scam against GroundLM legit well
(AUC 0.990) but its score scale does not transfer: no threshold chosen without GroundLM rows gives an acceptable
false-positive rate (`docs/TEACHER_SUITE_RESULTS.md`, "Why"). W1e asks whether one licensed exposure to same-family
legit SMS (CloveAI, GroundLM-touching groups removed, plus capped FinEE) and a threshold set on logits from held-out
rows of that exposure, not from GroundLM, pass the deployment gates on GroundLM, SmishX and the campaign test.

**Label for every GroundLM result:** "same source family (CloveAI), template-disjoint". Never cross-source. CloveAI
is the family GroundLM's own test pool draws from. GroundLM-touching CloveAI split groups are removed from training,
but the family overlap remains.

## 2. Model and recipe (the sms_encoder_fold recipe, reused)

- Model `jhu-clsp/mmBERT-base` at revision `c5955035435e2bf121cde7f3c8863ef52ff35d82` (the pin in
  `kaggle/jobs/sms_encoder_fold.py`). Snapshot patterns `*.json, *.safetensors, *.bin, *.txt` (mmBERT ships `.bin`).
- Training function `sms_teacher.train_once` with the sms_encoder_fold defaults: lr 3e-5, up to 4 epochs,
  weight decay 0.01, warmup 0.1, max_len 128, batch 32, eval batch 128, fp16 autocast over fp32 weights, patience 2.
  Input is canonicalised (`canon_text`), as in the teacher.
- **Early stopping** uses the teacher fold's own calibration split (`source_holdout:holdout_scamshield`, cal, 2,754
  rows), exactly as sms_encoder_fold does. It does not use the exposure hold-out in section 4.
- Seeds **0, 1, 2** (training seed). Data seed **0** (`build_pools` and the hold-out below). Every seed trains on the
  same rows.

## 3. Training partition

Training = the teacher fold's train partition (24,391 rows, asserted) plus the **arm C extra rows** of
`rakshak.data.retrain_prep.build_pools(arm="C", seed=0)`, minus the calibration hold-out of section 4.

`build_pools` arm C, as already defined in `docs/SMS_RETRAIN_PLAN.md`:
- FinEE synthetic legit rows (ICICI-token filter, cluster caps), and CloveAI rows (MIT) with every split group that
  touches a GroundLM text removed.
- All test-only dedup rules (near-duplicates at Jaccard 0.8 of any test-only text removed, with the whole split
  group): campaign test, lang_shift and temporal test, GroundLM, SmishX, SMS-Shield and IEEE where present.
- The 5-per-cap-group cap and the round-robin total budget `min(12 000, 0.25 × campaign-train negatives)` for legit
  rows, and the same budget for CloveAI scam rows.

`build_pools` also holds out 15 % of FinEE split groups plus skyaara and Ridham115 as its own validation pools (its
`legit_val` and `val_scam`). **This experiment does not use those pools**, and their FinEE groups are not in
`train_extra`. Nothing from them is trained on, scored or used for a threshold.

## 4. Calibration hold-out (used only for the threshold and the recall report)

- Split groups (the `cluster` column of `build_pools`, the 0.7 connected components) of **CloveAI** and of **FinEE**,
  taken from `train_extra`. For each source separately, 20 % of its groups are drawn without replacement with
  `numpy.random.default_rng(4)` (sources in sorted order, groups sorted, `round(0.2 × n_groups)` per source).
- Every `train_extra` row whose split group was drawn is removed from training and placed in the **calibration
  hold-out** (CH). A split group is never in both training and CH (asserted; the test in section 9 checks it).
- CH legit (label 0) is the **threshold set**. CH scam (label 1, CloveAI only) is used **only** to report recall
  at τ. Neither set is trained on or used for early stopping.
- Asserted: CH legit has at least 100 rows, so that 1 % is a meaningful count. The count goes to `prep_report.json`.

## 5. Score and threshold (logits, not probabilities)

- Score `m = logit_1 − logit_0` (float64). For a two-class softmax this is the logit of p_scam: monotone in p, and
  it does not saturate at float32 1.0 (the W1a defect). Alarm rule: flag when `m ≥ τ`.
- **Primary τ:** the smallest τ with at most `k = floor(0.01 × n_CH_legit)` CH legit rows at `m ≥ τ`. Computed as
  `nextafter((k+1)-th largest CH legit margin, +inf)`. If `k ≥ n_CH_legit`, τ = −∞ (asserted impossible for the
  hold-out sizes).
- Reported at τ: CH scam recall (k / n) with Wilson 95 %.
- **Reported second rule τ_cal:** the same definition on the teacher fold's **own** cal split (2,754 rows) at
  FPR ≤ 1 %. Not gated.
- τ and τ_cal are written to `thresholds.json` (per seed, and the combined file) **before any test set is scored**.
  `sms_retrain.run_seed_pipeline` enforces the order. The job fails if a test set is scored before the threshold
  event.

## 6. Sets and arm (as in `sms_teacher_suite`, unchanged definitions)

- Test sets: campaign test (all and clean), lang_shift, temporal, GroundLM policy negatives (1,133), GroundLM
  positives with text (91), SmishX smishing (226), SmishX ham (606).
- **Clean** campaign rows exclude any row whose id or canonical text is in the **new** training partition (fold
  train plus the extra rows of section 3, not the hold-out). Contamination counts are reported per set.
- One arm: model E (the exposure model) alone. The student is not part of this experiment. Both τ rules are reported
  for E; the gates use the primary τ.

## 7. Gates (pre-registered; a gate holds when the seed median passes and at least 2 of 3 seeds pass)

| # | Gate | Pass condition |
|---|---|---|
| G1 | GroundLM policy negatives flagged | ≤ 11 of 1,133 |
| G2 | SmishX smishing flagged | ≥ 157 of 226 |
| G3 | GroundLM positives recall (91 rows) | ≥ 0.90, i.e. ≥ 82 flagged |
| G4 | Campaign clean-row F1 | ≥ 0.9713 |

The verdict is pass only when G1 to G4 all hold. The job reports "incomplete" unless all three seeds are present.
Shipping also needs W1c (ONNX parity) and W1d (Pi). This job does not decide shipping.

**Relation to the earlier gates:** G1, G2 and G4 are the W1a and sms_retrain numbers. G3 is 0.90 here (W1a used 0.95
on the same 91 rows). Both are stated so the comparison is explicit. The 0.90 is pre-registered now, before any run.

## 8. Reported, not gated

- For each seed and both τ rules: counts and Wilson intervals for G1 to G3, CH scam recall, SmishX ham FPR (606),
  lang_shift F1 (campaign-clustered bootstrap, 2,000 resamples, `default_rng(0)`, percentile 95 %), temporal precision
  (same bootstrap), campaign all-rows and clean-row F1 with intervals.
- Paired McNemar (exact, two-sided) against the released GroundLM XLM-R decision (`pred`, its τ = 0.5), on the 1,133
  negatives and on the 91 positives. Per-category GroundLM false-positive counts.
- **Post-hoc comparison row (Amendment 2 protocol applied to these models; not a gated result).** The GroundLM
  policy negatives are split into 2 template-disjoint folds with `rakshak.eval.groupneg.negative_clusters` and
  `assign_folds` (deterministic; the fold sizes are checked against the 567 / 566 of Amendment 2 and the result
  logged). For each fold f, τ_f is the smallest margin with at most floor(1 % × |f|) false positives on fold f's
  negatives. τ_f is applied to the other fold's negatives. Pooled over both held-out folds, and SmishX and GroundLM
  positives reported at τ_0 and τ_1 separately. This is the calibration-only protocol of `docs/TEACHER_SUITE_PLAN.md`
  Amendment 2, run on E. The W1a teacher row from `docs/TEACHER_SUITE_RESULTS.md` is the reference.
- Contamination counts per set, CH and training counts, `prep_report.json`.

## 9. Tests (pure, synthetic; `tests/test_sms_encoder_exposure.py`)

- Calibration hold-out: disjoint by split group; deterministic for a seed; the fraction per source is 20 % (rounded);
  no split group appears on both sides.
- Logit threshold rule: the smallest τ with FP ≤ 1 %; `m ≥ τ` is monotone with p ≥ sigmoid(τ); the all-negatives
  edge gives −∞; the rule never reads scam or test scores.
- Gate aggregation: median plus 2-of-3 for each gate; a single bad seed does not fail the gate when the median and
  two seeds pass; an incomplete seed list reports incomplete.
- Ordering: with `run_seed_pipeline` and fake scorers, `threshold` comes before every test set.
- Arguments: default seeds 0 1 2; the revision and training defaults equal `sms_encoder_fold`.

## 10. Caveats (fixed before the run)

- Same-family exposure: CloveAI is the family of GroundLM's test pool. Gains here are not cross-source.
- FinEE provenance is unverified (`docs/SMS_RETRAIN_PLAN.md` section 8); a passing result would need a licence and
  provenance review before any shipping. CloveAI is MIT.
- 91 GroundLM positives: the G3 interval is wide and is reported.
- The threshold set is about 20 % of the CloveAI/FinEE pool after the budget, so its size depends on the cap; the
  count is in the report. If CH legit is below 100, the run stops.
- Three seeds, one fold, one data seed. A seed-level pass is not a population claim.
- The early-stopping calibration (the teacher fold cal) and the threshold hold-out are different sets, by design.
