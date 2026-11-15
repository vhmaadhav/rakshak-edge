# Teacher suite W1a: full-suite scoring of the mmBERT-base fold (pre-registration)

Written 2026-11-12, before `kaggle/jobs/sms_teacher_suite.py` exists and before any test set is scored. Part of
W1a in `docs/SOTA_PUSH_PLAN.md`, under Maadhav's delegation of plan decisions (2026-11-11). Maadhav has not
reviewed this file line by line. Frozen as of 2026-11-12; no threshold, set, gate or arm changes after the run.

## 1. Question

`docs/TEACHER_XSOURCE_RESULTS.md` showed that the mmBERT-base fold (`source_holdout:holdout_scamshield`, seed 0)
separates SmishX scam from GroundLM legitimate SMS (AUC 0.995, 1 to 6 false positives in 1,133) but scored only
SmishX label-1 rows as positives. Does this teacher, alone or combined with the shipped v0 student, pass the
deployment gates on GroundLM positives and negatives, SmishX, and the campaign test, with a threshold chosen
without any test row?

## 2. Teacher and its training partition (the same as sms_teacher_xsource)

- Checkpoint: `best_checkpoint/model.safetensors`, sha256 `673b970e9f4746dbbbb5a00aa84eb056d6750cbc2897ba712ab1aa1b234abc7e`,
  located under `--teacher-dir` by `sms_teacher_xsource.locate_teacher`. Provenance from the teacher job's
  `metrics.json` `meta`: model `jhu-clsp/mmBERT-base`, run `source_holdout:holdout_scamshield`, seed 0,
  `train_input` canon. The job stops if the run or the input form differs.
- Training partition: `rakshak.eval.data.load_runs` on the `source_holdout` split file with `mask = eval_mask`,
  the run named above. Size asserted at **24,391** rows (the `train` count of the teacher's logits file).
- Calibration split: the same run's `cal` partition, masked by `eval_mask` as `sms_teacher` did. Size asserted
  at **2,754** rows (the `cal` count of the same logits file). The job fails if either count differs.
- Input form: `rakshak.eval.data.canon_text` applied to the text as stored in the table, the form the teacher was
  trained on (`sms_teacher` builds its training and cal inputs with `canon_text` when `train_input` is canon).
  The text is the stored scrubbed form for every set. This suite uses no inbox rows and no surrogate fill. fp16
  autocast over fp32 weights, max_len 128, eval batch 128 (the `sms_teacher` defaults, as in xsource).
- Contamination: a test row is contaminated when its id is in the training partition, or its canonical text equals
  a canonical training text. Counts are reported per set. The clean subset of the campaign test (G4) excludes
  contaminated rows.

## 3. Thresholds (calibration split only, written before any test set is scored)

Teacher probability p_T, flag when p_T >= tau.

- **tau_maxF1 (primary):** `metrics.threshold_max_f1(y_cal, p_T_cal)`, the lowest threshold that reaches the best
  F1 on the cal split.
- **tau_cal1% (secondary):** `metrics.threshold_for_fpr(y_cal, p_T_cal, 0.01)["threshold"]`, the threshold with
  the highest recall among those with cal FPR <= 1%.

Both are written to `thresholds.json` before any test set is scored. The ordering is enforced with
`sms_retrain.run_seed_pipeline`. The job fails if either threshold is not finite. Teacher scores saturate near
1.0, so ties are possible; the achieved flag counts are reported.

The student keeps its shipped threshold **0.35127517580986023** (`baseline_v0.SMS_THRESHOLD`). It is not
re-tuned.

## 4. Sets

All sets use the `eval_ok` mask (and not synthetic), the same as the other SMS jobs.

- **GroundLM negatives (G1 set):** `policy_negative` and `text_available` and `eval_ok`. 1,133 rows, asserted.
- **GroundLM positives (G3 set):** `label_binary == 1` and `text_available` and `eval_ok`. 91 rows, asserted. The
  `policy_negative` flag is False on every positive and `label_policy` is empty for them, so "policy positive"
  here means the scam rows with text. The XLM-R join uses these 91 rows.
- **SmishX smishing (G2 set):** `eval_ok` and `label_fine == smishing`. 226 rows, asserted.
- **SmishX ham (reported):** `eval_ok` and `label_fine == legitimate`. 606 rows.
- **Campaign test (G4):** the `campaign` run's test partition. Reported on all rows and on the clean rows.
- **lang_shift (reported):** the test partitions of the `lang_shift` split file.
- **temporal (reported):** the test partitions of the `temporal` split file.

Campaign cal is not a test set and is not in `probs.parquet`.

## 5. Arms

For each threshold rule r in {tau_maxF1, tau_cal1%}, using the student threshold S_t = 0.35127517580986023:

- **T:** flag if p_T >= tau_r.
- **S and T:** flag if p_S >= S_t and p_T >= tau_r.
- **S or T:** flag if p_S >= S_t or p_T >= tau_r.

The student alone at S_t is reported as a reference, not an arm.

## 6. Gates (an arm passes when all four hold)

- **G1:** GroundLM negatives flagged <= 11 of 1,133.
- **G2:** SmishX smishing flagged >= 157 of 226.
- **G3:** GroundLM positives recall >= 0.95 (flagged / 91 >= 0.95; at least 87 flags). This is GroundLM's own release gate.
- **G4:** F1 on the clean campaign test rows >= 0.9713 (the sms_retrain G3 campaign gate), at the arm's threshold.

The primary verdict uses the tau_maxF1 arms. The tau_cal1% arms are secondary. The job reports a pass or fail per
arm. It does not decide whether to ship. Shipping also needs W1c (ONNX parity) and W1d (Pi).

## 7. Reported, not gated

- SmishX ham FPR (606 rows).
- lang_shift F1 and temporal precision, for each arm.
- GroundLM negatives flagged per category (`bucket_counts`).
- Paired against the released GroundLM XLM-R decision (`pred`, its own threshold 0.5), joined on GroundLM `row_key`
  (`sms_retrain.xlmr_reference`): a 2x2 table with the exact two-sided McNemar p-value, on the 1,133 negatives and
  on the 91 positives, for each arm.
- Intervals: Wilson 95 % for every count rate (`baseline_v0.wilson_interval`). For F1 and precision on the campaign,
  lang_shift and temporal sets: a campaign-clustered bootstrap, 2,000 resamples, `numpy.random.default_rng(0)`,
  percentile 95 %.
- Contamination counts per set, and the cal and training counts.

## 8. Caveats (fixed before the run)

- GroundLM and SmishX rows are cross-source: the fold never saw them. The XLM-R reference was trained on GroundLM's
  own in-distribution split, so the paired comparison is not like for like.
- Expected contamination from `docs/TEACHER_XSOURCE_RESULTS.md`: 0 of 1,133 GroundLM negatives, 0 of 1,200 SmishX
  text rows. The campaign test has about 5,101 contaminated rows of 22,145 by text. The job computes these counts
  and does not assert them.
- One checkpoint, one seed, one fold. Only 91 GroundLM positives, so G3's interval is wide and is reported.
- The teacher scores saturate near 1.0, so the threshold can keep more rows than its target.
- The G4 F1 gate is applied to scrubbed text, the stored form the teacher was trained on. Reported only.

## Amendment 1 (2026-11-12, before any test set is scored)

Source: coordinator instruction, from `research_notes/SOTA_IDEAS_2026-11-12.md` idea 1. Nothing has been
scored when this amendment is written. The gates of section 6 and the arms of section 5 are unchanged.

Adds a **reported, not gated** two-band output, for the teacher alone (T) and for the teacher gated by the student
(S and T):

- **tau_high** = tau_cal1% (section 3): visible SCAM when the score is at or above it.
- **tau_low** = the largest threshold whose calibration recall is at least 0.95. Computed as the k-th largest
  calibration positive score with k = ceil(0.95 x n_pos), integer arithmetic (the recall convention of
  `kaggle/jobs/sms_teacher_xsource.py`). Ties can keep more rows than k.
- **VERIFY band** (shown as "verify independently", the UNCERTAIN state): tau_low <= score < tau_high.
- **Benign**: below tau_low.
- For **S and T**: the band is computed on the teacher score and gated by the student. SCAM needs p_S >= 0.35127517580986023
  and p_T >= tau_high. VERIFY needs p_S >= 0.35127517580986023 and tau_low <= p_T < tau_high. Everything else is benign.
- Both thresholds come from the calibration split only, and are written to `thresholds.json` before any test set is scored.
- Counts per band are reported for: GroundLM negatives (visible FP and verify-band FP reported separately, each with a Wilson
  interval), GroundLM positives, SmishX smishing, SmishX ham, and campaign clean rows.
- If tau_low > tau_high the bands are inverted. The job records `ordered: false` and the VERIFY band is empty. This is
  reported, not a failure.

## Amendment 2 (2026-11-12, written AFTER the gated arms failed and BEFORE the analysis below was computed)

The gated result is final: every arm fails (calibration-split thresholds 0.040 / 0.102 give 679-725 GroundLM FP).
The row-level scores show the teacher ranks well (AUC SmishX smishing vs GroundLM legit 0.990) but its scores on
Indian legit SMS sit far above its calibration split's negatives, so no GroundLM-free threshold transfers.

Added analysis, labelled **post-hoc protocol, calibration-only exposure, not a gated result**: GroundLM's 1 133
scoreable negatives are split into 2 template-disjoint folds with the existing clusterer
(`rakshak.eval.groupneg.negative_clusters` + `assign_folds`, seed 0). For each fold F, τ_F = the smallest threshold
with at most floor(1 % × |F|) false positives on F's negatives (no positives used). τ_F is applied to the OTHER
fold's negatives, and to SmishX smishing (226), SmishX ham (606) and GroundLM's 91 positives. Results are pooled
over the two held-out folds. Models: teacher alone, v0 student alone, and S∧T (each with its own fold threshold).
This reports what one calibration-only exposure to real Indian legit SMS buys. It is compared with
`sms_groupneg` (training exposure, 11 / 1 133) and GroundLM XLM-R (in-distribution training, 5 / 1 294).
