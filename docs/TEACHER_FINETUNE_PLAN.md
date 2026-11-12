# Teacher fine-tune on filled data and inbox negatives T1 (pre-registration)

Written 2026-11-11 after `docs/TEACHER_XSOURCE_RESULTS.md`, before the job exists. **Approved 2026-11-11 by
Claude under Maadhav's delegation of plan decisions; Maadhav has not reviewed this file line by line.**
Frozen as of 2026-11-11; later changes go in dated amendments.

## 1. Question

D1 showed a pretrained encoder (mmBERT-base, trained on 24,391 public rows) separates scam from GroundLM's
legitimate Indian SMS (AUC 0.995, 1–6 false positives in 1,133) where the byte-level student cannot (0.71),
but does not separate scam from a real inbox it never saw (AUC 0.72 on filled text). The student could not
use the inbox negatives without losing recall (S1). Can the encoder?

## 2. Arms (Kaggle T4, one kernel per arm, run in parallel)

| Arm | Training data | Role |
|---|---|---|
| T0 | Campaign train split of `sms.parquet`, filled | control: the encoder without inbox data |
| T1 | T0 + inbox train donor (phones 2+3), filled; the pools and caps of S1 | the test |

Both: `jhu-clsp/mmBERT-base` from the Hugging Face hub (not the D1 checkpoint), the fixed hyperparameters
of `kaggle/jobs/sms_teacher.py` (lr 3e-5, up to 4 epochs, weight decay 0.01, warmup 0.1, max_len 128, batch
32, fp16 autocast, class-balanced cross-entropy, early stopping on cal PR-AUC with patience 2), input form
canon, no HPO, no adversarial augmentation. Seed 0. Data preparation, fill (seed 20261107), splits, the
test-only text exclusion and the inbox pools are those of the S-arms (`kaggle/jobs/sms_inbox_retrain.py`).
A `--smoke` run on a small subset checks the pipeline first; its numbers are not results.

## 3. Threshold rule and gates

Threshold: `rakshak.data.retrain_prep.threshold_rule` on the filled inbox held-out legit set plus
campaign-cal, set before any test set is scored (as S0–S2).

Gates, unchanged in definition and limit: G1 GroundLM FP ≤ 11 / 1,133 · G2 SmishX TP ≥ 157 / 226 · G3
campaign F1 ≥ 0.971 on the filled campaign test split · G5 test-donor legit FPR ≤ 1%. G4 (ONNX ≤ 10 MB)
does not apply to a teacher; the checkpoint size is reported.

An arm **passes provisionally** if its one seed meets G1, G2, G3 and G5. A provisional pass is confirmed
only by two more seeds (1 and 2) under the 3-seed rule of the earlier plans (median and at least 2 of 3).
Nothing is exported or distilled from a provisional pass.

## 4. Also reported, without a gate

- The D1 measures on the same rows: A1, A2, AUC against SmishX ham, campaign AUC, with bootstrap intervals
  for A1 and A2.
- A1 and A2 with only the 226 G2 smishing rows as positives (D1 used all 578 label-1 rows, which include
  spam).
- Test-donor legit FPR split by Codex policy category (benign_transactional, promotion), with Wilson
  intervals, and test-donor scam recall (n = 14, Clopper–Pearson).
- Scores of the scrubbed versions of the inbox sets, for the tag-shift comparison.
- The share of rows at the saturated score 1.0, and thresholds reported to full float precision.

## 5. Hypotheses (frozen)

- **H7:** T1 meets G5 (test-donor legit FPR ≤ 1%) and G1 (GroundLM FP ≤ 11).
- **H8:** T1 meets G2 (SmishX TP ≥ 157 of 226) at the same threshold.
- **H9:** T1 passes provisionally (G1, G2, G3, G5).

## 6. Stop rules and what follows

1. T1 passes provisionally: run seeds 1 and 2 of T1. If the 3-seed rule holds, the next plan is deployment:
   either the encoder itself on the Pi (INT8, latency and memory measured beside the Llama) or a smaller
   encoder distilled from it, decided by measurement.
2. T1 meets H7 but not H8 (false positives fixed, recall lost): the threshold rule is the binding
   constraint. Report the ranking measures; the next plan would revisit the rule openly, not this one.
3. T1 fails H7: the inbox negatives are not learnable from one donor even by the encoder. Report and stop;
   more donors is then the only route.
4. No arm, seed, hyperparameter or threshold is added beyond the above.

## 7. Limits

One seed until confirmed. One surrogate realisation. Train and test donors are one person, so G5 measures
transfer across that person's phones, not across people. Codex labels. 14 test-donor scam rows. G3 uses the
filled split, on which the shipped student scores 0.958.

## Amendment 1 (2026-11-11, before any run)

Choices made while building the job, before any score exists:

- G3's limit is 0.9713, the value the S-arms use in code (`sms_retrain.CAMPAIGN_MIN_F1`); section 3 wrote it
  rounded as 0.971.
- If H7 and H8 hold and only G3 fails, none of stop rules 1–3 applies. The arm does not pass, no seed is
  added (rule 4), and the result is reported with the ranking measures.
- The v0 reference scoring is dropped from this job (D1 already has v0 on the same sets).
- The smoke run trains on 1,920 rows for at most 60 steps.
- "Clean" rows are those whose canonical text is not in the arm's own training partition; bootstrap
  intervals use 1,000 resamples on clean rows.
- The 226-positive variant uses SmishX rows with `label_fine` smishing, `eval_ok` and text available; the
  count is reported.
- Campaign test is scored filled only. T0's stop rule is marked not applicable.
