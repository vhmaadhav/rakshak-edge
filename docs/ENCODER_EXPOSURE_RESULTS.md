# W1e: encoder with licensed same-family exposure, results (2026-11-12)

Plan: `docs/ENCODER_EXPOSURE_PLAN.md` (pre-registered). Kaggle T4 kernel `maadhavvh/rakshak-sms-encoder-exposure-w1e`
(the first launch failed on a list-indexing bug; relaunched v2), 4 493 s, status ok. mmBERT-base, fold
`source_holdout:holdout_scamshield` training partition + build_pools arm C extra rows (FinEE capped, CloveAI with
GroundLM-touching split groups removed), minus 20 % of CloveAI/FinEE split groups held out for calibration.
GroundLM label: same source family (CloveAI), template-disjoint.

## Gated result: KILLED (all three seeds)

| Seed | τ (logit margin, ≤ 1 % FPR on held-out CloveAI/FinEE legit) | G1 GroundLM FP / 1 133 | G2 SmishX / 226 | G3 GroundLM positives / 91 | G4 campaign clean F1 |
|---|---|---|---|---|---|
| 0 | -11.31 | 773 | 226 | 91 | 0.599 |
| 1 | -12.72 | 748 | 226 | 91 | 0.619 |
| 2 | -8.87 | 620 | 226 | 91 | 0.595 |

Why: the held-out calibration rows come from the same sources the model trained on, so they are easy (scores far
below 0). "Smallest τ with ≤ 1 % FPR" lands at a margin of -9 to -13, which flags most of GroundLM's legit SMS and
much of the campaign test. Same-source legit is not a usable calibration set for a different Indian corpus.

## Diagnostic (not gated): the ranking improved

Same post-hoc rule as `docs/TEACHER_SUITE_PLAN.md` Amendment 2 (τ on one template-disjoint half of GroundLM's
negatives, ≤ 1 % FP there, applied to the other half; 653 clusters). Margins, per seed:

| Model | AUC SmishX smishing vs GroundLM legit | AUC GroundLM positives vs legit | Held-out GroundLM FP / 1 133 | SmishX / 226 (fold τ0, τ1) | GroundLM positives / 91 | SmishX ham FP / 606 |
|---|---|---|---|---|---|---|
| Base teacher (no exposure) | 0.990 | 0.944 | 6 | 194, 194 | 58, 58 | 46, 46 |
| W1e seed 0 | 0.9969 | 0.9900 | 16 | 215, 208 | 66, 63 | 51, 46 |
| W1e seed 1 | 0.9965 | 0.9923 | 16 | 221, 194 | 88, 64 | 53, 37 |
| W1e seed 2 | 0.9981 | 0.9916 | **12 (1.06 %)** | **224, 223** | **79, 78** | 63, 60 |

Reading: licensed CloveAI/FinEE exposure raises the encoder's separation of GroundLM's own scams from its legit SMS
(AUC 0.944 → 0.990-0.992) and recall at a GroundLM-calibrated threshold (58 → 63-88 of 91), at a slightly higher
held-out FP (12-16 vs 6). This uses GroundLM legit SMS to place the threshold, so it is not a deployable or
cross-source result. For reference: GroundLM's in-distribution XLM-R gets 5 / 1 294 FP at 97 / 99.

## What is binding now

The model ranks well. A deployable threshold needs **real Indian legit SMS that is neither training data nor the test
corpus**. The only such data we hold is the consented inbox (`maadhavvh/rakshak-inbox`, phones 2+3, scrubbed then
filled). The next step (W1f) is to place τ on that, with no training on it, and test on GroundLM and the held-out
phone 1. It is pre-registered separately.
