# W1g: cross-fitted GroundLM calibration, confirmation on a fresh model (pre-registration)

Written 2026-11-12, before the model exists. Approved by Claude under Maadhav's delegation of plan decisions (2026-11-11);
he has not reviewed it line by line. Context: `docs/INBOX_CALIBRATION_RESULTS.md` (every GroundLM-free threshold source
failed), `docs/ENCODER_EXPOSURE_RESULTS.md` (post-hoc cross-fit on W1e seeds 0-2). The W1e seed 0-2 numbers were
seen before this plan was written, so they cannot confirm it. This plan is confirmed or killed by a **new seed (3)**
that no one has scored.

## Model

`kaggle/jobs/sms_encoder_exposure.py` recipe unchanged (mmBERT-base, fold `source_holdout:holdout_scamshield` +
build_pools arm C extra rows minus the 20 % CloveAI/FinEE calibration hold-out), **seed 3** only.

## Protocol (frozen)

1. GroundLM's 1 133 scoreable policy negatives (`policy_negative & eval_ok & text_available`) are split into 2
   template-disjoint folds exactly as `kaggle/jobs/sms_groupneg.py` does:
   `negative_clusters(template_key)` + `assign_folds(·, 2)`. Assert 653 clusters and fold sizes 567 / 566.
2. For each fold F: τ_F = the smallest logit margin with at most floor(0.01 × |F|) false positives on F's negatives.
   No positives are used. τ_F is written to `thresholds.json` before the other fold or any positive set is scored.
3. τ_F is applied to the other fold's negatives (pooled over both folds), and to SmishX smishing (226), SmishX ham
   (606), GroundLM policy positives with text (91), phone 1 test-donor legit and scam (filled), and the campaign test
   (clean rows).
4. The **deployment threshold** τ_dep is the same rule on all 1 133 negatives (≤ 11 FP). It is reported and stored
   for the service. Its in-sample FP is not a result.

## Gates (all must hold)

| # | Gate | Pass |
|---|---|---|
| C1 | Pooled held-out GroundLM FP | ≤ 11 / 1 133 (≤ 0.97 %) |
| C2 | SmishX smishing TP at each τ_F | ≥ 157 / 226 at both folds' τ |
| C3 | GroundLM positives recall at each τ_F | ≥ 0.90 (82 / 91) at both |
| C4 | Campaign clean F1 at each τ_F | ≥ 0.9713 at both |

Reported, not gated: SmishX ham FPR; phone-1 legit FPR and scam recall (scrubbed/filled, so expected to be poor:
`docs/INBOX_CALIBRATION_RESULTS.md`); per-category held-out GroundLM FP; paired McNemar vs GroundLM XLM-R on the
held-out negatives and the positives; AUCs; lang_shift and temporal at τ_dep.

## Labelling

"GroundLM-calibrated (threshold only, cross-fitted, template-disjoint); training data shares a source family
(CloveAI) with GroundLM; no GroundLM row in training." It is **not** cross-source and **not** the same regime as
GroundLM's in-distribution XLM-R (trained on GroundLM's sources, random split, test-sweep threshold).

If C1-C3 pass and C4 fails, the encoder ships as an Indian-SMS stage beside the byte student (which keeps campaign
coverage). That combination gets its own pre-registered system-level check before it ships.
