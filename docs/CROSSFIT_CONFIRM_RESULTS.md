# W1g: cross-fitted GroundLM calibration, confirmation on seed 3, results (2026-11-12)

Plan: `docs/CROSSFIT_CONFIRM_PLAN.md` (pre-registered before the model existed). Kaggle T4 kernel
`maadhavvh/rakshak-sms-crossfit-confirm`, 1 864 s, ok. Fresh W1e-recipe model, seed 3; folds as `sms_groupneg`
(653 clusters, 567 / 566). τ_0 = 12.387, τ_1 = 13.773 (margins), each written before the other fold was scored.

## Verdict: KILLED (C2, C3, C4 fail)

| Gate | Value | Limit | Pass |
|---|---|---|---|
| C1 pooled held-out GroundLM FP | **10 / 1 133** | ≤ 11 | ✔ |
| C2 SmishX smishing TP | τ_0: 148, τ_1: 114 | ≥ 157 at both | ✘ |
| C3 GroundLM positives recall | τ_0: 0.692, τ_1: 0.582 | ≥ 0.90 at both | ✘ |
| C4 campaign clean F1 | τ_0: 0.715, τ_1: 0.596 | ≥ 0.9713 at both | ✘ |

## Reading

The false-positive side holds (10 / 1 133 on template-held-out legit). Recall at that operating point varies a lot
between seeds of the same recipe: post-hoc seeds 0-2 gave 194-224 / 226 SmishX and 63-88 / 91 GroundLM positives,
and the fresh seed 3 gives 114-148 and 53-63. A single seed's tail ranking is not stable enough to ship on.

## Next

W1h (`docs/ENSEMBLE_CONFIRM_PLAN.md`): the mean margin of three fresh seeds (4, 5, 6) under the identical protocol and
gates. If W1h fails, SMS model work for Phase 1 stops, and the shipped detector stays v0 + rules + quiet mode
(`docs/SYSTEM_V1.md`).
