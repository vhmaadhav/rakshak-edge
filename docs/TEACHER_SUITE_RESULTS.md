# Teacher full-suite scoring W1a: results (2026-11-12)

Plan: `docs/TEACHER_SUITE_PLAN.md` (pre-registered; Amendment 1 adds the reported two-band arm, Amendment 2 the
post-hoc calibration-only cross-fit). Kaggle T4 kernel `maadhavvh/rakshak-sms-teacher-suite` v1, 266 s, status ok
(no `results.json` in the output; `metrics.json`, `report.md`, `thresholds.json` and `probs.parquet` were fetched).
Teacher: mmBERT-base, fold `source_holdout:holdout_scamshield`, sha `673b970e…`; student: shipped v0 ONNX at 0.35128.

## Gated arms: all KILLED

Thresholds from the fold's own calibration split (2 754 rows), written before any test scoring:
τ_maxF1 = 0.0405, τ_cal1% = 0.1024.

| Arm | G1 GroundLM FP / 1 133 (≤ 11) | G2 SmishX TP / 226 (≥ 157) | G3 GroundLM positives recall / 91 (≥ 0.95) | G4 campaign clean F1 (≥ 0.9713) | Pass |
|---|---|---|---|---|---|
| Teacher, τ_maxF1 | 719 | 226 | 91 | 0.883 | no |
| S∧T, τ_maxF1 | 465 | 157 | 81 | 0.965 | no |
| S∨T, τ_maxF1 | 725 | 226 | 91 | 0.897 | no |
| Teacher, τ_cal1% | 679 | 226 | 90 | 0.888 | no |
| S∧T, τ_cal1% | 465 | 157 | 81 | 0.961 | no |
| S∨T, τ_cal1% | 685 | 226 | 90 | 0.906 | no |
| Student only (v0) | 471 | 157 | 81 | 0.978 | no |

The two-band arm (Amendment 1) is ill-defined: τ_low (cal recall ≥ 0.95) = 0.99999917 lies above τ_high = 0.102.

## Why: the ranking transfers, the score scale does not

Teacher score quantiles (5 / 25 / 50 / 75 / 95 %): GroundLM legit 0.000 / 0.002 / **0.721** / **0.9996** / 1.000;
GroundLM positives ≥ 0.9999 at the 5th percentile; SmishX smishing 1.000 throughout. AUC SmishX smishing vs GroundLM
legit **0.990**; GroundLM positives vs GroundLM legit 0.944. Indian legit SMS score far above the calibration split's
negatives, so any threshold from GroundLM-free data lands where hundreds of them are flagged
(FP 601 / 331 / 123 at τ = 0.5 / 0.999 / 0.99999).

**Correction to `docs/TEACHER_XSOURCE_RESULTS.md`:** its "1 of 1 133 at 88.8 % kept" operating point exists only
because thousands of scores tie at float32 1.0. It is not reachable by any threshold rule that does not look at
GroundLM. `docs/SOTA_PUSH_PLAN.md`'s premise ("almost separates") holds for ranking (AUC), not for a deployable
operating point.

## Post-hoc, calibration-only exposure (Amendment 2; not a gated result)

The threshold is set on one template-disjoint half of GroundLM's negatives (≤ 1 % FP on that half; 653 clusters,
folds 567 / 566; no positives used) and applied to the other half. Results pooled over both held-out halves.

| Model | Held-out GroundLM FP / 1 133 | SmishX smishing / 226 | GroundLM positives / 91 | SmishX ham FP / 606 | τ |
|---|---|---|---|---|---|
| **Teacher** | **6 (0.53 %)** | **194 (85.8 %)** | 58 (63.7 %) | 46 (7.6 %) | 0.99999976 (both folds) |
| v0 student | 12 (1.06 %) | 13-15 (6 %) | 3 | 6 | 0.9977-0.9979 |
| S∧T | 1 (0.09 %) | 13-15 | 3 | 6 | both |

Reading:
1. At matched ~1 % Indian-legit FPR, the encoder keeps 86 % of SmishX smishing; the byte student keeps 6 %. The model
   class matters once the threshold is calibrated on real Indian legit SMS.
2. Recall on GroundLM's own positives is 64 % (their in-distribution XLM-R: 97/99). Those Indian scams are close to
   Indian legit in the teacher's space.
3. The teacher's τ is 2.4e-7 below 1.0 in probability: in the last float32 steps. A deployable version must
   threshold on logits, and INT8 export must preserve the top of the logit range.
4. This uses GroundLM legit data for calibration, so it is not cross-source. It sits between cross-source (all arms
   killed) and training exposure (`sms_groupneg`, 11 / 1 133 for the student).

## Next (W1e, to be pre-registered)

Licensed same-family exposure for the encoder: fine-tune the teacher fold on CloveAI (MIT; GroundLM-touching clusters
removed) plus capped FinEE, with thresholds on logits from held-out CloveAI legit template groups. GroundLM stays
test-only, labelled "same source family" as in `docs/SMS_RETRAIN_PLAN.md` arm C.

## mmBERT-small fold, seed 0 (suite run 2026-11-12, kernel `maadhavvh/rakshak-sms-teacher-suite-small-s0`, 196 s)

Checkpoint sha `f98fb3f3…` (140.6 M params, same fold and recipe). Thresholds from the fold's cal split only:
τ_maxF1 0.0661, τ_cal1% 0.5354. **All gated arms KILLED** (teacher alone: 911 / 726 GroundLM FP; S∧T 465).

In-job calibration-only cross-fit (Amendment 2 rule): pooled held-out GroundLM FP **13 / 1 133 (1.15 %)**; SmishX
smishing 86-104 / 226 (38-46 %); GroundLM positives 4-7 / 91; SmishX ham FP 11 / 606. Logit and probability thresholds
give the same counts. **Caveat:** the in-job fold build gave 1 004 clusters (folds 566 / 567), not the 653 of the base
analysis (`matches_amendment2: false`), so its folds are not identical to Amendment 2's. Treat it as indicative.

Reading: at about 1 % Indian-legit FPR the small encoder keeps roughly half the SmishX smishing that the base encoder
keeps (86-104 vs 194) and far fewer GroundLM positives (4-7 vs 58). On this fold, the cross-source ranking quality
comes with model size. Seeds 1 and 2 were not scored.
