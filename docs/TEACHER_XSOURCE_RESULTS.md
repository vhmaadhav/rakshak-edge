# Teacher cross-source ranking check D1: results (2026-11-11)

Plan: `docs/TEACHER_XSOURCE_PLAN.md` with Amendment 1 (both written before the run). Kaggle T4 kernel
`maadhavvh/rakshak-sms-teacher-xsource` version 1, 379 s, tree `d41bc7ae9815` (dirty). Scoring only. Output:
`kaggle/out/sms_teacher_xsource/output/` (`metrics.json`, `report.md`; no row-level file).

Teacher: `jhu-clsp/mmBERT-base`, fold `source_holdout:holdout_scamshield`, seed 0 (sha256 `673b970e…`),
trained on 24,391 rows of canonicalised text, none from `scamshield`. No GroundLM, SmishX or inbox row is in
its training partition (0 of 1,133, 0 of 1,200, 0 of 2,990). 5,101 of the 22,145 campaign test rows are, and
are excluded from the "clean" campaign AUC.

## Verdict by the frozen rule: "does not transfer" (A2 = 0.7225 < 0.85). A1 passes on its own.

| Measure (clean rows) | Teacher (mmBERT-base fold) | v0 student | Best S1 student seed |
|---|---|---|---|
| **A1** AUC, SmishX label-1 (578) vs GroundLM legit (1,133) | **0.9947** [0.9918, 0.9971] | 0.7146 | 0.915 |
| **A2** AUC, SmishX label-1 vs filled test-donor legit (2,976) | **0.7225** [0.7073, 0.7377] | 0.4022 | 0.807 |
| AUC vs scrubbed test-donor legit | 0.4596 | 0.1295 | – |
| AUC vs SmishX ham (622) | 0.9555 | 0.9342 | 0.966 |
| Campaign test AUC, scrubbed / filled | 0.9717 / 0.9719 | 0.9975 / 0.9939 (all rows) | 0.995 |

Operating points. The teacher's scores saturate, so many rows tie at 1.000000 and the "keep 70%" and "keep
90%" thresholds both land on 1.0 and keep more than asked.

| Threshold | SmishX label-1 kept | GroundLM legit FPR | Test-donor legit FPR, filled | Test-donor scam kept, filled |
|---|---|---|---|---|
| Teacher, "keep 70%" | 513 of 578 = 88.8% | **1 of 1,133 = 0.09%** [0.02, 0.50] | 1,336 of 2,976 = 44.9% | 13 of 14 |
| Teacher, "keep 90%" | 529 of 578 = 91.5% | **6 of 1,133 = 0.53%** [0.24, 1.15] | 1,801 of 2,976 = 60.5% | 14 of 14 |
| v0, keep 70% | 405 of 578 = 70.1% | 460 of 1,133 = 40.6% | 2,416 of 2,976 = 81.2% | 14 of 14 |
| v0, keep 90% | 521 of 578 = 90.1% | 488 of 1,133 = 43.1% | 2,902 of 2,976 = 97.5% | 14 of 14 |

## What this says

1. **On GroundLM the model class is the difference.** A pretrained encoder trained on 24,391 public rows
   separates SmishX scam and spam from GroundLM's legitimate Indian SMS almost completely (AUC 0.995, 1 to 6
   false positives in 1,133), where the byte-level student gets 0.71 and 460 false positives. Neither model
   saw GroundLM. This is the first cross-source result here that reaches the 1% target on GroundLM.
2. **The real inbox is still not separated** (AUC 0.72 filled, 0.46 scrubbed). The teacher never saw inbox
   data, and it was trained on tagged text. 1,336 false positives is more than the 518 promotion rows in
   that inbox, so transactional messages are flagged too.
3. **The frozen rule gives "does not transfer"**, because both AUCs had to pass. As Amendment 1 said before
   the run, that verdict covers this fold, not the model class.

## Not a claim

- This is not a like-for-like comparison with the published GroundLM model (270M XLM-R, 0.39% FPR at 0.980
  recall, in-distribution random split). The false-positive side here is cross-source and on the same 1,133
  policy negatives; the recall side is SmishX label-1 rows, not GroundLM's positives, which this job did
  not score.
- SmishX label-1 includes spam, and the inbox's legit rows include promotions. Part of the A2 gap may be
  that overlap. The job wrote no row-level file, so it cannot be split by category after the fact.

## Next step (by the plan's rule)

Fine-tune the teacher on the S1 data (filled campaign train plus the filled inbox train donor) and score
the same sets with the same gates: `docs/TEACHER_FINETUNE_PLAN.md`.

## Limits

One fold, one seed, one surrogate realisation, one donor person, Codex labels, 14 test-donor scam rows.
