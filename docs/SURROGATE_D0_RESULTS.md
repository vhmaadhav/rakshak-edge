# Surrogate diagnostic D0: results (2026-11-11)

Plan: `docs/SURROGATE_RETRAIN_PLAN.md` section 4 (approved 2026-11-11, Amendment 1). Kaggle CPU kernel
`maadhavvh/rakshak-sms-surrogate-d0` version 1, 143 s, tree `d3b99976013b` (dirty), data
`maadhavvh/rakshak-data` and the private `maadhavvh/rakshak-inbox`. Output:
`kaggle/out/sms_surrogate_d0/output/` (`metrics.json`, `report.md`). No training. The shipped v0 student
(sha256 `5341e14c…`) is scored on each set as stored (scrubbed) and with tags filled
(`rakshak.data.surrogates`, seed 20261107).

## Verdict: H1 fails. Stop rule 1 applies: S0, S1 and S2 of that plan do not run.

H1 said filling the tags would cut the phone-1 legit false-positive rate at least 10 times. It cut it by
8% to 28% in relative terms at the shipped threshold and at R0 seeds 1 and 3.

| Set | Rows | Threshold | Scrubbed | Filled |
|---|---|---|---|---|
| Phone 1 (test donor), legit FPR | 2,976 | shipped 0.3513 | 2,935 = 98.6% | 2,707 = 91.0% |
| | | R0 seed 1, 0.5128 | 97.5% | 84.1% |
| | | R0 seed 3, 0.7533 (verdict) | 94.8% | 68.0% |
| | | R0 seed 2, 0.999998 | 0.37% | 0.13% |
| Phones 2+3 held-out legit, FPR | 433 | shipped | 430 = 99.3% | 403 = 93.1% |
| Phone 1 scam recall | 14 | shipped | 14 of 14 | 14 of 14 |
| IMC25 campaign test, scam recall | 4,947 | shipped | 4,841 = 97.9% | 4,343 = 87.8% |
| | | R0 seed 3, 0.7533 | 95.1% | 67.5% |

Median score of phone-1 legit rows: 0.990 scrubbed, 0.906 filled. 2,871 of 2,990 phone-1 rows and 4,099
of 4,947 IMC25 test rows held a tag before filling.

## What this says

1. **The tag shortcut is real but it is not the main cause.** With realistic values in place of tags the
   shipped student still flags 91% of a real Indian inbox's legitimate messages. This agrees with the
   41.6% on GroundLM, which has no tags. The student treats the Indian transactional and promotional
   register as scam.
2. **v0's scam recall partly rests on the tags.** On IMC25 test rows, recall at the shipped threshold
   falls from 97.9% to 87.8% once the tags are filled. So the campaign F1 of 0.98 is inflated by the same
   shortcut, and filled data is the fairer test split from here on.
3. **Not tested by D0:** whether the inbox arms R1 and R2 lost SmishX recall because of the reverse
   shortcut ("tag means legit"). D0 scores v0 only. That question is the subject of
   `docs/INBOX_FILLED_PLAN.md`.

## Limits

- One surrogate realisation. Surrogate realism is a modelling choice; an unrealistic fill would also keep
  scores high, and D0 cannot separate that from the model's own weakness. The GroundLM figure (real text,
  no tags) is the reason to believe the weakness is real.
- Phone 1 and phones 2+3 belong to one person. Labels are Codex's, not human. 14 scam rows.
- The R0 thresholds were calibrated on R0's scores and are applied here to v0, as the plan states.
