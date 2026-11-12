# SMS student retrain on consented inbox data: results (2026-11-07)

Plan: `docs/INBOX_RETRAIN_PLAN.md` with addenda 1–3 (Codex labelling, three phones, no export cap,
phone 1 held out as the test donor). Kaggle T4, 3 seeds per arm, kernels
`rakshak-sms-inbox-retrain-{r0,r1,r2}`. Nothing exported to the Pi.

**Verdict: all three arms fail the pre-registered gates. No winner. A confound, explained below,
means these numbers should not be read as how the arms would do on raw phone text.**

## Data

| | Rows | Legit | Scam |
|---|---|---|---|
| Train donor (phones 2+3) | 5,048 | 4,886 | 160 |
| Test donor (phone 1) | 2,992 | 2,976 | 14 |

Labels are from Codex on scrubbed text, not human labels. Train and test come from the same person.

## Gates (per seed 1 / 2 / 3)

| Gate | v0 / R0 | R1 (+ inbox) | R2 (+ inbox + sender feature) | Target |
|---|---|---|---|---|
| G1 GroundLM FP / 1,133 | 449 / 0 / 469 | 244 / 99 / 171 | 322 / 89 / 182 | ≤ 11 |
| G2 SmishX TP / 226 | 145 / 0 / 123 | 51 / 37 / 73 | 70 / 27 / 53 | ≥ 157 |
| G3 campaign F1 | 0.975 / 0.388 / 0.975 | 0.943 / 0.938 / 0.950 | 0.954 / 0.913 / 0.946 | ≥ 0.971 |
| G4 ONNX bytes | 388 K | 388 K | 389 K | ≤ 10 MB ✓ |
| G5 test-donor legit FPR | 99.5% / 0.5% / 97.5% | 1.51% / 1.41% / 1.75% | 1.34% / 0.40% / 1.24% | ≤ 1% |
| Test-donor scam recall (n = 14) | — | 0 / 0 / 0 | 0 / 0 / 0 | (reported) |

R0 seed 2 collapsed (τ ≈ 1.0, nothing flagged), and seeds 1 and 3 hit the threshold-rule fallback.

**R2 vs R1, paired over 314 test-donor sender groups:** test-donor legit FPR −0.27 points
(95% CI −0.66 to −0.03). The sender/DLT feature lowers false alarms a little, and the interval
excludes zero. It does not rescue any gate.

## Confound: the scrub placeholders act as a scam shortcut

| Text | Share containing a `<TAG>` placeholder |
|---|---|
| Training SMS, legit (n = 68,686) | 0.7% |
| Training SMS, scam (n = 68,467) | 40.8% |
| Inbox, transactional / promotion / scam | 95.1% / 99.1% / 99.4% |

The training corpus's scam sources were de-identified with tags (`<NAMED_ENTITY>`, `<URL>`,
`<DATE_TIME>`, `<PHONE_NUMBER>`…), and its legit sources mostly were not. So v0 learned "has a tag →
scam". The inbox export scrubs every message into tags, which is why R0 scores inbox legit at a median
of 0.999 while it scores GroundLM legit at 0.04. R1 and R2 then learn the reverse for inbox-style tags,
"tag → legit". That plausibly explains the lost SmishX recall and 0/14 test-donor scam recall. On a phone
the model sees raw text, never tags, so none of the inbox numbers above measure deployed behaviour.

## Next step (needs a new pre-registered plan)

Replace placeholders with realistic surrogate values before training and scoring, the standard
de-identification practice: `<AMOUNT>` → a plausible rupee amount, `<URL>` → a synthetic domain,
`<PHONE>` → a random valid-format number, `<NAME>` → a sampled Indian name. Do this in **both** the
inbox data and the training scam sources, then re-run R0/R1/R2 with the same gates. Fixing the
training side also removes a shortcut v0 already depends on.
