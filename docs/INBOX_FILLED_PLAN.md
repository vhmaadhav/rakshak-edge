# Inbox retrain on filled text (pre-registration)

Written 2026-11-11 after `docs/SURROGATE_D0_RESULTS.md`, before any of these arms has run. **Approved
2026-11-11 by Claude under Maadhav's delegation of plan decisions that day ("you're free to redesign the
plan or architecture whichever u think is best"; "you dont need my permission for kaggle"). Maadhav has
not reviewed this file line by line.** Frozen as of 2026-11-11; later changes go in dated amendments.

## 1. Why a new plan, and what it does not do

`docs/SURROGATE_RETRAIN_PLAN.md` rested on H1 (the tag shortcut is the main cause of the inbox false
positives). D0 falsified H1, and its stop rule cancelled that plan's arms. This plan does not revive
them under the old hypothesis. It tests a different question that D0 did not touch:

R1 and R2 (inbox data added, scrubbed text) brought the test donor's legit FPR to 0.4–1.75% but lost
SmishX recall (27–73 of 226, against 157 needed) and campaign F1. In those arms almost every inbox row
carried a tag and almost every inbox row was legit, so "tag means legit" was available as a shortcut,
while SmishX and GroundLM have no tags. If that shortcut caused the recall loss, the same arms trained on
filled text should keep the FPR gain and recover recall.

The arms, data splits, caps, threshold rule and gates are those of `docs/INBOX_RETRAIN_PLAN.md`
(addenda 1–3) and `docs/SURROGATE_RETRAIN_PLAN.md` sections 3, 5, 6 and 7. Nothing is re-tuned. The only
change from the cancelled plan is the hypothesis set and the stop rules below.

## 2. Hypotheses (frozen)

- **H4 (the recall loss was the reverse shortcut):** S1 (inbox train donor added, filled text) has a
  higher SmishX TP count than R1 on every seed (R1: 51 / 37 / 73 of 226), and its seed median is at least
  100 of 226.
- **H5 (the FPR gain survives filling):** S1's test-donor legit FPR (G5) is at most 3% on the seed
  median (R1: 1.41–1.75%), and S1's GroundLM FP count (G1) is below S0's on the seed median.
- **H6 (S1 passes):** S1 passes all of G1–G5 under the section-7 rule. This is the only outcome that
  leads to an export.

## 3. Arms (3 seeds each: 1, 2, 3)

| Arm | Training data | Sender feature | Role |
|---|---|---|---|
| S0 | v0's data, filled | no | control for the fill alone |
| S1 | S0 + inbox train donor (phones 2+3), filled | no | the test of H4–H6 |
| S2 | S1 | yes | runs only if S1 meets H4 |

S0 and S1 run in parallel. `kaggle/jobs/sms_inbox_retrain.py` with the S-arm names (which imply
`--fill-surrogates`), Kaggle T4, one kernel per arm.

## 4. Gates and threshold rule

Unchanged: G1 GroundLM FP ≤ 11 / 1,133 · G2 SmishX TP ≥ 157 / 226 · G3 campaign F1 ≥ 0.971 (on the filled
campaign test split; v0 on the same filled split is reported beside it) · G4 ONNX ≤ 10 MB · G5 test-donor
legit FPR ≤ 1%. An arm passes if each gate holds on the seed median and on at least 2 of 3 seeds. The
threshold rule is `rakshak.data.retrain_prep.threshold_rule`, set before any test set is scored.

Note on G3, stated before the runs: D0 showed v0's IMC25 recall falls from 97.9% to 87.8% on filled text,
so v0 itself may no longer meet 0.971 on the filled split. The gate is not moved. If no arm can meet G3
on filled text while v0 also fails it there, that is reported as a property of the split, and no arm is
called a winner.

## 5. Stop rules

1. If S1 fails H4 (SmishX recall not recovered), the reverse-shortcut explanation is wrong. S2 does not
   run. Report and stop.
2. If S1 meets H4 but fails a gate, S2 runs once (the sender feature lowered FPR in R2 vs R1). No other
   arm, seed or threshold is added.
3. Nothing is exported to the Pi unless an arm passes every gate.

## 6. What a pass and a fail would mean

- **S1 or S2 passes:** export the median seed, check Pi parity and latency, and report cross-source
  numbers (GroundLM, SmishX, phone 1) next to v0.
- **H4 met, gates failed:** filled inbox data is the right direction and more real legit data or a
  larger student is the next variable. That would need a new plan.
- **H4 failed:** adding inbox negatives costs scam recall for a reason other than tags (for example the
  student's capacity, or label noise in Codex's labels). The next plan would test a larger student
  distilled from the teacher on the same filled data.

## 7. Limits

Same as the cancelled plan's section 9: one surrogate realisation, one donor person, Codex labels, 14
test-donor scam rows.
