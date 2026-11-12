# Inbox retrain on filled text: results (2026-11-11)

Plan: `docs/INBOX_FILLED_PLAN.md` (pre-registered and frozen before launch). Kaggle T4 kernels
`maadhavvh/rakshak-sms-inbox-retrain-s0` and `-s1`, version 1, tree `72af0a86922f` (dirty), 3 seeds each,
about 12 minutes per kernel. Outputs: `kaggle/out/sms_inbox_retrain-{s0,s1}/output/`. Nothing exported to
the Pi.

## Verdict: H4, H5 and H6 fail. No winner. Stop rule 1 applies: S2 does not run.

| Gate (seeds 1 / 2 / 3) | S0 (filled, no inbox) | S1 (filled + inbox) | R1 (scrubbed + inbox, earlier) | Target |
|---|---|---|---|---|
| τ | 0.99999 / 1.00000 / 0.144 (fallback) | 0.99958 / 0.99970 / 0.99959 | – | rule |
| G1 GroundLM FP / 1,133 | 0 / 0 / 523 | 5 / 5 / 4 ✔ | 244 / 99 / 171 | ≤ 11 |
| G2 SmishX TP / 226 | 0 / 0 / 208 | 29 / 23 / 28 ✘ | 51 / 37 / 73 | ≥ 157 |
| G3 campaign F1 (filled test) | 0.136 / 0.182 / 0.973 | 0.782 / 0.722 / 0.730 ✘ | 0.943 / 0.938 / 0.950 (scrubbed) | ≥ 0.971 |
| G4 ONNX bytes | 388 K ✔ | 388 K ✔ | 388 K | ≤ 10 MB |
| G5 test-donor legit FPR | 0.24% / 0.71% / 99.5% | 4.70% / 4.87% / 3.73% ✘ | 1.51% / 1.41% / 1.75% | ≤ 1% |
| Test-donor scam recall (n = 14) | – | 0 / 1 / 0 | 0 / 0 / 0 | reported |

- **H4 fails.** S1's SmishX count is lower than R1's on every seed (29 / 23 / 28 against 51 / 37 / 73), and
  its median is 28, not 100. Filling the tags did not bring recall back, so the reverse shortcut ("tag
  means legit") was not what cost R1 its recall.
- **H5 fails.** S1's test-donor legit FPR median is 4.70% (limit 3%), and its GroundLM FP median (5) is not
  below S0's (0).
- **H6 fails.** S1 passes G1 and G4 only. G1 passes because τ is about 0.9996, at which the model flags
  almost nothing; it is not evidence that the false positives are fixed.
- S0 repeats R0's pattern: two seeds end at τ ≈ 1.0 and flag nothing, and the third hits the threshold
  rule's fallback and flags everything.
- **v0 on the filled campaign test split: F1 0.958** (n = 22,145, shipped threshold), against 0.981 on the
  scrubbed split. v0 itself does not meet G3 on filled text, as section 4 of the plan warned.
- Source-identity probe on training scam rows: accuracy 0.887 scrubbed, 0.871 filled (majority baseline
  0.399). The fill hides little of the source style.

## Diagnostic only, not gated and not pre-registered: ranking quality from `probs.parquet`

AUC of SmishX label-1 rows (smishing and spam, n = 578) against each legit set, per seed, and the false-positive rate
at the threshold that keeps 70% of those SmishX rows.

| Arm | AUC vs GroundLM legit (1,133) | AUC vs test-donor legit (2,976) | AUC vs SmishX ham | Campaign test AUC | GroundLM FPR at 70% SmishX recall | Test-donor FPR at 70% SmishX recall |
|---|---|---|---|---|---|---|
| S0 | 0.825 / 0.791 / 0.810 | 0.447 / 0.338 / 0.450 | 0.935 / 0.953 / 0.936 | 0.995 / 0.995 / 0.995 | 26.4% / 29.3% / 28.4% | 79.5% / 87.7% / 77.6% |
| S1 | 0.867 / 0.899 / 0.915 | 0.687 / 0.765 / 0.807 | 0.942 / 0.966 / 0.943 | 0.995 / 0.995 / 0.995 | 19.4% / 13.5% / 9.4% | 41.5% / 26.8% / 23.1% |

What the diagnostic says (not a claim):

1. **No threshold fixes this.** The student ranks its own campaign split almost perfectly (AUC 0.995) and
   ranks scam against Indian legit messages from another source poorly. Without inbox data it scores a
   real Indian inbox's legit messages above SmishX scam more often than not (AUC below 0.5).
2. **Real inbox negatives help the ranking** (test-donor AUC 0.34–0.45 to 0.69–0.81; GroundLM 0.79–0.83
   to 0.87–0.92), but 4,886 legit rows from one person do not make a 388 KB byte-level student separate
   the two classes across sources.
3. The gates fail for the same reason in every round so far (`SMS_RETRAIN_RESULTS`, `INBOX_RETRAIN_RESULTS`
   and this one): the threshold rule asks for 1% FPR on held-out Indian legit, the model cannot rank well
   enough to give it, and τ goes to about 1.0.

## Open next step (needs a new pre-registered plan; nothing is decided here)

The untested variable is the model, not the data mix or the threshold. Before any more student
retraining: score the existing SMS teacher (and the multilingual research candidate) on the same sets
(GroundLM, SmishX, filled test donor) and report the same AUCs. If the teacher ranks cross-source well
(for example AUC ≥ 0.98 against GroundLM and the test donor), distil a larger student from it on filled
data. If the teacher ranks as badly as the student, the limit is the training data, and more real Indian
legit SMS (more donors) is the only route.

## Limits

One surrogate realisation; one donor person; Codex labels; 14 test-donor scam rows. The AUC diagnostic
uses all 578 SmishX label-1 rows (smishing and spam) in `probs.parquet` per seed, not the 226 `eval_ok` smishing rows of G2.
