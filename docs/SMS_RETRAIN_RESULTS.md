# SMS student retrain: results (2026-11-07)

Plan: `docs/SMS_RETRAIN_PLAN.md`, pre-registered and approved before launch, with two amendments made before launch
(two-level clustering; SMS-Shield and IEEE DataPort unavailable). Kaggle T4 kernels `maadhavvh/rakshak-sms-retrain-{a,b,c}`,
version 1, data `maadhavvh/rakshak-data` (version of 2026-11-07 with `external/indian_legit/`). Outputs are in
`kaggle/out/sms_retrain-{a,b,c}/output/` (`metrics.json`, `thresholds.json`, `prep_report.json`, `probs.parquet`, one
ONNX per seed). Each kernel ended with `status: error` (`KeyError: 'fallback'`) in the final `report.md` render, after
`metrics.json`, `probs.parquet` and the ONNX files were written. The key is fixed in `kaggle/jobs/sms_retrain.py`. No
number below depends on the failed render.

## Verdict: all three arms KILLED on the pre-registered gates

No arm passes G2 (SmishX smishing recall) or G3 (campaign F1). So there is no winner, and plan step 4 (export, Pi
parity, Pi latency) is not run. Per the plan, the gates are not re-tuned and the arms are not re-run.

### Gated results (pre-registered threshold rule, §4)

| Arm | Training extra | τ per seed | G1 GroundLM FP / 1 133 (≤ 11) | G2 SmishX TP / 226 (≥ 157) | G3 campaign F1 (≥ 0.9713) | G4 ONNX | Pass |
|---|---|---|---|---|---|---|---|
| v0 (shipped) | – | 0.3513 | 471 | 157 | 0.9813 | 388 KB | reference |
| A: + FinEE synthetic legit (cross-source) | 9 387 legit | 0.9996 / 0.9989 / 0.9952 | 2 / 6 / 1 ✔ | 18 / 27 / 31 ✘ | 0.852 / 0.879 / 0.907 ✘ | 388 KB ✔ | **KILL** |
| B: A + 2 000 minimal pairs (cross-source) | 9 387 legit + 2 000 scam twins | 0.9989 / 1.0000 / 0.9965 | 3 / 0 / 1 ✔ | 22 / 19 / 16 ✘ | 0.859 / 0.839 / 0.843 ✘ | 388 KB ✔ | **KILL** |
| C: A + CloveAI, GroundLM-stripped (same-source) | 9 387 legit + 157 scam | 0.9981 / 0.9934 / 0.9842 | 2 / 2 / 1 ✔ | 21 / 28 / 27 ✘ | 0.882 / 0.900 / 0.879 ✘ | 388 KB ✔ | **KILL** |

G1 "passes" only as a side effect of τ ≈ 0.99+, at which the models flag almost nothing. It is not evidence that
the false-positive problem is fixed.

**Why τ went to about 0.99.** The rule asks for ≤ 1 % FPR on the held-out Indian legit validation set (11 882 rows:
9 held-out FinEE split groups, skyaara, Ridham115 safe). At each model's campaign-cal max-F1 threshold, that set's
FPR is about 5-7 % (601-812 / 11 882). Getting it to 1 % takes a near-1.0 threshold. The validation set (unseen
synthetic templates plus Ridham's deliberately bank-like "safe" look-alikes) is much harder for the student than
campaign cal, and the rule as frozen does not trade recall against that.

### Diagnostic only, not gated: each model at its campaign-cal max-F1 threshold

These are not pre-registered and are not results. They show where each arm lands at a v0-like operating point.
SmishX here counts all 259 text-available smishing rows, not the 226 `eval_ok` rows used by G2.

| Arm | Regime for GroundLM | GroundLM FP / 1 133 (seeds) | SmishX smishing / 259 | Campaign F1 |
|---|---|---|---|---|
| A | cross-source | 362 / 275 / 312 (24-32 %) | 206 / 227 / 235 | 0.973-0.975 |
| B | cross-source | 372 / 325 / 304 (27-33 %) | 183 / 185 / 167 | 0.973-0.975 |
| C | same-source (CloveAI) | 44 / 112 / 95 (3.9-9.9 %) | 202 / 206 / 209 | 0.974 |
| `sms_groupneg` cross-fit (reference) | in-distribution, template-held-out | 11 (0.97 %) | – | – |
| GroundLM XLM-R 270M (published) | in-distribution, random split | 5 / 1 294 (0.39 %) | – | – |

What the diagnostics say (not claims):
1. **Synthetic licensed legit data does not transfer to GroundLM's real messages.** A and B move GroundLM FPR from
   v0's 41.6 % to about 24-33 %. That is far from 1 %.
2. **Minimal pairs (B) cost SmishX recall** without improving GroundLM over A.
3. **Same-source data (C, CloveAI) moves GroundLM a lot** (4-10 %), but it is the source GroundLM is drawn from, so this
   is not a cross-source result. Real template-held-out exposure (`sms_groupneg`) gets about 1 %.

The pattern is consistent: how close the training negatives are to the test's real distribution matters more than
volume or pairing.

## Data actually used (from `prep_report.json`)

- FinEE synthetic: 100 572 staged rows (24 507 ICICI-token rows dropped, 4 261 email/statement rows, 23 079 exact
  dups). 1 694 split groups, 28 407 cap groups. 9 387 rows in train (cap min(12 000, 0.25 × 37 550 negatives)).
- CloveAI (arm C): 19 230 staged. 184 split groups (10 113 rows) dropped for touching test-only text, then capped.
  157 scam rows entered train; the legit budget was already filled.
- Minimal pairs (arm B): 2 000 scam twins, and all six edit families fire their rule evidence.
- Test-only pool: 37 275 texts (campaign test, lang_shift, temporal, GroundLM, SmishX). No test text survived into
  train or validation (0 contaminated groups for arms A/B).

## Not done, and why

- Step 4 (ONNX export of a winner, Pi parity, Pi latency): no winner.
- SMS-Shield and IEEE DataPort test sets: not available (repository not found; paid subscription).
- Per-source breakdown of the legit-validation FPR: the job's row ids do not match the staged ids, so FinEE vs
  Ridham115 vs skyaara cannot be split from `probs.parquet`. Needs a source column in a future job.

## Open options (need a new pre-registration; nothing is decided here)

1. **Real, consented Indian legit SMS.** No licensed public corpus exists (OpenResearch audit, 2026-11-07). The
   repo already has a local-only scrub and manual-review export path (`docs/INBOX_EXPORT.md`). Team-donated inbox
   messages, scrubbed and reviewed, are the only route to real negatives we can ship.
2. **Re-specify the threshold validation set** (for example, drop Ridham115's adversarial look-alikes, or set the rule
   on a recall floor plus an FPR target). This would be a new pre-registered experiment, not a fix to this one.
3. **Rules-first gating for the SMS path** (as in the NIDS change). Rules alone get 12 / 1 133 = 1.06 % on GroundLM,
   and the shipped quiet mode shows 0.18 % visible warnings. The student's false positives are already mostly hidden
   at the system level. A system-level gate may be the right target for the competition demo.
