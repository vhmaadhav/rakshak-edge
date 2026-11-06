# GroundLM false-positive rate of the full edge decision path (laptop, no LLM)

Date: 2026-11-06. Script: `deploy/pi/service/eval_groundlm.py`. Tests: `tests/test_eval_groundlm.py`.

**In-sample. GroundLM is exposed:** `docs/GROUNDLM_FPR.md` records that these rows informed rule
design (the urgency downgrade and the apk_install follow-up). Nothing here is an independent test,
and no threshold below is a shipped threshold.

## What was measured

- Decision path: SMS student ONNX (`kaggle/out/sms_student/output/best_student.onnx`, sha256 matches
  the registry) with the repo's featurisation (`rbench.runners.onnx.SmsRunner`, batch size 1), then
  `run_rules`, `edge_decision.decide_edge` (the state `score >= 0.35127517580986023` gives scam, else
  uncertain if any evidence, else benign; `build_package`; the detector-only mode; `choose_action`).
  The Llama explainer is not run. It does not change risk or action.
- `decide()` calls `edge_decision.decide_edge`, the same function `edge_pipeline.EdgeAnalyzer` calls,
  so there is no copy to drift. `tests/test_eval_groundlm.py` checks that both give the same decision
  on the same inputs, in both modes.
- **Detector-only mode (2026-11-06 decision: quiet).** `--mode quiet` (the default) gives a
  detector-only flag (score at or above the threshold, no rule evidence) risk LOW and action IGNORE,
  with no visible notice. `--mode verify` gives UNCERTAIN and VERIFY_INDEPENDENTLY, which is a notice.
  The cached scores hold the probability and the evidence-derived fields. `apply_mode` sets the
  decision for the mode, and only the no-evidence scam rows change. Checked: all 24,476 cached rows
  match a fresh `decide` in verify mode (0 mismatches), and `tests/test_eval_groundlm.py` checks
  `apply_mode` against a fresh decide in both modes.
- Inputs: `data/processed/external/external_groundlm.parquet` and `external_smishx.parquet`,
  sha256 equal to the registry and to `kaggle/out/rakshak-data/external/`. Campaign test:
  `data/processed/sms.parquet` with `splits/campaign.json` (`test`) and `eval_mask`.
- Run: ONNX on CPU, 1 thread, about 25 s for 24.5k messages. Output is outside the repo.

## Counts check (passed, matches the datacard)

| Quantity | Got | Expected |
|---|---:|---:|
| Rows | 1,393 | 1,393 |
| Policy negatives | 1,294 | 1,294 |
| Scoreable negatives (text and eval_ok) | 1,133 | 1,133 |
| Hash-only (no text) | 169 | 169 |

Discrepancy in the older doc: `docs/GROUNDLM_FPR.md` section 3 says "text present 1,233". The file has
1,393 − 169 = **1,224** rows with text (1,133 of them negatives). The 1,233 figure is wrong and the
counts above are correct.

## User-visible warning: quiet (default) vs verify (2026-11-06)

What the user would see. A warning is any action other than IGNORE (alert f). Scores are the cached
laptop ONNX run, unchanged; only the detector-only mode differs. Wilson 95% CI.

| Measure | quiet (default) | verify | Denominator |
|---|---|---|---|
| GroundLM visible warning FPR, alert (f) | **2 = 0.18% [0.05, 0.64]** (meets 1%) | 472 = 41.66% [38.82, 44.55] | 1,133 policy negatives |
| Detector-only flags among the negatives (hidden in quiet, shown in verify) | 470 hidden | 470 shown | 1,133 |
| SmishX recall, alert (f) | 53 = 23.45% [18.40, 29.39] | 173 = 76.55% [70.61, 81.60] | 226 smishing |
| SmishX ham FPR, alert (f) | 22 = 3.63% [2.41, 5.44] | 58 = 9.57% [7.48, 12.17] | 606 legitimate |
| Campaign test recall, alert (f) | 16.20% | 98.12% | 11,193 positives |
| Campaign test FPR, alert (f) | 0.61% | 2.22% | 10,952 negatives |

Detector-only counts (score at or above the threshold, no rule evidence, same in both modes):
GroundLM negatives 470, SmishX smishing 120 and legitimate 36, campaign test positives 9,170 and negatives 176.

**Reading the table.** Quiet meets the 1% visible-warning line on GroundLM (2 of 1,133). The price is recall:
student-only scams are not shown. SmishX recall on the visible notice drops from 76.6% to 23.5%, and campaign
recall from 98.1% to 16.2%. Verify keeps the recall and shows about 42% of the GroundLM negatives. The
detector needs fixing on bank, UPI and telecom notices before verify mode can be shown. The threshold cannot do
it (see below). The numbers above are the descriptive run, and the GroundLM set is still in-sample.

## FPR on 1,133 scoreable policy negatives (Wilson 95% CI)

Rows (a) to (f) are the verify mode (the Fix B notice). The quiet row is the default decision.

| Alert | Definition | FP / 1,133 | Rate | Wilson 95% CI | ≤ 1%? |
|---|---|---:|---:|---|---|
| (a) | student score ≥ threshold | 471 | 41.57% | [38.73%, 44.46%] | no |
| (b) | any rule evidence | 12 | 1.06% | [0.61%, 1.84%] | no |
| (c) | risk ≥ MEDIUM (warning) | 1 | 0.09% | [0.02%, 0.50%] | yes |
| (d) | risk ≥ UNCERTAIN | 472 | 41.66% | [38.82%, 44.55%] | no |
| (e) | risk = HIGH only | 1 | 0.09% | [0.02%, 0.50%] | yes |
| (f) | action ≠ IGNORE | 472 | 41.66% | [38.82%, 44.55%] | no |
| (f), **quiet (default)** | action ≠ IGNORE, detector-only flags quiet | 2 | 0.18% | [0.05%, 0.64%] | **yes** |

The rule-only numbers reproduce `docs/GROUNDLM_FPR.md` (12 with evidence, 1 above LOW).

**Reading the table.** In verify mode only (c) and (e) meet 1%, and only because the rules-driven warnings
are rare. The student alone is far above the line. In quiet mode the visible warning, (f), also meets 1%
(2 of 1,133): the 470 student-only flags are no longer shown. The 2 that remain are rule evidence (one
MEDIUM, one UNCERTAIN). This is the number the user would see. Of the 472 negatives at UNCERTAIN or above, 470 have
no rule evidence. Fix B turns those into UNCERTAIN with VERIFY_INDEPENDENTLY, so a student-only
false alarm shows up in (d) and (f), not in (c). With the shipped threshold, the student-only flags
are the whole problem.

## SmishX (independent, non-Indian, eval_ok rows)

Recall is on smishing (n = 226). FPR is on legitimate (n = 606). Spam (n = 275) is promotion and
is not a scam or a ham here.

| Alert | Recall (smishing 226) | Ham FPR (legitimate 606) |
|---|---|---|
| (a) student | 157 = 69.47% [63.18, 75.11] | 55 = 9.08% [7.04, 11.63] |
| (b) rule evidence | 58 = 25.66% [20.41, 31.73] | 30 = 4.95% [3.49, 6.98] |
| (c) risk ≥ MEDIUM | 39 = 17.26% [12.89, 22.72] | 19 = 3.14% [2.02, 4.84] |
| (d) risk ≥ UNCERTAIN | 173 = 76.55% [70.61, 81.60] | 58 = 9.57% [7.48, 12.17] |
| (e) HIGH only | 5 = 2.21% [0.95, 5.07] | 9 = 1.49% [0.78, 2.80] |
| (f) action ≠ IGNORE | 173 = 76.55% [70.61, 81.60] | 58 = 9.57% [7.48, 12.17] |
| (f), **quiet (default)** | 53 = 23.45% [18.40, 29.39] | 22 = 3.63% [2.41, 5.44] |

Reconciliation: with legitimate plus promotion as negatives (881), (c) is 53/881 = 6.02% and (e) is
10/881 = 1.14%. These match the audit's SmishX warning 6.0159% and HIGH 1.1351% in
`docs/GROUNDLM_FPR.md`.

**Trade-off.** The rules-based warning (c) keeps SmishX FPR at 3.1% but catches only 17% of
smishing. The student's score catches 69% of smishing at 9.1% ham FPR. Neither reaches 1% FPR with
useful recall on SmishX.

## Where the FPs are

By rule type (among the 472 flagged at (d)):
- 470 have no rule evidence: student-only.
- 12 total evidence hits: 11 urgency-only (at (b)), 1 apk_install (a bare "download" verb in a
  Devanagari conversational message).

By category (flagged / all negatives in that category):

| Category | Flagged at (d) |
|---|---|
| Bank/account | 161 / 161 (100%) |
| UPI/Payment | 146 / 146 (100%) |
| Other | 161 / 800 |
| EMI/loan | 4 / 5 |
| URL/click, OTP, Reward/lottery | 0 / 21 |

Bank and UPI legitimate messages are flagged by the student at every score that matters. That is
not a rules problem.

By student-score band (negatives; FP under (a) and (d)):

| Score band | n | FP (a) | FP (d) |
|---|---:|---:|---:|
| [0, 0.05) | 596 | 0 | 1 |
| [0.05, 0.20) | 58 | 0 | 0 |
| [0.20, 0.3513) | 8 | 0 | 0 |
| [0.3513, 0.5) | 8 | 8 | 8 |
| [0.5, 0.8) | 60 | 60 | 60 |
| [0.8, 1.0] | 403 | 403 | 403 |

Most false alarms are confident (403 of 471 at ≥ 0.8), so this is a calibration and training
problem, not a threshold-placement one.

## Top 10 FP patterns (repo template groups; paraphrased, masked)

Counts are among the 472 flagged at (d). Templates are the repo's `template_key`, with numbers,
amounts and phones already masked.

1. Bank account balance statement, with masked account and amount (n = 34).
2. Bank debit notification with available balance (n = 27).
3. Telecom data-balance and active-plan notice, four variants (n = 18 + 12 + 12 + 10).
4. Telecom plan-renewal reminder, three variants (n = 11 + 8 + 7).
5. Utility bill due notice with a UPI pay instruction (n = 7).

The ten listed templates account for 146 of 472. The rest are spread over other bank, telecom,
and UPI templates.

## Descriptive threshold check (not a shipped threshold)

Smallest student threshold with FPR ≤ 1% on (a): 0.99793 (11 of 1,133 = 0.97%).

| Threshold | GroundLM FPR (a) | GroundLM fraud recall (91) | SmishX scam recall | SmishX ham FPR | Campaign test recall | Campaign FPR |
|---|---:|---:|---:|---:|---:|---:|
| 0.3513 (shipped) | 41.6% | 89.0% | 69.5% | 9.1% | 98.0% | 1.80% |
| 0.5 | 40.9% | 86.8% | 61.5% | 8.3% | 97.2% | 1.18% |
| 0.8 | 35.6% | 67.0% | 35.8% | 5.3% | 94.0% | 0.47% |
| 0.95 | 19.2% | 33.0% | 20.8% | 2.8% | 89.1% | 0.19% |
| **0.9979** | **0.97%** | **3.3%** | **5.8%** | **1.0%** | **71.9%** | **0.04%** |

Reaching 1% on GroundLM by threshold alone costs about 26 points of campaign-test recall and
drops SmishX scam recall to 6%. It would also remove almost all fraud recall on GroundLM. A
threshold cannot fix this. The student needs retraining or the bank and UPI negatives need
calibration data. The campaign test recall is the repo's own held-out split. Both recall numbers
are descriptive and the threshold was not chosen on any of these sets.

**Caveat.** GroundLM is in-sample (`docs/GROUNDLM_FPR.md`). Any threshold chosen to meet 1% on it
is not an independent test. The independent check is SmishX, and that is the only independent
evidence in this doc.

## Reproduce

```bash
.venv/bin/python deploy/pi/service/eval_groundlm.py run --out-dir <scratch dir> --rescore [--mode quiet|verify]
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q tests/test_eval_groundlm.py tests/test_rbench_metrics.py
```

Outputs (`scores.parquet` and `metrics.json`) are written to the `--out-dir` given. They are not
committed.

## Limits

- Scores come from the laptop ONNX run, not the Pi. The Pi agent should confirm the decision
  matches on a few rows before quoting Pi numbers.
- The explainer and the verifier are not in this path, and they do not change risk or action.
- The student threshold is the shipped one, not re-fit.
