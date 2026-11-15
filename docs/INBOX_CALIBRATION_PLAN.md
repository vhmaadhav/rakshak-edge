# Inbox calibration of the W1e encoder: threshold on real Indian legit SMS, no training (pre-registration W1f)

Written 2026-11-12, before `kaggle/jobs/sms_inbox_calibration.py` exists and before any model is scored on the
inbox, the calibration set, or any test set for this plan. Follows `docs/ENCODER_EXPOSURE_RESULTS.md`, "What is
binding now". Kaggle only (T4); no Modal. Frozen once the job is built; later changes are dated amendments written
before the result they could affect.

## 1. Question

W1e (`docs/ENCODER_EXPOSURE_RESULTS.md`) ranks well, but its threshold was placed on held-out CloveAI/FinEE rows that
come from the same sources it trained on, so it sat at a logit margin of -9 to -13 and was killed. The open question:
does a threshold placed on **real Indian legit SMS that no model trained on** (the consented train donor, phones 2
and 3) give an operating point that passes the deployment gates on GroundLM, SmishX and the held-out test donor
(phone 1)? No model is trained in this plan. Only the threshold is new.

## 2. Models (no training)

| Name | Checkpoint (sha256, from its kernel outputs) | Role |
|---|---|---|
| `w1e-s0` | `355b6519680a79d3d05ee5917692b3cfa18b878c0610b494ca6f2b2231458189` | W1e seed 0 (gated) |
| `w1e-s1` | `91cb21ee392bbab5ef08f63433d0215ceed3bb4ec4394ad37ba68caf6f029313` | W1e seed 1 (gated) |
| `w1e-s2` | `2408940091dd9cc765c39804494f4f758119bacfeac2afef28df01a13128eed1` | W1e seed 2 (gated) |
| `teacher` | `673b970e9f4746dbbbb5a00aa84eb056d6750cbc2897ba712ab1aa1b234abc7e` | base mmBERT teacher, reference only, no verdict |

The W1e checkpoints live in the W1e kernel output (`maadhavvh/rakshak-sms-encoder-exposure-w1e`, seed*/best_checkpoint).
The job finds each one by its sha256 under the attached inputs and asserts exactly one match per pin. The pins are
the `checkpoint_sha256` values in that kernel's `metrics.json`, and a unit test checks them against the local copy.
Scoring uses `sms_teacher_suite.teacher_margins` (the logit margin logit_1 - logit_0, canonical input), as in W1e.

## 3. Calibration sets (the threshold set; the only inbox text any model sees)

Source: `maadhavvh/rakshak-inbox`, `train_donor.parquet` (phones 2 and 3, Codex-labelled). The labels and filter are
those of `sms_inbox_retrain.label_inbox`: `benign_transactional` and `promotion` are legit (0); `scam` (1) is
dropped here; `benign_personal` and `unclear` are dropped.

- **Primary, filled.** The text goes through `sms_inbox_retrain.fill_inbox_frame` (rakshak.data.surrogates.fill_many,
  seed `SURROGATE_SEED` 20261107, keyed by row id), exactly as the S arms did, then `label_inbox`. Rule 5 of the
  surrogate plan applies: no `<TAG>` may remain, or the job stops.
- **Secondary, raw.** The scrubbed text as stored, with `label_inbox` only.
- **Set:** every legit train-donor row after the filter. No sender-ID hold-out and no dedup, because none of the four
  models trained on inbox rows (W1e used CloveAI and FinEE only; the teacher used the campaign train split). Counts
  go to `prep_report.json`.
- **Stop rule (before any scoring):** fewer than 500 legit rows after the filter stops the job with
  `status: "stopped: insufficient data"`, the same rule as `docs/INBOX_RETRAIN_PLAN.md` section 4.

## 4. Threshold (set before any test scoring)

- Score `m = logit_1 - logit_0` (float64) per row; alarm rule `m >= tau`.
- `tau_filled` (primary) and `tau_raw` (secondary) are the smallest margin with at most `floor(0.01 x n)` calibration
  legit rows at `m >= tau`, computed by `sms_encoder_exposure.logit_threshold`. The value is `-inf` when every
  negative may flag, and that is reported, not hidden.
- For all four models, **both calibration sets are scored first, both thresholds are computed, and
  `thresholds.json` is written. Only then is any test set scored.** The job checks this order, and the unit test
  for the order runs with fake scorers.
- Nothing is chosen from test scores. No arm, seed, threshold or set is added after a result.

## 5. Test sets (scored once, after thresholds.json is written)

| Set | Form | Used in |
|---|---|---|
| GroundLM policy negatives (1,133) | scrubbed (as stored) | G1; per-category FP; McNemar vs XLM-R |
| GroundLM positives with text (91) | scrubbed | G3; McNemar vs XLM-R |
| SmishX smishing (226) and ham (606) | scrubbed | G2; ham FPR (reported) |
| Campaign test, clean rows | scrubbed | G4; campaign all-row F1 (reported) |
| Campaign test, filled | surrogate fill of the campaign rows (the S-arm G3 surface) | reported only |
| lang_shift, temporal | scrubbed | reported |
| Phone 1 (test donor) legit and scam | **filled** (primary) and raw (secondary) | G5 (legit FPR); scam recall (reported) |

**Campaign clean rows.** For the teacher, "clean" means not in its own training partition (the fold train split,
from `sms_teacher_suite.build_sets`). For each W1e seed, "clean" is the `clean` column of W1e's own `probs.parquet`
campaign rows, which records clean against W1e's actual training partition (fold train plus the extra rows minus the
calibration hold-out). The job asserts that those ids match this job's campaign ids, in order. The clean mask is the
same for all three seeds, and the job checks that too.

**Why the campaign set is scrubbed.** The models trained on scrubbed text, and G4 is the W1e definition
(`ENCODER_EXPOSURE_PLAN.md` section 7). The filled campaign F1 is reported beside it, so the S-arm surface is visible.

## 6. Gates (primary arm, W1e seeds only)

| # | Gate | Pass |
|---|---|---|
| G1 | GroundLM policy negatives flagged (at tau_filled) | <= 11 of 1,133 |
| G2 | SmishX smishing flagged | >= 157 of 226 |
| G3 | GroundLM positives recall | >= 0.90 (>= 82 of 91) |
| G4 | Campaign clean-row F1 (scrubbed) | >= 0.9713 |
| G5 | Phone 1 legit FPR (filled) | <= 1 % |

Each gate holds when its seed median passes **and** at least 2 of 3 seeds pass (the median of three values is the
middle one). The arm passes only when G1 to G5 all hold. The job reports "incomplete" unless all three seeds are present.
The teacher row is computed with the same gates and is labelled reference-only; it has no verdict.

## 7. Reported, not gated (per model and per arm)

- Counts, rates and Wilson 95 % intervals for G1, G2, G3, G5, the ham FPR, and scam recall on phone 1 (14 rows).
- Campaign all-row F1 and clean F1 with campaign-clustered bootstrap intervals (2,000 resamples, `default_rng(0)`).
- lang_shift F1 and temporal precision (same bootstrap).
- Paired McNemar (exact, two-sided) against the released GroundLM XLM-R decision on the 1,133 negatives and 91
  positives; per-category GroundLM false-positive counts.
- Phone 1: template-novel legit FPR, meaning phone-1 legit rows with no near-duplicate (Jaccard >= 0.8) among the
  calibration legit rows of the same arm. Counts are reported with the template-novel subset.
- The secondary (raw) arm, all of the above, at tau_raw.
- The campaign filled F1 at tau_filled.
- Contamination counts per set; `prep_report.json`.

## 8. Privacy (binding)

- No inbox text, sender string or inbox row id is written anywhere. Inbox rows are indexed `calib_legit_<i>` and
  `phone1_<i>`. Logs, `prep_report.json`, `thresholds.json`, `metrics.json`, `report.md` and `probs.parquet` carry
  counts, labels, margins, probabilities, thresholds and intervals only.
- `probs.parquet` has no text column and no sender column. A unit test scans every written output for sentinel text
  and sender strings.
- Errors are logged as exception type and frames only (`REDACT_ERRORS`, the same redaction as `sms_inbox_retrain`).
- Nobody opens inbox rows on the laptop. The job runs on Kaggle only.

## 9. Tests (`tests/test_sms_inbox_calibration.py`, synthetic inputs only)

- Threshold: smallest tau with at most floor(0.01 n) negatives flagged; `-inf` when every negative may flag.
- Calibration selection: filter and label map; `benign_personal` and `unclear` dropped; filled text has no `<TAG>`;
  raw text keeps the tags.
- Gate aggregation: median plus 2 of 3; one bad seed does not fail a gate when the median and two seeds pass; an
  incomplete seed list reports incomplete; G5 uses the 1 % limit.
- Two-pass order: with fake scorers, `threshold` is written before every test set is scored, and no test set is
  scored before it.
- Clean-mask alignment: W1e probs clean masks are accepted only when the ids match, in order.
- Pins: the job's W1e shas equal the W1e kernel's `metrics.json` (skipped if the local copy is absent).
- Privacy: the probs frame has no text or sender column; a full synthetic run writes every output and none contains
  a sentinel text or sender; a raised message carrying text is redacted.
- `kaggle/build_kernel.py` has a `JOB_DEPENDENCIES` entry for the job.

## 10. Limits (fixed before the run)

- Train donor and test donor are the same person (phones 2+3 and phone 1). Banks, templates and sender IDs overlap.
  G5 is an optimistic, same-person estimate, not a population estimate. The template-novel subset is reported for that
  reason.
- Labels are Codex labels on scrubbed text, not human labels. 14 test-donor scams.
- The calibration legit set is about 4,900 rows from one person; tau is set on it, so the threshold is only as good as
  that sample.
- W1e is same-family (CloveAI is the family of GroundLM's test pool). Gains on GroundLM are not cross-source.
- 91 GroundLM positives: the G3 interval is wide and is reported.
- Three seeds, one data seed, one fold. A pass here is not a population claim.
- A pass would still need W1c (ONNX parity), W1d (Pi), and a FinEE provenance review before any shipping. This job
  decides none of those.
