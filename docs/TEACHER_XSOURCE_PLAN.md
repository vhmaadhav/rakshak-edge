# Teacher cross-source ranking check D1 (pre-registration)

Written 2026-11-11 after `docs/INBOX_FILLED_RESULTS.md`, before the job exists. **Approved 2026-11-11 by
Claude under Maadhav's delegation of plan decisions; Maadhav answered "Yes, run it" to this check but has
not reviewed this file line by line.** Frozen as of 2026-11-11.

## 1. Question

Every student retrain so far fails for one reason: the 388 KB byte-level student ranks its own campaign
split almost perfectly (AUC 0.995) and ranks scam against Indian legitimate SMS from other sources
poorly (SmishX label-1 against GroundLM legit 0.79–0.92; against the filled test donor 0.34–0.81). Is that
a limit of the small student, or of the training data? The existing SMS teacher was trained on the same
public data as the student and has never been scored on GroundLM or the inbox.

## 2. What runs (scoring only, no training)

One Kaggle T4 kernel. Models, scored on the same rows:

- **Teacher:** `jhu-clsp/mmBERT-base` fine-tune, the `best_checkpoint` of kernel
  `maadhavvh/rakshak-sms-teacher` (`model.safetensors` sha256
  `673b970e9f4746dbbbb5a00aa84eb056d6750cbc2897ba712ab1aa1b234abc7e`). Inference exactly as
  `kaggle/jobs/sms_teacher.py` does it for the form the checkpoint was trained on. The job records which
  run (split) the checkpoint came from.
- **v0 student** (ONNX sha256 `5341e14c…`), for a same-row reference.

Sets, built with the same code as the S-arms (`kaggle/jobs/sms_inbox_retrain.py`): GroundLM policy
negatives (the 1,133 rows of G1), SmishX (the 226 rows of G2, all label-1 rows, and ham), test donor
(phone 1) scrubbed and filled, phones 2+3 held-out legit scrubbed and filled, campaign test scrubbed and
filled. Fill is `rakshak.data.surrogates`, seed 20261107. Inbox privacy rules as in the other inbox jobs:
counts, rates, AUCs and quantiles only.

Contamination check: the share of each evaluation set whose exact text (after canonicalisation) appears
in the teacher's training partition, and every AUC repeated on the rows that do not.

## 3. Measures (per model)

- AUC of SmishX label-1 rows against: GroundLM legit, filled test-donor legit, scrubbed test-donor legit,
  SmishX ham. Campaign test AUC, scrubbed and filled.
- False-positive rate on GroundLM legit and on filled test-donor legit at the thresholds that keep 70%
  and 90% of SmishX label-1 rows.
- Test-donor scam recall at those thresholds (n = 14, Clopper–Pearson 95%).
- Bootstrap 95% intervals for the two headline AUCs (2,000 resamples, rows resampled within class).

## 4. Decision rule (frozen before any number is seen)

Headline AUCs: **A1** = SmishX label-1 against GroundLM legit; **A2** = SmishX label-1 against filled
test-donor legit. Both on the non-contaminated rows.

- **Teacher transfers** if A1 ≥ 0.98 and A2 ≥ 0.98. Next plan: distil a larger student from this teacher on
  filled data, with the same gates G1–G5.
- **Teacher does not transfer** if A1 < 0.93 or A2 < 0.85 (no better than the best S1 seed, 0.915 and
  0.807). The public training data is then the limit for this model class too. Next plan: fine-tune the
  teacher on the S1 data (filled campaign train plus the inbox train donor) to test whether a pretrained
  encoder can use the inbox negatives where the student could not.
- **In between:** reported as partial. The next plan is the same fine-tune, and distillation is not
  started from the current checkpoint.

No threshold, set or measure is changed after the run. v0's numbers are reference only.

## 5. Limits

The teacher saw scrubbed (tagged) training text, so its scores on filled text mix transfer with the tag
shift; the scrubbed and filled columns are both reported for that reason. One donor person, Codex labels,
14 test-donor scam rows, one surrogate realisation. SmishX label-1 includes spam as well as smishing.

## Amendment 1 (2026-11-11, before the run)

Found while building the job, before any score exists:

- The kernel's `best_checkpoint` is not the campaign model. `kaggle/out/sms_teacher/output/metrics.json`
  (`meta.best_checkpoint`) says it is the fold `source_holdout:holdout_scamshield`, seed 0, trained on
  canonicalised text. The teacher job keeps only the checkpoint with the best cal PR-AUC across its runs.
  So this teacher never saw the `scamshield` source in training (85,680 of the 137,153 rows of
  `sms.parquet`, 36% scam). This kernel kept no campaign-split checkpoint (the Hugging Face store was not checked for one).
- The run goes ahead on this checkpoint as section 2 states, and the decision rule is unchanged. What
  changes is how far a negative result reaches: "does not transfer" here says this fold does not transfer,
  not that the model class cannot. The fine-tune named in section 4 is the test of the model class.
- The teacher's training partition for the contamination check is that fold's train rows.
- Choices made in the build that the plan left open: campaign test means the whole campaign test
  partition; thresholds for the 70% and 90% recall points are the k-th largest positive score with
  k = ceil(n × p / 100); the bootstrap uses `numpy.random.default_rng(0)` on the non-contaminated rows;
  phones 2+3 held-out legit is scored and reported but is not a decision measure; if A1 or A2 cannot be
  computed the job fails and gives no verdict.
