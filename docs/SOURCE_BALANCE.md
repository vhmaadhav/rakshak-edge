# Source-balanced SMS student controls

Experiment branch `ci/source-balance-20261101`, based on `2463245`. Completed results: [SOURCE_BALANCE_RESULTS.md](SOURCE_BALANCE_RESULTS.md). Both interventions
miss the predeclared ScamShield gate; no intervention or new model is shipped. PR #2 is left for Maadhav to review.

## Predeclaration

Hypothesis: equal training-source sampling, with balanced+group-DRO as a second arm, improves the
ScamShield-held-out dual-base student's F1 (~0.75; teacher ~0.87) without harming the other folds.
Go requires a positive lower 95% paired campaign-bootstrap bound on ScamShield mean F1 and no
other source_holdout fold loss greater than half its baseline F1 CI width. Otherwise do not ship.
Report every fold and campaign/lang_shift/temporal, including missing/degenerate results.

Arms: ERM; inverse source-frequency replacement sampling; that sampler plus exponentiated-gradient
source weights (eta=.01). Each has the same number of updates per epoch. Weights use training
source groups only. Groups absent from a minibatch get no update, and objective weights normalize
over present groups. This is an adaptation of [Sagawa et al.](https://arxiv.org/abs/1911.08731),
not an exact reproduction. Early stopping and weight decay remain fixed across arms.

Three fixed seeds 1/2/3 per arm. Same dual-base with rule features, AdamW lr .002/weight decay
.0001, max15 epochs/patience3 by cal PR-AUC, distillation alpha .7/temperature2. Teacher transfer
files are matched to each run and hash/membership-checked. The corrected experiment uses only
the original frozen training IDs and their teacher logits: `include_unlabeled=False`. The inherited
broad KD pool is audited for excluded-row copies/campaign neighbours/external overlaps and never
used for this comparison. Historical ~0.75 is a reference; fresh train-only ERM is the paired
control. No test/cal row is added to training.
Calibration selects stopping checkpoints and thresholds. Test/external outcomes select nothing.

**In-sample for rule-feature design:** ScamShield test rows informed the frozen evidence rules.
The learned-student training partitions are unchanged, but an end-to-end independent held-out
improvement claim would ignore that exposure. Report paired changes on the named frozen splits
with this caveat and keep independent Indian validation open.

The internal primary label remains the existing spam-inclusive `label_binary`, allowing comparison
with prior student numbers. This experiment does not resolve the spam-versus-scam labelling task.
Campaign-fitted models also report external GroundLM and SmishX FPR/recall for both available
label endpoints at calibration-frozen max-F1 and 1%-FPR thresholds. GroundLM rows informed rule
feature design: retain that in-sample caveat. SmishX is independent, non-Indian-specific. PhreshPhish
is a URL protocol and does not apply to this SMS model. Independent Indian validation stays open.

One campaign bootstrap draw is applied to both arms and all three seed predictions; the statistic
is the mean of the three seed F1 values. Intervals describe campaign uncertainty conditional on
these seeds. Per-seed metrics and training histories are retained alongside pooled comparisons;
this is not a confidence interval over an unlimited population of training seeds.

After successful completion, `sms_source_audit` applies the same acceptance rule to the
hash-pinned saved comparisons without inference or tuning. It requires all three frozen source folds (UCI, Mishra–Soni, ScamShield), and refuses
missing seeds or comparisons. Both intervention arms are reported; intervals/tests are unadjusted for the two
predeclared comparisons. A failed gate means do not ship that intervention. Full split and
per-seed external results remain in the source kernel's metrics regardless of the decision.

## Run register

`kaggle/launch.sh sms_source_balance --run-name c-src-20261101 --gpu t4 --internet
--datasets maadhavvh/rakshak-data maadhavvh/rakshak-teacher-logits
--job-args="--seeds 1 2 3 --epochs 15 --n-boot 1000 --teacher-seed 0"`

Initial private kernel version1, source tree `8a3a0189e52a`, dirty base `2463245`: **cancelled by
the user before model outcomes were inspected**, after code review identified the inherited
broad KD pool risk. It cannot support the source-generalization gate. Its event subscription
is retired on confirmed cancellation; a killed container cannot send the completion callback.
The corrected train-only run gets a new run name and one completion subscription.
Predictions and metrics stay in private Kaggle outputs.

Development tests: three passed, three skipped because local torch is absent. The job checks
synthetic group-loss gradients and distillation equivalence on Kaggle before reading corpus data
or training. Lint clean. No local fitting or benchmark evaluation.

Corrected run: `sms_source_balance-c-clean-20261101`, private version1/tree `62a788d27aa9`,
args unchanged, train-only pool. Cancellation of the old run was acknowledged once; its
partial checkpoint (2,274.3 s, runner status `running`) is quarantined, not a completed result.
No outcomes from it were used for model or parameter choices. One corrected-run event waiter.

Final decision: **kill balanced and DRO**. Retain the train-only KD guard; full negative external
results and the confirmed inherited-pool contamination are in the result report. No tuning on
test/external outputs. Independent Indian validation remains open.
