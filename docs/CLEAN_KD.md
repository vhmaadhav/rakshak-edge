# Clean distillation rerun — predeclaration 2026-11-01

Historical Hinglish KD0.80–0.82 is **WITHDRAWN** as clean evidence: transfer-pool
contamination reintroduced excluded rows. Existing train-only C controls all used
KD, and therefore do not answer whether KD beats a matched no-KD student.

Hypothesis: train-only soft teacher targets improve language-shift F1 relative to
hard-label-only training without harming any frozen fold. Experiment: six frozen
campaign/source_holdout(each of3)/lang_shift/temporal runs, seeds1/2/3 paired in
both arms. Existing dual/base student with rule side features; AdamW.002,
weight_decay.0001, batch128,15epoch cap/patience3, cal PR-AUC checkpoint selection,
cal-only max-F1 and1%-FPR thresholds. No test/external/human sample selection.
Control output arm `erm` means **hard-label-only, no KD** in THIS experiment;
`kd` uses alpha.7/temperature2. Config explicitly records teacher-enabled arms.
No teacher on control means full BCE, not.3-scaled BCE. Same ordered original
train IDs/labels/batch seeds/class weighting/early-stop recipe in both arms.

Use hash-manifested existing mmBERT seed0 logits, reindexed ONLY to each fold's
exact frozen train IDs by build_training_pool. Reject missing teacher coverage,
extra rows or any test/removed-transfer inclusion. Never use inherited unlabeled
transfer rows for training. Cached teacher targets are **in-sample teacher outputs
on its own training data**, not out-of-fold targets. Teacher checkpoint/HPO used
cal only. Source code confirms train_once receives run.train; other folds'
teachers cannot be naively ensembled because they may have trained on this fold's
test rows. A new inner-fold ensemble requires separate train-only teacher fits;
optional OOF work is deferred behind this minimal matched-control rerun and the
two existing subscriptions, with no OOF/ensemble claim from this run.

Go exactly per user requirement: lower95% campaign-clustered CI of KD lang_shift
mean F1 over the three fixed seeds must be **above the no-KD control point mean**;
every other frozen fold must lose no more than the no-KD CI half-width. Report
paired campaign-bootstrap KD-minus-control CIs and null-centred p-values for every
fold (one resampled campaign multiplicity shared by both arms/allseeds). A failure
kills this arm. Missing fold/seed/undefined required CI is unmeasurable, never go;
known failures still kill an incomplete run. Conditional-on-three-seeds CIs do not
estimate uncertainty over all random initialisations. No threshold/hyperparameter
changes after outcomes. No synthetic or test/external rows added to training.

Report all per-seed/cal-threshold metrics and both legacy spam-inclusive and known
fraud external endpoints as **student diagnostics**, not deployed/fused FPR.
GroundLM rule design and GroundLM/SmishX intervention feedback remain in-sample.
Full-system external claims must cite frozen v1 (newly fitted, not previously
shipped); these detector-only comparisons cannot replace it. Independent Indian
held-out validation remains OPEN. PhreshPhish URL-only protocol is unchanged by
this SMS job; no new PhreshPhish result is implied. Save each selected student
checkpoint with exact kernel/tree/run/seed/arm/teacher-enabled metadata privately;
no weights or scores in git. No GPU launch until P1 capacity and current jobs permit.
