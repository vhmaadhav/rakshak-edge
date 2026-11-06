# E0: three-seed URL controls, calibration outputs only — 2026-11-02

Hypothesis: reproducible, fully saved char-CNN controls and their calibration
predictions are needed to decide whether a stronger teacher justifies KD without
using PhreshPhish test results for that decision. This phase makes no improvement
claim and exposes no new test/benchmark outcomes.

Experiment: original charcnn-wide with URL side features, three seeds1/2/3,
original frozen url_domain and PhreshPhish official-train domain-grouped cal carve
(seed0,fraction.05), no training subsampling. Preserve original200-byte limit,
AdamW.002/weight_decay.0001,batch512,max8epochs/patience3, AMPfp16, calPR-AUC
stopping. Fit side standardization on original fit rows only. No teacher/KD in E0.
Source inspection found model construction preceded seeding in existingrun_unit;
E0 seeds BEFORE construction as well as training, with synthetic regression.
This corrects seed reproducibility; it is not an identical historical checkpoint.

Original single-class url_temporal test must remain disclosed as unmeasurable;
no fabricated negatives, altered split or oracle threshold. Required complete
controls are3seeds×2measurable datasets. Save all six selected CPU state dicts,
config/source/model/fit/cal/test-ID hashes, feature scaler and exact calibration
logits. Exact CPU serialization round-trip and complete ordered cal IDs required.
Do not run test prediction, protocol_section, evaluate_scores, quantized test
checks, or use benchmark/test outcomes to choose a teacher in this phase.
Existing dataset builders may transform rows using train-fitted statistics;
that does not authorize fitting or scoring test rows.

Technical GO for this preparatory phase only: six hash-bound, reloadable controls
with original train/cal membership and finite complete cal logits, exact seed
initialization independent of preceding RNG use, no test-score calls. Any failure
or missing required control blocks progression and is logged. No performance or
shipment GO from these criteria. Three-seed metrics and paired domain-clustered
PhreshPhish results will come from a later frozen-model evaluation, after the
separately declared teacher/cal-only KD eligibility decision.

Candidate teacher research is URL_TEACHER_RESEARCH.md. Benefit of pretrained
ByT5 on URLs remains UNVERIFIED. Its exact learning recipe and eligibility/scientific
gates must be declared before that teacher job; no test feedback exists from E0.
Historical URL benchmark feedback informed the overall intervention choice;
retain that exposure caveat and URL-only qualifier, no reproduced SOTA claim.
No SMS fusion expert replacement or new v1 FPR claim. Private Kaggle only,
max2kernels, oneevent/fetch; private artifacts verifiedisPrivate, HF publish pending.
