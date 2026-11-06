# E1/E2: fixed pretrained-byte teacher and train-only KD — predeclaration 2026-11-02

Declared while E0 is running, before reading its calibration predictions, and
before any new URL test/benchmark outcomes. E0 deliberately returns none. Primary
checks/revision/licence are in URL_TEACHER_RESEARCH.md. Hypothesis: pretrained byte
context can produce a teacher that improves rare-base-rate URL ranking and makes
train-only distillation useful. Both improvements are UNVERIFIED.

## Fixed teacher

One teacher seed0 per original measurable dataset (PhreshPhish and url_domain),
trained independently on that dataset's exact E0 fit IDs. No additional URL corpus,
benchmark/test rows, synthetic data or cross-dataset teacher transfer. Pinned
Apache-2.0 google/byt5-small@68377bdc18a2ffec8a0533fef03b1c513a4dd49d,
README/config/bin SHA checks from the research register. Deserialize weights_only,
require every exact T5EncoderModel encoder/shared parameter and shape; do not
execute or train the decoder/language-model head. Upstream pretraining membership
is unknown, not a downstream clean-membership guarantee. Keep original attribution.

Input: identical first200raw UTF-8 URL bytes as the charCNN, each mapped byte+3,
followed by EOS1, PAD0 to201positions. Masked mean of encoder last_hidden_state,
concatenate the same E0 train-standardized URL side features, Linear128/ReLU/
Dropout.1/Linear1. No alternate pooling, URL canonicalisation, features or length
sweep. All encoder parameters fine-tuned; seed before constructing the classifier.
AMPfp16, gradient checkpointing enabled, AdamW lr5e-5/weight_decay.01, batch16,
accumulation8 (effective128; normalize final partial accumulation by actual rows),
weighted BCE with original-fit neg/pos weight, max8epochs/patience3 on natural-cal
AP, gradient clip1, no scheduler/warmup. One GPU, T4 only. OOM/unsupported operations
or source mismatches stop unmeasurable; no batch/architecture sweep in this arm.

Freeze selected CPU teacher weights/config/scaler/source/fit-cal-test-ID hashes
and finite complete CAL predictions before eligibility. Never score test or
benchmark rows in this job. Training/cal prediction remain on Kaggle.

## Cal-only teacher eligibility, then KD

Compare teacher0 against the mean AP of E0's three fixed charCNN seeds, not AP of
their averaged scores. Phresh cal gets two separately reported prevalence-weighted
AP proxies: positive row weight pi/n_pos, negative (1-pi)/n_neg, pi=.0005/.001.
These are cal-only ranking proxies, not the paper's975-subset test protocol.
url_domain uses natural-cal AP. Same1000registered-domain cluster bootstrap draws,
seed13, applied to teacher/allthree students; renormalize class weights to pi on
each weighted-AP draw. Record invalid single-class draws, require>=950validdraws.

A dataset is KD-eligible ONLY if every required teacher-minus-control AP CI lower
bound>0. Missing results block eligibility; a nonpositive required lower bound
kills distillation for that dataset. This is **in-sample calibration model
selection**: checkpoints also used cal early stopping. No independent teacher-win
claim from it. Freeze eligibility/reasons/IDs/hashes BEFORE creating training
teacher targets or any test scoring. No teacher/threshold choice from benchmark
outcomes. Do not use test scores to rescue a cal-ineligible teacher.

Only eligible datasets: teacher0 logits on EXACT original fit IDs, never cal/test
or external URL targets; three fresh KD seeds1/2/3, same charCNN-wide/side/byte200/
seed-before-construction/8epoch/batch512/AdamW.002/weight_decay.0001/patience3/AMP
settings as E0. KD alpha.5/T2 with existing soft-BCE formulation. No unlabeled
extras/OOF/ensemble claim: these are in-sample single-teacher training logits.
Save selected weights and cal thresholds before test. Ineligible datasets keep
their unchanged E0 controls, explicitly labelled as no KD, not an invented gain.
The primary target is PhreshPhish; no post-hoc promotion of the large teacher as
the Arm model. Report teacher even if cal-ineligible, in a later frozen eval only.

## Frozen evaluation and scientific go/kill

After all artifacts/eligibility/thresholds freeze, a separate Kaggle job scores
all original test rows/subsets with no tuning. Report each seed, official975subset
AP/P@R.90 averaged by every published base rate, plus url_domain thresholds/cal
FPR1%/maxF1 and domain-clustered CIs. Original single-class url_temporal is reported
as NA under the existing binary protocol, reconfirmed from frozen inputs; never
invent negatives. No benchmark subsets are used as calibration or training.

Paired1000registered-domain bootstrap resamples, seed13, shared across all models
and three fixed students. For Phresh, apply domain multiplicities within EACH
original subset, compute its AP/P@R.90, then average subsets and student seeds in
the original order. This preserves the unresampled protocol point estimate; do
not replace it with a pooled AP or the weighted-cal proxy. Report valid draws and
require>=950; missing needed CIs blocksGO. Confidence intervals condition on three
student seeds and one teacher seed, not general initialization uncertainty.

Primary **teacher GO** only if its paired AP gain CI lower bound>0 vs E0 mean at
BOTH0.05%/0.1%; otherwise reject the stronger-teacher claim. Primary **KD GO** only
if Phresh was cal-eligible, teacher GO holds, KD-minus-E0 mean AP CI lower bound>0
at BOTH rare rates, and KD P@R.90 at both rates loses no more than E0 CIhalf.
Companion url_domain KD (where cal-eligible) must not lose F1>its matched E0CIhalf;
otherwise global candidateKILL. An ineligible companion remains an explicit
unchanged control, not a KD/transfer-generalization result. Missing required Phresh
seed/fold/artifact/CI blocksGO; observed required failure KILL. Report natural-base
and allother published rates even if they are negative. If teacher or KD loses,
log it; no benchmark-chosen retry, alpha/length/learning-rate sweep or model switch.

ScientificGO is not shipment: Arm/format/quantisation quality/latency remain OPEN.
All new system FPR still cites v1 or a separately named successor, never silently
substitute URL weights. Historical Phresh outcomes informed this intervention:
**in-sample intervention selection**, no untouched benchmark/SOTA claim. GTE uses
URL+HTML; keep its paper row separate from this URL-only experiment. No Indian
held-out validation claim. HF publish pending; private Kaggle only, <=2kernels,
one event/fetch per job; immutable private artifact owner/isPrivate verification.

Official APIs used: [weighted AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html),
[T5EncoderModel](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel).
The API sources support implementation, not the new teacher or KD hypothesis.

Operational split2026-11-02, before E1 launch: run each dataset as its own unique
Kaggle kernel, same recipe/eligibility, fixed11-hour deadline per kernel. Budget
exhaustion aborts incomplete/unmeasurable; no shortened fit is promoted. Freeze
that dataset's eligibility before its train targets; the later test evaluation
waits for both datasets and all required artifacts. No added scientific gate or
parameter search. AMP loss normalized by the actual update-window rows BEFORE
scaling/backward, including a final partial window; record skippedAMPupdates.

### Fixed teacher numerical failure and remaining control report

Phresh teacher p1-20261102 v1/tree84d35409ac4a failed with nonfinite FP32
loss after FP16 forward, before a selected checkpoint/eligibility/targets existed.
Primary teacher and KD performance are **UNMEASURABLE**, not a measured loss.
Stop this fixed teacher arm; no precision/hyperparameter switch and no url_domain
teacher follow-up can rescue the missing primary arm. Underlying numerical cause
is UNVERIFIED. Retain exact error and immutable source in the report.

Proceed with the already requested three-seed E0 control evaluation only, without
new training or candidate selection. Same official subsets/1000domain draws/seed13,
mean per-subset AP and P@R.90, all rates + full benchmark/test and url_domain;
no teacher/KD win can be claimed. All six original checkpoint/source/input/scaler/
fit-cal-test memberships and cached calibration IDs/hashes are checked. Require
reconstructed full-cal logits to match E0 within1e-5 before outcomes (otherwise
unmeasurable). Freeze maxF1 and cal1%-FPR thresholds for all six models in a new
manifest before any test prediction. Save frozen weights unchanged, scores and
aggregate metrics privately. Confirm original url_temporal single-class status
from frozen inputs; no synthetic negatives. These measurements do not replace
fusionv1 or establish an independent benchmark/SOTA claim.

Implementation optimization: weighted AP changes only at positive-score knots;
P@R.90 can attain its maximum there. Reuse the shared benign cumulative mass at
those knots, preserving every original subset and tied score. Synthetic tests
compare against explicit repeated rows and sklearn, including zero-weight/tied
knots and the fixed-seed domain resamples. A rate bootstrap draw that loses a
class in ANY original subset is invalid, not silently averaged over survivors.
No new statistical endpoint or gate is introduced by this optimization.
