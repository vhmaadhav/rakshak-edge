# Exact-checkpoint measurement reference — 2026-11-01

Optional format preference is pending. This is a reviewable **preparation**, not
an adopted model, launched experiment or published v1. The ONNX artifact attempt
and its diagnostic remain **KILL** under the unchanged1e-5 logit gate; see
FUSION_V1_PARITY.md. A checkpoint reference does not retroactively pass that export.

Hypothesis: exact CPU float32 parameter serialization can provide a stable
measurement reference without cross-runtime ONNX arithmetic differences.
Experiment, if selected: same original train-only dual/base seed1/teacher0/15epoch
recipe, same cal stopping, four cal-only fusion stages/alpha.01/beta.05, same six
frozen folds (campaign primary + five validation companions). Save actual PyTorch
checkpoint, reload every selected parameter strictly, require unchanged <=1e-5
maximum logit difference on ALL cal rows and finite outputs before fitting
fusion. No weights/training-precision/decision-threshold change based on outcomes.
No ONNX equivalence claim from a same-framework serialization test.

Freeze full state/provenance/artifact/source manifests before any test/external
outcome, immutable v1 primary binding companion hashes. Actual inference uses the
reloaded CPU checkpoint, fixed two Torch threads, full eval mode, same canonical
features and pinned URL expert. Exact tensor/schema/model hash verification is
required; no randomly initialized parameter may remain after reload. Detector
scientific go/kill gates remain identical; absolute safety failures still fail.

Format: pytorch_checkpoint. **Measurement reference only; Arm/Android portability
and latency remain UNVERIFIED.** Llama+ExecuTorch remain mandatory for advisory;
this alternative concerns the small detector only. If policy later passes, its
candidate Phase1 code must retain deployment-format/Arm validation as OPEN.
Publishing a reference is not shipment approval. No test/external tuning or
independent Indian claim; GroundLM/SmishX intervention feedback is in-sample.
Private Kaggle publication with verified privacy; HF publish pending.

## Default implementation choice after the optional response window

No format preference arrived after an extended opportunity to reply. Proceed
with this checkpoint format as the default **next measurement-reference attempt**
when a slot frees; the owner can still steer otherwise. This is an implementation
assumption within the authorized frozen-baseline task, not a waiver of the failed
ONNX gate. Every scientific criterion and numerical round-trip tolerance above
stays fixed. No artifact/result/Arm claim exists before actual Kaggle completion.

## Completed outcome

The default next attempt completed on Kaggle: artifact **GO**, safety **KILL**.
Six exact CPU checkpoint round-trips had maximum cal logit difference0; this does
not pass the earlier ONNX equivalence failure. See `FUSION_V1_RESULTS.md` for
all folds, external endpoints, thresholds and immutable hashes. Reference only,
not shipment approval; Arm/Android portability remains UNVERIFIED.
