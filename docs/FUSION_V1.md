# Frozen fusion v1 protocol — 2026-11-01

User confirmed there is **no previously frozen/shipped fusion configuration**.
Label this baseline **newly fitted (not previously shipped)**. Earlier PR#6
cal-only score audits remain historical experiments, not deployable artifacts.

Hypothesis: the existing fixed baseline recipe can be made into one portable,
hash-bound reference without using external outcomes to choose parameters.
Experiment: fresh campaign ERM student, seed1 (chosen now, before arm(a) outcomes),
exact frozen train IDs and their matched seed0 teacher logits only. Same dual/base
student, epoch cap15, cal PR early stopping, existing TrainConfig defaults. No
synthetic augmentation, source balancing or candidate(b) context features. Export
its selected weights plus ONNX. Require cal logit parity <=1e-5 before fusion;
fit fusion using ONNX probabilities so inference matches the saved expert.

Fit four fixed campaign-disjoint cal stages using existing fit_cal_only: logistic
stacker, Platt calibrator, empirical t_safe/t_scam at alpha=.01/beta=.05, then
CRC diagnostic only. Existing checkpoint selection used full cal; CRC therefore
has no independent-selection guarantee. Freeze JSON BEFORE test/external scoring:
stacker/calibrator parameters, exact feature order, thresholds, alpha/beta,
cal-stage IDs, training-ID digest, git SHA AND dirty bundle digest, private Kaggle
kernel slug/version, SHA256 of student checkpoint/ONNX, URL ONNX/scaler and the
bundled canonicaliser/rules/preprocessing/fusion source archive. Include source
kernel versions and git/bundle hashes for each expert, library versions, and
inference policy (max URL score; missing URL NaN+presence; boundary ties).

Artifact go = all required files/hashes/provenance present, cal-only membership
and model parity verified, state round-trip equivalent and manifest unchanged
across scoring. Any missing item kills publication as frozen v1. This is an
artifact gate, **not safety approval**. Safety go remains GroundLM SCAM FPR<=1%,
>=UNCERTAIN FPR<=10%, SmishX SCAM recall>=same student at cal1%-FPR; report
campaign CIs/recall/coverage and explicit failure. A valid but unsafe baseline may
be archived privately for comparison, with shipment approval false. No threshold
or expert selection based on these external results; changes require a new
version, never overwrite v1. Fold-specific student results in PR#6 are diagnostics,
not claims that the campaign v1 weights trained on their holdout splits.

GroundLM informed earlier rules; GroundLM/SmishX aggregate feedback motivated the
intervention programme: **in-sample for intervention selection**. No independent
held-out improvement claim. Independent Indian validation stays OPEN. Future
external FPR claims must identify the immutable v1 manifest SHA (or a separately
frozen successor); old experiment numbers must retain their original provenance.
Publish manifest and necessary inference artifacts together as a **private**
Kaggle model bundle (HF org access currently404), verify privacy. No messages,
weights, evaluation artifacts or secrets in git. No third kernel while the two
existing subscriptions are pending. No new benchmark computation on laptop.

## Fold-validation companion freeze (before any v1/policy run)

The policy requires every frozen-fold guard. The campaign model cannot be reused
on full source-holdout test folds it may have trained on. Freeze seed1 instances
of this SAME recipe separately for the other five canonical splits, each from
its own train IDs/cal only, with separate student weights/state/provenance. Freeze
ALL six before any test/external decision. The campaign manifest binds these
validation manifests by SHA256; private publication includes all six. They are
fold-validation companions, not a different deployed configuration or three-seed
claim. External v1 results still use ONLY the campaign instance. Do not tune on
the new policy's external results. HF publish pending until org scope is fixed.

## Parity diagnostic predeclaration after artifact failure, before any test outcomes

Original run79e16... failed the unchanged <=1e-5 ONNX logit parity artifact gate
on the ScamShield companion, after lang_shift and Mishra passed. No test/external
outcome was scored; no frozen campaign v1 was produced. The failure remains
recorded, not promoted or excused as a safety success. Numeric error magnitude
was not saved by the original helper; do not invent it.

Hypothesis: CPU backend/fusion arithmetic, rather than changed weights, causes
the failed absolute logit check. UNVERIFIED until a diagnostic. Reconstruct the
same parameter tensors from each saved ONNX initializer with strict exact-name,
shape and dtype binding to the known dual/base architecture; no training or
weight modification. Calibration rows only, exact original fold IDs. Evaluate
four pre-fixed runtime pairs: (Torch MKLDNN on,ORT all); (on,ORT basic);
(on,ORT disabled); (off,ORT disabled), in this preference order. Record each
fold's maximum logit discrepancy and sigmoid discrepancy, including failures.
Go only if strict weight reconstruction succeeds AND a single pair meets the
SAME <=1e-5 maximum logit difference on every available companion's full cal
sample. No threshold relaxation, no test/external selection, no quantisation.
If a pair passes, fix that runtime in the new baseline artifact attempt and
repeat all six full-cal checks before any outcomes. If none passes, artifact
publication remains killed/unmeasured; don't claim v1 exists. No B quantisation
or runtime recipe changes are authorized by this separate student diagnostic.

## Completed outcome

The original ONNX attempt failed its unchanged artifact gate; see FUSION_V1_PARITY.md. The owner confirms CPU PyTorch checkpoint as frozen v1 measurement reference, ONNX deferred to Phase2. The checkpoint replacement has artifactGO/safetyKILL; all completed results, privacy verification and exact hashes are in FUSION_V1_RESULTS.md and STATUS.md. No shipment approval or Arm portability claim.
