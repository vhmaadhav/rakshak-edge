# Last P1 arm(c): template-family calibration — predeclaration 2026-11-02

Hard-negative(a) and context-fusion(b) failed their unchanged gates. The original
escalation audit stays UNMEASURABLE; its native-cal successor evidence-required
alert has now emitted a KILL completion event. This arm is declared **before
reading that successor's detailed outcome table**, with no policy-family tuning
from it. Independent Indian validation remains OPEN. No per-family fit has run.

Hypothesis: calibration errors differ across the five task-defined transactional
lexical families, and cal-only family maps can reduce false alarms without losing
recall. Sparse cal families may make this impossible; the fixed fallback below
must remain visible, never import external/test/synthetic calibration examples.

## Experts, controls and model identity

Use all3 already-trained **no-KD** seeds1/2/3 from the completed clean control
`sms_clean_kd-kd1-20261101`v1, exact frozen train IDs/all6folds, full hard-labelBCE.
This is the owner-requested clean≈.41 student recipe, **not** the rejected KD arm
or a newly trained model. Checkpoint hashes + training metadata and source-tree/
kernel/git/version must be verified. No teacher/extras/synthetic data are added.
Actual selected checkpoints were saved before their original test outcomes. No
seed/checkpoint selection from this calibration arm's outcome.

These are a named experimental cohort `template-cal-c1`, with matched new global
and family calibration states using **identical experts per seed/fold**. They are
not the v1 experts silently replaced: bind immutable v1 manifest
`c310d3ccbc1c1e50ce9436f217c66ee9934f778985bfef487a034cc3070b79fa`
and its unchanged reference metrics explicitly. Every external comparison names
this cohort or v1. No previously shipped fusion state existed. PyTorch checkpoint
measurement references only; Arm/Android portability UNVERIFIED, ONNX Phase2.

Fixed URL weights/scaler/rules are the same pinned v1 versions. Use exact CPU
checkpoint reload, finite outputs, hash/provenance guards; on ALLcal rows require
maximum probability difference<=1e-5 against the cached source GPUcal scores.
This verifies checkpoint/source identity, not ONNX or Arm equivalence. If that
technical check fails, stop unmeasurable without changing tolerance.

## Fixed family assignment and calibration

Use the existing `transactional_profile.PATTERNS` task-category regexes unchanged;
no new strings/template rows from any benchmark. Disjoint priority: otp, upi,
bank, delivery, telecom_utility, elseother. First matching family wins. These are
lexical proxies, not human transactional labels or evidence of Indian provenance.
No sender or private message example is used. Record the exact source hash.

For each6fold×3seed, fit one global unchanged baseline stacker/calibrator and
SAFE/SCAM thresholds on that fold's original known-fraud cal rows, using existing
four campaign-separated stages `fit_cal_only`/`campaign_parts` (fixed seed0).
No new model/stacker features. Global and family roles share the same fitted
logistic stacker. Candidate replaces ONLY its probability/threshold calibration
for eligible named families:
- stage2: fit the existing sigmoid/Platt map on that family's raw global-model
  logits, requiring at least200negatives and20positives in this stage;
- stage3: choose family's SAFE/SCAM thresholds with the existing search rule,
  alpha.01/beta.05, independently requiring at least200negatives/20positives;
- if either stage is too sparse or threshold search infeasible, use the EXACT
  global map+thresholds, with the fallback reason/counts recorded; no threshold
  fitting on pooled test/external rows or on another family's examples;
- other always uses global map+thresholds. No hyperparameter/min-count/family-order
  changes after results. No pruning a poorly performing family after outcomes.

Cal stage4 is diagnostic only, never external tuning or certified risk under drift;
checkpoint selection reused cal. Freeze all18model/role states, params, stageIDs,
family eligibilities/counts/fallbacks and exact expert/source hashes before ANY
new test/external decision. No family vocabulary/regex/cal selection from outcomes.

## Go/kill, endpoints and honest reporting

Every seed must satisfy GroundLM unconditional SCAMFPR<=1%, >=UNCERTAINFPR<=10%,
AND neither FPR may increase against its matched global-fusion control. SmishX
candidateSCAM recall must be>=the same control student's frozen cal1%-FPR recall.
On every frozen fold, candidateSCAM recall loss versus matched global fusion may
not exceed control recallCIhalf. Require legacy-spam guard on every fold and
known-fraud guard where positives exist; UCI fraud recall explicitlyNA, legacy
mandatory. Any observed failure KILL; missing required result blocksGO. No
student/alert recall endpoint substitution to rescue these original P1 caps.
If all families fall back, report it, never call unchanged global output a fix.

All3seeds, allsource_holdout folds, campaign/lang_shift/temporal, GroundLM/SmishX
fraud+legacy endpoints, all-negative FPRdenominators, recall atSCAM andalert,
coverage/UNCERTAIN shares, campaign-clustered95%CIs and paired changes/p-values,
1,000 shared campaign resamples per pair. No new URL-only PhreshPhish/NIDS claim.
Confidence intervals conditional on fixed seeds; original temporal overlap stays.
GroundLM/ScamShield informed rules; aggregate GroundLM/SmishX/old experiment
feedback informed this design: **in-sample intervention selection**, no held-out
improvement claim. Independent consented/scrubbed Indian inbox validation OPEN.

Private Kaggle/T4 only,<=2kernels, one wait_event+matching fetch per job. Publish
allowlisted models/manifests/aggregate reports privately, verifyisPrivate;
**HF publish pending**. No raw messages/weights/secrets in git. ScientificGO would
only authorize a candidate successor, not shipment without device/safety checks.
After this last P1 arm's outcome, move capacity to the predeclared multilingual
student, thenE/F as instructed; no unregistered template-threshold fishing.
