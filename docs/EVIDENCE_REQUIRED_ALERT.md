# Evidence-required alert — predeclaration 2026-11-01

Declared on the owner's explicit instruction **before reading any result of the
running evidence_escalation-gate-20261101 audit**. That audit and its original
15% alert-FPR/student-recall gate remain unchanged. This is a distinct arm,
not a post-hoc gate change. User confirms PyTorch checkpoint=frozen v1 measurement
reference; ONNX work deferred to Phase2. Arm/Android portability is UNVERIFIED.

Hypothesis: requiring qualifying rule/URL evidence for **every alert**, rather
than only SCAM, reduces transactional false alarms while preserving at least the
frozen rules-only detector's recall. No new model, stacker or externally selected
threshold. GroundLM/SmishX feedback motivates this policy: **in-sample policy
selection**, never an independent held-out improvement. Independent Indian
validation stays OPEN pending new consented, scrubbed inbox data after freeze.

Use exactly the frozen v1 campaign experts for external reporting, and its own
five fold-validation companions for frozen internal folds. Same named qualifying
evidence/weights as EVIDENCE_ESCALATION.md; no new families or weight changes:
credential_request, upi_collect_or_pay, brand_domain_mismatch, apk_install,
remote_access_app at existing weight>=.85; kyc_block_urgency/account_suspension
at existing weight>=.75 AND extracted link; URL expert score at its own cal-only
known-fraud URL-present1%-FPR threshold (both classes required). Urgency or lexical
suspicious_url alone is insufficient. Freeze all six URL thresholds and exact
parent/policy/source hashes before any test/external decision is read.

Decision for every row:
- no qualifying evidence: SAFE at every fused/student risk, with no metric-exempt
  alert or warning. Any optional information note is counted SAFE in all metrics;
- qualifying evidence and frozen fused risk>=t_scam: SCAM;
- qualifying evidence and lower frozen fused risk: UNCERTAIN.

Predeclared GO requires GroundLM SCAM FPR<=1% and >=UNCERTAIN FPR<=15% on all
eligible known-fraud negatives. Recall gate: on SmishX and every frozen internal
fold, policy SCAM-or-UNCERTAIN recall may not fall below **rules-only alert
recall** by more than the rules-only campaign-clustered CI half-width. Report
student-only SCAM recall at the identical parent's cal1%-FPR threshold as well;
student recall is a comparison, **not the new GO guard**. KILL on any observed
failure. Missing required results block GO, never become success. Require legacy
spam-inclusive endpoint and known fraud endpoint wherever positives exist; UCI
fraud recall explicitly NA, mandatory legacy-spam recall guard remains.

Rules-only comparator is fixed before outcomes: canonicaliser -> run_rules ->
build_package with confidence_state='benign', no model score; risk HIGH maps
SCAM, risk MEDIUM/UNCERTAIN maps UNCERTAIN, LOW maps SAFE. Thus rules-only alerts
are risk!=LOW, matching the earlier above-LOW evidence audit. No URL-model input
in this comparator (its lexical/domain rules remain). Report its SCAM and alert
FPR/recall plus both student and policy endpoints; do not choose a weaker rules
baseline after results. Qualifying-evidence alert recall is not assumed to equal
rules-only recall because some existing rule families do not qualify.

Campaign-clustered95%CIs and paired policy-minus-rules and policy-minus-student
recall tests (1,000 shared campaign bootstrap draws per pair), all six folds and
both external sets, all-negative FPR denominators, exact counts, coverage and
UNCERTAIN share. Preserve descriptive risk–coverage curves without tuning from
them. Same parent models/rows/thresholds, no training/eval/split changes. Unit tests
use synthetic fixtures only. Real evaluation only private Kaggle, maximum2GPU
kernels, one existing event subscription per job and one matching fetch.

Only GO authorizes a candidate Phase1 successor/v2 and end-to-end/ELA decision
wiring. UNCERTAIN must remain VERIFY_INDEPENDENTLY through package construction.
An artifact manifest is not shipment approval; Arm runtime/latency remains OPEN.
If both evidence policies KILL, log negatives before the last per-template cal-only
arm(c). No per-template rule may be derived from test/external rows.

## Owner-approved URL calibration resolution before this arm's compute

The earlier escalation audit stopped before outcomes because an SMS fold's
URL-present known-fraud cal subset lacked both classes. It remains
**UNMEASURABLE** with its original gate and calibration requirement unchanged.
The owner explicitly authorizes this new arm to instead use **only the URL
expert's original frozen URL calibration split**, never URL test, PhreshPhish
benchmark/test subsets, or any SMS/external row. This supersedes only the new
arm's proposed per-SMS-fold URL threshold paragraph above, before this arm's
compute; the declared evidence families and scientific go/kill gates do not change.

Reconstruct the original exported charcnn-wide/s0 expert's native PhreshPhish
train-only grouped cal carve (seed0, fraction.05, same registered-domain/normalised
URL fallback grouping and original partition IDs). Filter parquet to official
train IDs before loading URL/label columns; no benchmark columns loaded. Use the
cal-only1%-FPR threshold with existing tie handling, one identical threshold for
all6 SMS folds/external sets. Require both classes on that native cal split;
otherwise this arm also remains unmeasurable, with no further fallback.
Freeze exact calibration split identifier, source partition SHA256, ordered cal
IDs+SHA256, labels/counts, original URL kernel/model/source/algorithm hashes and
selected threshold in the policy manifest BEFORE SMS/test/external scoring.
No parameter/source/threshold change from this arm's outcome.

## Native-URL adapter correction before any outcome

The first native-cal attempt used the SMS extractor entry point, which cannot
score every original native URL record, and failed before threshold freeze or
SMS/external outcomes. Recover using the original URL model's direct raw URL
byte/side interface on **every identical native cal row**, not SMS URL extraction.
No row exclusion, model/precision/feature/scaler/split or scientific-gate change.
Core-score row-count/finite-output guards must pass first. The original failed
attempt and the earlier UNMEASURABLE escalation audit remain recorded separately.
