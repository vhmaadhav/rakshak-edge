# Evidence-gated escalation — predeclaration 2026-11-01

Hypothesis: student-only high risk creates many transactional false alarms;
requiring independent rule/URL corroboration reduces SCAM while retaining alert
recall. This is a **decision-policy change**, no new model or external fitting.
GroundLM/SmishX false-alarm feedback motivated this policy: **in-sample policy
selection**. Independent Indian validation remains OPEN pending consented,
privacy-scrubbed new inbox data; never tune on that independent collection.

Use immutable newly fitted (not previously shipped) fusion v1 campaign experts
for external outcomes. Use its seed1 fold-validation companion experts/cal states
for every frozen internal fold, to avoid training on the tested holdout rows.
Record parent manifest hashes and verify every expert/source artifact. No external
row is used to select thresholds, evidence families, weights or a checkpoint.

Pre-fixed corroboration:
- credential_request / upi_collect_or_pay / brand_domain_mismatch / apk_install /
  remote_access_app evidence with existing weight>=.85;
- kyc_block_urgency / account_suspension with existing weight>=.75 AND an extracted
  link (these existing weights are.8/.75, explicit category exception);
- suspicious_url from the URL expert: present URL score>=its cal-only threshold.
  Set that threshold separately per fold using known fraud-labelled URL-present
  calibration rows at empirical1% FPR, maximum recall, existing threshold_for_fpr
  tie handling. Both classes are required; otherwise that required calibration
  is unmeasurable, not external-fitted. Lexical shortened_url/urgency alone never
  corroborate. A lexical suspicious_url flag alone is insufficient.

SCAM iff frozen fused probability>=t_scam AND corroboration. High risk without
corroboration becomes UNCERTAIN, never SCAM. Existing uncertain stays UNCERTAIN.
Baseline SAFE with corroboration becomes UNCERTAIN (caution despite low score);
low-risk/no corroboration stays SAFE. No threshold, model or stacker refitting.
Freeze candidate-policy JSON (all six cal URL thresholds + parent hashes + policy
source SHA + version) BEFORE reading test/external decisions. This policy can
only preserve/increase the alert set; demoting SCAM to UNCERTAIN cannot lower
>=UNCERTAIN FPR. Report that constraint honestly; no go inferred from SCAM alone.

Go (all required endpoints): GroundLM unconditional SCAM FPR<=1%,
>=UNCERTAIN FPR<=15%; SmishX SCAM-or-UNCERTAIN recall>=student-alone SCAM recall
at its frozen cal1%-FPR threshold. On every internal fold, policy alert recall
must not lose >baseline fused alert-recall CI half-width; require legacy spam
endpoint everywhere and known fraud endpoint wherever it has positives. UCI fraud
recall is explicitly NA; UCI legacy-spam guard remains mandatory. Report both
label endpoints, campaign-clustered CIs, paired changes, coverage and uncertain
share per split, all risk–coverage curves. Compare with the same parent fold
model/threshold, never substitute a historical student's score. Any observed gate
failure kills; missing required results block go. No multiple testing adjustment
or independent held-out improvement claim.

Risk–coverage is descriptive: confidence-ranked curves at fixed policy decisions
(condition on auto-decided rows separately), plus operational SAFE/UNCERTAIN/SCAM
coverage and conditional decided error. Ranking curves are not alternate deployable
thresholds and cannot select a new policy. Save counts/denominators per endpoint.

Only GO authorizes candidate Phase1 wiring and a separately frozen v2 manifest
referencing v1 weights + exact policy. UNCERTAIN must preserve decision semantics
in the evidence package and recommend VERIFY_INDEPENDENTLY. Existing package
rules may otherwise raise risk, so integration requires explicit tests rather
than assuming the old package builder preserves this new decision. If KILL,
retain negative results and predeclare per-template-family cal-only arm(c) last.
Private Kaggle dataset publication with verified privacy; **HF publish pending**.
