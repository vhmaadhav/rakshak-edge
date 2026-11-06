# Frozen fused-system audit plan — 2026-11-01

Hypothesis: the full canonicaliser → student+URL+rules → calibrated fusion →
SAFE/UNCERTAIN/SCAM pipeline suppresses transactional false alarms that the
student-only cal-1%-FPR operating point exposes. The rule/evidence GroundLM
0.0883% is not a student or fused-system measurement.

Experiment: replay the exact frozen expert scores/model versions, fusion state,
calibrator and SAFE/SCAM thresholds on GroundLM and SmishX. Never fit an object
or choose a threshold on these sets. Hash every input/state. Report fraud-policy
and legacy endpoints separately, FPR at SCAM and >=UNCERTAIN over **all** eligible
negatives, recall over **all** positives, UNCERTAIN-only fractions and coverage.
Also report the library's conditional-on-auto-decision FPR with its different
denominator; it must not replace unconditional FPR. Campaign bootstrap is primary;
IID Wilson reference remains separate, especially when all-zero bootstrap CIs collapse.

Go: GroundLM SCAM FPR<=1%, >=UNCERTAIN FPR<=10%, and SmishX SCAM recall at
least the paired student-only recall at its frozen cal-1%-FPR threshold. This
conservatively interprets the user's UNCERTAIN<=10% gate as **>=UNCERTAIN**,
not just UNCERTAIN-only. Point-estimate gate, not a population-FPR guarantee.
Otherwise fail this baseline gate; preserve CIs/paired recall differences.
No externally selected seed or operating point. GroundLM remains in-sample for
rule-feature design; an audit subset does not restore independence.

**Artifact status:** no frozen fusion state/threshold file was found in the
checkout. Private HF inventory via connector and local account both returned404.
User clarification is pending. Never invent shipped state or call a newly fitted
cal-only baseline shipped. Code preparation and aggregate diagnostic definitions
can proceed while the required configuration is unavailable.

Future candidate arms are separate predeclarations: train-only deterministic
hard negatives from independent published transaction/safety semantics; cal-only
fusion with protective transactional features; family calibration only if those
fail. Three seeds/all frozen folds, cal-only fitting, external reporting. Kill
increased fused GroundLM FPR or any frozen-fold recall loss exceeding its baseline
CI half-width. External audit outcomes used for design change exposure status;
record that in the design-set register rather than retaining an independent claim.

## Diagnostic preparation (no benchmark run yet)

`sms_transactional_audit` binds the exact corrected source-control kernel tree
`62a788d27aa9db20b84964f00b8208d3025c3be8f5128352d5c76c53b5ae589e` and metrics
SHA `db3811b2ede62d577e3dad40684276d0ec0580d231d0505cafe3d5d54fb46538`.
It uses all three ERM seeds and their original cal-1%-FPR thresholds; no new
inference, training or threshold search. Saved external score IDs must match the
full eligible population exactly. Output is aggregate only, with campaign CIs.

Fixed overlapping proxies: bank/UPI/OTP/delivery/telecom-utility keywords,
OTP+do-not-share+no-URL composition, normalised length (<80/80–159/>=160), digit
share (none/<10%/>=10%), URL presence and sender style (missing stays missing).
Compare counts in frozen campaign train fraud negatives, reporting unknown-policy
rows and source counts. These proxies cannot establish real-world Indian origin,
transaction authenticity or a causal training-data gap. No representative message
examples are returned. External diagnostics used to design later arms must enter
the design-set register; keep them separate from claims of untouched validation.

Future hard-negative template semantics may cite independently checked public
[RBI digital banking safeguards](https://rbikehtahai.rbi.org.in/safeguards-for-digital-banking.html)
and [NPCI fraud-awareness guidance](https://www.npci.org.in/fraud-awareness), checked
2026-11-01. They support security semantics (e.g. credentials should not be shared),
not a claim that generated SMS are authentic bank templates. No message templates
have been generated or copied from benchmarks/external sets.

## Candidate (b), fixed before any external audit outcomes

Hypothesis: a logistic stacker fitted only to the original cal IDs can learn from
rule evidence and explicit protective OTP context, reducing transactional false
alarms without an imposed safety override. Compare the existing fusion feature
schema against an opt-in schema adding three binary inputs: do-not-share text,
OTP+do-not-share+no-URL, and transaction-related keywords (not authenticated notifications). No external-derived
phrases, per-family thresholds, manual probability caps or test-selected parameters.

Use the corrected three ERM student seeds1/2/3 unchanged, fixed URL expert and every
source_holdout/campaign/lang_shift/temporal split. Four campaign-disjoint cal stages
(seed0) for fusion, Platt probability calibration, empirical SAFE/SCAM thresholds
(alpha.01/beta.05), reserved CRC reporting. All stage IDs and resulting artifacts
recorded. Infeasible stages/thresholds are failures to measure, not hidden fallbacks.
Underlying student early stopping already used its original cal partition; these
subdivisions do not make that expert selection independent of cal.

Freeze both schemas/thresholds before reading any external outcome. All external
fraud/legacy endpoints reported. Keep candidate(b) only if it passes the top fused
point gates, does not raise fused GroundLM SCAM or >=UNCERTAIN FPR against its
paired baseline, and no frozen-fold SCAM recall loss exceeds that baseline's
campaign CI half-width. Show each seed and paired campaign differences; no seed
selection. A newly fitted baseline remains labelled **new cal-only baseline**, not
shipped. Missing deployed artifacts still block an actual shipped-system audit.
No runtime/quality claim or job launched by this preparation.

Exposure clarification before outcomes: the user's aggregate GroundLM/SmishX student
FPR feedback motivated candidate planning. This is **in-sample for intervention
selection**, even though no message rows were inspected to author features. SmishX
remains non-Indian-specific; its earlier independent label is scoped to the frozen
reviewed rules/control. Do not claim untouched independent validation for a new
feedback-motivated candidate. Register updated in GROUNDLM_FPR.md.

The fixed URL source for the **new** cal-only baseline/candidate is private
`maadhavvh/rakshak-url-student`, successful tree
`5131673567fcf4d3be5eda9f43d28194632d2cce7e2d0940fc06a400eede36db`, ONNX SHA
`1bbd2c57cb541a7cbc287887ef51976d54240a353655827a511c63ec56213fc7`, scaler SHA
`c6b3fc968b03339a33052fe814a9bb851d3035cb302b06458955c2a5e5b6501e`.
It is the PhreshPhish-fitted charcnn-wide, seed0, length200, raw URL + standardised
lexical side features. Missing URL remains NaN with an explicit absence indicator;
multiple URLs use fixed max probability. This aggregation is a new baseline choice,
not evidence that an unidentified shipped configuration uses it. Source/output
hashes are checked before any inference. No inference is run locally.

Candidate decision is deliberately seed-conservative: **every seed1/2/3** must
pass the absolute external point gates, no-increase GroundLM SCAM/>=UNCERTAIN
FPR guards and every frozen-fold recall CI-half-width guard. All six exact folds
and every seed/arm are mandatory; missing or infeasible endpoints are
unmeasurable, not go or a hidden dropped fold. Paired campaign differences are
reported per seed for internal recall and external FPR/alert/recall.
