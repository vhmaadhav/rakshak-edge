# Fusion v1 ONNX artifact gate — negative result 2026-11-01

**Artifact decision: KILL.** No complete campaign v1 was frozen or published.
No test/external decision was read by the failed fitting attempt or diagnostic.
This is an export-reference check, not a measured detector/FPR failure or safety pass.
The numerical tolerance was predeclared before fitting and remains unchanged.

The first seed1 fitting job stopped on the ScamShield validation companion. Two earlier
companions passed. It had not saved the failed numeric difference; the subsequent diagnostic
recovered it from the exact same exported parameter tensors, not by retraining or changing
weights. Every expected initializer name, shape, dtype and finite tensor was checked.

| Fold | MKLDNN | ORT optimization | Cal rows | Max absolute logit difference | Max probability difference | ≤1e-5 logit gate |
|---|---|---|---:|---:|---:|---|
| lang_shift | False | disabled | 8454 | 7.62939453125e-06 | 1.16229057312e-06 | pass |
| lang_shift | True | all | 8454 | 7.62939453125e-06 | 8.94069671631e-07 | pass |
| lang_shift | True | basic | 8454 | 7.62939453125e-06 | 8.94069671631e-07 | pass |
| lang_shift | True | disabled | 8454 | 7.62939453125e-06 | 8.94069671631e-07 | pass |
| source_holdout:holdout_mishra_soni_smsphishing | False | disabled | 10876 | 7.62939453125e-06 | 1.43051147461e-06 | pass |
| source_holdout:holdout_mishra_soni_smsphishing | True | all | 10876 | 9.53674316406e-06 | 1.25169754028e-06 | pass |
| source_holdout:holdout_mishra_soni_smsphishing | True | basic | 10876 | 9.53674316406e-06 | 1.25169754028e-06 | pass |
| source_holdout:holdout_mishra_soni_smsphishing | True | disabled | 10876 | 9.53674316406e-06 | 1.25169754028e-06 | pass |
| source_holdout:holdout_scamshield | False | disabled | 2754 | 1.52587890625e-05 | 8.64267349243e-07 | FAIL |
| source_holdout:holdout_scamshield | True | all | 2754 | 2.28881835938e-05 | 7.7486038208e-07 | FAIL |
| source_holdout:holdout_scamshield | True | basic | 2754 | 2.28881835938e-05 | 7.7486038208e-07 | FAIL |
| source_holdout:holdout_scamshield | True | disabled | 2754 | 2.28881835938e-05 | 7.7486038208e-07 | FAIL |

All four fixed runtime pairs fail on ScamShield. A small probability difference does not
override the logit criterion. Even a small probability change can cross an operational
decision boundary; no unmeasured threshold-equivalence or performance claim is made.

PyTorch documents that floating-point calculations can vary with backend/execution choices.
[Official PyTorch2.11 numerical-accuracy notes](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html).
ONNX Runtime enables all graph optimizations by default and provides basic/disabled levels.
[Official graph-optimization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html).
The diagnostic measures discrepancies under those settings; it does not establish a complete
causal decomposition of arithmetic differences. No raw SMS, training label, teacher or rule
was changed; no synthetic evaluation row, external calibration or quantization was introduced.

## Traceability

- Fitting: fusion_v1-freeze-20261101 v1,error546.9s,
  tree79e16fe8bdaf722ecd7b202198b9a555201ded44bd90fba13b43161b90ea13f3.
- Diagnostic: onnx_parity_audit-p1-20261101 v1,ok274.5s,
  tree5895a880af8957eb891ddad4f1fda45905e72463e9f456b3db8bd377354c2cba.
- Metrics SHA256:2a6611fe8a37b81c2a34dfc79fbd3db2474fd58c56f1a6540ec490562700a68c.
- One original completion subscription and one verified fetch for each job.
- Original source ONNX hashes and all runtime results are in private diagnostic metrics.json.

The owner confirmed the exact PyTorch checkpoint as frozen v1 measurement reference;
ONNX work is deferred to Phase2. The replacement artifact completed with artifactGO and
safetyKILL, without retroactively passing this ONNX gate; see FUSION_V1_RESULTS.md.
Arm deployment/quality/performance remains UNVERIFIED. A reference is not shipment
approval. Hard-negative and policy audits use its immutable manifest; no gate changes.

HF publish pending. These failed/incomplete outputs stay private on Kaggle; no complete v1
dataset is published from them. No model/data/secrets in git.
