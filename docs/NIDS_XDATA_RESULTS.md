# NIDS cross-dataset generalisation: results (leave-one-dataset-out, NetFlow-v2)

Plan: `docs/NIDS_XDATA_PLAN.md` (committed before any target result, as `dbafa03`). Compute-budget
amendment: same file, section "Amendment (compute budget)" (committed as `7ef37d3`, before the
relaunch). Metrics: `docs/results/nids_xdata/metrics.json`. Job: `kaggle/jobs/nids_xdata.py`; code:
`src/rakshak/nids/xdata.py`; tests: `tests/test_nids_xdata.py`.

## Verdict

- **Gate as written (plan section 5): `lgbm/c` passes all three criteria.** Mean MCC over the four
  targets is 0.051, which is 0.159 above the LightGBM baseline (`lgbm/a`, -0.108). Worst-target MCC is
  -0.181 (BoT-IoT), which is 0.286 above the baseline's worst (-0.467, UNSW). The paired 95% interval
  of the mean-MCC difference is [0.152, 0.161].
- **The pass does not support deployment.** Three things make it weak.
  1. The FPR check (plan section 4, item 8, FPR at most 1% on each target's benign flows) fails on all
     four targets for `lgbm/c`. The thresholds were not changed to meet it, as pre-registered.
  2. The result is picked from six non-baseline arms. The pass is the best of several, not a single
     pre-declared arm.
  3. The absolute MCC is near zero. The baseline it beats is negative on UNSW (-0.467), so most of the
     gain is against a failing baseline, not against a working model.
- **MLP arms:** `mlp/d` (quantile normalisation with a per-site head) has mean MCC 0.053 and passes the
  mean criterion, but its worst target (CSE-CIC-2018, -0.344) fails the worst-target criterion. It
  does not pass the gate. `mlp/b` is degenerate: its threshold sits at 1.0 and every flow is predicted
  benign, so MCC is 0 on every target.
- **No ONNX export and no Pi change.** The gate pass is not a deployable pass (FPR fails), so the
  best arm was not exported. The LightGBM model is also not in ONNX form, and no model-to-Pi path is
  proposed here.

Conclusion: under this protocol, cross-dataset generalisation of the NetFlow-v2 detector remains poor.
The one arm that clears the gate's numeric thresholds still produces 6% to 100% false positives on
held-out targets, so it is not a usable detector.

## Per-target results (seed mean +/- sd over 3 seeds; test rows after dedupe and head removal)

Targets: UNSW (47,062 test rows), CSE-CIC-2018 (validation and test capped at 100k per class),
BoT-IoT, ToN-IoT. Prevalence is the target attack share in the test rows.

| arm | target | MCC | F1 | PR-AUC | FPR | prevalence |
|---|---|---|---|---|---|---|
| lgbm/a (baseline) | unsw | -0.467 +/- 0.049 | 0.222 | 0.503 | 0.623 | 0.671 |
| lgbm/a | cse_cic_ids2018 | -0.048 +/- 0.099 | 0.713 | 0.789 | 0.933 | 0.589 |
| lgbm/a | bot_iot | 0.019 +/- 0.055 | 0.127 | 0.635 | 0.060 | 0.591 |
| lgbm/a | ton_iot | 0.065 +/- 0.051 | 0.051 | 0.711 | 0.006 | 0.618 |
| lgbm/b | unsw | -0.013 +/- 0.067 | 0.179 | 0.674 | 0.110 | 0.671 |
| lgbm/b | cse_cic_ids2018 | 0.140 +/- 0.224 | 0.536 | 0.804 | 0.403 | 0.589 |
| lgbm/b | bot_iot | -0.237 +/- 0.065 | 0.118 | 0.502 | 0.243 | 0.591 |
| lgbm/b | ton_iot | 0.070 +/- 0.062 | 0.115 | 0.695 | 0.027 | 0.618 |
| lgbm/c | unsw | 0.062 +/- 0.009 | 0.172 | 0.647 | 0.060 | 0.671 |
| lgbm/c | cse_cic_ids2018 | 0.338 +/- 0.088 | 0.406 | 0.791 | 0.011 | 0.589 |
| lgbm/c | bot_iot | -0.181 +/- 0.083 | 0.152 | 0.507 | 0.222 | 0.591 |
| lgbm/c | ton_iot | -0.014 +/- 0.009 | 0.762 | 0.705 | 0.999 | 0.618 |
| lgbm/d | unsw | 0.033 +/- 0.051 | 0.804 | 0.838 | 0.996 | 0.671 |
| lgbm/d | cse_cic_ids2018 | -0.720 +/- 0.008 | 0.272 | 0.420 | 0.993 | 0.589 |
| lgbm/d | bot_iot | 0.000 +/- 0.000 | 0.000 | 0.622 | 0.000 | 0.591 |
| lgbm/d | ton_iot | -0.091 +/- 0.005 | 0.021 | 0.587 | 0.037 | 0.618 |
| mlp/a (baseline) | unsw | -0.012 +/- 0.052 | 0.266 | 0.667 | 0.332 | 0.671 |
| mlp/a | cse_cic_ids2018 | -0.052 +/- 0.193 | 0.662 | 0.568 | 0.828 | 0.589 |
| mlp/a | bot_iot | -0.163 +/- 0.190 | 0.084 | 0.572 | 0.149 | 0.591 |
| mlp/a | ton_iot | -0.001 +/- 0.115 | 0.735 | 0.677 | 0.878 | 0.618 |
| mlp/b | all four | 0.000 (threshold 1.0, all benign) | | | | |
| mlp/c | unsw | -0.129 +/- 0.167 | 0.272 | 0.606 | 0.362 | 0.671 |
| mlp/c | cse_cic_ids2018 | -0.271 +/- 0.223 | 0.363 | 0.480 | 0.643 | 0.589 |
| mlp/c | bot_iot | -0.158 +/- 0.103 | 0.173 | 0.506 | 0.238 | 0.591 |
| mlp/c | ton_iot | 0.010 +/- 0.017 | 0.509 | 0.672 | 0.663 | 0.618 |
| mlp/d | unsw | 0.208 +/- 0.256 | 0.730 | 0.730 | 0.574 | 0.671 |
| mlp/d | cse_cic_ids2018 | -0.344 +/- 0.147 | 0.303 | 0.502 | 0.595 | 0.589 |
| mlp/d | bot_iot | 0.232 +/- 0.124 | 0.763 | 0.678 | 0.897 | 0.591 |
| mlp/d | ton_iot | 0.117 +/- 0.035 | 0.263 | 0.669 | 0.082 | 0.618 |

Aggregate (mean over four targets of the seed-mean MCC; worst target in brackets):

| arm | mean MCC | worst MCC | gate |
|---|---|---|---|
| lgbm/a (baseline) | -0.108 | -0.467 (unsw) | reference |
| lgbm/b | -0.010 | -0.237 (bot_iot) | fail (mean diff 0.098 < 0.10) |
| **lgbm/c** | **0.051** | -0.181 (bot_iot) | **pass on the three criteria** |
| lgbm/d | -0.195 | -0.720 (cse_cic_ids2018) | fail |
| mlp/a (baseline) | -0.057 | -0.163 (bot_iot) | reference |
| mlp/b | 0.000 | 0.000 (unsw) | fail (degenerate) |
| mlp/c | -0.137 | -0.271 (cse_cic_ids2018) | fail |
| mlp/d | 0.053 | -0.344 (cse_cic_ids2018) | fail (worst target) |

Gate detail for `lgbm/c`: mean-MCC difference 0.159 (threshold 0.10); worst-target difference 0.286
(threshold 0.05); paired 95% CI of the mean-MCC difference [0.152, 0.161] (excludes 0).
Gate detail for `mlp/d`: mean-MCC difference 0.110 (passes); worst-target difference -0.181 (fails).

How to read the intervals: as plan section 4 item 7 specifies, the paired bootstrap uses the **seed-0
model only**, while the mean and worst differences are means over three seeds. So an interval can fail to
contain its point difference when seed 0 differs from the seed mean. Examples from `metrics.json`:
`lgbm/b` has a difference of 0.098 and an interval of [0.028, 0.037]; `mlp/b` has 0.057 and [0.110,
0.116]. The intervals are a seed-0 check on row-sampling noise, not an interval for the 3-seed
difference. The FPR check (plan section 4 item 8) is reported but is not one of the three gate criteria
in section 5. That is why `metrics.json` records `"verdict": "PASS: lgbm/c"` even though every target
fails it.

### Reference: shipped-style single-source model (a1)

LightGBM 300 trees / 63 leaves trained on CSE-CIC-2018 alone, threshold = max-F1 on CSE's own
held-out 15%, scored on the other targets (3 seeds):

| target | MCC |
|---|---|
| unsw | -0.128 +/- 0.076 |
| bot_iot | 0.175 +/- 0.030 |
| ton_iot | 0.078 +/- 0.014 |

The single-source reference is comparable to the multi-source baselines on BoT and ToN, and it is
better than the multi-source LightGBM on UNSW. It is context only and is not part of the gate.

## Model size and latency (seed 0, target fold; single flow, one thread)

| arm | size | p50 latency |
|---|---|---|
| lgbm/c (best arm) | 0.35 to 0.69 MB (text model, not ONNX) | 0.21 to 0.24 ms |
| mlp/d (INT8 ONNX) | about 4 KB | about 0.006 ms |
| lgbm/a (baseline) | about 2.0 MB (text model) | about 0.25 ms |

The MLP fp32 vs INT8 agreement at threshold 0.5 on 4,096 rows: UNSW 0.986, CSE-CIC 0.834, BoT 0.995,
ToN 0.998. The CSE fold shows the largest drop, so the INT8 graph is not a free approximation there.

## Deviations from the plan and caveats (all stated up front)

1. **Feature count.** The dhoogla NF-*-v2 copies have 41 feature columns (43 total with `Label` and
   `Attack`), not 43, and no IPv4 address columns. The job records the columns it actually used.
2. **Shared schema.** The 33-feature shared schema (`rakshak.nids.schema`) already excludes IPs, L4
   ports, TTL min and max, TCP window and timestamps. The hygiene arm (b) therefore removes only
   `L7_PROTO` and `DNS_TTL_ANSWER`, and adds five ratio features. The port, TTL and window drops in the
   plan are satisfied by the schema, not by new code.
3. **Gate reference.** The gate reference for each model family is that family's baseline arm (`a`):
   LightGBM 300 trees / 63 leaves, or a 64-32 MLP, on the 33 features, LODO over three sources. The
   shipped CSE-only model is reported as a1, outside the gate.
4. **Deduplication removes most rows.** Duplicates on the 33 shared features were removed before
   splitting. The removal was large: UNSW kept 52,062 of 374,951 rows, CSE 153,023 of 600,000, BoT
   37,272 of 429,228, ToN 102,844 of 599,867. The 100k-per-class evaluation cap therefore did not bind
   on UNSW (47,062 test rows) or BoT. Small test sets make the per-seed spread and the paired intervals
   wider than the 100k cap would suggest.
5. **Sampling caps.** The stage caps were 300k benign and 300k attack per dataset (seed 0, rare
   attacks kept in full by water-filling). BoT-IoT has only 129,437 benign rows in total, all kept.
   Prevalence in every target is set by these caps, not by the source datasets' natural prevalence.
6. **Compute-budget amendment.** The first run was aborted after one result (`unsw_lgbm_a_s0`, MCC
   -0.414) because it projected to about 20 h. After the thread fix, the run used: training cap
   30,000 benign and 30,000 attack per source (unchanged), evaluation cap 100,000 per class for
   validation and test, bootstrap draws 100 (was 200), LightGBM and MLP at 3 seeds each. Arms, features,
   threshold rule, hyperparameter grids, metrics and the gate were not changed. The amendment section
   in the plan states this.
7. **MLP is sklearn, not torch.** The MLP arms use sklearn `MLPClassifier`, so the thread cap is set
   through the BLAS environment variables. `torch.set_num_threads` is not used by this job.
8. **Threshold transfer.** Inner thresholds come from models trained on two sources, but are applied
   to the outer model trained on three. The score scale can differ between the two. This is the known
   limitation stated in the plan, and it was not corrected.
9. **Label noise and lab data.** NetFlow-v2 sets are lab captures with label noise (Engelen et al.).
   CSE-CIC-2018 is never the only source. Results are for these samples only.
10. **Bootstrap intervals.** Intervals use 100 resamples of each target's test rows, paired across arms
    within a target. The mean-over-targets interval combines the four targets' draws by index, which
    treats the four targets as independent.

## Tests

`PYTHONPATH=src:deploy/pi/service pytest -q tests/test_nids_xdata.py tests/test_nids*.py` on the VM:
161 passed, 3 skipped (31 s). The VM copy includes `deploy/pi/service` and `kaggle/jobs`, which the
existing NIDS tests import.

## Reproduce

```
python ../runjob.py nids_xdata <workdir> --mode stage --raw <parquet> --key <key> --out-dir ../nids_data
python ../runjob.py nids_xdata <workdir> --mode fold --target <key> --data-dir ../nids_data \
    --out-dir ../out/xdata --seeds 0 1 2 --n-boot 100
python ../runjob.py nids_xdata <workdir> --mode aggregate --out-dir ../out/xdata
```

Run with `PYTHONPATH=src:kaggle/jobs` from the VM copy of the repo. Per-fold JSON and `metrics.json`
are written to the out directory. The committed copy is `docs/results/nids_xdata/metrics.json`.
