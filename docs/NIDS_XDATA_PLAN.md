# NIDS cross-dataset generalisation: pre-registered plan

Status: written before any target-dataset evaluation. Adapts section 6 of the research brief
(`nids.md`) to the four NetFlow-v2 sets we hold. Numbers in `NIDS_XDATA_RESULTS.md` must follow this
document. Changes after the first target result are reported as deviations, not silently applied.

## 1. Data

- Sets: NF-UNSW-NB15-v2, NF-CSE-CIC-IDS2018-v2, NF-BoT-IoT-v2, NF-ToN-IoT-v2 (Kaggle mirrors
  `dhoogla/nfunswnb15v2`, `dhoogla/nfcsecicids2018v2`, `dhoogla/nfbotiotv2`, `dhoogla/nftoniotv2`).
  Binary label: benign = 0, any attack = 1.
- Staging (one dataset at a time, on the VM; `kaggle/jobs/nids_xdata.py --mode stage`):
  stratified sample capped at 300,000 benign and 300,000 attack rows per dataset. Attack strings are
  water-filled with `splits.plan_quotas`, so rare attacks keep all their rows. Seed 0. Columns kept:
  every numeric NetFlow-v2 feature column (43 expected) plus `Label` and `Attack`.
- Experiment sample: duplicates removed on the 33 shared features (as `sample_dataset`).
- Per-source training subsample (fixed, seed 0): 30,000 benign + 30,000 attack, or all rows when a
  class has fewer. Equal per source, so no source dominates.
- Target test set: the target sample minus the 5,000-flow unlabelled calibration head
  (`baseline.calibration_slice`). The head is removed from the test set for every arm, so all arms are
  scored on identical rows. The head's labels are never read.
- Target prevalence is whatever the cap produces. It is reported with every result, because MCC, F1
  and FPR depend on it.

## 2. Features

Frozen before any target result. Built from sources only.

- **Base (arm a):** the 33 shared features of `rakshak.nids.schema.SHARED_FEATURES`, transformed by
  `schema.coerce_matrix` (caps, then log1p on non-categorical columns). Same as `nids_baselines`.
- **Hygiene (arm b):** base minus `L7_PROTO` and `DNS_TTL_ANSWER` (service and TTL artefacts). The
  raw columns dropped from every arm are IPv4 addresses, L4 ports, MIN_TTL, MAX_TTL, TCP_WIN_MAX_IN/OUT,
  FLOW_START/END_MILLISECONDS, FLOW_ID and DNS_QUERY_ID, if present in the file. Added engineered
  features, computed on raw values and then log1p(x):
  bytes per packet `(IN_BYTES+OUT_BYTES)/(IN_PKTS+OUT_PKTS)`, in/out byte ratio
  `IN_BYTES/(OUT_BYTES+1)`, in/out packet ratio `IN_PKTS/(OUT_PKTS+1)`, and duration ratios
  `DURATION_IN/(FLOW_DURATION_MILLISECONDS+1)` and `DURATION_OUT/(FLOW_DURATION_MILLISECONDS+1)`.
  Only the NetFlow columns above are used; no host, port or time column enters.
- **Stability selection (arm c):** inside each outer LODO fold, for each of the three source datasets
  separately, rank the hygiene features by `mutual_info_classif` on that source's training subsample
  (50,000-row cap, random_state 0). Keep the features ranked in the top k = 20 on every source. If that
  leaves fewer than 8, keep the 8 features with the best worst-source rank. The held-out target is
  never used.
- **Quantile normalisation (arm d):** arm c's features, each dataset mapped by its own per-feature
  quantile transform (uniform output, categorical columns passed through, `baseline.fit_site_baseline`).
  Sources: fitted on the source's benign rows of the sample. Target: fitted on the 5,000 unlabelled
  head only (site calibration, RADAR-style). Reported separately as the adaptation arm.

## 3. Arms and models

| Arm | Features | Normalisation | LightGBM | MLP |
|---|---|---|---|---|
| a1 (reference only) | base, CSE-only source | none | 300 trees, 63 leaves | not run |
| a (baseline) | base, LODO on 3 sources | none | 300 trees, 63 leaves | 64-32, sklearn default alpha |
| b | hygiene + engineered | none | <=200 trees, leaves in {15, 31}, lr 0.1 | 2x32, alpha in {1e-4, 1e-2} |
| c | b, stability-selected | none | same | same |
| d | c | per-dataset benign quantiles | same | same |

- a1 is the shipped recipe reduced to its training data: a LightGBM on CSE-CIC-2018 alone. It is
  scored on the three non-CSE targets only, and its threshold is the max-F1 point on a 15% held-out
  CSE split. It gives context only and is not in the gate.
- Gate reference per model family: a-LightGBM for the LightGBM arms, a-MLP for the MLP arms.
- Every MLP is exported to ONNX (fp32, then INT8 with `onnxruntime` dynamic quantisation, as the
  existing student) and all reported MLP metrics come from the INT8 graph. For arm d the quantile map
  is applied before the graph; the on-device cost of that map is reported separately.

## 4. Protocol

1. **Outer loop:** leave-one-dataset-out over the four sets. Each set is the target once; the other
   three are sources.
2. **Inner loop (threshold and hyperparameters):** inside each outer fold, leave-one-source-out over the
   three sources: train on two, validate on the third. Three inner folds.
3. **Hyperparameter choice:** the grid value with the highest mean inner MCC at its best threshold.
   Ties go to the lower mean inner FPR. The baseline arm has no grid.
4. **Threshold rule:** candidate thresholds are 200 quantiles of the pooled inner validation scores.
   Choose the threshold that maximises the mean inner MCC across the three inner folds. Ties go to the
   higher threshold (lower FPR). Fixed before any outer result. The outer model (trained on all three
   sources) uses this score threshold unchanged. Known limitation: inner models are trained on two
   sources, so their score scale can differ from the outer model; this is reported, not corrected.
5. **Seeds:** 3 seeds (0, 1, 2) for every configuration. The seed changes model fitting and the
   subsamples only, not the data split. Report mean and standard deviation across seeds.
6. **Metrics:** primary MCC. Secondary F1 of the attack class, PR-AUC (average precision), FPR on the
   target's benign flows at the chosen threshold, TPR, and target prevalence.
7. **Confidence intervals:** 200-resample bootstrap over the target test rows (seed-0 model). Paired
   intervals for the MCC difference against the gate reference use the same resampled rows.
8. **FPR check:** at the chosen threshold the target's benign-flow FPR must be <= 1%. The actual FPR
   is reported, and the threshold is never changed to meet it.

## 5. Success gate (pre-registered)

For each non-baseline arm (b, c, d) and each model family, compared with its gate reference:

- mean over the four targets of the seed-averaged MCC is at least 0.10 higher than the reference, and
- the worst-target MCC is at least 0.05 higher than the reference's worst-target MCC, and
- the paired bootstrap 95% interval of the mean-MCC difference excludes zero.

All arms are reported. Reporting every arm makes the selection of a best arm visible, so a pass by
the best of several arms is reported with that caveat. A failure is reported as a failure.

## 6. Known limits (stated in advance)

- NF-v2 sets are lab captures with recognised label noise (Engelen et al.). CSE-CIC-2018 is the set
  with the most noise concern, so it is never used as the only source.
- Sampling caps change prevalence and the attack mix; results are for these samples only.
- The 5,000-flow calibration head is unlabelled but drawn at random, so its attack share is
  whatever the set contains.
- Hyperparameter grids are small on purpose. A larger search would need its own pre-registration.
