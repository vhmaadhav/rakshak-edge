# Source-control results and KD-pool audit — 2026-11-01

**Decision: kill source-balanced sampling and balanced + group-DRO for this target.** Neither improves ScamShield under the predeclared paired campaign gate. Keep the train-only KD correction and report the negative intervention results. No new model checkpoint is shipped by this experiment.

**In-sample for rule-feature design:** ScamShield test rows informed the frozen rules. These are comparisons on named frozen splits, not a clean independent held-out improvement claim. Selection/stopping/thresholds use calibration only; all three seeds (1/2/3) and both predeclared intervention arms are reported, without multiplicity adjustment. Campaign intervals are conditional on these three seeds, not population intervals over future training seeds.

## Provenance

| Private kernel / version | Tree SHA256 | Outcome |
|---|---|---|
| `sms_source_balance-c-clean-20261101` / 1 | `62a788d27aa9db20b84964f00b8208d3025c3be8f5128352d5c76c53b5ae589e` | ok, 5316.6 s; six splits, 54 successful fit units |
| `sms_source_audit-c-audit-20261101` / 1 | `e83b49d3e41ca29a849b1c0247bfe95ed9b140e918f565a6f9fa3c8bd5166239` | ok, 74.1 s; both arms kill |

Source metrics SHA256: `db3811b2ede62d577e3dad40684276d0ec0580d231d0505cafe3d5d54fb46538`. Base commit `2463245`, dirty bundles; one event wait and one fetch for each. Original broad-pool `c-src-20261101` was cancelled before inspecting outcomes; its partial outputs are quarantined and excluded.

Training args: `--seeds 1 2 3 --epochs 15 --n-boot 1000 --teacher-seed 0`. Private inputs `maadhavvh/rakshak-data` and `maadhavvh/rakshak-teacher-logits`. All arms use exactly the frozen train IDs and matched teacher logits; no extra unlabeled rows. Dual-base, KD alpha .7 / T2, AdamW .002, weight decay .0001, patience3 on cal PR-AUC. See `SOURCE_BALANCE.md` for the full predeclaration. Protocol correction preceded outcomes; historical ~.75 was a reference, not an assumed replication. Fresh train-only ERM is the paired control.

## Mean F1 over the three seeds

Absolute CIs and differences resample whole campaigns, with the same draw applied to both arms and every seed (1,000 draws, seed13). F1 uses each model's calibration-max-F1 threshold. All internal endpoints remain legacy spam-inclusive `label_binary`.

| Split | ERM mean [95% CI] | Balanced mean [95% CI] | DRO mean [95% CI] |
|---|---|---|---|
| campaign | 0.9753 [0.9732, 0.9774] | 0.9735 [0.9713, 0.9758] | 0.9727 [0.9704, 0.9749] |
| lang_shift | 0.4000 [0.3606, 0.4376] | 0.3367 [0.3028, 0.3684] | 0.4214 [0.3834, 0.4571] |
| source_holdout: mishra_soni_smsphishing | 0.9344 [0.9149, 0.9493] | 0.9429 [0.9254, 0.9569] | 0.9330 [0.9129, 0.9489] |
| source_holdout: scamshield | 0.7503 [0.7424, 0.7602] | 0.7437 [0.7357, 0.7536] | 0.7399 [0.7319, 0.7497] |
| source_holdout: uci_sms_spam | 0.8872 [0.8685, 0.9034] | 0.8818 [0.8616, 0.8988] | 0.8912 [0.8737, 0.9075] |
| temporal | 0.8513 [0.7907, 0.9025] | 0.8875 [0.8317, 0.9369] | 0.8946 [0.8387, 0.9420] |

| Split | Intervention | Delta F1 [paired 95% CI] | Null-centred p |
|---|---|---|---|
| campaign | balanced | -0.0018 [-0.0027, -0.0008] | 0.0010 |
| campaign | dro | -0.0027 [-0.0037, -0.0017] | 0.0010 |
| lang_shift | balanced | -0.0633 [-0.0771, -0.0509] | 0.0010 |
| lang_shift | dro | +0.0214 [0.0010, 0.0392] | 0.0280 |
| source_holdout: mishra_soni_smsphishing | balanced | +0.0085 [0.0009, 0.0174] | 0.0480 |
| source_holdout: mishra_soni_smsphishing | dro | -0.0014 [-0.0089, 0.0060] | 0.7223 |
| source_holdout: scamshield | balanced | -0.0065 [-0.0079, -0.0052] | 0.0010 |
| source_holdout: scamshield | dro | -0.0104 [-0.0122, -0.0088] | 0.0010 |
| source_holdout: uci_sms_spam | balanced | -0.0054 [-0.0163, 0.0049] | 0.2987 |
| source_holdout: uci_sms_spam | dro | +0.0041 [-0.0041, 0.0128] | 0.3796 |
| temporal | balanced | +0.0361 [0.0177, 0.0615] | 0.0080 |
| temporal | dro | +0.0433 [0.0172, 0.0771] | 0.0090 |

ScamShield required a strictly positive delta lower CI. Balanced instead gives −0.0065 [−0.0079, −0.0052]; DRO −0.0104 [−0.0122, −0.0088]. Both other-source guards pass (allowed loss is the ERM CI half-width), but that cannot compensate for failure on the primary target. Improvements on temporal or DRO's language shift do not change the decision.

## Every seed

| Split | Arm | Seed1 F1 | Seed2 F1 | Seed3 F1 | Test rows / positives |
|---|---|---:|---:|---:|---|
| campaign | erm | 0.9752 | 0.9746 | 0.9762 | 22145 / 11193 |
| campaign | balanced | 0.9728 | 0.9751 | 0.9728 | 22145 / 11193 |
| campaign | dro | 0.9738 | 0.9723 | 0.9718 | 22145 / 11193 |
| lang_shift | erm | 0.3851 | 0.4939 | 0.3209 | 1677 / 357 |
| lang_shift | balanced | 0.2960 | 0.3006 | 0.4133 | 1677 / 357 |
| lang_shift | dro | 0.3145 | 0.6083 | 0.3414 | 1677 / 357 |
| source_holdout: mishra_soni_smsphishing | erm | 0.9406 | 0.9326 | 0.9299 | 941 / 417 |
| source_holdout: mishra_soni_smsphishing | balanced | 0.9585 | 0.9159 | 0.9543 | 941 / 417 |
| source_holdout: mishra_soni_smsphishing | dro | 0.9513 | 0.9215 | 0.9261 | 941 / 417 |
| source_holdout: scamshield | erm | 0.7474 | 0.7648 | 0.7385 | 75973 / 29029 |
| source_holdout: scamshield | balanced | 0.7129 | 0.7863 | 0.7320 | 75973 / 29029 |
| source_holdout: scamshield | dro | 0.7474 | 0.7756 | 0.6965 | 75973 / 29029 |
| source_holdout: uci_sms_spam | erm | 0.8854 | 0.8819 | 0.8942 | 5159 / 642 |
| source_holdout: uci_sms_spam | balanced | 0.8722 | 0.9026 | 0.8705 | 5159 / 642 |
| source_holdout: uci_sms_spam | dro | 0.8855 | 0.9008 | 0.8874 | 5159 / 642 |
| temporal | erm | 0.8552 | 0.8378 | 0.8609 | 11029 / 77 |
| temporal | balanced | 0.9078 | 0.8531 | 0.9014 | 11029 / 77 |
| temporal | dro | 0.9078 | 0.8841 | 0.8921 | 11029 / 77 |

No splits were skipped. Temporal has only 77 positive test messages and its source/label confound remains. Its original frozen train partition also shares three campaign IDs with the test partition; the split was not changed. Temporal is not a campaign-disjoint or clean source-shift claim.

## Inherited broad KD pool: confirmed protocol contamination

Counts below concern the old extra pool, **not** the corrected training rows. Categories overlap and must not be added. Exact-key/campaign membership checks do not measure all possible near-duplicate overlap. External count combines GroundLM and SmishX exact keys; it is not a per-set count.

| Split | Extra rows | Exact test copies | Exact cal copies | Test campaigns | Held-out source | External exact | Frozen train in test campaigns |
|---|---:|---:|---:|---:|---:|---:|---:|
| campaign | 27345 | 4147 | 2033 | 4147 | 0 | 2138 | 0 |
| lang_shift | 68417 | 14 | 1759 | 14 | 0 | 2340 | 0 |
| source_holdout: mishra_soni_smsphishing | 28469 | 730 | 1903 | 3181 | 5030 | 2349 | 0 |
| source_holdout: scamshield | 34035 | 2122 | 1531 | 11137 | 9707 | 1994 | 0 |
| source_holdout: uci_sms_spam | 29007 | 14299 | 576 | 16695 | 415 | 1937 | 0 |
| temporal | 85613 | 3019 | 1468 | 6491 | 0 | 2152 | 3 |

Historical `sms_student-kd` runs with unlabeled extras cannot establish campaign-disjoint, source-disjoint, language-shift or independent-external generalization. In particular the previous ~0.80 language-shift claim is withdrawn as clean evidence. These counts do not by themselves estimate the causal size of that bias; the new three-seed train-only ERM changes the pool and is reported separately.

`sms_student` now defaults to train-only KD and refuses `--kd-unlabeled`. `build_training_pool` enforces that policy even for programmatic callers; the legacy broad-pool builder remains for membership auditing. Frozen split files and external rows are unchanged. A future unlabeled KD pipeline needs an independent pool with campaign/external-overlap exclusions before use.

## External tests at calibration-frozen thresholds

Each row uses a campaign-fitted model and its own frozen calibration threshold. FPR/recall show events/denominator, percent and campaign-bootstrap 95% CI; ordinary Wilson references remain in the private metrics. `fpr_1pct` is a **calibration target**, not an external-FPR guarantee. No external result selected an arm, seed, stopping epoch or threshold.

GroundLM is **in-sample for rule-feature design**. SmishX is **independent, non-Indian-specific** for these corrected train-only models. It has 1,107 eligible messages; fraud endpoint: 881 non-fraud /226 fraud. The legacy spam-inclusive endpoint has 606 negatives /501 positives. Promotions are negatives only for the fraud endpoint; do not conflate them.

### groundlm / label_scam

GroundLM `label_binary` and `label_scam` endpoint metrics are identical for all nine models; this table reports both.

| Arm | Seed | Operating point | Threshold | FPR [95% CI] | Recall [95% CI] |
|---|---:|---|---:|---|---|
| erm | 1 | max_f1 | 0.458124 | 458/1133 = 40.42% [26.65, 52.69] | 78/91 = 85.71% [74.35, 92.91] |
| erm | 1 | fpr_1pct | 0.944833 | 407/1133 = 35.92% [21.80, 48.54] | 69/91 = 75.82% [61.02, 87.40] |
| erm | 2 | max_f1 | 0.251412 | 498/1133 = 43.95% [29.46, 56.42] | 73/91 = 80.22% [62.50, 92.56] |
| erm | 2 | fpr_1pct | 0.930649 | 415/1133 = 36.63% [23.19, 48.65] | 56/91 = 61.54% [39.75, 80.59] |
| erm | 3 | max_f1 | 0.489942 | 484/1133 = 42.72% [29.08, 55.21] | 74/91 = 81.32% [64.41, 90.00] |
| erm | 3 | fpr_1pct | 0.962241 | 406/1133 = 35.83% [21.47, 48.00] | 59/91 = 64.84% [44.93, 81.01] |
| balanced | 1 | max_f1 | 0.222038 | 443/1133 = 39.10% [25.33, 51.46] | 74/91 = 81.32% [61.64, 91.21] |
| balanced | 1 | fpr_1pct | 0.875912 | 356/1133 = 31.42% [18.04, 43.78] | 60/91 = 65.93% [45.76, 80.51] |
| balanced | 2 | max_f1 | 0.450325 | 427/1133 = 37.69% [23.59, 50.78] | 68/91 = 74.73% [55.93, 87.41] |
| balanced | 2 | fpr_1pct | 0.951204 | 333/1133 = 29.39% [16.02, 41.18] | 47/91 = 51.65% [27.40, 74.79] |
| balanced | 3 | max_f1 | 0.663300 | 487/1133 = 42.98% [29.76, 54.58] | 71/91 = 78.02% [59.37, 88.24] |
| balanced | 3 | fpr_1pct | 0.969812 | 354/1133 = 31.24% [18.33, 43.66] | 59/91 = 64.84% [40.27, 82.83] |
| dro | 1 | max_f1 | 0.074611 | 461/1133 = 40.69% [27.68, 52.55] | 69/91 = 75.82% [62.10, 87.50] |
| dro | 1 | fpr_1pct | 0.467025 | 381/1133 = 33.63% [20.04, 45.56] | 54/91 = 59.34% [37.86, 80.67] |
| dro | 2 | max_f1 | 0.326318 | 455/1133 = 40.16% [26.26, 52.46] | 72/91 = 79.12% [63.33, 88.78] |
| dro | 2 | fpr_1pct | 0.827981 | 377/1133 = 33.27% [20.25, 45.92] | 55/91 = 60.44% [40.00, 77.62] |
| dro | 3 | max_f1 | 0.675872 | 469/1133 = 41.39% [27.79, 53.40] | 72/91 = 79.12% [60.65, 91.77] |
| dro | 3 | fpr_1pct | 0.943614 | 381/1133 = 33.63% [20.48, 45.83] | 62/91 = 68.13% [48.28, 84.09] |

### smishx / label_scam

| Arm | Seed | Operating point | Threshold | FPR [95% CI] | Recall [95% CI] |
|---|---:|---|---:|---|---|
| erm | 1 | max_f1 | 0.458124 | 315/881 = 35.75% [30.74, 40.34] | 161/226 = 71.24% [61.64, 81.19] |
| erm | 1 | fpr_1pct | 0.944833 | 267/881 = 30.31% [25.81, 34.67] | 119/226 = 52.65% [43.71, 62.72] |
| erm | 2 | max_f1 | 0.251412 | 328/881 = 37.23% [32.24, 42.00] | 201/226 = 88.94% [80.90, 94.90] |
| erm | 2 | fpr_1pct | 0.930649 | 282/881 = 32.01% [27.15, 36.38] | 154/226 = 68.14% [59.60, 77.35] |
| erm | 3 | max_f1 | 0.489942 | 327/881 = 37.12% [32.27, 41.72] | 186/226 = 82.30% [74.51, 89.32] |
| erm | 3 | fpr_1pct | 0.962241 | 296/881 = 33.60% [28.65, 38.08] | 151/226 = 66.81% [57.53, 76.32] |
| balanced | 1 | max_f1 | 0.222038 | 320/881 = 36.32% [31.41, 40.87] | 190/226 = 84.07% [76.19, 90.48] |
| balanced | 1 | fpr_1pct | 0.875912 | 257/881 = 29.17% [24.57, 33.59] | 141/226 = 62.39% [53.36, 71.88] |
| balanced | 2 | max_f1 | 0.450325 | 334/881 = 37.91% [32.90, 42.62] | 192/226 = 84.96% [78.45, 91.59] |
| balanced | 2 | fpr_1pct | 0.951204 | 280/881 = 31.78% [27.28, 36.52] | 136/226 = 60.18% [49.43, 71.43] |
| balanced | 3 | max_f1 | 0.663300 | 335/881 = 38.02% [33.13, 42.86] | 206/226 = 91.15% [86.59, 95.43] |
| balanced | 3 | fpr_1pct | 0.969812 | 307/881 = 34.85% [29.81, 39.62] | 170/226 = 75.22% [66.55, 84.48] |
| dro | 1 | max_f1 | 0.074611 | 313/881 = 35.53% [30.83, 40.09] | 170/226 = 75.22% [64.66, 86.45] |
| dro | 1 | fpr_1pct | 0.467025 | 241/881 = 27.36% [23.04, 31.73] | 111/226 = 49.12% [40.08, 60.47] |
| dro | 2 | max_f1 | 0.326318 | 306/881 = 34.73% [29.94, 39.52] | 164/226 = 72.57% [64.43, 80.00] |
| dro | 2 | fpr_1pct | 0.827981 | 241/881 = 27.36% [23.12, 31.79] | 120/226 = 53.10% [43.22, 63.02] |
| dro | 3 | max_f1 | 0.675872 | 309/881 = 35.07% [30.26, 39.66] | 170/226 = 75.22% [65.16, 85.88] |
| dro | 3 | fpr_1pct | 0.943614 | 238/881 = 27.01% [22.54, 31.57] | 114/226 = 50.44% [41.21, 62.12] |

### smishx / label_binary

| Arm | Seed | Operating point | Threshold | FPR [95% CI] | Recall [95% CI] |
|---|---:|---|---:|---|---|
| erm | 1 | max_f1 | 0.458124 | 64/606 = 10.56% [8.26, 13.06] | 412/501 = 82.24% [76.02, 87.99] |
| erm | 1 | fpr_1pct | 0.944833 | 50/606 = 8.25% [6.10, 10.67] | 336/501 = 67.07% [60.04, 73.83] |
| erm | 2 | max_f1 | 0.251412 | 64/606 = 10.56% [8.26, 13.16] | 465/501 = 92.81% [89.01, 96.14] |
| erm | 2 | fpr_1pct | 0.930649 | 41/606 = 6.77% [4.94, 8.76] | 395/501 = 78.84% [73.18, 84.18] |
| erm | 3 | max_f1 | 0.489942 | 67/606 = 11.06% [8.62, 13.53] | 446/501 = 89.02% [84.77, 92.86] |
| erm | 3 | fpr_1pct | 0.962241 | 53/606 = 8.75% [6.62, 10.95] | 394/501 = 78.64% [73.12, 84.29] |
| balanced | 1 | max_f1 | 0.222038 | 64/606 = 10.56% [8.11, 13.04] | 446/501 = 89.02% [85.14, 92.66] |
| balanced | 1 | fpr_1pct | 0.875912 | 48/606 = 7.92% [5.81, 10.23] | 350/501 = 69.86% [63.55, 75.68] |
| balanced | 2 | max_f1 | 0.450325 | 73/606 = 12.05% [9.54, 14.71] | 453/501 = 90.42% [86.70, 94.04] |
| balanced | 2 | fpr_1pct | 0.951204 | 53/606 = 8.75% [6.61, 11.02] | 363/501 = 72.46% [65.17, 78.98] |
| balanced | 3 | max_f1 | 0.663300 | 73/606 = 12.05% [9.55, 14.83] | 468/501 = 93.41% [90.25, 96.12] |
| balanced | 3 | fpr_1pct | 0.969812 | 55/606 = 9.08% [6.93, 11.37] | 422/501 = 84.23% [78.83, 89.46] |
| dro | 1 | max_f1 | 0.074611 | 63/606 = 10.40% [8.11, 12.94] | 420/501 = 83.83% [77.17, 89.61] |
| dro | 1 | fpr_1pct | 0.467025 | 45/606 = 7.43% [5.45, 9.69] | 307/501 = 61.28% [54.23, 69.05] |
| dro | 2 | max_f1 | 0.326318 | 70/606 = 11.55% [9.21, 14.29] | 400/501 = 79.84% [74.53, 84.84] |
| dro | 2 | fpr_1pct | 0.827981 | 53/606 = 8.75% [6.62, 11.15] | 308/501 = 61.48% [54.50, 68.05] |
| dro | 3 | max_f1 | 0.675872 | 64/606 = 10.56% [8.25, 12.99] | 415/501 = 82.83% [75.91, 88.84] |
| dro | 3 | fpr_1pct | 0.943614 | 46/606 = 7.59% [5.61, 9.75] | 306/501 = 61.08% [53.64, 68.24] |

External FPR remains far above the 1% reference on GroundLM and the SmishX fraud endpoint. No Indian-safety or independent held-out improvement claim is supported. Independent Indian validation remains **OPEN**: collect consented, privacy-scrubbed real bank/UPI/OTP/delivery messages and received scams after rules are frozen, human label/group them, keep them private and never tune on them. PhreshPhish is a URL protocol; no URL model was changed or evaluated in this SMS experiment.

## Milestone validation

One complete local synthetic-fixture unit-suite invocation (no corpus/model benchmark):

```text
830 passed, 21 skipped, 144 warnings in 1376.39s (0:22:56)
```

Full `ruff check .`:

```text
All checks passed!
```

Tests used `nice -n 19 taskset -c 0-3`, `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`,
with a temporary user-scope CPU quota15% raised to25% while cool, and an85 C stop guard.
Pre-run Tctl38.2 C; final38.4 C. Three new source-objective tests are skipped locally because
Torch is absent; the Kaggle job verified source-loss gradients/KD equivalence before its54
successful fits. The skipped tests are not counted as passes. No source code changed after
these checks. Data, predictions, checkpoints and logits remain private and uncommitted.
