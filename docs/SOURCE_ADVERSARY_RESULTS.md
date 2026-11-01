# Source-adversary results — 2026-11-01

**Decision: keep for further validation (go on the recorded C gate).** ScamShield delta +0.0153 [0.0136,0.0174], paired campaign bootstrap; both other source guards pass. No deployment or independent held-out improvement claim.

**In-sample for rule-feature design and repeated benchmark comparisons.** ScamShield rows informed frozen rule features. GroundLM is design-exposed. SmishX is independent, non-Indian-specific for these train-only models; external outcomes select no arm, seed, epoch or threshold. Independent Indian validation remains OPEN.

## Provenance

Training: `sms_source_adversary-c-adv-20261101` v1, tree `a09fb20ed34903e9605e7f7e48c26aed055a4da18ae4a41a0d7c8588214b9573`, ok 3635.9s; 36 fits, six splits, zero failed units. Audit: `sms_source_audit-c-adv-audit-20261101` v1, tree `f50b5a0c3bbd5b55d085b046d1ee1df0d7a21516ca6bae98facf45e3a5ae21a1`, ok 60.3s. One wait_event and one fetch for each. Metrics SHA256 `f58cef333d6c870cb30fcad0cb9416385a845c7bd547a40bb91737a0acc705c4`.

Both arms: frozen train IDs only, matched teacher logits, seeds1/2/3, epochs15, KD alpha.7/T2, AdamW.002, patience3 on cal PR-AUC. Cal-only max-F1/1%-FPR operating points; no split changes. Full predeclaration: SOURCE_ADVERSARY.md. Fresh ERM is the paired comparator; it differs from prior source-control ERM. The prior run is not substituted into any paired test.

## All splits

F1 uses legacy spam-inclusive label_binary. Intervals resample campaigns jointly across arms and all three fixed seeds, 1000 draws/seed13. These condition on the three training seeds, not future seed populations. Tests are unadjusted. Temporal has only77 positives and three original train/test campaign overlaps.

| Split | ERM mean [95% CI] | Adversary mean [95% CI] | Paired gain [95% CI] | p |
|---|---|---|---|---|
| campaign | 0.9748 [0.9725, 0.9769] | 0.9756 [0.9735, 0.9777] | +0.0008 [-0.0001, 0.0018] | 0.093906 |
| lang_shift | 0.3698 [0.3327, 0.4052] | 0.3966 [0.3599, 0.4308] | +0.0269 [0.0136, 0.0387] | 0.000999 |
| source_holdout:holdout_mishra_soni_smsphishing | 0.9373 [0.9178, 0.9525] | 0.9459 [0.9264, 0.9622] | +0.0086 [-0.0003, 0.0175] | 0.059940 |
| source_holdout:holdout_scamshield | 0.7340 [0.7257, 0.7445] | 0.7493 [0.7415, 0.7592] | +0.0153 [0.0136, 0.0174] | 0.000999 |
| source_holdout:holdout_uci_sms_spam | 0.8829 [0.8636, 0.9002] | 0.8913 [0.8731, 0.9083] | +0.0084 [-0.0000, 0.0173] | 0.054945 |
| temporal | 0.8710 [0.8125, 0.9212] | 0.8666 [0.8091, 0.9155] | -0.0044 [-0.0190, 0.0118] | 0.554446 |

## Every seed

| Split | Arm | Seed1 | Seed2 | Seed3 | Test / positive |
|---|---|---|---|---|---|
| campaign | erm | 0.9737 | 0.9740 | 0.9766 | 22145 / 11193 |
| campaign | adversarial | 0.9757 | 0.9748 | 0.9762 | 22145 / 11193 |
| lang_shift | erm | 0.3749 | 0.3569 | 0.3775 | 1677 / 357 |
| lang_shift | adversarial | 0.3447 | 0.4380 | 0.4072 | 1677 / 357 |
| source_holdout:holdout_mishra_soni_smsphishing | erm | 0.9507 | 0.9151 | 0.9460 | 941 / 417 |
| source_holdout:holdout_mishra_soni_smsphishing | adversarial | 0.9563 | 0.9335 | 0.9479 | 941 / 417 |
| source_holdout:holdout_scamshield | erm | 0.7130 | 0.7527 | 0.7362 | 75973 / 29029 |
| source_holdout:holdout_scamshield | adversarial | 0.7496 | 0.7587 | 0.7396 | 75973 / 29029 |
| source_holdout:holdout_uci_sms_spam | erm | 0.8877 | 0.8887 | 0.8721 | 5159 / 642 |
| source_holdout:holdout_uci_sms_spam | adversarial | 0.8899 | 0.8910 | 0.8931 | 5159 / 642 |
| temporal | erm | 0.8951 | 0.8531 | 0.8649 | 11029 / 77 |
| temporal | adversarial | 0.8951 | 0.8158 | 0.8889 | 11029 / 77 |

## Frozen-threshold external results

All campaign models and seeds, fraud and legacy endpoints shown. GroundLM is in-sample for rule-feature design. SmishX fraud endpoint has881 negatives/226 positives; legacy has606/501. Promotions are fraud negatives. FPR denominators are all eligible endpoint negatives; recall denominators are all positives. The 1% label denotes the calibration target, not an external guarantee. CIs are campaign-clustered. No fused system is evaluated here; low rule-layer FPR does not describe this student.

### groundlm / label_scam

| Arm | Seed | Operating point | Threshold | FPR [95% CI] | Recall [95% CI] |
|---|---|---|---|---|---|
| erm | 1 | max_f1 | 0.564250 | 454/1133 = 40.07% [26.2661, 52.0351] | 71/91 = 78.02% [59.5695, 88.9764] |
| erm | 1 | fpr_1pct | 0.874459 | 398/1133 = 35.13% [21.3869, 47.2862] | 60/91 = 65.93% [46.8085, 79.8361] |
| erm | 2 | max_f1 | 0.730319 | 480/1133 = 42.37% [28.7152, 54.2589] | 68/91 = 74.73% [58.3333, 87.5000] |
| erm | 2 | fpr_1pct | 0.975429 | 394/1133 = 34.77% [21.5382, 47.3809] | 51/91 = 56.04% [32.9787, 77.9735] |
| erm | 3 | max_f1 | 0.722350 | 477/1133 = 42.10% [28.0713, 54.9300] | 71/91 = 78.02% [61.1088, 89.9106] |
| erm | 3 | fpr_1pct | 0.973559 | 424/1133 = 37.42% [22.8473, 50.8305] | 59/91 = 64.84% [42.1010, 82.3566] |
| adversarial | 1 | max_f1 | 0.480751 | 472/1133 = 41.66% [27.1315, 54.1102] | 72/91 = 79.12% [62.2951, 90.1879] |
| adversarial | 1 | fpr_1pct | 0.974515 | 401/1133 = 35.39% [21.4394, 48.2819] | 55/91 = 60.44% [37.8900, 78.3393] |
| adversarial | 2 | max_f1 | 0.229850 | 472/1133 = 41.66% [27.8462, 53.9874] | 70/91 = 76.92% [63.4263, 88.3351] |
| adversarial | 2 | fpr_1pct | 0.747670 | 397/1133 = 35.04% [22.7015, 46.0011] | 59/91 = 64.84% [46.9867, 80.0093] |
| adversarial | 3 | max_f1 | 0.235730 | 481/1133 = 42.45% [28.0144, 55.3987] | 79/91 = 86.81% [72.8697, 93.7516] |
| adversarial | 3 | fpr_1pct | 0.941799 | 385/1133 = 33.98% [20.2847, 45.7043] | 64/91 = 70.33% [51.6607, 82.4442] |

### smishx / label_scam

| Arm | Seed | Operating point | Threshold | FPR [95% CI] | Recall [95% CI] |
|---|---|---|---|---|---|
| erm | 1 | max_f1 | 0.564250 | 318/881 = 36.10% [31.1733, 40.6181] | 180/226 = 79.65% [70.0780, 89.6739] |
| erm | 1 | fpr_1pct | 0.874459 | 288/881 = 32.69% [27.6995, 37.2306] | 138/226 = 61.06% [51.2629, 72.5667] |
| erm | 2 | max_f1 | 0.730319 | 297/881 = 33.71% [29.0093, 38.1741] | 149/226 = 65.93% [56.6026, 76.2628] |
| erm | 2 | fpr_1pct | 0.975429 | 184/881 = 20.89% [17.0323, 24.8133] | 96/226 = 42.48% [32.9194, 54.2475] |
| erm | 3 | max_f1 | 0.722350 | 322/881 = 36.55% [31.7201, 41.2363] | 187/226 = 82.74% [76.7323, 88.3842] |
| erm | 3 | fpr_1pct | 0.973559 | 305/881 = 34.62% [29.8231, 39.3432] | 162/226 = 71.68% [64.0163, 79.3537] |
| adversarial | 1 | max_f1 | 0.480751 | 335/881 = 38.02% [33.1361, 42.6305] | 193/226 = 85.40% [79.1301, 91.5688] |
| adversarial | 1 | fpr_1pct | 0.974515 | 280/881 = 31.78% [27.2125, 36.3639] | 141/226 = 62.39% [53.7626, 71.6585] |
| adversarial | 2 | max_f1 | 0.229850 | 311/881 = 35.30% [30.2835, 39.9133] | 183/226 = 80.97% [72.9625, 88.5845] |
| adversarial | 2 | fpr_1pct | 0.747670 | 275/881 = 31.21% [26.5544, 35.8141] | 137/226 = 60.62% [50.1939, 71.3517] |
| adversarial | 3 | max_f1 | 0.235730 | 326/881 = 37.00% [31.8913, 41.5622] | 178/226 = 78.76% [71.4875, 86.6968] |
| adversarial | 3 | fpr_1pct | 0.941799 | 249/881 = 28.26% [23.8033, 32.6848] | 114/226 = 50.44% [40.1684, 61.7443] |

### smishx / label_binary

| Arm | Seed | Operating point | Threshold | FPR [95% CI] | Recall [95% CI] |
|---|---|---|---|---|---|
| erm | 1 | max_f1 | 0.564250 | 64/606 = 10.56% [8.2366, 13.0938] | 434/501 = 86.63% [80.6318, 91.7674] |
| erm | 1 | fpr_1pct | 0.874459 | 50/606 = 8.25% [6.1258, 10.5611] | 376/501 = 75.05% [67.7822, 81.7642] |
| erm | 2 | max_f1 | 0.730319 | 62/606 = 10.23% [8.0697, 12.8506] | 384/501 = 76.65% [70.7179, 82.3795] |
| erm | 2 | fpr_1pct | 0.975429 | 33/606 = 5.45% [3.6424, 7.2868] | 247/501 = 49.30% [41.7909, 56.5685] |
| erm | 3 | max_f1 | 0.722350 | 65/606 = 10.73% [8.2508, 13.2019] | 444/501 = 88.62% [84.6442, 92.2776] |
| erm | 3 | fpr_1pct | 0.973559 | 54/606 = 8.91% [6.6113, 11.1856] | 413/501 = 82.44% [77.6065, 87.0669] |
| adversarial | 1 | max_f1 | 0.480751 | 72/606 = 11.88% [9.2865, 14.5909] | 456/501 = 91.02% [87.1780, 94.4683] |
| adversarial | 1 | fpr_1pct | 0.974515 | 51/606 = 8.42% [6.2914, 10.6741] | 370/501 = 73.85% [67.9704, 79.7959] |
| adversarial | 2 | max_f1 | 0.229850 | 61/606 = 10.07% [7.8070, 12.5206] | 433/501 = 86.43% [81.4036, 90.8723] |
| adversarial | 2 | fpr_1pct | 0.747670 | 47/606 = 7.76% [5.7756, 10.0333] | 365/501 = 72.85% [65.9299, 79.2096] |
| adversarial | 3 | max_f1 | 0.235730 | 69/606 = 11.39% [8.9082, 13.9585] | 435/501 = 86.83% [82.2619, 91.0914] |
| adversarial | 3 | fpr_1pct | 0.941799 | 49/606 = 8.09% [6.0922, 10.2479] | 314/501 = 62.67% [55.6123, 69.6847] |

PhreshPhish is a URL protocol, not evaluated by this SMS intervention. No frozen split/external set changed. Synthetic Torch invariants ran on Kaggle before corpus access; local Torch skip is not counted as a pass. No model checkpoint is shipped by this experiment harness; deployability and fused false alarms remain open. Next priority is the full frozen fusion audit, followed by clean KD vs no-KD.
