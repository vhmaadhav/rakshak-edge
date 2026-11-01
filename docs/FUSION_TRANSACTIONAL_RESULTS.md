# Cal-only fusion baseline and transactional-context candidate — 2026-11-01

**Operational decision: kill candidate(b).** The original registered runner decision is **unmeasurable** because UCI has no fraud-labelled positives. The saved-output supplement preserves that decision and separately lists14 observable failures and3 NA endpoints. The candidate already violates the explicit external safety constraints; NA UCI recall cannot rescue it. No post-outcome fitting, relabelling or threshold changes.

**These are newly fitted cal-only models, not an identified shipped configuration.** No deployed fusion state/threshold was located; private HF access returned404. The shipped-system audit remains OPEN pending artifact identification. This failure does not establish the performance of an unidentified deployed bundle.

GroundLM is in-sample for rule-feature design. User-provided aggregate GroundLM/SmishX FPR feedback motivated this new candidate selection; these are **in-sample for intervention selection/repeated benchmark feedback**, not untouched validation. Independent Indian validation remains OPEN. No external/test row was copied into candidate features or synthetic messages.

## Provenance and fixed experiment

Private `fusion_transactional-p1-cal-20261101` v1, tree `67b2545f3d290a11a565003c15b15639ed70ecaf87861b7c22b44ef32b8c6d84`, ok 361.4s; reporting `fusion_final_audit-p1-final-20261101` v1, tree `7efd60c0756370f5facecd16e5eaf6386be113c9be55fad7693a22403b83b0c7`, ok 63.1s. One wait/fetch each. Args `--n-boot 1000` for the model comparison, no reporting args.

Source metrics SHA `1ccaf29eb90e4542dc93881282255d6f4670651d888bd50abc5e2d3d68378984`; frozen states SHA `87b5da994a3c7d579362243fc8cfcf8bfa827cd08621842478cff25f02218ce6`. Original ERM scores source tree `62a788d27aa9db20b84964f00b8208d3025c3be8f5128352d5c76c53b5ae589e`; metrics SHA `db3811b2ede62d577e3dad40684276d0ec0580d231d0505cafe3d5d54fb46538`. URL source tree `5131673567fcf4d3be5eda9f43d28194632d2cce7e2d0940fc06a400eede36db`, ONNX SHA `1bbd2c57cb541a7cbc287887ef51976d54240a353655827a511c63ec56213fc7`, scaler SHA `c6b3fc968b03339a33052fe814a9bb851d3035cb302b06458955c2a5e5b6501e`.

Corrected frozen-train-only ERM student seeds1/2/3 unchanged; fixed PhreshPhish-fitted URL charcnn-wide seed0, offline raw bytes+side features, max over present URLs. No URL retraining or new PhreshPhish protocol claim; cross-task URL-training overlap has not been audited. Missing URLs stay missing with an explicit absence flag. Rules/canonicaliser frozen. Baseline uses existing fusion features; candidate adds do-not-share, OTP+do-not-share+no-URL and transaction keywords. These are lexical clues, not authenticated sender/transaction evidence or a handwritten risk cap.

LR stacker, Platt, empirical thresholds(alpha.01/beta.05), reserved CRC diagnostics use four label-blind campaign-disjoint cal stages(seed0). Every state was saved/hashed before test/external decisions. The underlying student was already early-stopped on its original full cal partition; subdivisions do not undo that selection exposure. CRC is diagnostic, not a certified guarantee under reused expert-selection data or domain shift. No externally selected seed, threshold or feature revision.

Go required every seed to satisfy GroundLM SCAM<=1%, >=UNCERTAIN<=10%, SmishX SCAM recall>=student at its original cal1%-FPR threshold, no GroundLM FPR increase against paired baseline, and no frozen-fold recall loss beyond baseline CI half-width. Every exact fold/seed is reported.

## External results, frozen thresholds

Unconditional FPR includes every known endpoint negative, including abstentions. Recall includes every positive; coverage is the fraction auto-decided. All CIs are campaign-clustered percentile95%,1000 draws/seed13; fitted objects held fixed. IID Wilson references and conditional-on-auto-decision FPR remain in private JSON. Zero-event bootstrap CIs can collapse and do not prove zero population risk. GroundLM uses1224 scoreable/eval-eligible rows (1133 negatives/91 positives); unavailable/excluded release rows are not silently scored. GroundLM positive adjudication has the source paper’s LLM-related limitation.

### groundlm / label_scam

| Arm | Seed | SCAM FPR [95% CI] | >=UNCERTAIN FPR [95% CI] | SCAM recall [95% CI] | Coverage [95% CI] |
|---|---:|---|---|---|---|
| baseline | 1 | 398/1133=35.13% [21.26, 47.44] | 419/1133=36.98% [23.09, 49.88] | 40/91=43.96% [24.54, 64.29] | 1196/1224=97.71% [96.55, 98.89] |
| transactional | 1 | 418/1133=36.89% [22.27, 49.75] | 423/1133=37.33% [23.32, 50.37] | 65/91=71.43% [56.51, 84.03] | 1214/1224=99.18% [98.52, 99.80] |
| baseline | 2 | 421/1133=37.16% [23.50, 49.60] | 445/1133=39.28% [25.24, 52.17] | 31/91=34.07% [18.18, 56.67] | 1169/1224=95.51% [91.75, 98.15] |
| transactional | 2 | 434/1133=38.31% [24.10, 50.95] | 454/1133=40.07% [25.46, 52.73] | 31/91=34.07% [18.18, 56.67] | 1173/1224=95.83% [92.02, 98.36] |
| baseline | 3 | 396/1133=34.95% [20.86, 47.21] | 443/1133=39.10% [25.16, 51.57] | 30/91=32.97% [19.32, 51.06] | 1161/1224=94.85% [91.84, 97.67] |
| transactional | 3 | 433/1133=38.22% [24.28, 50.77] | 441/1133=38.92% [24.85, 51.36] | 35/91=38.46% [20.45, 60.25] | 1187/1224=96.98% [93.16, 99.27] |

### groundlm / label_binary

| Arm | Seed | SCAM FPR [95% CI] | >=UNCERTAIN FPR [95% CI] | SCAM recall [95% CI] | Coverage [95% CI] |
|---|---:|---|---|---|---|
| baseline | 1 | 398/1133=35.13% [21.26, 47.44] | 419/1133=36.98% [23.09, 49.88] | 40/91=43.96% [24.54, 64.29] | 1196/1224=97.71% [96.55, 98.89] |
| transactional | 1 | 418/1133=36.89% [22.27, 49.75] | 423/1133=37.33% [23.32, 50.37] | 65/91=71.43% [56.51, 84.03] | 1214/1224=99.18% [98.52, 99.80] |
| baseline | 2 | 421/1133=37.16% [23.50, 49.60] | 445/1133=39.28% [25.24, 52.17] | 31/91=34.07% [18.18, 56.67] | 1169/1224=95.51% [91.75, 98.15] |
| transactional | 2 | 434/1133=38.31% [24.10, 50.95] | 454/1133=40.07% [25.46, 52.73] | 31/91=34.07% [18.18, 56.67] | 1173/1224=95.83% [92.02, 98.36] |
| baseline | 3 | 396/1133=34.95% [20.86, 47.21] | 443/1133=39.10% [25.16, 51.57] | 30/91=32.97% [19.32, 51.06] | 1161/1224=94.85% [91.84, 97.67] |
| transactional | 3 | 433/1133=38.22% [24.28, 50.77] | 441/1133=38.92% [24.85, 51.36] | 35/91=38.46% [20.45, 60.25] | 1187/1224=96.98% [93.16, 99.27] |

### smishx / label_scam

| Arm | Seed | SCAM FPR [95% CI] | >=UNCERTAIN FPR [95% CI] | SCAM recall [95% CI] | Coverage [95% CI] |
|---|---:|---|---|---|---|
| baseline | 1 | 37/881=4.20% [2.34, 6.49] | 200/881=22.70% [18.47, 27.06] | 27/226=11.95% [7.48, 18.47] | 918/1107=82.93% [79.44, 86.04] |
| transactional | 1 | 48/881=5.45% [3.59, 7.81] | 61/881=6.92% [4.81, 9.40] | 56/226=24.78% [16.14, 35.43] | 1088/1107=98.28% [97.26, 99.13] |
| baseline | 2 | 22/881=2.50% [1.17, 4.18] | 239/881=27.13% [22.39, 31.61] | 24/226=10.62% [6.34, 16.83] | 841/1107=75.97% [72.39, 79.84] |
| transactional | 2 | 25/881=2.84% [1.41, 4.63] | 221/881=25.09% [20.60, 29.67] | 27/226=11.95% [7.00, 18.83] | 848/1107=76.60% [73.08, 80.23] |
| baseline | 3 | 23/881=2.61% [1.33, 4.23] | 216/881=24.52% [20.09, 28.85] | 26/226=11.50% [5.93, 19.16] | 887/1107=80.13% [76.71, 83.46] |
| transactional | 3 | 27/881=3.06% [1.72, 4.74] | 47/881=5.33% [3.62, 7.25] | 28/226=12.39% [7.14, 19.53] | 1050/1107=94.85% [93.01, 96.40] |

### smishx / label_binary

| Arm | Seed | SCAM FPR [95% CI] | >=UNCERTAIN FPR [95% CI] | SCAM recall [95% CI] | Coverage [95% CI] |
|---|---:|---|---|---|---|
| baseline | 1 | 18/606=2.97% [1.65, 4.46] | 34/606=5.61% [3.80, 7.61] | 46/501=9.18% [5.76, 13.11] | 918/1107=82.93% [79.44, 86.04] |
| transactional | 1 | 31/606=5.12% [3.47, 7.05] | 34/606=5.61% [3.81, 7.60] | 73/501=14.57% [10.30, 19.84] | 1088/1107=98.28% [97.26, 99.13] |
| baseline | 2 | 13/606=2.15% [1.15, 3.33] | 37/606=6.11% [4.29, 8.09] | 33/501=6.59% [3.79, 9.61] | 841/1107=75.97% [72.39, 79.84] |
| transactional | 2 | 16/606=2.64% [1.48, 3.97] | 40/606=6.60% [4.66, 8.73] | 36/501=7.19% [4.09, 10.52] | 848/1107=76.60% [73.08, 80.23] |
| baseline | 3 | 13/606=2.15% [1.14, 3.33] | 36/606=5.94% [4.13, 7.87] | 36/501=7.19% [3.86, 10.76] | 887/1107=80.13% [76.71, 83.46] |
| transactional | 3 | 17/606=2.81% [1.64, 4.28] | 31/606=5.12% [3.46, 6.94] | 38/501=7.58% [4.25, 11.20] | 1050/1107=94.85% [93.01, 96.40] |

SmishX fraud endpoint881 negatives/226 positives; legacy606/501. Promotions are not fraud. Seed1 student recall119/226=52.65% [43.71,62.72], whereas baseline fusion27/226=11.95% and candidate56/226=24.78%. Lower FPR from losing scams cannot satisfy the gate. All other student references remain in JSON.

| Dataset | Seed | Endpoint | Candidate-minus-baseline pp [paired95% CI] | p |
|---|---:|---|---|---|
| groundlm | 1 | fpr_alert | +0.35 [-0.18, 0.86] | 0.216783 |
| groundlm | 1 | fpr_scam | +1.77 [0.52, 3.02] | 0.004995 |
| groundlm | 1 | recall_scam | +27.47 [2.08, 57.34] | 0.046953 |
| groundlm | 2 | fpr_alert | +0.79 [0.20, 1.52] | 0.018981 |
| groundlm | 2 | fpr_scam | +1.15 [0.17, 2.50] | 0.056943 |
| groundlm | 2 | recall_scam | +0.00 [0.00, 0.00] | 1.000000 |
| groundlm | 3 | fpr_alert | -0.18 [-0.74, 0.37] | 0.566434 |
| groundlm | 3 | fpr_scam | +3.27 [0.92, 5.94] | 0.012987 |
| groundlm | 3 | recall_scam | +5.49 [0.00, 13.54] | 0.233766 |
| smishx | 1 | fpr_alert | -15.78 [-19.78, -11.97] | 0.000999 |
| smishx | 1 | fpr_scam | +1.25 [-0.11, 2.61] | 0.077922 |
| smishx | 1 | recall_scam | +12.83 [6.47, 19.55] | 0.000999 |
| smishx | 2 | fpr_alert | -2.04 [-3.47, -0.69] | 0.005994 |
| smishx | 2 | fpr_scam | +0.34 [0.00, 0.81] | 0.126873 |
| smishx | 2 | recall_scam | +1.33 [-1.11, 3.93] | 0.361638 |
| smishx | 3 | fpr_alert | -19.18 [-23.31, -15.04] | 0.000999 |
| smishx | 3 | fpr_scam | +0.45 [0.11, 0.96] | 0.059940 |
| smishx | 3 | recall_scam | +0.88 [0.00, 2.14] | 0.226773 |

## Every frozen fold and seed

Fraud(label_scam) and legacy spam-inclusive(label_binary) endpoints are separate. UCI has642 legacy spam positives but zero mapped fraud positives: fraud recall is NA, not zero or a substituted spam metric. This mapping does not establish that UCI contains no real scams. Temporal has77 positives and the original three train/test campaign overlaps; no split was changed.

### label_scam

| Fold | Arm | Seed | SCAM FPR [95% CI] | >=UNCERTAIN FPR [95% CI] | SCAM recall [95% CI] | Coverage [95% CI] |
|---|---|---:|---|---|---|---|
| campaign | baseline | 1 | 121/10986=1.10% [0.86, 1.38] | 168/10986=1.53% [1.26, 1.84] | 5473/5926=92.36% [91.68, 93.06] | 16708/16912=98.79% [98.62, 98.95] |
| campaign | transactional | 1 | 130/10986=1.18% [0.92, 1.46] | 150/10986=1.37% [1.09, 1.66] | 5541/5926=93.50% [92.87, 94.13] | 16836/16912=99.55% [99.44, 99.65] |
| campaign | baseline | 2 | 127/10986=1.16% [0.92, 1.44] | 168/10986=1.53% [1.25, 1.85] | 5480/5926=92.47% [91.78, 93.16] | 16645/16912=98.42% [98.24, 98.61] |
| campaign | transactional | 2 | 128/10986=1.17% [0.93, 1.45] | 172/10986=1.57% [1.29, 1.89] | 5509/5926=92.96% [92.26, 93.60] | 16692/16912=98.70% [98.54, 98.87] |
| campaign | baseline | 3 | 109/10986=0.99% [0.76, 1.29] | 170/10986=1.55% [1.28, 1.89] | 5435/5926=91.71% [91.02, 92.46] | 16641/16912=98.40% [98.20, 98.58] |
| campaign | transactional | 3 | 123/10986=1.12% [0.88, 1.44] | 152/10986=1.38% [1.12, 1.71] | 5493/5926=92.69% [92.04, 93.36] | 16755/16912=99.07% [98.93, 99.21] |
| lang_shift | baseline | 1 | 322/1322=24.36% [21.93, 26.69] | 423/1322=32.00% [29.09, 34.80] | 76/101=75.25% [58.87, 90.91] | 1320/1423=92.76% [91.33, 94.05] |
| lang_shift | transactional | 1 | 320/1322=24.21% [21.80, 26.60] | 430/1322=32.53% [29.74, 35.37] | 76/101=75.25% [58.73, 91.11] | 1310/1423=92.06% [90.66, 93.42] |
| lang_shift | baseline | 2 | 13/1322=0.98% [0.38, 1.80] | 14/1322=1.06% [0.38, 2.00] | 66/101=65.35% [50.00, 82.02] | 1422/1423=99.93% [99.79, 100.00] |
| lang_shift | transactional | 2 | 32/1322=2.42% [1.44, 3.63] | 36/1322=2.72% [1.61, 4.12] | 72/101=71.29% [55.81, 86.91] | 1419/1423=99.72% [99.43, 99.93] |
| lang_shift | baseline | 3 | 672/1322=50.83% [47.85, 53.75] | 930/1322=70.35% [67.27, 73.42] | 69/101=68.32% [52.84, 84.45] | 1160/1423=81.52% [79.53, 83.62] |
| lang_shift | transactional | 3 | 631/1322=47.73% [44.65, 50.57] | 993/1322=75.11% [72.08, 77.92] | 70/101=69.31% [54.09, 85.39] | 1055/1423=74.14% [71.84, 76.44] |
| source_holdout:holdout_mishra_soni_smsphishing | baseline | 1 | 162/668=24.25% [20.54, 27.81] | 162/668=24.25% [20.54, 27.81] | 267/273=97.80% [95.82, 99.29] | 941/941=100.00% [100.00, 100.00] |
| source_holdout:holdout_mishra_soni_smsphishing | transactional | 1 | 162/668=24.25% [20.54, 27.81] | 162/668=24.25% [20.54, 27.81] | 267/273=97.80% [95.82, 99.29] | 941/941=100.00% [100.00, 100.00] |
| source_holdout:holdout_mishra_soni_smsphishing | baseline | 2 | 132/668=19.76% [16.67, 23.09] | 149/668=22.31% [19.04, 25.72] | 228/273=83.52% [77.69, 88.71] | 905/941=96.17% [94.70, 97.42] |
| source_holdout:holdout_mishra_soni_smsphishing | transactional | 2 | 131/668=19.61% [16.51, 22.94] | 134/668=20.06% [16.94, 23.46] | 234/273=85.71% [80.23, 90.61] | 938/941=99.68% [99.28, 100.00] |
| source_holdout:holdout_mishra_soni_smsphishing | baseline | 3 | 167/668=25.00% [21.40, 28.66] | 174/668=26.05% [22.46, 29.89] | 268/273=98.17% [96.20, 99.63] | 932/941=99.04% [98.41, 99.60] |
| source_holdout:holdout_mishra_soni_smsphishing | transactional | 3 | 167/668=25.00% [21.40, 28.66] | 174/668=26.05% [22.46, 29.89] | 268/273=98.17% [96.20, 99.63] | 932/941=99.04% [98.41, 99.60] |
| source_holdout:holdout_scamshield | baseline | 1 | 5752/46944=12.25% [11.90, 12.60] | 7938/46944=16.91% [16.48, 17.35] | 1325/5568=23.80% [21.22, 26.85] | 49357/52512=93.99% [93.73, 94.24] |
| source_holdout:holdout_scamshield | transactional | 1 | 6269/46944=13.35% [12.92, 13.82] | 8352/46944=17.79% [17.32, 18.30] | 3005/5568=53.97% [48.75, 60.24] | 49802/52512=94.84% [94.61, 95.06] |
| source_holdout:holdout_scamshield | baseline | 2 | 4715/46944=10.04% [9.74, 10.37] | 7349/46944=15.65% [15.25, 16.07] | 1345/5568=24.16% [21.54, 27.25] | 48709/52512=92.76% [92.45, 93.05] |
| source_holdout:holdout_scamshield | transactional | 2 | 5238/46944=11.16% [10.77, 11.59] | 7735/46944=16.48% [16.04, 16.97] | 3003/5568=53.93% [48.76, 59.96] | 49179/52512=93.65% [93.37, 93.93] |
| source_holdout:holdout_scamshield | baseline | 3 | 7494/46944=15.96% [15.55, 16.39] | 7639/46944=16.27% [15.87, 16.72] | 2166/5568=38.90% [35.90, 42.28] | 52249/52512=99.50% [99.32, 99.62] |
| source_holdout:holdout_scamshield | transactional | 3 | 7882/46944=16.79% [16.35, 17.30] | 8001/46944=17.04% [16.60, 17.54] | 3423/5568=61.48% [56.92, 66.87] | 52342/52512=99.68% [99.62, 99.73] |
| source_holdout:holdout_uci_sms_spam | baseline | 1 | 18/4517=0.40% [0.22, 0.58] | 18/4517=0.40% [0.22, 0.58] | NA (n=0) | 4517/4517=100.00% [100.00, 100.00] |
| source_holdout:holdout_uci_sms_spam | transactional | 1 | 19/4517=0.42% [0.24, 0.62] | 19/4517=0.42% [0.24, 0.62] | NA (n=0) | 4517/4517=100.00% [100.00, 100.00] |
| source_holdout:holdout_uci_sms_spam | baseline | 2 | 42/4517=0.93% [0.64, 1.22] | 44/4517=0.97% [0.71, 1.26] | NA (n=0) | 4515/4517=99.96% [99.89, 100.00] |
| source_holdout:holdout_uci_sms_spam | transactional | 2 | 43/4517=0.95% [0.66, 1.24] | 43/4517=0.95% [0.66, 1.24] | NA (n=0) | 4517/4517=100.00% [100.00, 100.00] |
| source_holdout:holdout_uci_sms_spam | baseline | 3 | 28/4517=0.62% [0.40, 0.84] | 31/4517=0.69% [0.44, 0.91] | NA (n=0) | 4514/4517=99.93% [99.85, 100.00] |
| source_holdout:holdout_uci_sms_spam | transactional | 3 | 24/4517=0.53% [0.33, 0.73] | 29/4517=0.64% [0.42, 0.86] | NA (n=0) | 4512/4517=99.89% [99.78, 99.98] |
| temporal | baseline | 1 | 87/10952=0.79% [0.64, 0.97] | 3685/10952=33.65% [32.70, 34.51] | 65/77=84.42% [75.68, 92.00] | 7420/11029=67.28% [66.33, 68.22] |
| temporal | transactional | 1 | 123/10952=1.12% [0.94, 1.32] | 3719/10952=33.96% [33.03, 34.80] | 65/77=84.42% [75.68, 92.00] | 7421/11029=67.29% [66.37, 68.20] |
| temporal | baseline | 2 | 106/10952=0.97% [0.79, 1.14] | 2843/10952=25.96% [25.11, 26.79] | 66/77=85.71% [77.21, 93.42] | 8284/11029=75.11% [74.27, 75.92] |
| temporal | transactional | 2 | 120/10952=1.10% [0.92, 1.27] | 2881/10952=26.31% [25.46, 27.14] | 66/77=85.71% [77.21, 93.42] | 8258/11029=74.88% [74.02, 75.68] |
| temporal | baseline | 3 | 150/10952=1.37% [1.16, 1.59] | 1427/10952=13.03% [12.44, 13.69] | 70/77=90.91% [83.78, 96.30] | 9747/11029=88.38% [87.73, 88.95] |
| temporal | transactional | 3 | 166/10952=1.52% [1.31, 1.74] | 1484/10952=13.55% [12.95, 14.23] | 70/77=90.91% [83.78, 96.30] | 9704/11029=87.99% [87.32, 88.57] |

### label_binary

| Fold | Arm | Seed | SCAM FPR [95% CI] | >=UNCERTAIN FPR [95% CI] | SCAM recall [95% CI] | Coverage [95% CI] |
|---|---|---:|---|---|---|---|
| campaign | baseline | 1 | 98/10952=0.89% [0.67, 1.18] | 140/10952=1.28% [1.02, 1.59] | 9559/11193=85.40% [84.75, 86.04] | 21542/22145=97.28% [97.06, 97.49] |
| campaign | transactional | 1 | 107/10952=0.98% [0.73, 1.27] | 126/10952=1.15% [0.90, 1.47] | 9810/11193=87.64% [87.02, 88.26] | 22011/22145=99.39% [99.29, 99.49] |
| campaign | baseline | 2 | 105/10952=0.96% [0.73, 1.26] | 140/10952=1.28% [1.03, 1.59] | 9515/11193=85.01% [84.34, 85.71] | 21083/22145=95.20% [94.91, 95.50] |
| campaign | transactional | 2 | 106/10952=0.97% [0.74, 1.28] | 144/10952=1.31% [1.06, 1.64] | 9560/11193=85.41% [84.77, 86.07] | 21381/22145=96.55% [96.28, 96.79] |
| campaign | baseline | 3 | 87/10952=0.79% [0.59, 1.09] | 144/10952=1.31% [1.07, 1.64] | 9458/11193=84.50% [83.82, 85.19] | 21421/22145=96.73% [96.49, 96.96] |
| campaign | transactional | 3 | 101/10952=0.92% [0.71, 1.22] | 130/10952=1.19% [0.95, 1.52] | 9572/11193=85.52% [84.89, 86.20] | 21771/22145=98.31% [98.15, 98.48] |
| lang_shift | baseline | 1 | 320/1320=24.24% [21.87, 26.59] | 421/1320=31.89% [29.19, 34.47] | 247/357=69.19% [60.26, 77.75] | 1569/1677=93.56% [92.39, 94.62] |
| lang_shift | transactional | 1 | 319/1320=24.17% [21.80, 26.49] | 429/1320=32.50% [29.72, 35.08] | 245/357=68.63% [59.74, 77.33] | 1558/1677=92.90% [91.70, 94.07] |
| lang_shift | baseline | 2 | 12/1320=0.91% [0.30, 1.73] | 13/1320=0.98% [0.30, 1.98] | 233/357=65.27% [56.17, 74.92] | 1676/1677=99.94% [99.82, 100.00] |
| lang_shift | transactional | 2 | 30/1320=2.27% [1.29, 3.53] | 34/1320=2.58% [1.48, 4.00] | 244/357=68.35% [59.48, 77.81] | 1673/1677=99.76% [99.52, 99.94] |
| lang_shift | baseline | 3 | 671/1320=50.83% [47.79, 53.58] | 928/1320=70.30% [67.26, 73.22] | 231/357=64.71% [55.90, 73.91] | 1404/1677=83.72% [81.87, 85.54] |
| lang_shift | transactional | 3 | 630/1320=47.73% [44.90, 50.46] | 992/1320=75.15% [72.15, 77.90] | 231/357=64.71% [55.39, 74.16] | 1293/1677=77.10% [74.89, 79.21] |
| source_holdout:holdout_mishra_soni_smsphishing | baseline | 1 | 30/524=5.73% [3.63, 8.01] | 30/524=5.73% [3.63, 8.01] | 399/417=95.68% [93.42, 97.56] | 941/941=100.00% [100.00, 100.00] |
| source_holdout:holdout_mishra_soni_smsphishing | transactional | 1 | 30/524=5.73% [3.63, 8.01] | 30/524=5.73% [3.63, 8.01] | 399/417=95.68% [93.42, 97.56] | 941/941=100.00% [100.00, 100.00] |
| source_holdout:holdout_mishra_soni_smsphishing | baseline | 2 | 36/524=6.87% [4.55, 9.42] | 42/524=8.02% [5.51, 10.67] | 324/417=77.70% [72.14, 82.73] | 905/941=96.17% [94.70, 97.42] |
| source_holdout:holdout_mishra_soni_smsphishing | transactional | 2 | 33/524=6.30% [4.00, 8.72] | 36/524=6.87% [4.55, 9.42] | 332/417=79.62% [74.49, 84.58] | 938/941=99.68% [99.28, 100.00] |
| source_holdout:holdout_mishra_soni_smsphishing | baseline | 3 | 31/524=5.92% [3.84, 8.17] | 38/524=7.25% [5.11, 9.71] | 404/417=96.88% [94.92, 98.41] | 932/941=99.04% [98.41, 99.60] |
| source_holdout:holdout_mishra_soni_smsphishing | transactional | 3 | 31/524=5.92% [3.84, 8.17] | 38/524=7.25% [5.11, 9.71] | 404/417=96.88% [94.92, 98.41] | 932/941=99.04% [98.41, 99.60] |
| source_holdout:holdout_scamshield | baseline | 1 | 5752/46944=12.25% [11.90, 12.60] | 7938/46944=16.91% [16.48, 17.35] | 4616/29029=15.90% [15.16, 16.73] | 69150/75973=91.02% [90.72, 91.31] |
| source_holdout:holdout_scamshield | transactional | 1 | 6269/46944=13.35% [12.92, 13.82] | 8352/46944=17.79% [17.32, 18.30] | 8964/29029=30.88% [29.20, 33.22] | 70167/75973=92.36% [92.11, 92.61] |
| source_holdout:holdout_scamshield | baseline | 2 | 4715/46944=10.04% [9.74, 10.37] | 7349/46944=15.65% [15.25, 16.07] | 4675/29029=16.10% [15.28, 16.94] | 66489/75973=87.52% [87.18, 87.87] |
| source_holdout:holdout_scamshield | transactional | 2 | 5238/46944=11.16% [10.77, 11.59] | 7735/46944=16.48% [16.04, 16.97] | 9040/29029=31.14% [29.46, 33.33] | 67616/75973=89.00% [88.68, 89.32] |
| source_holdout:holdout_scamshield | baseline | 3 | 7494/46944=15.96% [15.55, 16.39] | 7639/46944=16.27% [15.87, 16.72] | 9198/29029=31.69% [30.65, 32.71] | 75491/75973=99.37% [99.23, 99.47] |
| source_holdout:holdout_scamshield | transactional | 3 | 7882/46944=16.79% [16.35, 17.30] | 8001/46944=17.04% [16.60, 17.54] | 12483/29029=43.00% [41.42, 44.84] | 75641/75973=99.56% [99.52, 99.61] |
| source_holdout:holdout_uci_sms_spam | baseline | 1 | 18/4517=0.40% [0.22, 0.58] | 18/4517=0.40% [0.22, 0.58] | 510/642=79.44% [75.90, 83.29] | 5159/5159=100.00% [100.00, 100.00] |
| source_holdout:holdout_uci_sms_spam | transactional | 1 | 19/4517=0.42% [0.24, 0.62] | 19/4517=0.42% [0.24, 0.62] | 508/642=79.13% [75.38, 83.14] | 5158/5159=99.98% [99.94, 100.00] |
| source_holdout:holdout_uci_sms_spam | baseline | 2 | 42/4517=0.93% [0.64, 1.22] | 44/4517=0.97% [0.71, 1.26] | 508/642=79.13% [75.65, 82.89] | 5152/5159=99.86% [99.75, 99.94] |
| source_holdout:holdout_uci_sms_spam | transactional | 2 | 43/4517=0.95% [0.66, 1.24] | 43/4517=0.95% [0.66, 1.24] | 511/642=79.60% [76.02, 83.48] | 5158/5159=99.98% [99.94, 100.00] |
| source_holdout:holdout_uci_sms_spam | baseline | 3 | 28/4517=0.62% [0.40, 0.84] | 31/4517=0.69% [0.44, 0.91] | 523/642=81.46% [77.85, 84.99] | 5151/5159=99.84% [99.71, 99.96] |
| source_holdout:holdout_uci_sms_spam | transactional | 3 | 24/4517=0.53% [0.33, 0.73] | 29/4517=0.64% [0.42, 0.86] | 509/642=79.28% [75.72, 82.75] | 5142/5159=99.67% [99.51, 99.82] |
| temporal | baseline | 1 | 87/10952=0.79% [0.64, 0.97] | 3685/10952=33.65% [32.70, 34.51] | 65/77=84.42% [75.68, 92.00] | 7420/11029=67.28% [66.33, 68.22] |
| temporal | transactional | 1 | 123/10952=1.12% [0.94, 1.32] | 3719/10952=33.96% [33.03, 34.80] | 65/77=84.42% [75.68, 92.00] | 7421/11029=67.29% [66.37, 68.20] |
| temporal | baseline | 2 | 106/10952=0.97% [0.79, 1.14] | 2843/10952=25.96% [25.11, 26.79] | 66/77=85.71% [77.21, 93.42] | 8284/11029=75.11% [74.27, 75.92] |
| temporal | transactional | 2 | 120/10952=1.10% [0.92, 1.27] | 2881/10952=26.31% [25.46, 27.14] | 66/77=85.71% [77.21, 93.42] | 8258/11029=74.88% [74.02, 75.68] |
| temporal | baseline | 3 | 150/10952=1.37% [1.16, 1.59] | 1427/10952=13.03% [12.44, 13.69] | 70/77=90.91% [83.78, 96.30] | 9747/11029=88.38% [87.73, 88.95] |
| temporal | transactional | 3 | 166/10952=1.52% [1.31, 1.74] | 1484/10952=13.55% [12.95, 14.23] | 70/77=90.91% [83.78, 96.30] | 9704/11029=87.99% [87.32, 88.57] |

| Fold | Seed | Fraud recall delta pp [paired95% CI] | p |
|---|---:|---|---|
| campaign | 1 | +1.15 [0.81, 1.47] | 0.000999000999000999 |
| campaign | 2 | +0.49 [0.30, 0.69] | 0.000999000999000999 |
| campaign | 3 | +0.98 [0.71, 1.23] | 0.000999000999000999 |
| lang_shift | 1 | +0.00 [-2.35, 2.92] | 1.0 |
| lang_shift | 2 | +5.94 [1.06, 12.00] | 0.03596403596403597 |
| lang_shift | 3 | +0.99 [0.00, 3.23] | 0.5214785214785215 |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | +0.00 [0.00, 0.00] | 1.0 |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | +2.20 [0.00, 5.04] | 0.1048951048951049 |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | +0.00 [0.00, 0.00] | 1.0 |
| source_holdout:holdout_scamshield | 1 | +30.17 [25.07, 36.55] | 0.000999000999000999 |
| source_holdout:holdout_scamshield | 2 | +29.78 [24.60, 36.10] | 0.000999000999000999 |
| source_holdout:holdout_scamshield | 3 | +22.58 [17.04, 29.50] | 0.000999000999000999 |
| source_holdout:holdout_uci_sms_spam | 1 | NA [NA, NA] | None |
| source_holdout:holdout_uci_sms_spam | 2 | NA [NA, NA] | None |
| source_holdout:holdout_uci_sms_spam | 3 | NA [NA, NA] | None |
| temporal | 1 | +0.00 [0.00, 0.00] | 1.0 |
| temporal | 2 | +0.00 [0.00, 0.00] | 1.0 |
| temporal | 3 | +0.00 [0.00, 0.00] | 1.0 |

## Observable failed checks and unavailable endpoints

- 1/groundlm_scam_le_1pct
- 1/groundlm_alert_le_10pct
- 1/groundlm_scam_no_increase
- 1/groundlm_alert_no_increase
- 1/smishx_recall_ge_student
- 2/groundlm_scam_le_1pct
- 2/groundlm_alert_le_10pct
- 2/groundlm_scam_no_increase
- 2/groundlm_alert_no_increase
- 2/smishx_recall_ge_student
- 3/groundlm_scam_le_1pct
- 3/groundlm_alert_le_10pct
- 3/groundlm_scam_no_increase
- 3/smishx_recall_ge_student

Unmeasurable fraud-recall guards: 1/source_holdout:holdout_uci_sms_spam/fraud_recall, 2/source_holdout:holdout_uci_sms_spam/fraud_recall, 3/source_holdout:holdout_uci_sms_spam/fraud_recall.

Candidate(b) is withheld. Keep the measurement/provenance tools for reproduction; default fusion remains unchanged. Next candidate(a) needs its own predeclaration, independently sourced deterministic training-only hard negatives, three seeds, cal-only thresholds and all frozen endpoints. Do not introduce per-family calibration until(a)/(b) fail. Shipped-state identification and independent Indian validation remain OPEN.
