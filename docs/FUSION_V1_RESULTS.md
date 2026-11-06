# Frozen fusion v1: measurement reference, safety gate KILL

**Newly fitted (not previously shipped).** Exact-checkpoint artifact gate **GO**; detector safety gate **KILL**. Shipment is not approved. The student is a CPU PyTorch checkpoint: Arm/Android portability and latency remain **UNVERIFIED**. The earlier ONNX attempt remains KILL under its unchanged numerical gate.

All six seed-1 experts and their cal-only stacker, Platt calibrator and thresholds were frozen before any test or external decision. Each fold uses only its own frozen train IDs; teacher0 targets are cached single-teacher outputs, not OOF. Four disjoint cal stages fit fusion/probability/threshold/CRC diagnostics; the underlying checkpoint selection reused full cal, so no CRC population guarantee is claimed. Alpha=0.01, beta=0.05. Full JSON parameters and expert versions are private.

**Exposure:** frozen test labels were not used for this fit. GroundLM and ScamShield informed rule-feature design; GroundLM/SmishX aggregate false-alarm feedback motivated these interventions. These audits are in-sample for intervention selection. SmishX is independent of training after overlap removal, non-Indian-specific; it is no longer untouched for policy selection. Independent Indian validation stays OPEN.

Rates below use all eligible endpoint negatives/positives, including abstentions. Intervals are saved 95% campaign-clustered percentile CIs (1,000 draws), not iid message intervals. Coverage is the fraction assigned SAFE or SCAM. Conditional auto-decided FPR must not replace the all-negative FPR. Legacy `label_binary` includes spam; `label_scam` is the known fraud-policy endpoint. Unknown labels excluded.

| Population | Label | SCAM FPR [CI] | >=UNCERTAIN FPR [CI] | SCAM recall [CI] | >=UNCERTAIN recall [CI] |
|---|---|---|---|---|---|
| campaign | label_binary | 91/10952=0.83% [0.62,1.13] | 100/10952=0.91% [0.70,1.20] | 9839/11193=87.90% [87.28,88.51] | 10350/11193=92.47% [91.96,92.99] |
| campaign | label_scam | 118/10986=1.07% [0.82,1.36] | 128/10986=1.17% [0.92,1.46] | 5510/5926=92.98% [92.30,93.64] | 5648/5926=95.31% [94.72,95.85] |
| lang_shift | label_binary | 570/1320=43.18% [40.35,46.26] | 836/1320=63.33% [60.36,66.33] | 249/357=69.75% [61.30,78.19] | 271/357=75.91% [67.62,83.84] |
| lang_shift | label_scam | 571/1322=43.19% [40.06,46.11] | 837/1322=63.31% [60.21,66.36] | 77/101=76.24% [63.79,89.02] | 81/101=80.20% [67.79,93.10] |
| source_holdout:holdout_mishra_soni_smsphishing | label_binary | 23/524=4.39% [2.50,6.49] | 23/524=4.39% [2.50,6.49] | 397/417=95.20% [92.71,97.27] | 397/417=95.20% [92.71,97.27] |
| source_holdout:holdout_mishra_soni_smsphishing | label_scam | 157/668=23.50% [19.82,27.15] | 157/668=23.50% [19.82,27.15] | 263/273=96.34% [93.52,98.46] | 263/273=96.34% [93.52,98.46] |
| source_holdout:holdout_scamshield | label_binary | 5512/46944=11.74% [11.40,12.10] | 7826/46944=16.67% [16.24,17.12] | 4278/29029=14.74% [13.96,15.61] | 9192/29029=31.66% [30.69,32.67] |
| source_holdout:holdout_scamshield | label_scam | 5512/46944=11.74% [11.40,12.10] | 7826/46944=16.67% [16.24,17.12] | 1295/5568=23.26% [20.68,26.33] | 2283/5568=41.00% [37.55,44.78] |
| source_holdout:holdout_uci_sms_spam | label_binary | 31/4517=0.69% [0.46,0.95] | 33/4517=0.73% [0.49,1.02] | 533/642=83.02% [79.47,86.31] | 539/642=83.96% [80.42,87.19] |
| source_holdout:holdout_uci_sms_spam | label_scam | 31/4517=0.69% [0.46,0.95] | 33/4517=0.73% [0.49,1.02] | NA (n=0) | NA (n=0) |
| temporal | label_binary | 47/10952=0.43% [0.31,0.56] | 3713/10952=33.90% [33.06,34.79] | 64/77=83.12% [74.02,91.14] | 74/77=96.10% [91.25,100.00] |
| temporal | label_scam | 47/10952=0.43% [0.31,0.56] | 3713/10952=33.90% [33.06,34.79] | 64/77=83.12% [74.02,91.14] | 74/77=96.10% [91.25,100.00] |
| groundlm | label_binary | 354/1133=31.24% [16.91,43.57] | 362/1133=31.95% [17.58,44.22] | 29/91=31.87% [19.32,52.56] | 53/91=58.24% [37.35,79.42] |
| groundlm | label_scam | 354/1133=31.24% [16.91,43.57] | 362/1133=31.95% [17.58,44.22] | 29/91=31.87% [19.32,52.56] | 53/91=58.24% [37.35,79.42] |
| smishx | label_binary | 29/606=4.79% [3.15,6.56] | 41/606=6.77% [4.80,8.91] | 197/501=39.32% [32.36,47.14] | 290/501=57.88% [50.37,65.92] |
| smishx | label_scam | 183/881=20.77% [16.51,25.33] | 247/881=28.04% [23.60,32.47] | 43/226=19.03% [12.88,26.90] | 84/226=37.17% [28.02,48.11] |

| Population | Label | Coverage [CI] | UNCERTAIN share [CI] | Student recall at cal-1%-FPR [CI] | SCAM-minus-student recall delta [CI], paired p |
|---|---|---|---|---|---|
| campaign | label_binary | 21625/22145=97.65% [97.46,97.84] | 520/22145=2.35% [2.16,2.54] | 10676/11193=95.38% [95.01,95.76] | -7.48pp [-7.98,-6.97], p=0.000999001 |
| campaign | label_scam | 16764/16912=99.12% [98.98,99.27] | 148/16912=0.88% [0.73,1.02] | 5690/5926=96.02% [95.51,96.51] | -3.04pp [-3.49,-2.59], p=0.000999001 |
| lang_shift | label_binary | 1389/1677=82.83% [80.95,84.66] | 288/1677=17.17% [15.34,19.05] | 255/357=71.43% [62.59,80.67] | -1.68pp [-4.52,+0.81], p=0.22977 |
| lang_shift | label_scam | 1153/1423=81.03% [78.88,83.16] | 270/1423=18.97% [16.84,21.12] | 75/101=74.26% [61.32,87.78] | +1.98pp [-2.94,+6.48], p=0.481518 |
| source_holdout:holdout_mishra_soni_smsphishing | label_binary | 941/941=100.00% [100.00,100.00] | 0/941=0.00% [0.00,0.00] | 380/417=91.13% [87.80,94.17] | +4.08pp [+1.99,+6.42], p=0.002997 |
| source_holdout:holdout_mishra_soni_smsphishing | label_scam | 941/941=100.00% [100.00,100.00] | 0/941=0.00% [0.00,0.00] | 258/273=94.51% [91.02,97.37] | +1.83pp [+0.32,+3.89], p=0.0619381 |
| source_holdout:holdout_scamshield | label_binary | 68745/75973=90.49% [90.20,90.78] | 7228/75973=9.51% [9.22,9.80] | 21269/29029=73.27% [72.06,74.47] | -58.53pp [-59.74,-57.30], p=0.000999001 |
| source_holdout:holdout_scamshield | label_scam | 49210/52512=93.71% [93.44,93.97] | 3302/52512=6.29% [6.03,6.56] | 4734/5568=85.02% [82.37,87.76] | -61.76pp [-65.34,-58.08], p=0.000999001 |
| source_holdout:holdout_uci_sms_spam | label_binary | 5151/5159=99.84% [99.73,99.94] | 8/5159=0.16% [0.06,0.27] | 551/642=85.83% [82.74,88.80] | -2.80pp [-4.28,-1.44], p=0.000999001 |
| source_holdout:holdout_uci_sms_spam | label_scam | 4515/4517=99.96% [99.89,100.00] | 2/4517=0.04% [0.00,0.11] | NA (n=0) | NA (no positive rows) |
| temporal | label_binary | 7353/11029=66.67% [65.72,67.59] | 3676/11029=33.33% [32.41,34.28] | 69/77=89.61% [82.28,95.89] | -6.49pp [-13.33,-1.32], p=0.040959 |
| temporal | label_scam | 7353/11029=66.67% [65.72,67.59] | 3676/11029=33.33% [32.41,34.28] | 69/77=89.61% [82.28,95.89] | -6.49pp [-13.33,-1.32], p=0.040959 |
| groundlm | label_binary | 1192/1224=97.39% [93.69,99.58] | 32/1224=2.61% [0.42,6.31] | 54/91=59.34% [39.42,79.55] | -27.47pp [-56.31,-1.69], p=0.043956 |
| groundlm | label_scam | 1192/1224=97.39% [93.69,99.58] | 32/1224=2.61% [0.42,6.31] | 54/91=59.34% [39.42,79.55] | -27.47pp [-56.31,-1.69], p=0.043956 |
| smishx | label_binary | 1002/1107=90.51% [88.31,92.62] | 105/1107=9.49% [7.38,11.69] | 338/501=67.47% [60.44,74.36] | -28.14pp [-33.05,-23.16], p=0.000999001 |
| smishx | label_scam | 1002/1107=90.51% [88.31,92.62] | 105/1107=9.49% [7.38,11.69] | 113/226=50.00% [40.81,60.93] | -30.97pp [-38.38,-23.77], p=0.000999001 |

UCI fraud recall is **NA**, not a relabelled spam endpoint. The corresponding legacy-spam endpoint is reported separately. The absolute baseline safety gate does not need a UCI recall endpoint: all three external checks failed.

| Recorded safety check | Pass |
|---|---|
| groundlm_at_least_uncertain_le_10pct | False |
| groundlm_scam_le_1pct | False |
| smishx_recall_ge_student | False |

Decision: **keep the immutable measurement reference; kill this configuration as a shipment candidate**. Do not confuse an artifact GO with safety GO. Future full-system FPR claims must name this v1 manifest or a named successor. No thresholds were changed from external outcomes.

| Fold | Cal rows | Checkpoint round-trip max absolute logit difference | Manifest SHA256 |
|---|---|---|---|
| campaign | 11072 | 0.0 | `c310d3ccbc1c1e50ce9436f217c66ee9934f778985bfef487a034cc3070b79fa` |
| lang_shift | 8454 | 0.0 | `923a40e928d7abdd781ef28cda7f07e5b3aa79aa39521bfd047dcbfd3fe1ee65` |
| source_holdout:holdout_mishra_soni_smsphishing | 10876 | 0.0 | `3d13d4d81b97012441b0f74d5f52ba1a324715b6388f1ff95d74179ecd3f5ae4` |
| source_holdout:holdout_scamshield | 2754 | 0.0 | `c9ad09d019a1ffc051b70d695f8b2d8419a6af9898d3fbd4692c95e518cabf3b` |
| source_holdout:holdout_uci_sms_spam | 10350 | 0.0 | `9b515a83408b31b2340c02b12403b0aef01843de054140f9810814db6d347da4` |
| temporal | 5515 | 0.0 | `d84ef1115a7f3aa881bc62a834271d0090c46853978dc2fa50e4c772d9c12a60` |

Round-trip comparisons are same-framework serialization checks, **not ONNX equivalence**. All model tensors were strictly compared on reload. The original ONNX failure and cal-only runtime diagnostic remain in `FUSION_V1_PARITY.md`.

Job: `fusion_torch_ref-ref-20261101`, kernel `maadhavvh/rakshak-fusion-torch-ref-ref-20261101` version1; args `--kernel maadhavvh/rakshak-fusion-torch-ref-ref-20261101 --kernel-version 1 --n-boot 1000`. Outcome `ok`, 1148.9s. One completion subscription and one matching fetch.

Git `6d4669f68c194cafdd9d7827e86d0264d34ba9d3`; dirty source tree `0cf9f00c86d115f31a92ad61a4376d501bd36cbed5a396b29fad1857374fe205`.

Primary manifest SHA256 `c310d3ccbc1c1e50ce9436f217c66ee9934f778985bfef487a034cc3070b79fa`; metrics SHA256 `aada644fd94eeaea681ef3ec3f53928c78d029f496e0319966377c192857377f`.

Primary frozen thresholds: t_safe=0.9312007756147571, t_scam=0.9456365206144762. Student standalone cal-1%-FPR threshold 0.8724082112312317. These are not deployment-approved thresholds.

| Expert/artifact | SHA256 | Bytes |
|---|---|---|
| inference_source | `88f9dd4726eafa145a2e66b8e73c1444aa48d317eb52c2d5b99263b0241e257a` | 195114 |
| student_checkpoint | `f1d0c7df814ed3a08c49e753f6400b3802f1df5f91b8327c69cd3e03639df377` | 374573 |
| url_onnx | `1bbd2c57cb541a7cbc287887ef51976d54240a353655827a511c63ec56213fc7` | 2741232 |
| url_scaler | `c6b3fc968b03339a33052fe814a9bb851d3035cb302b06458955c2a5e5b6501e` | 2083 |

Private Kaggle model bundle `maadhavvh/rakshak-fusion-v1-20261101`v1 verified ownermaadhavvh, `isPrivate=true`,23581454bytes; receipt and the initial processing-time403 are recorded in STATUS.md; **HF publish pending**. No scores, message rows, weights or secrets belong in git. All statistical results above are formatted from the Kaggle aggregate JSON; no local model inference or resampling was performed.
