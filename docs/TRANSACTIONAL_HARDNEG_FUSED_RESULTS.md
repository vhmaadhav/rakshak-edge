# Hard-negative arm(a): fused audit KILL

**KILL under the predeclared gates.** Every augmented seed lowers GroundLM FPR relative to its fresh no-augmentation fused control, but all three still fail the1%SCAM/10%alert caps and SmishX SCAM-recall-versus-control-student gate. Seven fold/label recall guards also fail. Sixteen failures total, no unavailable required result; UCI fraud recall explicitly NA for all3seeds with legacy-spam guard retained. No externally selected seed or threshold; the generator/weights/hyperparameters/gates were not changed from outcomes.

Hypothesis: original deterministic benign transactional templates improve context handling. Training-only synthetic prefixes,6families/EN+Hinglish, min(12000,floor.25*original-negative-count), generatorseed20261101, all3modelseeds. Original train IDs with teacher0 only; synthetic fullBCE/no teacher. No test/external rows used as templates or labels.200-item sample was generated for manual audit but **human review is not completed**. This cannot establish real Indian authenticity or close independent validation.

Experimental named fusion arm(a) uses fresh ERM/augmented student scores, the same frozen URL expert and unchanged fusion schema (candidateb context features disabled in both arms), separate four-stage cal-only stackers/calibrators/thresholds per fold/role/seed. All states froze before test/external decisions. These experimental controls differ from immutable v1; the untouched v1 manifest/results are included as a reference, never silently refitted/replaced. No shipped configuration existed; all are newly fitted (not previously shipped).

**Exposure:** GroundLM/ScamShield informed rule-feature design; GroundLM/SmishX aggregate feedback motivated interventions. In-sample for intervention selection, **no held-out improvement claim**. SmishX independent of train after overlap removal, non-Indian-specific, but no longer untouched for policy selection. Independent Indian validation stays OPEN. Original temporal overlaps and77positives unchanged. No new URL-only PhreshPhish experiment.

Rates use all eligible label-endpoint negatives/positives (including abstentions). CIs are95% campaign-clustered percentile intervals,1,000 draws. Legacy-spam and known-fraud endpoints remain distinct. `baseline`=fresh no-augmentation ERM; `transactional`=hardneg(a). Student recall column is always the paired **control** student at its own frozen cal1%-FPR threshold, as required by the gate, not the augmented student.

| Population | Seed | Role | Label | SCAM FPR[CI] | >=UNCERTAIN FPR[CI] | SCAM recall[CI] | >=UNCERTAIN recall[CI] |
|---|---|---|---|---|---|---|---|
| campaign | 1 | baseline | label_binary | 87/10952=0.79% [0.58,1.08] | 131/10952=1.20% [0.95,1.49] | 9502/11193=84.89% [84.23,85.57] | 10323/11193=92.23% [91.78,92.73] |
| campaign | 1 | baseline | label_scam | 110/10986=1.00% [0.77,1.27] | 157/10986=1.43% [1.15,1.75] | 5476/5926=92.41% [91.74,93.10] | 5663/5926=95.56% [95.00,96.10] |
| campaign | 1 | transactional | label_binary | 86/10952=0.79% [0.57,1.08] | 139/10952=1.27% [1.01,1.59] | 9445/11193=84.38% [83.71,85.07] | 10175/11193=90.91% [90.36,91.40] |
| campaign | 1 | transactional | label_scam | 107/10986=0.97% [0.76,1.26] | 166/10986=1.51% [1.23,1.81] | 5422/5926=91.50% [90.77,92.20] | 5628/5926=94.97% [94.36,95.53] |
| campaign | 2 | baseline | label_binary | 92/10952=0.84% [0.63,1.14] | 218/10952=1.99% [1.69,2.38] | 9372/11193=83.73% [83.03,84.44] | 10188/11193=91.02% [90.51,91.53] |
| campaign | 2 | baseline | label_scam | 115/10986=1.05% [0.81,1.33] | 246/10986=2.24% [1.92,2.57] | 5397/5926=91.07% [90.30,91.81] | 5674/5926=95.75% [95.19,96.27] |
| campaign | 2 | transactional | label_binary | 119/10952=1.09% [0.84,1.40] | 154/10952=1.41% [1.14,1.74] | 9679/11193=86.47% [85.88,87.10] | 10193/11193=91.07% [90.53,91.61] |
| campaign | 2 | transactional | label_scam | 141/10986=1.28% [1.02,1.58] | 181/10986=1.65% [1.37,1.95] | 5551/5926=93.67% [92.98,94.29] | 5659/5926=95.49% [94.93,96.05] |
| campaign | 3 | baseline | label_binary | 96/10952=0.88% [0.66,1.16] | 121/10952=1.10% [0.87,1.43] | 9599/11193=85.76% [85.13,86.37] | 10026/11193=89.57% [88.99,90.12] |
| campaign | 3 | baseline | label_scam | 119/10986=1.08% [0.85,1.37] | 145/10986=1.32% [1.08,1.63] | 5516/5926=93.08% [92.39,93.75] | 5614/5926=94.74% [94.12,95.32] |
| campaign | 3 | transactional | label_binary | 93/10952=0.85% [0.63,1.16] | 157/10952=1.43% [1.17,1.76] | 9458/11193=84.50% [83.84,85.20] | 10235/11193=91.44% [90.94,91.96] |
| campaign | 3 | transactional | label_scam | 116/10986=1.06% [0.81,1.35] | 186/10986=1.69% [1.38,2.01] | 5437/5926=91.75% [91.06,92.48] | 5669/5926=95.66% [95.14,96.20] |
| lang_shift | 1 | baseline | label_binary | 867/1320=65.68% [62.71,68.64] | 869/1320=65.83% [62.89,68.80] | 246/357=68.91% [59.90,78.11] | 247/357=69.19% [60.29,78.57] |
| lang_shift | 1 | baseline | label_scam | 868/1322=65.66% [62.61,68.58] | 870/1322=65.81% [62.80,68.76] | 78/101=77.23% [61.67,92.22] | 78/101=77.23% [61.67,92.22] |
| lang_shift | 1 | transactional | label_binary | 1033/1320=78.26% [75.13,81.26] | 1205/1320=91.29% [88.30,93.88] | 228/357=63.87% [54.61,73.21] | 250/357=70.03% [61.89,78.75] |
| lang_shift | 1 | transactional | label_scam | 1034/1322=78.21% [74.92,81.15] | 1206/1322=91.23% [88.10,93.77] | 68/101=67.33% [52.17,83.73] | 73/101=72.28% [58.57,86.21] |
| lang_shift | 2 | baseline | label_binary | 54/1320=4.09% [2.99,5.23] | 143/1320=10.83% [9.10,12.56] | 230/357=64.43% [55.49,73.85] | 238/357=66.67% [57.48,76.32] |
| lang_shift | 2 | baseline | label_scam | 55/1322=4.16% [3.14,5.22] | 145/1322=10.97% [9.25,12.61] | 70/101=69.31% [53.78,85.23] | 72/101=71.29% [56.09,87.36] |
| lang_shift | 2 | transactional | label_binary | 31/1320=2.35% [1.49,3.35] | 48/1320=3.64% [2.43,5.24] | 232/357=64.99% [55.94,74.39] | 239/357=66.95% [57.77,76.49] |
| lang_shift | 2 | transactional | label_scam | 33/1322=2.50% [1.62,3.53] | 50/1322=3.78% [2.62,5.27] | 66/101=65.35% [50.00,81.40] | 70/101=69.31% [53.79,84.89] |
| lang_shift | 3 | baseline | label_binary | 100/1320=7.58% [6.14,9.00] | 217/1320=16.44% [14.37,18.34] | 232/357=64.99% [56.04,74.28] | 245/357=68.63% [60.00,77.61] |
| lang_shift | 3 | baseline | label_scam | 102/1322=7.72% [6.35,9.15] | 219/1322=16.57% [14.65,18.60] | 72/101=71.29% [56.10,87.36] | 74/101=73.27% [58.19,89.53] |
| lang_shift | 3 | transactional | label_binary | 758/1320=57.42% [54.22,60.30] | 804/1320=60.91% [57.67,63.97] | 235/357=65.83% [56.37,75.78] | 237/357=66.39% [56.67,76.26] |
| lang_shift | 3 | transactional | label_scam | 760/1322=57.49% [54.20,60.52] | 806/1322=60.97% [57.73,64.01] | 69/101=68.32% [52.84,84.45] | 70/101=69.31% [53.78,85.41] |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | baseline | label_binary | 18/524=3.44% [1.74,5.40] | 19/524=3.63% [1.89,5.60] | 394/417=94.48% [91.85,96.71] | 395/417=94.72% [92.11,96.98] |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | baseline | label_scam | 146/668=21.86% [18.30,25.48] | 148/668=22.16% [18.58,25.82] | 266/273=97.44% [95.02,99.30] | 266/273=97.44% [95.02,99.30] |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | transactional | label_binary | 20/524=3.82% [2.09,5.81] | 20/524=3.82% [2.09,5.81] | 393/417=94.24% [91.45,96.65] | 394/417=94.48% [91.69,96.80] |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | transactional | label_scam | 146/668=21.86% [18.15,25.41] | 147/668=22.01% [18.40,25.56] | 267/273=97.80% [95.47,99.63] | 267/273=97.80% [95.47,99.63] |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | baseline | label_binary | 36/524=6.87% [4.64,9.52] | 43/524=8.21% [5.77,10.82] | 286/417=68.59% [61.52,74.83] | 307/417=73.62% [66.84,79.16] |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | baseline | label_scam | 108/668=16.17% [13.39,19.28] | 125/668=18.71% [15.68,21.91] | 214/273=78.39% [70.54,85.37] | 225/273=82.42% [75.00,89.16] |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | transactional | label_binary | 34/524=6.49% [4.29,8.70] | 36/524=6.87% [4.57,9.14] | 393/417=94.24% [91.64,96.67] | 393/417=94.24% [91.64,96.67] |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | transactional | label_scam | 162/668=24.25% [20.53,27.79] | 164/668=24.55% [20.76,28.01] | 265/273=97.07% [94.52,98.95] | 265/273=97.07% [94.52,98.95] |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | baseline | label_binary | 31/524=5.92% [3.77,8.40] | 31/524=5.92% [3.77,8.40] | 399/417=95.68% [93.52,97.56] | 399/417=95.68% [93.52,97.56] |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | baseline | label_scam | 162/668=24.25% [20.44,28.10] | 162/668=24.25% [20.44,28.10] | 268/273=98.17% [96.24,99.64] | 268/273=98.17% [96.24,99.64] |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | transactional | label_binary | 20/524=3.82% [1.93,5.88] | 20/524=3.82% [1.93,5.88] | 373/417=89.45% [86.18,92.68] | 375/417=89.93% [86.70,93.06] |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | transactional | label_scam | 142/668=21.26% [17.74,24.81] | 142/668=21.26% [17.74,24.81] | 251/273=91.94% [88.09,95.43] | 253/273=92.67% [88.97,96.04] |
| source_holdout:holdout_scamshield | 1 | baseline | label_binary | 5553/46944=11.83% [11.48,12.18] | 8004/46944=17.05% [16.63,17.49] | 4415/29029=15.21% [14.43,16.08] | 9871/29029=34.00% [32.93,35.05] |
| source_holdout:holdout_scamshield | 1 | baseline | label_scam | 5553/46944=11.83% [11.48,12.18] | 8004/46944=17.05% [16.63,17.49] | 1325/5568=23.80% [21.21,26.92] | 2382/5568=42.78% [39.23,46.70] |
| source_holdout:holdout_scamshield | 1 | transactional | label_binary | 8066/46944=17.18% [16.81,17.57] | 9405/46944=20.03% [19.58,20.51] | 8574/29029=29.54% [28.59,30.49] | 11820/29029=40.72% [39.62,41.83] |
| source_holdout:holdout_scamshield | 1 | transactional | label_scam | 8066/46944=17.18% [16.81,17.57] | 9405/46944=20.03% [19.58,20.51] | 2171/5568=38.99% [35.67,42.46] | 2731/5568=49.05% [45.33,52.88] |
| source_holdout:holdout_scamshield | 2 | baseline | label_binary | 5197/46944=11.07% [10.75,11.42] | 6399/46944=13.63% [13.23,14.06] | 4542/29029=15.65% [14.86,16.50] | 7603/29029=26.19% [25.27,27.08] |
| source_holdout:holdout_scamshield | 2 | baseline | label_scam | 5197/46944=11.07% [10.75,11.42] | 6399/46944=13.63% [13.23,14.06] | 1265/5568=22.72% [20.15,25.73] | 1756/5568=31.54% [28.74,34.80] |
| source_holdout:holdout_scamshield | 2 | transactional | label_binary | 5883/46944=12.53% [12.20,12.87] | 7348/46944=15.65% [15.24,16.05] | 7431/29029=25.60% [24.69,26.59] | 11220/29029=38.65% [37.51,39.78] |
| source_holdout:holdout_scamshield | 2 | transactional | label_scam | 5883/46944=12.53% [12.20,12.87] | 7348/46944=15.65% [15.24,16.05] | 1718/5568=30.85% [27.97,34.28] | 2506/5568=45.01% [41.49,48.72] |
| source_holdout:holdout_scamshield | 3 | baseline | label_binary | 7470/46944=15.91% [15.50,16.34] | 7674/46944=16.35% [15.94,16.79] | 9009/29029=31.03% [29.97,32.05] | 9530/29029=32.83% [31.86,33.87] |
| source_holdout:holdout_scamshield | 3 | baseline | label_scam | 7470/46944=15.91% [15.50,16.34] | 7674/46944=16.35% [15.94,16.79] | 2103/5568=37.77% [34.65,41.36] | 2303/5568=41.36% [38.86,44.28] |
| source_holdout:holdout_scamshield | 3 | transactional | label_binary | 6863/46944=14.62% [14.26,14.98] | 8684/46944=18.50% [18.10,18.94] | 7538/29029=25.97% [25.06,26.97] | 11552/29029=39.79% [38.59,40.94] |
| source_holdout:holdout_scamshield | 3 | transactional | label_scam | 6863/46944=14.62% [14.26,14.98] | 8684/46944=18.50% [18.10,18.94] | 1901/5568=34.14% [31.11,37.50] | 2714/5568=48.74% [44.71,52.83] |
| source_holdout:holdout_uci_sms_spam | 1 | baseline | label_binary | 26/4517=0.58% [0.38,0.80] | 26/4517=0.58% [0.38,0.80] | 528/642=82.24% [78.34,85.88] | 529/642=82.40% [78.50,85.95] |
| source_holdout:holdout_uci_sms_spam | 1 | baseline | label_scam | 26/4517=0.58% [0.38,0.80] | 26/4517=0.58% [0.38,0.80] | NA(n=0) | NA(n=0) |
| source_holdout:holdout_uci_sms_spam | 1 | transactional | label_binary | 35/4517=0.77% [0.53,1.04] | 37/4517=0.82% [0.58,1.11] | 536/642=83.49% [79.94,87.00] | 541/642=84.27% [80.77,87.54] |
| source_holdout:holdout_uci_sms_spam | 1 | transactional | label_scam | 35/4517=0.77% [0.53,1.04] | 37/4517=0.82% [0.58,1.11] | NA(n=0) | NA(n=0) |
| source_holdout:holdout_uci_sms_spam | 2 | baseline | label_binary | 26/4517=0.58% [0.35,0.82] | 26/4517=0.58% [0.35,0.82] | 472/642=73.52% [69.08,78.12] | 475/642=73.99% [69.76,78.48] |
| source_holdout:holdout_uci_sms_spam | 2 | baseline | label_scam | 26/4517=0.58% [0.35,0.82] | 26/4517=0.58% [0.35,0.82] | NA(n=0) | NA(n=0) |
| source_holdout:holdout_uci_sms_spam | 2 | transactional | label_binary | 48/4517=1.06% [0.75,1.37] | 49/4517=1.08% [0.78,1.39] | 531/642=82.71% [79.40,85.97] | 532/642=82.87% [79.56,86.12] |
| source_holdout:holdout_uci_sms_spam | 2 | transactional | label_scam | 48/4517=1.06% [0.75,1.37] | 49/4517=1.08% [0.78,1.39] | NA(n=0) | NA(n=0) |
| source_holdout:holdout_uci_sms_spam | 3 | baseline | label_binary | 30/4517=0.66% [0.44,0.91] | 30/4517=0.66% [0.44,0.91] | 493/642=76.79% [72.96,80.80] | 493/642=76.79% [72.96,80.80] |
| source_holdout:holdout_uci_sms_spam | 3 | baseline | label_scam | 30/4517=0.66% [0.44,0.91] | 30/4517=0.66% [0.44,0.91] | NA(n=0) | NA(n=0) |
| source_holdout:holdout_uci_sms_spam | 3 | transactional | label_binary | 35/4517=0.77% [0.51,1.04] | 37/4517=0.82% [0.55,1.08] | 493/642=76.79% [72.40,80.76] | 493/642=76.79% [72.40,80.76] |
| source_holdout:holdout_uci_sms_spam | 3 | transactional | label_scam | 35/4517=0.77% [0.51,1.04] | 37/4517=0.82% [0.55,1.08] | NA(n=0) | NA(n=0) |
| temporal | 1 | baseline | label_binary | 33/10952=0.30% [0.20,0.41] | 6106/10952=55.75% [54.78,56.73] | 61/77=79.22% [69.86,87.67] | 77/77=100.00% [100.00,100.00] |
| temporal | 1 | baseline | label_scam | 33/10952=0.30% [0.20,0.41] | 6106/10952=55.75% [54.78,56.73] | 61/77=79.22% [69.86,87.67] | 77/77=100.00% [100.00,100.00] |
| temporal | 1 | transactional | label_binary | 90/10952=0.82% [0.66,1.00] | 4781/10952=43.65% [42.75,44.67] | 67/77=87.01% [78.66,93.67] | 76/77=98.70% [95.89,100.00] |
| temporal | 1 | transactional | label_scam | 90/10952=0.82% [0.66,1.00] | 4781/10952=43.65% [42.75,44.67] | 67/77=87.01% [78.66,93.67] | 76/77=98.70% [95.89,100.00] |
| temporal | 2 | baseline | label_binary | 106/10952=0.97% [0.80,1.15] | 2489/10952=22.73% [21.91,23.52] | 68/77=88.31% [79.99,94.81] | 74/77=96.10% [91.66,100.00] |
| temporal | 2 | baseline | label_scam | 106/10952=0.97% [0.80,1.15] | 2489/10952=22.73% [21.91,23.52] | 68/77=88.31% [79.99,94.81] | 74/77=96.10% [91.66,100.00] |
| temporal | 2 | transactional | label_binary | 65/10952=0.59% [0.45,0.74] | 4604/10952=42.04% [41.09,43.03] | 64/77=83.12% [74.02,91.14] | 76/77=98.70% [95.95,100.00] |
| temporal | 2 | transactional | label_scam | 65/10952=0.59% [0.45,0.74] | 4604/10952=42.04% [41.09,43.03] | 64/77=83.12% [74.02,91.14] | 76/77=98.70% [95.95,100.00] |
| temporal | 3 | baseline | label_binary | 133/10952=1.21% [1.01,1.43] | 1224/10952=11.18% [10.58,11.80] | 68/77=88.31% [80.76,94.87] | 72/77=93.51% [87.84,98.68] |
| temporal | 3 | baseline | label_scam | 133/10952=1.21% [1.01,1.43] | 1224/10952=11.18% [10.58,11.80] | 68/77=88.31% [80.76,94.87] | 72/77=93.51% [87.84,98.68] |
| temporal | 3 | transactional | label_binary | 134/10952=1.22% [1.03,1.42] | 1629/10952=14.87% [14.20,15.57] | 67/77=87.01% [78.67,93.75] | 72/77=93.51% [87.84,98.68] |
| temporal | 3 | transactional | label_scam | 134/10952=1.22% [1.03,1.42] | 1629/10952=14.87% [14.20,15.57] | 67/77=87.01% [78.67,93.75] | 72/77=93.51% [87.84,98.68] |
| groundlm | 1 | baseline | label_binary | 395/1133=34.86% [20.97,47.16] | 424/1133=37.42% [23.49,49.66] | 31/91=34.07% [17.32,52.08] | 56/91=61.54% [47.94,69.61] |
| groundlm | 1 | baseline | label_scam | 395/1133=34.86% [20.97,47.16] | 424/1133=37.42% [23.49,49.66] | 31/91=34.07% [17.32,52.08] | 56/91=61.54% [47.94,69.61] |
| groundlm | 1 | transactional | label_binary | 83/1133=7.33% [0.96,14.88] | 135/1133=11.92% [2.86,22.38] | 27/91=29.67% [14.94,52.55] | 37/91=40.66% [22.14,68.06] |
| groundlm | 1 | transactional | label_scam | 83/1133=7.33% [0.96,14.88] | 135/1133=11.92% [2.86,22.38] | 27/91=29.67% [14.94,52.55] | 37/91=40.66% [22.14,68.06] |
| groundlm | 2 | baseline | label_binary | 415/1133=36.63% [22.96,49.07] | 479/1133=42.28% [27.23,55.53] | 28/91=30.77% [18.70,54.70] | 51/91=56.04% [29.91,81.93] |
| groundlm | 2 | baseline | label_scam | 415/1133=36.63% [22.96,49.07] | 479/1133=42.28% [27.23,55.53] | 28/91=30.77% [18.70,54.70] | 51/91=56.04% [29.91,81.93] |
| groundlm | 2 | transactional | label_binary | 208/1133=18.36% [7.31,29.57] | 232/1133=20.48% [8.61,32.02] | 37/91=40.66% [21.19,61.44] | 42/91=46.15% [25.00,67.80] |
| groundlm | 2 | transactional | label_scam | 208/1133=18.36% [7.31,29.57] | 232/1133=20.48% [8.61,32.02] | 37/91=40.66% [21.19,61.44] | 42/91=46.15% [25.00,67.80] |
| groundlm | 3 | baseline | label_binary | 418/1133=36.89% [22.83,50.21] | 436/1133=38.48% [24.12,51.60] | 28/91=30.77% [17.32,48.98] | 35/91=38.46% [21.42,60.66] |
| groundlm | 3 | baseline | label_scam | 418/1133=36.89% [22.83,50.21] | 436/1133=38.48% [24.12,51.60] | 28/91=30.77% [17.32,48.98] | 35/91=38.46% [21.42,60.66] |
| groundlm | 3 | transactional | label_binary | 142/1133=12.53% [3.68,23.48] | 205/1133=18.09% [6.60,29.73] | 30/91=32.97% [16.18,54.32] | 46/91=50.55% [27.27,75.51] |
| groundlm | 3 | transactional | label_scam | 142/1133=12.53% [3.68,23.48] | 205/1133=18.09% [6.60,29.73] | 30/91=32.97% [16.18,54.32] | 46/91=50.55% [27.27,75.51] |
| smishx | 1 | baseline | label_binary | 17/606=2.81% [1.49,4.31] | 33/606=5.45% [3.79,7.43] | 33/501=6.59% [3.70,9.53] | 221/501=44.11% [36.93,52.06] |
| smishx | 1 | baseline | label_scam | 26/881=2.95% [1.51,4.76] | 205/881=23.27% [18.89,27.72] | 24/226=10.62% [6.43,16.89] | 49/226=21.68% [15.41,30.46] |
| smishx | 1 | transactional | label_binary | 13/606=2.15% [1.15,3.32] | 34/606=5.61% [3.80,7.58] | 33/501=6.59% [3.60,9.61] | 180/501=35.93% [29.37,44.35] |
| smishx | 1 | transactional | label_scam | 23/881=2.61% [1.24,4.38] | 170/881=19.30% [15.24,23.60] | 23/226=10.18% [5.78,16.28] | 44/226=19.47% [13.23,27.91] |
| smishx | 2 | baseline | label_binary | 13/606=2.15% [0.99,3.45] | 37/606=6.11% [4.30,8.25] | 21/501=4.19% [2.44,6.12] | 180/501=35.93% [28.92,43.92] |
| smishx | 2 | baseline | label_scam | 16/881=1.82% [0.93,2.74] | 176/881=19.98% [15.71,24.57] | 18/226=7.96% [4.27,13.81] | 41/226=18.14% [11.49,26.87] |
| smishx | 2 | transactional | label_binary | 17/606=2.81% [1.48,4.45] | 42/606=6.93% [4.94,9.03] | 46/501=9.18% [5.66,13.50] | 257/501=51.30% [43.95,59.24] |
| smishx | 2 | transactional | label_scam | 33/881=3.75% [1.98,5.96] | 224/881=25.43% [21.13,29.67] | 30/226=13.27% [8.13,20.52] | 75/226=33.19% [24.54,44.06] |
| smishx | 3 | baseline | label_binary | 14/606=2.31% [1.30,3.48] | 32/606=5.28% [3.61,7.10] | 44/501=8.78% [5.52,12.48] | 222/501=44.31% [36.58,52.24] |
| smishx | 3 | baseline | label_scam | 30/881=3.41% [1.86,5.29] | 202/881=22.93% [18.70,27.32] | 28/226=12.39% [7.42,19.54] | 52/226=23.01% [14.77,34.15] |
| smishx | 3 | transactional | label_binary | 16/606=2.64% [1.48,3.97] | 49/606=8.09% [6.09,10.43] | 33/501=6.59% [3.63,9.91] | 310/501=61.88% [54.24,69.94] |
| smishx | 3 | transactional | label_scam | 26/881=2.95% [1.61,4.63] | 261/881=29.63% [25.00,34.36] | 23/226=10.18% [5.36,16.81] | 98/226=43.36% [34.05,53.81] |

| Population | Seed | Role | Label | Coverage[CI] | UNCERTAIN share[CI] | Paired control student recall[CI] |
|---|---|---|---|---|---|---|
| campaign | 1 | baseline | label_binary | 21280/22145=96.09% [95.81,96.34] | 865/22145=3.91% [3.66,4.19] | 10621/11193=94.89% [94.49,95.34] |
| campaign | 1 | baseline | label_scam | 16678/16912=98.62% [98.43,98.78] | 234/16912=1.38% [1.22,1.57] | 5689/5926=96.00% [95.50,96.52] |
| campaign | 1 | transactional | label_binary | 21362/22145=96.46% [96.21,96.70] | 783/22145=3.54% [3.30,3.79] | 10621/11193=94.89% [94.49,95.34] |
| campaign | 1 | transactional | label_scam | 16647/16912=98.43% [98.24,98.61] | 265/16912=1.57% [1.39,1.76] | 5689/5926=96.00% [95.50,96.52] |
| campaign | 2 | baseline | label_binary | 21203/22145=95.75% [95.47,96.02] | 942/22145=4.25% [3.98,4.53] | 10614/11193=94.83% [94.45,95.25] |
| campaign | 2 | baseline | label_scam | 16504/16912=97.59% [97.35,97.81] | 408/16912=2.41% [2.19,2.65] | 5687/5926=95.97% [95.41,96.44] |
| campaign | 2 | transactional | label_binary | 21596/22145=97.52% [97.30,97.72] | 549/22145=2.48% [2.28,2.70] | 10614/11193=94.83% [94.45,95.25] |
| campaign | 2 | transactional | label_scam | 16764/16912=99.12% [98.98,99.26] | 148/16912=0.88% [0.74,1.02] | 5687/5926=95.97% [95.41,96.44] |
| campaign | 3 | baseline | label_binary | 21693/22145=97.96% [97.77,98.15] | 452/22145=2.04% [1.85,2.23] | 10610/11193=94.79% [94.39,95.19] |
| campaign | 3 | baseline | label_scam | 16788/16912=99.27% [99.14,99.39] | 124/16912=0.73% [0.61,0.86] | 5686/5926=95.95% [95.40,96.49] |
| campaign | 3 | transactional | label_binary | 21304/22145=96.20% [95.93,96.46] | 841/22145=3.80% [3.54,4.07] | 10610/11193=94.79% [94.39,95.19] |
| campaign | 3 | transactional | label_scam | 16610/16912=98.21% [98.02,98.41] | 302/16912=1.79% [1.59,1.98] | 5686/5926=95.95% [95.40,96.49] |
| lang_shift | 1 | baseline | label_binary | 1674/1677=99.82% [99.58,100.00] | 3/1677=0.18% [0.00,0.42] | 253/357=70.87% [61.69,80.32] |
| lang_shift | 1 | baseline | label_scam | 1421/1423=99.86% [99.64,100.00] | 2/1423=0.14% [0.00,0.36] | 78/101=77.23% [61.67,92.22] |
| lang_shift | 1 | transactional | label_binary | 1483/1677=88.43% [86.93,89.79] | 194/1677=11.57% [10.21,13.07] | 253/357=70.87% [61.69,80.32] |
| lang_shift | 1 | transactional | label_scam | 1246/1423=87.56% [85.79,89.30] | 177/1423=12.44% [10.70,14.21] | 78/101=77.23% [61.67,92.22] |
| lang_shift | 2 | baseline | label_binary | 1580/1677=94.22% [93.07,95.36] | 97/1677=5.78% [4.64,6.93] | 233/357=65.27% [55.53,75.24] |
| lang_shift | 2 | baseline | label_scam | 1331/1423=93.53% [92.16,94.75] | 92/1423=6.47% [5.25,7.84] | 69/101=68.32% [53.39,84.09] |
| lang_shift | 2 | transactional | label_binary | 1653/1677=98.57% [97.92,99.16] | 24/1677=1.43% [0.84,2.08] | 233/357=65.27% [55.53,75.24] |
| lang_shift | 2 | transactional | label_scam | 1402/1423=98.52% [97.83,99.15] | 21/1423=1.48% [0.85,2.17] | 69/101=68.32% [53.39,84.09] |
| lang_shift | 3 | baseline | label_binary | 1547/1677=92.25% [90.93,93.49] | 130/1677=7.75% [6.51,9.07] | 243/357=68.07% [59.19,77.30] |
| lang_shift | 3 | baseline | label_scam | 1304/1423=91.64% [90.19,93.02] | 119/1423=8.36% [6.98,9.81] | 71/101=70.30% [55.16,86.05] |
| lang_shift | 3 | transactional | label_binary | 1629/1677=97.14% [96.34,97.91] | 48/1677=2.86% [2.09,3.66] | 243/357=68.07% [59.19,77.30] |
| lang_shift | 3 | transactional | label_scam | 1376/1423=96.70% [95.64,97.60] | 47/1423=3.30% [2.40,4.36] | 71/101=70.30% [55.16,86.05] |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | baseline | label_binary | 939/941=99.79% [99.47,100.00] | 2/941=0.21% [0.00,0.53] | 385/417=92.33% [89.17,95.01] |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | baseline | label_scam | 939/941=99.79% [99.47,100.00] | 2/941=0.21% [0.00,0.53] | 264/273=96.70% [94.02,98.82] |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | transactional | label_binary | 940/941=99.89% [99.67,100.00] | 1/941=0.11% [0.00,0.33] | 385/417=92.33% [89.17,95.01] |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | transactional | label_scam | 940/941=99.89% [99.67,100.00] | 1/941=0.11% [0.00,0.33] | 264/273=96.70% [94.02,98.82] |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | baseline | label_binary | 913/941=97.02% [95.79,98.15] | 28/941=2.98% [1.85,4.21] | 388/417=93.05% [90.19,95.50] |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | baseline | label_scam | 913/941=97.02% [95.79,98.15] | 28/941=2.98% [1.85,4.21] | 263/273=96.34% [93.61,98.58] |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | transactional | label_binary | 939/941=99.79% [99.47,100.00] | 2/941=0.21% [0.00,0.53] | 388/417=93.05% [90.19,95.50] |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | transactional | label_scam | 939/941=99.79% [99.47,100.00] | 2/941=0.21% [0.00,0.53] | 263/273=96.34% [93.61,98.58] |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | baseline | label_binary | 941/941=100.00% [100.00,100.00] | 0/941=0.00% [0.00,0.00] | 391/417=93.76% [91.07,96.14] |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | baseline | label_scam | 941/941=100.00% [100.00,100.00] | 0/941=0.00% [0.00,0.00] | 264/273=96.70% [94.35,98.64] |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | transactional | label_binary | 939/941=99.79% [99.45,100.00] | 2/941=0.21% [0.00,0.55] | 391/417=93.76% [91.07,96.14] |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | transactional | label_scam | 939/941=99.79% [99.45,100.00] | 2/941=0.21% [0.00,0.55] | 264/273=96.70% [94.35,98.64] |
| source_holdout:holdout_scamshield | 1 | baseline | label_binary | 68066/75973=89.59% [89.29,89.91] | 7907/75973=10.41% [10.09,10.71] | 20269/29029=69.82% [68.52,71.08] |
| source_holdout:holdout_scamshield | 1 | baseline | label_scam | 49004/52512=93.32% [93.05,93.58] | 3508/52512=6.68% [6.42,6.95] | 4537/5568=81.48% [78.46,84.65] |
| source_holdout:holdout_scamshield | 1 | transactional | label_binary | 71388/75973=93.96% [93.74,94.19] | 4585/75973=6.04% [5.81,6.26] | 20269/29029=69.82% [68.52,71.08] |
| source_holdout:holdout_scamshield | 1 | transactional | label_scam | 50613/52512=96.38% [96.18,96.58] | 1899/52512=3.62% [3.42,3.82] | 4537/5568=81.48% [78.46,84.65] |
| source_holdout:holdout_scamshield | 2 | baseline | label_binary | 71710/75973=94.39% [94.15,94.61] | 4263/75973=5.61% [5.39,5.85] | 17153/29029=59.09% [57.54,60.63] |
| source_holdout:holdout_scamshield | 2 | baseline | label_scam | 50819/52512=96.78% [96.55,96.98] | 1693/52512=3.22% [3.02,3.45] | 3807/5568=68.37% [64.05,73.39] |
| source_holdout:holdout_scamshield | 2 | transactional | label_binary | 70719/75973=93.08% [92.84,93.31] | 5254/75973=6.92% [6.69,7.16] | 17153/29029=59.09% [57.54,60.63] |
| source_holdout:holdout_scamshield | 2 | transactional | label_scam | 50259/52512=95.71% [95.48,95.94] | 2253/52512=4.29% [4.06,4.52] | 3807/5568=68.37% [64.05,73.39] |
| source_holdout:holdout_scamshield | 3 | baseline | label_binary | 75248/75973=99.05% [98.87,99.18] | 725/75973=0.95% [0.82,1.13] | 20395/29029=70.26% [68.87,71.53] |
| source_holdout:holdout_scamshield | 3 | baseline | label_scam | 52108/52512=99.23% [99.00,99.40] | 404/52512=0.77% [0.60,1.00] | 4524/5568=81.25% [78.23,84.48] |
| source_holdout:holdout_scamshield | 3 | transactional | label_binary | 70138/75973=92.32% [92.07,92.55] | 5835/75973=7.68% [7.45,7.93] | 20395/29029=70.26% [68.87,71.53] |
| source_holdout:holdout_scamshield | 3 | transactional | label_scam | 49878/52512=94.98% [94.76,95.20] | 2634/52512=5.02% [4.80,5.24] | 4524/5568=81.25% [78.23,84.48] |
| source_holdout:holdout_uci_sms_spam | 1 | baseline | label_binary | 5158/5159=99.98% [99.94,100.00] | 1/5159=0.02% [0.00,0.06] | 552/642=85.98% [82.78,88.92] |
| source_holdout:holdout_uci_sms_spam | 1 | baseline | label_scam | 4517/4517=100.00% [100.00,100.00] | 0/4517=0.00% [0.00,0.00] | NA(n=0) |
| source_holdout:holdout_uci_sms_spam | 1 | transactional | label_binary | 5152/5159=99.86% [99.77,99.96] | 7/5159=0.14% [0.04,0.23] | 552/642=85.98% [82.78,88.92] |
| source_holdout:holdout_uci_sms_spam | 1 | transactional | label_scam | 4515/4517=99.96% [99.89,100.00] | 2/4517=0.04% [0.00,0.11] | NA(n=0) |
| source_holdout:holdout_uci_sms_spam | 2 | baseline | label_binary | 5156/5159=99.94% [99.86,100.00] | 3/5159=0.06% [0.00,0.14] | 529/642=82.40% [78.87,85.69] |
| source_holdout:holdout_uci_sms_spam | 2 | baseline | label_scam | 4517/4517=100.00% [100.00,100.00] | 0/4517=0.00% [0.00,0.00] | NA(n=0) |
| source_holdout:holdout_uci_sms_spam | 2 | transactional | label_binary | 5157/5159=99.96% [99.90,100.00] | 2/5159=0.04% [0.00,0.10] | 529/642=82.40% [78.87,85.69] |
| source_holdout:holdout_uci_sms_spam | 2 | transactional | label_scam | 4516/4517=99.98% [99.93,100.00] | 1/4517=0.02% [0.00,0.07] | NA(n=0) |
| source_holdout:holdout_uci_sms_spam | 3 | baseline | label_binary | 5159/5159=100.00% [100.00,100.00] | 0/5159=0.00% [0.00,0.00] | 538/642=83.80% [80.70,86.96] |
| source_holdout:holdout_uci_sms_spam | 3 | baseline | label_scam | 4517/4517=100.00% [100.00,100.00] | 0/4517=0.00% [0.00,0.00] | NA(n=0) |
| source_holdout:holdout_uci_sms_spam | 3 | transactional | label_binary | 5157/5159=99.96% [99.90,100.00] | 2/5159=0.04% [0.00,0.10] | 538/642=83.80% [80.70,86.96] |
| source_holdout:holdout_uci_sms_spam | 3 | transactional | label_scam | 4515/4517=99.96% [99.89,100.00] | 2/4517=0.04% [0.00,0.11] | NA(n=0) |
| temporal | 1 | baseline | label_binary | 4940/11029=44.79% [43.83,45.74] | 6089/11029=55.21% [54.26,56.17] | 65/77=84.42% [75.68,92.00] |
| temporal | 1 | baseline | label_scam | 4940/11029=44.79% [43.83,45.74] | 6089/11029=55.21% [54.26,56.17] | 65/77=84.42% [75.68,92.00] |
| temporal | 1 | transactional | label_binary | 6329/11029=57.39% [56.42,58.32] | 4700/11029=42.61% [41.68,43.58] | 65/77=84.42% [75.68,92.00] |
| temporal | 1 | transactional | label_scam | 6329/11029=57.39% [56.42,58.32] | 4700/11029=42.61% [41.68,43.58] | 65/77=84.42% [75.68,92.00] |
| temporal | 2 | baseline | label_binary | 8640/11029=78.34% [77.54,79.14] | 2389/11029=21.66% [20.86,22.46] | 69/77=89.61% [81.82,95.95] |
| temporal | 2 | baseline | label_scam | 8640/11029=78.34% [77.54,79.14] | 2389/11029=21.66% [20.86,22.46] | 69/77=89.61% [81.82,95.95] |
| temporal | 2 | transactional | label_binary | 6478/11029=58.74% [57.79,59.68] | 4551/11029=41.26% [40.32,42.21] | 69/77=89.61% [81.82,95.95] |
| temporal | 2 | transactional | label_scam | 6478/11029=58.74% [57.79,59.68] | 4551/11029=41.26% [40.32,42.21] | 69/77=89.61% [81.82,95.95] |
| temporal | 3 | baseline | label_binary | 9934/11029=90.07% [89.47,90.66] | 1095/11029=9.93% [9.34,10.53] | 68/77=88.31% [80.76,94.87] |
| temporal | 3 | baseline | label_scam | 9934/11029=90.07% [89.47,90.66] | 1095/11029=9.93% [9.34,10.53] | 68/77=88.31% [80.76,94.87] |
| temporal | 3 | transactional | label_binary | 9529/11029=86.40% [85.72,87.10] | 1500/11029=13.60% [12.90,14.28] | 68/77=88.31% [80.76,94.87] |
| temporal | 3 | transactional | label_scam | 9529/11029=86.40% [85.72,87.10] | 1500/11029=13.60% [12.90,14.28] | 68/77=88.31% [80.76,94.87] |
| groundlm | 1 | baseline | label_binary | 1170/1224=95.59% [92.16,98.38] | 54/1224=4.41% [1.62,7.84] | 57/91=62.64% [42.55,78.00] |
| groundlm | 1 | baseline | label_scam | 1170/1224=95.59% [92.16,98.38] | 54/1224=4.41% [1.62,7.84] | 57/91=62.64% [42.55,78.00] |
| groundlm | 1 | transactional | label_binary | 1162/1224=94.93% [92.05,97.51] | 62/1224=5.07% [2.49,7.95] | 57/91=62.64% [42.55,78.00] |
| groundlm | 1 | transactional | label_scam | 1162/1224=94.93% [92.05,97.51] | 62/1224=5.07% [2.49,7.95] | 57/91=62.64% [42.55,78.00] |
| groundlm | 2 | baseline | label_binary | 1137/1224=92.89% [89.76,95.86] | 87/1224=7.11% [4.14,10.24] | 60/91=65.93% [45.78,85.23] |
| groundlm | 2 | baseline | label_scam | 1137/1224=92.89% [89.76,95.86] | 87/1224=7.11% [4.14,10.24] | 60/91=65.93% [45.78,85.23] |
| groundlm | 2 | transactional | label_binary | 1195/1224=97.63% [96.56,98.65] | 29/1224=2.37% [1.35,3.44] | 60/91=65.93% [45.78,85.23] |
| groundlm | 2 | transactional | label_scam | 1195/1224=97.63% [96.56,98.65] | 29/1224=2.37% [1.35,3.44] | 60/91=65.93% [45.78,85.23] |
| groundlm | 3 | baseline | label_binary | 1199/1224=97.96% [96.99,98.88] | 25/1224=2.04% [1.12,3.01] | 52/91=57.14% [37.04,77.01] |
| groundlm | 3 | baseline | label_scam | 1199/1224=97.96% [96.99,98.88] | 25/1224=2.04% [1.12,3.01] | 52/91=57.14% [37.04,77.01] |
| groundlm | 3 | transactional | label_binary | 1145/1224=93.55% [89.83,97.08] | 79/1224=6.45% [2.92,10.17] | 52/91=57.14% [37.04,77.01] |
| groundlm | 3 | transactional | label_scam | 1145/1224=93.55% [89.83,97.08] | 79/1224=6.45% [2.92,10.17] | 52/91=57.14% [37.04,77.01] |
| smishx | 1 | baseline | label_binary | 903/1107=81.57% [78.18,84.81] | 204/1107=18.43% [15.19,21.82] | 363/501=72.46% [64.37,79.96] |
| smishx | 1 | baseline | label_scam | 903/1107=81.57% [78.18,84.81] | 204/1107=18.43% [15.19,21.82] | 129/226=57.08% [46.08,69.62] |
| smishx | 1 | transactional | label_binary | 939/1107=84.82% [81.67,87.79] | 168/1107=15.18% [12.21,18.33] | 363/501=72.46% [64.37,79.96] |
| smishx | 1 | transactional | label_scam | 939/1107=84.82% [81.67,87.79] | 168/1107=15.18% [12.21,18.33] | 129/226=57.08% [46.08,69.62] |
| smishx | 2 | baseline | label_binary | 924/1107=83.47% [79.73,86.87] | 183/1107=16.53% [13.13,20.27] | 267/501=53.29% [45.24,61.65] |
| smishx | 2 | baseline | label_scam | 924/1107=83.47% [79.73,86.87] | 183/1107=16.53% [13.13,20.27] | 85/226=37.61% [28.00,49.46] |
| smishx | 2 | transactional | label_binary | 871/1107=78.68% [75.19,81.92] | 236/1107=21.32% [18.08,24.81] | 267/501=53.29% [45.24,61.65] |
| smishx | 2 | transactional | label_scam | 871/1107=78.68% [75.19,81.92] | 236/1107=21.32% [18.08,24.81] | 85/226=37.61% [28.00,49.46] |
| smishx | 3 | baseline | label_binary | 911/1107=82.29% [78.83,85.57] | 196/1107=17.71% [14.43,21.17] | 389/501=77.64% [71.89,83.16] |
| smishx | 3 | baseline | label_scam | 911/1107=82.29% [78.83,85.57] | 196/1107=17.71% [14.43,21.17] | 148/226=65.49% [56.42,75.02] |
| smishx | 3 | transactional | label_binary | 797/1107=72.00% [68.44,75.91] | 310/1107=28.00% [24.09,31.56] | 389/501=77.64% [71.89,83.16] |
| smishx | 3 | transactional | label_scam | 797/1107=72.00% [68.44,75.91] | 310/1107=28.00% [24.09,31.56] | 148/226=65.49% [56.42,75.02] |

Paired internal SCAM-recall change (augmented minus fresh fused control), both label policies:

| Fold | Seed | Label | Difference[CI], paired p |
|---|---|---|---|
| campaign | 1 | label_binary | -0.51pp [-0.81,-0.21], p=0.000999001 |
| campaign | 1 | label_scam | -0.91pp [-1.27,-0.56], p=0.000999001 |
| campaign | 2 | label_binary | +2.74pp [+2.41,+3.10], p=0.000999001 |
| campaign | 2 | label_scam | +2.60pp [+2.15,+3.06], p=0.000999001 |
| campaign | 3 | label_binary | -1.26pp [-1.56,-0.96], p=0.000999001 |
| campaign | 3 | label_scam | -1.33pp [-1.74,-0.92], p=0.000999001 |
| lang_shift | 1 | label_binary | -5.04pp [-8.90,-1.68], p=0.00799201 |
| lang_shift | 1 | label_scam | -9.90pp [-16.67,-4.12], p=0.004995 |
| lang_shift | 2 | label_binary | +0.56pp [-2.08,+3.42], p=0.735265 |
| lang_shift | 2 | label_scam | -3.96pp [-9.18,+0.00], p=0.122877 |
| lang_shift | 3 | label_binary | +0.84pp [-1.97,+3.61], p=0.595405 |
| lang_shift | 3 | label_scam | -2.97pp [-6.38,+0.00], p=0.0969031 |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | label_binary | -0.24pp [-1.58,+0.94], p=0.787213 |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | label_scam | +0.37pp [+0.00,+1.23], p=0.545455 |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | label_binary | +25.66pp [+20.00,+32.48], p=0.000999001 |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | label_scam | +18.68pp [+12.00,+26.36], p=0.000999001 |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | label_binary | -6.24pp [-9.18,-3.55], p=0.000999001 |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | label_scam | -6.23pp [-9.85,-2.96], p=0.002997 |
| source_holdout:holdout_scamshield | 1 | label_binary | +14.33pp [+13.64,+15.00], p=0.000999001 |
| source_holdout:holdout_scamshield | 1 | label_scam | +15.19pp [+12.95,+17.49], p=0.000999001 |
| source_holdout:holdout_scamshield | 2 | label_binary | +9.95pp [+9.39,+10.51], p=0.000999001 |
| source_holdout:holdout_scamshield | 2 | label_scam | +8.14pp [+6.75,+9.61], p=0.000999001 |
| source_holdout:holdout_scamshield | 3 | label_binary | -5.07pp [-5.48,-4.64], p=0.000999001 |
| source_holdout:holdout_scamshield | 3 | label_scam | -3.63pp [-4.60,-2.75], p=0.000999001 |
| source_holdout:holdout_uci_sms_spam | 1 | label_binary | +1.25pp [-1.98,+4.23], p=0.3996 |
| source_holdout:holdout_uci_sms_spam | 1 | label_scam | NA(n=0) |
| source_holdout:holdout_uci_sms_spam | 2 | label_binary | +9.19pp [+5.60,+12.98], p=0.000999001 |
| source_holdout:holdout_uci_sms_spam | 2 | label_scam | NA(n=0) |
| source_holdout:holdout_uci_sms_spam | 3 | label_binary | +0.00pp [-3.75,+3.47], p=1 |
| source_holdout:holdout_uci_sms_spam | 3 | label_scam | NA(n=0) |
| temporal | 1 | label_binary | +7.79pp [+2.50,+14.47], p=0.00999001 |
| temporal | 1 | label_scam | +7.79pp [+2.50,+14.47], p=0.00999001 |
| temporal | 2 | label_binary | -5.19pp [-10.67,-1.27], p=0.048951 |
| temporal | 2 | label_scam | -5.19pp [-10.67,-1.27], p=0.048951 |
| temporal | 3 | label_binary | -1.30pp [-4.11,+0.00], p=0.561439 |
| temporal | 3 | label_scam | -1.30pp [-4.11,+0.00], p=0.561439 |

Paired external changes use the known-fraud endpoint; no paired superiority claim for the separately displayed legacy endpoints:

| External | Seed | Endpoint | Difference[CI], paired p |
|---|---|---|---|
| groundlm | 1 | fpr_alert | -25.51pp [-37.14,-12.94], p=0.000999001 |
| groundlm | 1 | fpr_scam | -27.54pp [-38.09,-16.36], p=0.000999001 |
| groundlm | 1 | recall_scam | -4.40pp [-20.00,+7.04], p=0.638362 |
| groundlm | 2 | fpr_alert | -21.80pp [-33.73,-9.27], p=0.001998 |
| groundlm | 2 | fpr_scam | -18.27pp [-28.05,-7.44], p=0.000999001 |
| groundlm | 2 | recall_scam | +9.89pp [-8.82,+25.00], p=0.292707 |
| groundlm | 3 | fpr_alert | -20.39pp [-30.76,-9.05], p=0.000999001 |
| groundlm | 3 | fpr_scam | -24.36pp [-34.86,-11.84], p=0.000999001 |
| groundlm | 3 | recall_scam | +2.20pp [-8.70,+12.67], p=0.712288 |
| smishx | 1 | fpr_alert | -3.97pp [-6.12,-2.03], p=0.000999001 |
| smishx | 1 | fpr_scam | -0.34pp [-1.12,+0.33], p=0.390609 |
| smishx | 1 | recall_scam | -0.44pp [-2.47,+1.35], p=0.734266 |
| smishx | 2 | fpr_alert | +5.45pp [+3.33,+7.66], p=0.000999001 |
| smishx | 2 | fpr_scam | +1.93pp [+0.46,+3.83], p=0.025974 |
| smishx | 2 | recall_scam | +5.31pp [+1.83,+9.95], p=0.013986 |
| smishx | 3 | fpr_alert | +6.70pp [+4.56,+8.94], p=0.000999001 |
| smishx | 3 | fpr_scam | -0.45pp [-1.57,+0.46], p=0.400599 |
| smishx | 3 | recall_scam | -2.21pp [-4.83,+0.00], p=0.0879121 |

| Failed predeclared check |
|---|
| 1/groundlm_scam_le_1pct |
| 1/groundlm_at_least_uncertain_le_10pct |
| 1/smishx_recall_ge_student |
| 1/campaign/label_scam/recall |
| 2/groundlm_scam_le_1pct |
| 2/groundlm_at_least_uncertain_le_10pct |
| 2/smishx_recall_ge_student |
| 3/groundlm_scam_le_1pct |
| 3/groundlm_at_least_uncertain_le_10pct |
| 3/smishx_recall_ge_student |
| 3/campaign/label_binary/recall |
| 3/campaign/label_scam/recall |
| 3/source_holdout:holdout_mishra_soni_smsphishing/label_binary/recall |
| 3/source_holdout:holdout_mishra_soni_smsphishing/label_scam/recall |
| 3/source_holdout:holdout_scamshield/label_binary/recall |
| 3/source_holdout:holdout_scamshield/label_scam/recall |

UCI known-fraud recall NA: 1/source_holdout:holdout_uci_sms_spam/label_scam/recall, 2/source_holdout:holdout_uci_sms_spam/label_scam/recall, 3/source_holdout:holdout_uci_sms_spam/label_scam/recall.

**Artifact limitation:** the original training job saved scores and aggregate receipts, but did not save the36selected student checkpoints. No reloadable/deployable augmented model is claimed. This KILL does not justify retraining merely to recover rejected weights. The private archive retains generated training data/manifests, aggregate training/fused results and hash-bound cal states; actual row-score files are excluded.200-item manual review remains pending.

All student-only per-seed/F1/external diagnostics and synthetic counts are in `TRANSACTIONAL_HARDNEG_STUDENT_RESULTS.md`; these are not frozen-v1 deployment claims. Full-system experimental results above explicitly name arm(a), binding unchanged parentv1 manifest `c310d3ccbc1c1e50ce9436f217c66ee9934f778985bfef487a034cc3070b79fa`. Future deployed external FPR claims must name v1 or a separately frozen successor.

Fused job `fusion_hardneg-a-audit-20261101`v1, tree `99c63db6d3110d23c9638e2dc9a36321303d944e65dd0e34bc787dd342d923f3`, git `6d4669f68c194cafdd9d7827e86d0264d34ba9d3`, ok367.3s. Original one event subscription and one provenance-matched fetch. Args bind source tree `1eb58925b36edc0f543d555c4007bc04289f1181a326e2ec7330a48fd315584f`, source metrics `0982ed2815454351417ccb9b91ae725bf24340d8b32a1858255f185f1d3738a7`, v1 manifest/metrics SHA and n_boot1000; exact launch in STATUS.md.

Fused metricsSHA `1c32e23643f594e722f5b15066c78d18962ff971d75ffc57be49c60a2b57109e`; frozen statesSHA `77949234736a99401eb8507d7b7dab45788e41e87cfa288e23a49d55d0ae78af`. Models/schema/thresholds never refit on external rows. **HF publish pending**; private archive `maadhavvh/rakshak-transactional-hardneg-20261101`v1 verified `isPrivate=true`,23311763bytes; privacy/manifest receipt in STATUS.md. All tables format saved Kaggle statistics only; no local inference/resampling.
