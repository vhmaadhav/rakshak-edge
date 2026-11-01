# Clean single-teacher KD versus no KD: KILL

**The historical Hinglish KD0.80–0.82 claim remains WITHDRAWN.** Both arms use only each fold’s frozen original train IDs, with no unlabeled transfer extras or synthetic rows. Cached mmBERT teacher0 targets are complete on KD train IDs only; these are in-sample teacher outputs, **not out-of-fold targets or a teacher ensemble**. No-KD receives full hard-label BCE, not a0.3-scaled loss. Same student, ordered IDs, seeds1/2/3, class weighting, AdamW.002/weight_decay.0001,15epoch cap/patience3, cal PR-AUC stopping. KDalpha.7/temperature2. Cal-only thresholds; no test/external selection.

Predeclared gate: KD lang_shift mean-F1 CI lower bound must exceed the matched no-KD point mean, and every fold loss must be within no-KD CI half-width. Result **KILL**: both language-shift checks fail; other fold loss checks pass. No gate/hyperparameter changes after results. Keep the train-only guard and full-BCE control, not this KD recipe as a language-shift fix.

CIs and paired tests use1,000 campaign-clustered bootstrap draws, one shared campaign multiplicity for both arms and all3 fixed seeds. These intervals are conditional on these seeds; they do not estimate all initialization uncertainty. A=KD, B=no-KD; paired p is null-centred. Frozen test membership stays unchanged.

GroundLM/ScamShield informed rule features; GroundLM/SmishX feedback motivated interventions. Reports are in-sample for intervention selection, with no held-out improvement claim. SmishX is independent of train after overlap removal, non-Indian-specific, but no longer untouched for policy selection. Temporal retains its original train/test campaign overlaps. Independent Indian validation remains OPEN.

| Fold | No-KD mean F1 [CI] | KD mean F1 [CI] | Paired KD-minus-control [CI] | p | No-fold-loss |
|---|---|---|---|---|---|
| campaign | 0.974358 [0.972272,0.976493] | 0.974835 [0.972665,0.976860] | 0.000477 [-0.000485,0.001455] | 0.306693 | True |
| lang_shift | 0.405842 [0.369672,0.441243] | 0.351977 [0.318139,0.384016] | -0.053865 [-0.069574,-0.038291] | 0.000999001 | False |
| source_holdout:holdout_mishra_soni_smsphishing | 0.938140 [0.918628,0.954345] | 0.936617 [0.917594,0.952134] | -0.001523 [-0.009290,0.007359] | 0.724276 | True |
| source_holdout:holdout_scamshield | 0.715195 [0.706331,0.726140] | 0.753329 [0.745836,0.762716] | 0.038134 [0.035297,0.040816] | 0.000999001 | True |
| source_holdout:holdout_uci_sms_spam | 0.867138 [0.846774,0.885429] | 0.896366 [0.878854,0.912469] | 0.029228 [0.018651,0.040057] | 0.000999001 | True |
| temporal | 0.865595 [0.805111,0.917742] | 0.869635 [0.812379,0.915976] | 0.004040 [-0.018911,0.026401] | 0.767233 | True |

| Fold | Arm | Seed | F1 at cal-max-F1 | cal-max-F1 threshold | Recall at cal-1%-FPR | FPR at cal-1%-FPR | cal-1%-FPR threshold | Teacher/train rows |
|---|---|---|---|---|---|---|---|---|
| campaign | erm | 1 | 0.974501 | 0.4875442386 | 0.952828 | 0.009953 | 0.8982028365 | 0/76591 |
| campaign | erm | 2 | 0.974722 | 0.4673939943 | 0.952381 | 0.011505 | 0.8626157045 | 0/76591 |
| campaign | erm | 3 | 0.973852 | 0.1584033519 | 0.949701 | 0.009770 | 0.6077079177 | 0/76591 |
| campaign | kd | 1 | 0.975764 | 0.3887766004 | 0.955240 | 0.011505 | 0.9289531112 | 76591/76591 |
| campaign | kd | 2 | 0.974487 | 0.3871260583 | 0.945859 | 0.009953 | 0.8827684522 | 76591/76591 |
| campaign | kd | 3 | 0.974254 | 0.4102430046 | 0.949254 | 0.011596 | 0.935045898 | 76591/76591 |
| lang_shift | erm | 1 | 0.346486 | 0.6205515265 | 0.697479 | 0.347727 | 0.9453421831 | 0/58605 |
| lang_shift | erm | 2 | 0.524358 | 0.4311504364 | 0.649860 | 0.029545 | 0.94250983 | 0/58605 |
| lang_shift | erm | 3 | 0.346683 | 0.3080346882 | 0.663866 | 0.088636 | 0.8909648657 | 0/58605 |
| lang_shift | kd | 1 | 0.353167 | 0.3280114233 | 0.661064 | 0.275758 | 0.9667152762 | 58605/58605 |
| lang_shift | kd | 2 | 0.370029 | 0.1895367503 | 0.694678 | 0.299242 | 0.8855559826 | 58605/58605 |
| lang_shift | kd | 3 | 0.332735 | 0.1015660539 | 0.703081 | 0.369697 | 0.7686860561 | 58605/58605 |
| source_holdout:holdout_mishra_soni_smsphishing | erm | 1 | 0.947735 | 0.4053137004 | 0.942446 | 0.030534 | 0.7830531597 | 0/96867 |
| source_holdout:holdout_mishra_soni_smsphishing | erm | 2 | 0.925234 | 0.1864462942 | 0.930456 | 0.062977 | 0.3324281275 | 0/96867 |
| source_holdout:holdout_mishra_soni_smsphishing | erm | 3 | 0.941452 | 0.7531033158 | 0.932854 | 0.038168 | 0.9212982059 | 0/96867 |
| source_holdout:holdout_mishra_soni_smsphishing | kd | 1 | 0.957143 | 0.332069993 | 0.949640 | 0.034351 | 0.7598038316 | 96867/96867 |
| source_holdout:holdout_mishra_soni_smsphishing | kd | 2 | 0.901001 | 0.7259193063 | 0.925659 | 0.066794 | 0.9785541892 | 96867/96867 |
| source_holdout:holdout_mishra_soni_smsphishing | kd | 3 | 0.951708 | 0.1167396158 | 0.920863 | 0.043893 | 0.3804236054 | 96867/96867 |
| source_holdout:holdout_scamshield | erm | 1 | 0.698034 | 0.1397853494 | 0.654966 | 0.227335 | 0.4829989672 | 0/24391 |
| source_holdout:holdout_scamshield | erm | 2 | 0.719751 | 0.07803329825 | 0.622068 | 0.188991 | 0.6139837503 | 0/24391 |
| source_holdout:holdout_scamshield | erm | 3 | 0.727800 | 0.06157945842 | 0.648317 | 0.208227 | 0.3268409967 | 0/24391 |
| source_holdout:holdout_scamshield | kd | 1 | 0.742740 | 0.02437748201 | 0.679011 | 0.223415 | 0.9095834494 | 24391/24391 |
| source_holdout:holdout_scamshield | kd | 2 | 0.766399 | 0.02525094524 | 0.666024 | 0.175677 | 0.9963447452 | 24391/24391 |
| source_holdout:holdout_scamshield | kd | 3 | 0.750849 | 0.7761918902 | 0.675704 | 0.206437 | 0.9972701669 | 24391/24391 |
| source_holdout:holdout_uci_sms_spam | erm | 1 | 0.854480 | 0.2556959391 | 0.788162 | 0.009962 | 0.5743978024 | 0/92637 |
| source_holdout:holdout_uci_sms_spam | erm | 2 | 0.860588 | 0.348770678 | 0.778816 | 0.010848 | 0.553352654 | 0/92637 |
| source_holdout:holdout_uci_sms_spam | erm | 3 | 0.886346 | 0.5327774286 | 0.809969 | 0.009741 | 0.8517301083 | 0/92637 |
| source_holdout:holdout_uci_sms_spam | kd | 1 | 0.899454 | 0.5116662383 | 0.864486 | 0.008634 | 0.7522962093 | 92637/92637 |
| source_holdout:holdout_uci_sms_spam | kd | 2 | 0.898240 | 0.26066643 | 0.819315 | 0.005313 | 0.7301368713 | 92637/92637 |
| source_holdout:holdout_uci_sms_spam | kd | 3 | 0.891403 | 0.2474481463 | 0.836449 | 0.009520 | 0.7023722529 | 92637/92637 |
| temporal | erm | 1 | 0.892086 | 0.9797132015 | 0.896104 | 0.010044 | 0.1539578289 | 0/34996 |
| temporal | erm | 2 | 0.845638 | 0.9670833945 | 0.831169 | 0.004200 | 0.909352839 | 0/34996 |
| temporal | erm | 3 | 0.859060 | 0.9780784249 | 0.857143 | 0.004931 | 0.7416427135 | 0/34996 |
| temporal | kd | 1 | 0.895105 | 0.03209104016 | 0.857143 | 0.008674 | 3.260778612e-05 | 34996/34996 |
| temporal | kd | 2 | 0.867647 | 0.6119151711 | 0.857143 | 0.009131 | 0.002241762588 | 34996/34996 |
| temporal | kd | 3 | 0.846154 | 0.461499542 | 0.870130 | 0.006848 | 0.07407360524 | 34996/34996 |

External student-only diagnostics below are **not fused/deployed FPR claims**. Full-system external claims must cite the frozen fusion-v1 manifest or a named successor. No new URL-only PhreshPhish experiment was run. Every campaign-model arm/seed, both frozen operating thresholds and both label policies are shown; no averages hide seed failures.

| Set | Label | Arm | Seed | Threshold type | Threshold | FPR [CI] | Recall [CI] |
|---|---|---|---|---|---|---|---|
| groundlm | label_binary | erm | 1 | fpr_1pct | 0.8982028365 | 454/1133=40.07% [25.42,53.03] | 67/91=73.63% [59.32,86.00] |
| groundlm | label_scam | erm | 1 | fpr_1pct | 0.8982028365 | 454/1133=40.07% [25.42,53.03] | 67/91=73.63% [59.32,86.00] |
| groundlm | label_binary | erm | 1 | max_f1 | 0.4875442386 | 476/1133=42.01% [27.17,55.19] | 82/91=90.11% [79.16,96.06] |
| groundlm | label_scam | erm | 1 | max_f1 | 0.4875442386 | 476/1133=42.01% [27.17,55.19] | 82/91=90.11% [79.16,96.06] |
| smishx | label_binary | erm | 1 | fpr_1pct | 0.8982028365 | 48/606=7.92% [5.91,10.18] | 326/501=65.07% [58.28,72.40] |
| smishx | label_scam | erm | 1 | fpr_1pct | 0.8982028365 | 256/881=29.06% [24.53,33.56] | 118/226=52.21% [42.67,63.70] |
| smishx | label_binary | erm | 1 | max_f1 | 0.4875442386 | 65/606=10.73% [8.54,13.35] | 420/501=83.83% [78.08,88.85] |
| smishx | label_scam | erm | 1 | max_f1 | 0.4875442386 | 318/881=36.10% [31.18,40.94] | 167/226=73.89% [64.93,83.34] |
| groundlm | label_binary | erm | 2 | fpr_1pct | 0.8626157045 | 347/1133=30.63% [17.58,42.54] | 57/91=62.64% [44.28,78.99] |
| groundlm | label_scam | erm | 2 | fpr_1pct | 0.8626157045 | 347/1133=30.63% [17.58,42.54] | 57/91=62.64% [44.28,78.99] |
| groundlm | label_binary | erm | 2 | max_f1 | 0.4673939943 | 407/1133=35.92% [21.73,48.44] | 73/91=80.22% [63.32,88.98] |
| groundlm | label_scam | erm | 2 | max_f1 | 0.4673939943 | 407/1133=35.92% [21.73,48.44] | 73/91=80.22% [63.32,88.98] |
| smishx | label_binary | erm | 2 | fpr_1pct | 0.8626157045 | 39/606=6.44% [4.47,8.54] | 313/501=62.48% [55.95,68.82] |
| smishx | label_scam | erm | 2 | fpr_1pct | 0.8626157045 | 236/881=26.79% [22.54,31.17] | 116/226=51.33% [41.39,61.58] |
| smishx | label_binary | erm | 2 | max_f1 | 0.4673939943 | 58/606=9.57% [7.41,11.97] | 405/501=80.84% [74.02,86.82] |
| smishx | label_scam | erm | 2 | max_f1 | 0.4673939943 | 304/881=34.51% [29.69,38.96] | 159/226=70.35% [58.21,81.40] |
| groundlm | label_binary | erm | 3 | fpr_1pct | 0.6077079177 | 339/1133=29.92% [17.92,40.81] | 66/91=72.53% [55.93,86.52] |
| groundlm | label_scam | erm | 3 | fpr_1pct | 0.6077079177 | 339/1133=29.92% [17.92,40.81] | 66/91=72.53% [55.93,86.52] |
| groundlm | label_binary | erm | 3 | max_f1 | 0.1584033519 | 500/1133=44.13% [30.33,56.82] | 81/91=89.01% [75.40,95.39] |
| groundlm | label_scam | erm | 3 | max_f1 | 0.1584033519 | 500/1133=44.13% [30.33,56.82] | 81/91=89.01% [75.40,95.39] |
| smishx | label_binary | erm | 3 | fpr_1pct | 0.6077079177 | 42/606=6.93% [5.12,9.05] | 350/501=69.86% [61.82,77.17] |
| smishx | label_scam | erm | 3 | fpr_1pct | 0.6077079177 | 270/881=30.65% [25.81,35.00] | 122/226=53.98% [43.46,66.84] |
| smishx | label_binary | erm | 3 | max_f1 | 0.1584033519 | 70/606=11.55% [9.11,14.19] | 463/501=92.42% [88.97,95.48] |
| smishx | label_scam | erm | 3 | max_f1 | 0.1584033519 | 335/881=38.02% [33.06,42.68] | 198/226=87.61% [80.43,93.18] |
| groundlm | label_binary | kd | 1 | fpr_1pct | 0.9289531112 | 379/1133=33.45% [19.70,45.59] | 64/91=70.33% [54.24,84.03] |
| groundlm | label_scam | kd | 1 | fpr_1pct | 0.9289531112 | 379/1133=33.45% [19.70,45.59] | 64/91=70.33% [54.24,84.03] |
| groundlm | label_binary | kd | 1 | max_f1 | 0.3887766004 | 464/1133=40.95% [27.55,52.81] | 78/91=85.71% [72.57,93.46] |
| groundlm | label_scam | kd | 1 | max_f1 | 0.3887766004 | 464/1133=40.95% [27.55,52.81] | 78/91=85.71% [72.57,93.46] |
| smishx | label_binary | kd | 1 | fpr_1pct | 0.9289531112 | 47/606=7.76% [5.78,9.94] | 338/501=67.47% [59.86,75.05] |
| smishx | label_scam | kd | 1 | fpr_1pct | 0.9289531112 | 285/881=32.35% [27.48,36.77] | 100/226=44.25% [35.32,55.50] |
| smishx | label_binary | kd | 1 | max_f1 | 0.3887766004 | 62/606=10.23% [7.96,12.79] | 405/501=80.84% [75.00,86.75] |
| smishx | label_scam | kd | 1 | max_f1 | 0.3887766004 | 315/881=35.75% [30.83,40.63] | 152/226=67.26% [58.33,77.53] |
| groundlm | label_binary | kd | 2 | fpr_1pct | 0.8827684522 | 419/1133=36.98% [23.55,49.02] | 53/91=58.24% [35.92,79.71] |
| groundlm | label_scam | kd | 2 | fpr_1pct | 0.8827684522 | 419/1133=36.98% [23.55,49.02] | 53/91=58.24% [35.92,79.71] |
| groundlm | label_binary | kd | 2 | max_f1 | 0.3871260583 | 486/1133=42.89% [29.10,54.86] | 72/91=79.12% [65.31,89.52] |
| groundlm | label_scam | kd | 2 | max_f1 | 0.3871260583 | 486/1133=42.89% [29.10,54.86] | 72/91=79.12% [65.31,89.52] |
| smishx | label_binary | kd | 2 | fpr_1pct | 0.8827684522 | 43/606=7.10% [5.13,9.23] | 236/501=47.11% [40.12,54.06] |
| smishx | label_scam | kd | 2 | fpr_1pct | 0.8827684522 | 191/881=21.68% [17.46,25.78] | 88/226=38.94% [30.47,50.32] |
| smishx | label_binary | kd | 2 | max_f1 | 0.3871260583 | 64/606=10.56% [8.28,13.22] | 369/501=73.65% [67.53,79.61] |
| smishx | label_scam | kd | 2 | max_f1 | 0.3871260583 | 285/881=32.35% [27.85,36.87] | 148/226=65.49% [57.78,74.49] |
| groundlm | label_binary | kd | 3 | fpr_1pct | 0.935045898 | 330/1133=29.13% [17.26,39.85] | 59/91=64.84% [41.43,81.99] |
| groundlm | label_scam | kd | 3 | fpr_1pct | 0.935045898 | 330/1133=29.13% [17.26,39.85] | 59/91=64.84% [41.43,81.99] |
| groundlm | label_binary | kd | 3 | max_f1 | 0.4102430046 | 466/1133=41.13% [26.71,53.98] | 73/91=80.22% [65.00,89.17] |
| groundlm | label_scam | kd | 3 | max_f1 | 0.4102430046 | 466/1133=41.13% [26.71,53.98] | 73/91=80.22% [65.00,89.17] |
| smishx | label_binary | kd | 3 | fpr_1pct | 0.935045898 | 48/606=7.92% [5.79,10.08] | 334/501=66.67% [59.45,73.54] |
| smishx | label_scam | kd | 3 | fpr_1pct | 0.935045898 | 263/881=29.85% [25.36,34.21] | 119/226=52.65% [44.44,63.45] |
| smishx | label_binary | kd | 3 | max_f1 | 0.4102430046 | 69/606=11.39% [9.08,13.86] | 437/501=87.23% [82.57,91.55] |
| smishx | label_scam | kd | 3 | max_f1 | 0.4102430046 | 326/881=37.00% [32.03,41.60] | 180/226=79.65% [71.68,86.76] |

| Selected model | SHA256 |
|---|---|
| campaign/erm/seed1 | `7a33504123874b7eeaf8f08b40ee0adf4ca03c2089313d3bf6a3f778b4e7360c` |
| campaign/erm/seed2 | `8c6eaf920d83ddd43229b52ef54260819bae95c6abd6d28e269f70238770fd4e` |
| campaign/erm/seed3 | `08ee9df2a597848025839022007a50ce83c6db54a44034b6516e99a1a54be33d` |
| campaign/kd/seed1 | `5cf7e9f4ff9001a89ffc58b1ecfa592a5ba5f69222d6932878889a22ba870498` |
| campaign/kd/seed2 | `c5c588e4684f62786fafdf50e60ec64e03edfd740838d48d8276206be8fcff9b` |
| campaign/kd/seed3 | `c65301140d4bee5e00075d8b2fd2b3a0aabeb2b1c1b5f65d8df802fae9a983d6` |
| lang_shift/erm/seed1 | `1fadf1c51238850b6fab5f63b940b1f24e0712f8d808baa15befa194d5f01e7b` |
| lang_shift/erm/seed2 | `d02f771e8780d48fea3ef7a7a969dbb911da576f057223b7a66e2f830331a674` |
| lang_shift/erm/seed3 | `f5417e89cc7c2726412c2e817d8e1f9c659d7c559d734b1dd6927c42bbdcd83d` |
| lang_shift/kd/seed1 | `4bee227765991d59c5aecaafdee7a2a8f4cbf920e5fbfe5b2bfbb97a3ed1966c` |
| lang_shift/kd/seed2 | `2f82a291323cac0d3d64bcbfaa3d90b48af3a1f4798c00c709696aed52b37e9e` |
| lang_shift/kd/seed3 | `961fff4e3ff3794bf6ac22d98946b55029f4067df0e18f2c7f54ffe18d16be70` |
| source_holdout:holdout_mishra_soni_smsphishing/erm/seed1 | `bff5097248421c5c9642a6b32d9f8ccce11e2f9e8f8aa300506187994a2ce0ac` |
| source_holdout:holdout_mishra_soni_smsphishing/erm/seed2 | `cc16665c82e9860fdc67313c098c5c13ad8cd56bfade5ea7f0b5a44d6a4adbc1` |
| source_holdout:holdout_mishra_soni_smsphishing/erm/seed3 | `d77caf08f2ffb1ca64645999f01a07d1240491bf8697469336e4482327efa659` |
| source_holdout:holdout_mishra_soni_smsphishing/kd/seed1 | `131ba77aee5a0aa74257f17f1c7cb233d4b3caf2806199482677215d27bd846a` |
| source_holdout:holdout_mishra_soni_smsphishing/kd/seed2 | `521464e63518eb29263675c35c9f770ae6b168f862fce72534bd4713cbd16896` |
| source_holdout:holdout_mishra_soni_smsphishing/kd/seed3 | `0944a9d6a86afb700a274d0555b9732fbc4ed8753bf6e5a02385ede80e5b441f` |
| source_holdout:holdout_scamshield/erm/seed1 | `0607022d81d0df21f17a514a2f01107b562f90c1f354c0e841a3907b5a2ad86e` |
| source_holdout:holdout_scamshield/erm/seed2 | `6278d973556d092e5af72096775ae56ec978eb210c2b952171cacd47bf31641a` |
| source_holdout:holdout_scamshield/erm/seed3 | `e1e213c5a3f34d9658193c5bd5193aefd12c540d21c36c436f30f8ccc376e670` |
| source_holdout:holdout_scamshield/kd/seed1 | `be85beb9b4f06db8ee67f7da73278675b461bfba9951b25c2660d2b4a50e3cf7` |
| source_holdout:holdout_scamshield/kd/seed2 | `57b2f7ee232045bd95a919b0ed1055ded44af90bb59788384b720d3be968b9e3` |
| source_holdout:holdout_scamshield/kd/seed3 | `57ab18297022d6024febd07972148b2cb495632f5201acc2d84e69869c74869a` |
| source_holdout:holdout_uci_sms_spam/erm/seed1 | `b57fbd6776f09e4ee0ecaeaf6fb93ebf7701460a1c82b59f27ed56f8402a289d` |
| source_holdout:holdout_uci_sms_spam/erm/seed2 | `3b336a68a7a17ef8b04e80011e33e5ff01053796cef2d4603a20042dbc10b506` |
| source_holdout:holdout_uci_sms_spam/erm/seed3 | `2817dfd546b8caa4a4ed8908a3e08aec8b0d798a48845bf77a99d386abeaca80` |
| source_holdout:holdout_uci_sms_spam/kd/seed1 | `9059754f459dfca9626ec44b5feca6afc4d3b20dc1145d4490f4778e17048c1d` |
| source_holdout:holdout_uci_sms_spam/kd/seed2 | `d6140f8137ef4b48ddb2453a8fce961bb6a870270326209db8fbd2633a2ad771` |
| source_holdout:holdout_uci_sms_spam/kd/seed3 | `10da39cb17ddf198c0cd7e65521e207d41d9408c37965b624bd56284b07276f5` |
| temporal/erm/seed1 | `cbd5dc20ad9b284b33a4b994a07dd774e4962a84a8ef8a631a8a43839b4e5ecc` |
| temporal/erm/seed2 | `6c74c99d34b69605a1b42650dc0209b889eab4867f725c3e92cacbda88f62045` |
| temporal/erm/seed3 | `c46582830553d55efaabd73cc47f875a3875dd8a6834930089c5dfcf61fc6515` |
| temporal/kd/seed1 | `1668c5f85f72ec3abbbeb4d8f55352c35df6dcd6724cd822b3dd436f8d873d25` |
| temporal/kd/seed2 | `91f151019ce277f2ad6c0df3326e0979a0f9ce2ba1675806230ae62e8733f57d` |
| temporal/kd/seed3 | `13bef632120dd34840d5efe20ed13e07057378dad52abf9d82b160b87864acba` |

Job `sms_clean_kd-kd1-20261101`v1, kernel `maadhavvh/rakshak-sms-clean-kd-kd1-20261101`; args `--seeds 1 2 3 --epochs 15 --n-boot 1000 --teacher-seed 0`. Result `ok`, 3108.8s;36 successful fits, no missing required gate results. One existing completion subscription and one provenance-matched fetch.

Git `6d4669f68c194cafdd9d7827e86d0264d34ba9d3`, dirty snapshot tree `4d8d927781cdfc90d92b1aad05413ba91b2676160e640e3b1f9b42b648052cb7`; aggregate metrics SHA256 `708b0fdc6e7b1646717c0bc572a56105100884c1eaaab6e44b47e8693cc8dcde`. Checkpoints saved privately before test/external scoring with exact run/arm/seed/teacher-enabled/git/tree metadata. No messages, row scores, logits or model weights in git. Private archive `maadhavvh/rakshak-clean-kd-20261101`v1 verified ownermaadhavvh, `isPrivate=true`,14567242bytes; initial processing-time403 and successful later owner verification in STATUS.md; **HF publish pending**. All tables are formatted saved aggregate JSON; no local model or statistical evaluation.
