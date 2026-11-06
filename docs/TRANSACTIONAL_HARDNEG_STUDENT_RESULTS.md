# Hard-negative student diagnostics — 2026-11-01

**Fused gate pending.** These are saved student-only diagnostic statistics, not a deployed
fusion-v1 safety claim. V1 is running separately; the final fused comparison will cite its
manifest and retain the original paired fresh-control gate. No template/model/threshold
choice changed after these outcomes. GroundLM/SmishX feedback motivated this intervention:
**in-sample for intervention selection**, no independent held-out improvement claim.
Independent Indian validation OPEN. Synthetic data is training-only; human-review sample
200 is present privately but human review has not been completed. HF publish pending.

## Student F1 at each arm/seed calibration max-F1 threshold

| Fold | Seed | Fresh control | Hard negatives |
|---|---:|---:|---:|
| campaign |1| 0.972724 | 0.974839 |
| campaign |2| 0.973443 | 0.973971 |
| campaign |3| 0.975256 | 0.973907 |
| lang_shift |1| 0.323164 | 0.307286 |
| lang_shift |2| 0.588652 | 0.586319 |
| lang_shift |3| 0.457002 | 0.300000 |
| source_holdout:holdout_mishra_soni_smsphishing |1| 0.951016 | 0.948626 |
| source_holdout:holdout_mishra_soni_smsphishing |2| 0.920598 | 0.925029 |
| source_holdout:holdout_mishra_soni_smsphishing |3| 0.951049 | 0.943713 |
| source_holdout:holdout_scamshield |1| 0.747235 | 0.710965 |
| source_holdout:holdout_scamshield |2| 0.734792 | 0.754723 |
| source_holdout:holdout_scamshield |3| 0.732204 | 0.695464 |
| source_holdout:holdout_uci_sms_spam |1| 0.906813 | 0.882748 |
| source_holdout:holdout_uci_sms_spam |2| 0.883721 | 0.883408 |
| source_holdout:holdout_uci_sms_spam |3| 0.878378 | 0.870384 |
| temporal |1| 0.901408 | 0.876712 |
| temporal |2| 0.867133 | 0.875000 |
| temporal |3| 0.900000 | 0.897959 |

## Paired mean F1 over the three fixed training seeds

| Fold | Control mean [CI] | Hard-negative mean [CI] | Delta [paired CI] | p |
|---|---|---|---|---:|
| campaign | 0.973808 [0.971510,0.975965] | 0.974239 [0.972075,0.976419] | 0.000431 [-0.000502,0.001344] | 0.414585 |
| lang_shift | 0.456273 [0.417433,0.492236] | 0.397868 [0.362604,0.431063] | -0.058404 [-0.075205,-0.040703] | 0.000999 |
| source_holdout:holdout_mishra_soni_smsphishing | 0.940888 [0.922203,0.956294] | 0.939122 [0.919675,0.955932] | -0.001765 [-0.010485,0.006746] | 0.684316 |
| source_holdout:holdout_scamshield | 0.738077 [0.730039,0.748345] | 0.720384 [0.711452,0.731198] | -0.017693 [-0.019871,-0.015719] | 0.000999 |
| source_holdout:holdout_uci_sms_spam | 0.889637 [0.871758,0.906879] | 0.878847 [0.858703,0.896493] | -0.010791 [-0.019931,-0.002135] | 0.016983 |
| temporal | 0.889514 [0.832071,0.938338] | 0.883224 [0.823300,0.928647] | -0.006290 [-0.030499,0.017222] | 0.600400 |

Campaign bootstrap resamples campaigns jointly across both arms/all three fixed seeds;
1,000 draws,seed13. Intervals condition on these initialisations. Tests are descriptive,
unadjusted. These F1 losses are not concealed by a benign-FPR gain on another endpoint.

## Student external endpoints at each arm/seed frozen cal1%-FPR threshold

| Set | Label endpoint | Seed | Arm | FPR events/n [campaign CI] | Recall events/n [campaign CI] |
|---|---|---:|---|---|---|
| groundlm | label_binary |1|erm| 397/1133 = 0.350397 [0.210387,0.471871] | 57/91 = 0.626374 [0.425532,0.780038] |
| groundlm | label_scam |1|erm| 397/1133 = 0.350397 [0.210387,0.471871] | 57/91 = 0.626374 [0.425532,0.780038] |
| smishx | label_binary |1|erm| 51/606 = 0.084158 [0.063018,0.106387] | 363/501 = 0.724551 [0.643699,0.799564] |
| smishx | label_scam |1|erm| 285/881 = 0.323496 [0.275927,0.370420] | 129/226 = 0.570796 [0.460770,0.696154] |
| groundlm | label_binary |1|hardneg| 101/1133 = 0.089144 [0.013826,0.179835] | 50/91 = 0.549451 [0.317003,0.740000] |
| groundlm | label_scam |1|hardneg| 101/1133 = 0.089144 [0.013826,0.179835] | 50/91 = 0.549451 [0.317003,0.740000] |
| smishx | label_binary |1|hardneg| 44/606 = 0.072607 [0.052893,0.095242] | 306/501 = 0.610778 [0.533674,0.689515] |
| smishx | label_scam |1|hardneg| 234/881 = 0.265607 [0.221578,0.309183] | 116/226 = 0.513274 [0.407274,0.643291] |
| groundlm | label_binary |2|erm| 435/1133 = 0.383936 [0.241054,0.508884] | 60/91 = 0.659341 [0.457831,0.852273] |
| groundlm | label_scam |2|erm| 435/1133 = 0.383936 [0.241054,0.508884] | 60/91 = 0.659341 [0.457831,0.852273] |
| smishx | label_binary |2|erm| 43/606 = 0.070957 [0.051325,0.090767] | 267/501 = 0.532934 [0.452378,0.616547] |
| smishx | label_scam |2|erm| 225/881 = 0.255392 [0.211913,0.299257] | 85/226 = 0.376106 [0.279954,0.494577] |
| groundlm | label_binary |2|hardneg| 193/1133 = 0.170344 [0.065004,0.277776] | 60/91 = 0.659341 [0.488365,0.816335] |
| groundlm | label_scam |2|hardneg| 193/1133 = 0.170344 [0.065004,0.277776] | 60/91 = 0.659341 [0.488365,0.816335] |
| smishx | label_binary |2|hardneg| 47/606 = 0.077558 [0.054363,0.100333] | 343/501 = 0.684631 [0.606780,0.752050] |
| smishx | label_scam |2|hardneg| 270/881 = 0.306470 [0.262285,0.353222] | 120/226 = 0.530973 [0.421010,0.641518] |
| groundlm | label_binary |3|erm| 406/1133 = 0.358341 [0.216717,0.485651] | 52/91 = 0.571429 [0.370350,0.770142] |
| groundlm | label_scam |3|erm| 406/1133 = 0.358341 [0.216717,0.485651] | 52/91 = 0.571429 [0.370350,0.770142] |
| smishx | label_binary |3|erm| 50/606 = 0.082508 [0.061054,0.104132] | 389/501 = 0.776447 [0.718922,0.831602] |
| smishx | label_scam |3|erm| 291/881 = 0.330306 [0.284851,0.374729] | 148/226 = 0.654867 [0.564180,0.750169] |
| groundlm | label_binary |3|hardneg| 133/1133 = 0.117387 [0.034790,0.221302] | 56/91 = 0.615385 [0.389831,0.793893] |
| groundlm | label_scam |3|hardneg| 133/1133 = 0.117387 [0.034790,0.221302] | 56/91 = 0.615385 [0.389831,0.793893] |
| smishx | label_binary |3|hardneg| 55/606 = 0.090759 [0.069304,0.113678] | 380/501 = 0.758483 [0.687613,0.824076] |
| smishx | label_scam |3|hardneg| 297/881 = 0.337117 [0.289409,0.383096] | 138/226 = 0.610619 [0.502182,0.726377] |

Known fraud and legacy spam-inclusive labels are reported separately; no external
relabelling or threshold tuning. These saved student endpoints do not include fusion and
cannot pass its predeclared SCAM/alert FPR, recall and coverage gate.

Run: sms_transactional_hardneg-a1-20261101 v1,ok3341.3s,36fits; one event/fetch.
Tree:1eb58925b36edc0f543d555c4007bc04289f1181a326e2ec7330a48fd315584f.
Metrics SHA256:0982ed2815454351417ccb9b91ae725bf24340d8b32a1858255f185f1d3738a7.
