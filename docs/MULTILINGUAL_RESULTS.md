# Multilingual compact-input experiment: research GO, deployment open

Private Kaggle `sms_multilingual-m3-20261102` version1 completed54fits in4228.1s.
Exact source `93ee0115cc22af3a2bedeaacc82df2c10ebb52ed13f2e30387b72747da90d12e`,
baseb17021f (dirty snapshot). Manifest/metric hashes identify this run; no main push or merge.

**Recorded candidate gate GO; this is in-sample intervention selection, not independent Indian validation.**
Legacy spam-inclusive mean F1 improves on the frozen lang_shift sample from fresh byte control
0.4633 [0.4268,0.4951] to multilingual KD0.8916 [0.8630,0.9171], paired gain
+0.4283 [0.3909,0.4622], p0.000999. All6fold loss guards and18weight-size guards pass.
Confidence intervals condition on these three fixed seeds; they do not estimate initialization uncertainty.

**No safe/shipped model claim:** GroundLM candidate cal1%-FPR fraud false-positive rates
are39.19/34.86/38.31% for seeds1/2/3; SmishX25.09/23.50/29.06%. Recall remains weak.
Fusionv1 weights/calibration are unchanged. These are student-only diagnostics, not fused FPR.
Independent Indian validation stays OPEN; Arm/runtime/export/tokenizer integration remain UNVERIFIED.

## Hypothesis and unchanged protocol

Compact MIT mmBERT-small-derived token embeddings beside dual byte CNNs may close the language-shift gap.
Full preregistration: MULTILINGUAL_STUDENT.md. Pinned jhu-clsp/mmBERT-small revision
abc32620dd4f6ab06f5fbe905dc25f310618e09f; card/config/bin hashes checked before corpus access.
No transformer layers deployed. Source pretraining membership is unknown.
Train-token-only vocabulary<=16384, PCA384→64 fitted only on selected training embeddings,
128tokens, masked mean/max pooling plus existing byte/rule features; same fixed three-seed recipe.
Only exact original frozen train IDs/teacher0 targets, no transfer extras or OOF/ensemble claim.
Original synthetic training rows retained equally; no new synthetic augmentation. Cal stopping/thresholds only.
Vocabulary and classifier/threshold manifests were frozen before corresponding outcome transformations/scoring.

The scientific gate compares KDhybrid to the fresh byte control. NoKDhybrid is diagnostic and NOTpromotable.
Its high F1 does not establish a separate KD benefit: no paired KD-versus-noKDhybrid test was predeclared.
Historical clean byte0.4058/KD0.3520/teacher0.802 are different reference runs, not reproduced controls here.
The fresh byte control is0.4633, so do not substitute historical0.4058 in the paired gain.
Contaminated historical≈0.81 remains WITHDRAWN; this experiment does not rehabilitate its transfer pool.

## Matched mean-F1 comparisons

1000paired campaign draws, seed13; mean over3fixed training seeds. p-values unadjusted, exploratory outside
primary language endpoint. Non-significant fold deltas are not wins. Temporal remains only77positives
and the inherited3train/test campaign overlaps; frozen memberships unchanged.

| Fold | Fresh byte mean F1 [CI] | KDhybrid mean F1 [CI] | Paired delta [CI] | p | NoKDhybrid diagnostic [CI] |
|---|---|---|---|---|---|
| campaign | 0.9747 [0.9726, 0.9767] | 0.9764 [0.9742, 0.9785] | 0.0017 [0.0003, 0.0031] | 0.0190 | 0.9762 [0.9740, 0.9783] |
| lang_shift | 0.4633 [0.4268, 0.4951] | 0.8916 [0.8630, 0.9171] | 0.4283 [0.3909, 0.4622] | 0.0010 | 0.8726 [0.8468, 0.8953] |
| source_holdout:holdout_mishra_soni_smsphishing | 0.9423 [0.9232, 0.9574] | 0.9517 [0.9352, 0.9658] | 0.0094 [-0.0021, 0.0222] | 0.1329 | 0.9458 [0.9275, 0.9620] |
| source_holdout:holdout_scamshield | 0.7126 [0.7036, 0.7234] | 0.7857 [0.7780, 0.7942] | 0.0730 [0.0686, 0.0779] | 0.0010 | 0.7573 [0.7488, 0.7666] |
| source_holdout:holdout_uci_sms_spam | 0.8587 [0.8383, 0.8774] | 0.8737 [0.8531, 0.8915] | 0.0150 [0.0004, 0.0285] | 0.0420 | 0.8728 [0.8530, 0.8909] |
| temporal | 0.8877 [0.8310, 0.9373] | 0.8644 [0.8032, 0.9179] | -0.0233 [-0.0574, 0.0072] | 0.1708 | 0.8459 [0.7755, 0.9014] |

## Every seed and fold

F1 and its campaign CI use the cal-maxF1 threshold. AP CI is threshold-free.
FPR/recall in the maxF1 columns have corresponding campaign CIs below. Cal1%-FPR recall/FPR are point
estimates; do not reuse the existing test-oracle recall@FPR bootstrap for those fixed-cal endpoints.

| Fold / arm | Seed | Test n/positives | MaxF1 threshold | F1 [CI] | AP [CI] | MaxF1 FPR [CI] | MaxF1 recall [CI] | Cal1% threshold | Cal1% FPR / recall |
|---|---|---|---|---|---|---|---|---|---|
| campaign / byte_nokd | 1 | 22145/11193 | 0.4754 | 0.9741 [0.9717, 0.9763] | 0.9967 [0.9962, 0.9972] | 0.0268 [0.0232, 0.0303] | 0.9743 [0.9713, 0.9772] | 0.9030 | 0.0101 / 0.9484 |
| campaign / byte_nokd | 2 | 22145/11193 | 0.2274 | 0.9749 [0.9726, 0.9771] | 0.9968 [0.9963, 0.9972] | 0.0208 [0.0177, 0.0242] | 0.9703 [0.9670, 0.9734] | 0.6480 | 0.0098 / 0.9473 |
| campaign / byte_nokd | 3 | 22145/11193 | 0.5495 | 0.9752 [0.9728, 0.9773] | 0.9966 [0.9959, 0.9971] | 0.0268 [0.0236, 0.0307] | 0.9767 [0.9742, 0.9795] | 0.9557 | 0.0094 / 0.9431 |
| campaign / multilingual_kd | 1 | 22145/11193 | 0.1092 | 0.9771 [0.9749, 0.9792] | 0.9971 [0.9965, 0.9975] | 0.0206 [0.0177, 0.0239] | 0.9745 [0.9715, 0.9777] | 0.6082 | 0.0100 / 0.9518 |
| campaign / multilingual_kd | 2 | 22145/11193 | 0.4759 | 0.9764 [0.9741, 0.9785] | 0.9972 [0.9967, 0.9976] | 0.0203 [0.0174, 0.0236] | 0.9728 [0.9697, 0.9758] | 0.9077 | 0.0103 / 0.9547 |
| campaign / multilingual_kd | 3 | 22145/11193 | 0.1109 | 0.9758 [0.9736, 0.9779] | 0.9970 [0.9965, 0.9974] | 0.0250 [0.0218, 0.0286] | 0.9761 [0.9734, 0.9790] | 0.7244 | 0.0103 / 0.9530 |
| campaign / multilingual_nokd | 1 | 22145/11193 | 0.2533 | 0.9760 [0.9736, 0.9782] | 0.9972 [0.9967, 0.9976] | 0.0205 [0.0176, 0.0237] | 0.9722 [0.9691, 0.9755] | 0.6450 | 0.0098 / 0.9495 |
| campaign / multilingual_nokd | 2 | 22145/11193 | 0.6752 | 0.9763 [0.9741, 0.9783] | 0.9971 [0.9967, 0.9976] | 0.0169 [0.0142, 0.0200] | 0.9694 [0.9664, 0.9729] | 0.8759 | 0.0101 / 0.9518 |
| campaign / multilingual_nokd | 3 | 22145/11193 | 0.3742 | 0.9762 [0.9740, 0.9783] | 0.9969 [0.9965, 0.9974] | 0.0220 [0.0190, 0.0254] | 0.9741 [0.9712, 0.9771] | 0.7394 | 0.0106 / 0.9517 |
| lang_shift / byte_nokd | 1 | 1677/357 | 0.3473 | 0.3588 [0.3348, 0.3822] | 0.7246 [0.6542, 0.8011] | 0.7136 [0.6831, 0.7443] | 0.7955 [0.7119, 0.8827] | 0.8289 | 0.3295 / 0.7031 |
| lang_shift / byte_nokd | 2 | 1677/357 | 0.6358 | 0.6858 [0.6531, 0.7197] | 0.8790 [0.8404, 0.9181] | 0.1811 [0.1566, 0.2052] | 0.8711 [0.8232, 0.9157] | 0.9725 | 0.0098 / 0.6695 |
| lang_shift / byte_nokd | 3 | 1677/357 | 0.2843 | 0.3453 [0.3192, 0.3769] | 0.7263 [0.6599, 0.7977] | 0.8985 [0.8703, 0.9241] | 0.9020 [0.8429, 0.9514] | 0.8614 | 0.4917 / 0.7339 |
| lang_shift / multilingual_kd | 1 | 1677/357 | 0.1326 | 0.8861 [0.8595, 0.9138] | 0.9504 [0.9321, 0.9679] | 0.0371 [0.0276, 0.0482] | 0.9048 [0.8688, 0.9419] | 0.5869 | 0.0098 / 0.8235 |
| lang_shift / multilingual_kd | 2 | 1677/357 | 0.2607 | 0.9068 [0.8813, 0.9306] | 0.9577 [0.9406, 0.9727] | 0.0273 [0.0186, 0.0375] | 0.9132 [0.8772, 0.9477] | 0.9290 | 0.0023 / 0.7255 |
| lang_shift / multilingual_kd | 3 | 1677/357 | 0.9168 | 0.8819 [0.8423, 0.9165] | 0.9448 [0.9225, 0.9648] | 0.0273 [0.0184, 0.0364] | 0.8683 [0.8015, 0.9299] | 0.9813 | 0.0114 / 0.7759 |
| lang_shift / multilingual_nokd | 1 | 1677/357 | 0.7196 | 0.8475 [0.8162, 0.8717] | 0.9604 [0.9442, 0.9740] | 0.0788 [0.0647, 0.0927] | 0.9496 [0.9288, 0.9708] | 0.9444 | 0.0076 / 0.8263 |
| lang_shift / multilingual_nokd | 2 | 1677/357 | 0.2350 | 0.8979 [0.8702, 0.9216] | 0.9642 [0.9485, 0.9771] | 0.0485 [0.0364, 0.0617] | 0.9608 [0.9396, 0.9794] | 0.8387 | 0.0023 / 0.7255 |
| lang_shift / multilingual_nokd | 3 | 1677/357 | 0.8685 | 0.8724 [0.8353, 0.9054] | 0.9369 [0.9118, 0.9606] | 0.0341 [0.0247, 0.0449] | 0.8711 [0.8081, 0.9271] | 0.9405 | 0.0045 / 0.7143 |
| source_holdout:holdout_mishra_soni_smsphishing / byte_nokd | 1 | 941/417 | 0.5468 | 0.9454 [0.9264, 0.9618] | 0.9721 [0.9489, 0.9887] | 0.0706 [0.0471, 0.0956] | 0.9760 [0.9597, 0.9887] | 0.9105 | 0.0420 / 0.9353 |
| source_holdout:holdout_mishra_soni_smsphishing / byte_nokd | 2 | 941/417 | 0.2235 | 0.9325 [0.9122, 0.9512] | 0.9775 [0.9649, 0.9866] | 0.0649 [0.0424, 0.0865] | 0.9448 [0.9172, 0.9677] | 0.5464 | 0.0382 / 0.9185 |
| source_holdout:holdout_mishra_soni_smsphishing / byte_nokd | 3 | 941/417 | 0.3592 | 0.9488 [0.9293, 0.9649] | 0.9830 [0.9710, 0.9918] | 0.0668 [0.0433, 0.0922] | 0.9784 [0.9609, 0.9907] | 0.8174 | 0.0420 / 0.9281 |
| source_holdout:holdout_mishra_soni_smsphishing / multilingual_kd | 1 | 941/417 | 0.1566 | 0.9567 [0.9406, 0.9718] | 0.9860 [0.9726, 0.9949] | 0.0553 [0.0345, 0.0786] | 0.9808 [0.9667, 0.9929] | 0.7771 | 0.0248 / 0.9113 |
| source_holdout:holdout_mishra_soni_smsphishing / multilingual_kd | 2 | 941/417 | 0.1283 | 0.9434 [0.9245, 0.9605] | 0.9838 [0.9741, 0.9915] | 0.0592 [0.0371, 0.0826] | 0.9592 [0.9373, 0.9776] | 0.7056 | 0.0401 / 0.9185 |
| source_holdout:holdout_mishra_soni_smsphishing / multilingual_kd | 3 | 941/417 | 0.6125 | 0.9550 [0.9366, 0.9715] | 0.9853 [0.9748, 0.9929] | 0.0458 [0.0267, 0.0653] | 0.9664 [0.9390, 0.9869] | 0.9243 | 0.0324 / 0.9209 |
| source_holdout:holdout_mishra_soni_smsphishing / multilingual_nokd | 1 | 941/417 | 0.5559 | 0.9502 [0.9324, 0.9674] | 0.9816 [0.9692, 0.9911] | 0.0496 [0.0304, 0.0716] | 0.9616 [0.9370, 0.9814] | 0.8229 | 0.0286 / 0.8897 |
| source_holdout:holdout_mishra_soni_smsphishing / multilingual_nokd | 2 | 941/417 | 0.1958 | 0.9413 [0.9209, 0.9608] | 0.9822 [0.9721, 0.9905] | 0.0477 [0.0286, 0.0692] | 0.9424 [0.9106, 0.9683] | 0.5637 | 0.0382 / 0.8993 |
| source_holdout:holdout_mishra_soni_smsphishing / multilingual_nokd | 3 | 941/417 | 0.6424 | 0.9458 [0.9259, 0.9627] | 0.9858 [0.9767, 0.9926] | 0.0401 [0.0231, 0.0577] | 0.9424 [0.9111, 0.9685] | 0.7891 | 0.0344 / 0.9233 |
| source_holdout:holdout_scamshield / byte_nokd | 1 | 75973/29029 | 0.0478 | 0.6790 [0.6686, 0.6915] | 0.5723 [0.5565, 0.5927] | 0.2523 [0.2474, 0.2576] | 0.7238 [0.7118, 0.7360] | 0.2463 | 0.2279 / 0.6299 |
| source_holdout:holdout_scamshield / byte_nokd | 2 | 75973/29029 | 0.0120 | 0.7133 [0.7040, 0.7239] | 0.6353 [0.6162, 0.6589] | 0.2637 [0.2587, 0.2692] | 0.7908 [0.7815, 0.8004] | 0.1629 | 0.1892 / 0.5988 |
| source_holdout:holdout_scamshield / byte_nokd | 3 | 75973/29029 | 0.2971 | 0.7455 [0.7369, 0.7552] | 0.7262 [0.7028, 0.7511] | 0.2432 [0.2379, 0.2485] | 0.8280 [0.8171, 0.8373] | 0.7317 | 0.2029 / 0.6770 |
| source_holdout:holdout_scamshield / multilingual_kd | 1 | 75973/29029 | 0.6856 | 0.7859 [0.7788, 0.7943] | 0.6905 [0.6770, 0.7064] | 0.2334 [0.2282, 0.2389] | 0.8917 [0.8861, 0.8974] | 0.9342 | 0.2105 / 0.8385 |
| source_holdout:holdout_scamshield / multilingual_kd | 2 | 75973/29029 | 0.8180 | 0.7859 [0.7785, 0.7949] | 0.7481 [0.7335, 0.7644] | 0.1957 [0.1907, 0.2011] | 0.8523 [0.8450, 0.8597] | 0.9601 | 0.1787 / 0.8001 |
| source_holdout:holdout_scamshield / multilingual_kd | 3 | 75973/29029 | 0.7346 | 0.7851 [0.7777, 0.7938] | 0.7074 [0.6939, 0.7237] | 0.2240 [0.2190, 0.2293] | 0.8803 [0.8739, 0.8872] | 0.9801 | 0.1957 / 0.7942 |
| source_holdout:holdout_scamshield / multilingual_nokd | 1 | 75973/29029 | 0.5770 | 0.7633 [0.7552, 0.7727] | 0.7596 [0.7410, 0.7797] | 0.2274 [0.2224, 0.2328] | 0.8442 [0.8373, 0.8520] | 0.8570 | 0.2029 / 0.7748 |
| source_holdout:holdout_scamshield / multilingual_nokd | 2 | 75973/29029 | 0.0508 | 0.7506 [0.7420, 0.7608] | 0.7621 [0.7421, 0.7834] | 0.1921 [0.1871, 0.1975] | 0.7874 [0.7782, 0.7974] | 0.0967 | 0.1782 / 0.7396 |
| source_holdout:holdout_scamshield / multilingual_nokd | 3 | 75973/29029 | 0.2157 | 0.7581 [0.7495, 0.7675] | 0.7710 [0.7511, 0.7921] | 0.2058 [0.2010, 0.2110] | 0.8136 [0.8037, 0.8231] | 0.3575 | 0.1934 / 0.7702 |
| source_holdout:holdout_uci_sms_spam / byte_nokd | 1 | 5159/642 | 0.2232 | 0.8417 [0.8204, 0.8614] | 0.9289 [0.9096, 0.9461] | 0.0385 [0.0334, 0.0439] | 0.9237 [0.8999, 0.9452] | 0.5638 | 0.0082 / 0.7586 |
| source_holdout:holdout_uci_sms_spam / byte_nokd | 2 | 5159/642 | 0.2559 | 0.8646 [0.8424, 0.8857] | 0.9368 [0.9191, 0.9521] | 0.0221 [0.0183, 0.0264] | 0.8801 [0.8495, 0.9077] | 0.4593 | 0.0095 / 0.7788 |
| source_holdout:holdout_uci_sms_spam / byte_nokd | 3 | 5159/642 | 0.4504 | 0.8698 [0.8493, 0.8899] | 0.9325 [0.9134, 0.9509] | 0.0270 [0.0223, 0.0319] | 0.9159 [0.8936, 0.9382] | 0.8812 | 0.0093 / 0.7975 |
| source_holdout:holdout_uci_sms_spam / multilingual_kd | 1 | 5159/642 | 0.3693 | 0.8751 [0.8529, 0.8950] | 0.9400 [0.9228, 0.9554] | 0.0204 [0.0164, 0.0248] | 0.8894 [0.8615, 0.9160] | 0.6796 | 0.0115 / 0.8380 |
| source_holdout:holdout_uci_sms_spam / multilingual_kd | 2 | 5159/642 | 0.2190 | 0.8685 [0.8483, 0.8889] | 0.9449 [0.9285, 0.9593] | 0.0312 [0.0261, 0.0361] | 0.9361 [0.9130, 0.9560] | 0.8463 | 0.0091 / 0.8053 |
| source_holdout:holdout_uci_sms_spam / multilingual_kd | 3 | 5159/642 | 0.5486 | 0.8774 [0.8578, 0.8978] | 0.9431 [0.9261, 0.9577] | 0.0241 [0.0197, 0.0285] | 0.9143 [0.8904, 0.9361] | 0.8635 | 0.0113 / 0.8551 |
| source_holdout:holdout_uci_sms_spam / multilingual_nokd | 1 | 5159/642 | 0.4711 | 0.8739 [0.8517, 0.8951] | 0.8880 [0.8551, 0.9192] | 0.0190 [0.0151, 0.0230] | 0.8801 [0.8505, 0.9094] | 0.6514 | 0.0128 / 0.8318 |
| source_holdout:holdout_uci_sms_spam / multilingual_nokd | 2 | 5159/642 | 0.1351 | 0.8661 [0.8466, 0.8849] | 0.9299 [0.9088, 0.9492] | 0.0350 [0.0295, 0.0401] | 0.9517 [0.9341, 0.9683] | 0.4927 | 0.0108 / 0.8302 |
| source_holdout:holdout_uci_sms_spam / multilingual_nokd | 3 | 5159/642 | 0.5065 | 0.8783 [0.8551, 0.8992] | 0.9196 [0.8925, 0.9440] | 0.0170 [0.0133, 0.0208] | 0.8769 [0.8471, 0.9064] | 0.6720 | 0.0111 / 0.8162 |
| temporal / byte_nokd | 1 | 11029/77 | 0.9819 | 0.8857 [0.8296, 0.9342] | 0.8622 [0.7881, 0.9273] | 0.0001 [0.0000, 0.0003] | 0.8052 [0.7200, 0.8889] | 0.3387 | 0.0121 / 0.8831 |
| temporal / byte_nokd | 2 | 11029/77 | 0.9724 | 0.8759 [0.8160, 0.9300] | 0.8787 [0.8108, 0.9383] | 0.0000 [0.0000, 0.0000] | 0.7792 [0.6892, 0.8691] | 0.3736 | 0.0089 / 0.8701 |
| temporal / byte_nokd | 3 | 11029/77 | 0.9987 | 0.9014 [0.8485, 0.9474] | 0.8785 [0.8086, 0.9383] | 0.0001 [0.0000, 0.0003] | 0.8312 [0.7467, 0.9103] | 0.8868 | 0.0111 / 0.8961 |
| temporal / multilingual_kd | 1 | 11029/77 | 0.3620 | 0.8630 [0.8076, 0.9172] | 0.8539 [0.7719, 0.9208] | 0.0005 [0.0002, 0.0010] | 0.8182 [0.7397, 0.9000] | 0.0245 | 0.0033 / 0.8571 |
| temporal / multilingual_kd | 2 | 11029/77 | 0.9262 | 0.8671 [0.8060, 0.9250] | 0.8883 [0.8265, 0.9433] | 0.0004 [0.0000, 0.0007] | 0.8052 [0.7215, 0.8947] | 0.0158 | 0.0059 / 0.8831 |
| temporal / multilingual_kd | 3 | 11029/77 | 0.2277 | 0.8630 [0.8058, 0.9231] | 0.8572 [0.7797, 0.9218] | 0.0005 [0.0002, 0.0010] | 0.8182 [0.7375, 0.9025] | 0.0216 | 0.0041 / 0.8571 |
| temporal / multilingual_nokd | 1 | 11029/77 | 0.9998 | 0.8511 [0.7906, 0.9128] | 0.8670 [0.8004, 0.9281] | 0.0004 [0.0001, 0.0008] | 0.7792 [0.6923, 0.8718] | 0.9957 | 0.0046 / 0.8571 |
| temporal / multilingual_nokd | 2 | 11029/77 | 0.9723 | 0.8841 [0.8281, 0.9380] | 0.8882 [0.8294, 0.9412] | 0.0000 [0.0000, 0.0000] | 0.7922 [0.7066, 0.8832] | 0.7809 | 0.0106 / 0.9221 |
| temporal / multilingual_nokd | 3 | 11029/77 | 0.9578 | 0.8025 [0.7375, 0.8701] | 0.8734 [0.8095, 0.9317] | 0.0016 [0.0008, 0.0024] | 0.8182 [0.7375, 0.9025] | 0.8261 | 0.0072 / 0.8831 |

## External student-only endpoints, every seed

GroundLM is in-sample for rules and intervention selection. SmishX was originally independent/non-Indian-specific,
but previous feedback informed this intervention: current comparisons are in-sample intervention selection.
No thresholds fitted externally. External campaigns are approximate template/domain groups.
Both legacy binary and known-fraud labels are retained; promotions are not silently called fraud.
Rate CIs are campaign-clustered percentiles; Wilson bounds in archivedJSON are secondary IID bounds.

| Population / arm | Seed | Cal operating point | Threshold | Label endpoint | FPR events/n [95% CI] | Recall events/n [95% CI] |
|---|---|---|---|---|---|---|
| groundlm / byte_nokd | 1 | fpr_1pct | 0.9030 | label_binary | 440/1133 = 38.83% [24.03%, 51.57%] | 70/91 = 76.92% [60.00%, 89.41%] |
| groundlm / byte_nokd | 1 | fpr_1pct | 0.9030 | label_scam | 440/1133 = 38.83% [24.03%, 51.57%] | 70/91 = 76.92% [60.00%, 89.41%] |
| groundlm / byte_nokd | 1 | max_f1 | 0.4754 | label_binary | 480/1133 = 42.37% [27.32%, 55.27%] | 82/91 = 90.11% [80.25%, 97.09%] |
| groundlm / byte_nokd | 1 | max_f1 | 0.4754 | label_scam | 480/1133 = 42.37% [27.32%, 55.27%] | 82/91 = 90.11% [80.25%, 97.09%] |
| smishx / byte_nokd | 1 | fpr_1pct | 0.9030 | label_binary | 48/606 = 7.92% [5.80%, 10.18%] | 342/501 = 68.26% [61.00%, 75.58%] |
| smishx / byte_nokd | 1 | fpr_1pct | 0.9030 | label_scam | 269/881 = 30.53% [25.80%, 35.17%] | 121/226 = 53.54% [44.39%, 65.46%] |
| smishx / byte_nokd | 1 | max_f1 | 0.4754 | label_binary | 69/606 = 11.39% [8.96%, 14.05%] | 426/501 = 85.03% [79.23%, 90.00%] |
| smishx / byte_nokd | 1 | max_f1 | 0.4754 | label_scam | 325/881 = 36.89% [31.95%, 41.59%] | 170/226 = 75.22% [65.72%, 84.98%] |
| groundlm / byte_nokd | 2 | fpr_1pct | 0.6480 | label_binary | 304/1133 = 26.83% [15.82%, 37.38%] | 70/91 = 76.92% [61.42%, 87.00%] |
| groundlm / byte_nokd | 2 | fpr_1pct | 0.6480 | label_scam | 304/1133 = 26.83% [15.82%, 37.38%] | 70/91 = 76.92% [61.42%, 87.00%] |
| groundlm / byte_nokd | 2 | max_f1 | 0.2274 | label_binary | 421/1133 = 37.16% [23.65%, 49.51%] | 80/91 = 87.91% [73.33%, 94.69%] |
| groundlm / byte_nokd | 2 | max_f1 | 0.2274 | label_scam | 421/1133 = 37.16% [23.65%, 49.51%] | 80/91 = 87.91% [73.33%, 94.69%] |
| smishx / byte_nokd | 2 | fpr_1pct | 0.6480 | label_binary | 42/606 = 6.93% [4.97%, 9.08%] | 306/501 = 61.08% [53.95%, 68.34%] |
| smishx / byte_nokd | 2 | fpr_1pct | 0.6480 | label_scam | 242/881 = 27.47% [23.04%, 31.92%] | 106/226 = 46.90% [37.50%, 58.06%] |
| smishx / byte_nokd | 2 | max_f1 | 0.2274 | label_binary | 62/606 = 10.23% [7.95%, 12.79%] | 403/501 = 80.44% [74.72%, 85.80%] |
| smishx / byte_nokd | 2 | max_f1 | 0.2274 | label_scam | 309/881 = 35.07% [30.30%, 39.74%] | 156/226 = 69.03% [59.91%, 78.46%] |
| groundlm / byte_nokd | 3 | fpr_1pct | 0.9557 | label_binary | 373/1133 = 32.92% [21.32%, 43.37%] | 73/91 = 80.22% [63.74%, 91.97%] |
| groundlm / byte_nokd | 3 | fpr_1pct | 0.9557 | label_scam | 373/1133 = 32.92% [21.32%, 43.37%] | 73/91 = 80.22% [63.74%, 91.97%] |
| groundlm / byte_nokd | 3 | max_f1 | 0.5495 | label_binary | 488/1133 = 43.07% [28.30%, 56.16%] | 83/91 = 91.21% [79.66%, 96.83%] |
| groundlm / byte_nokd | 3 | max_f1 | 0.5495 | label_scam | 488/1133 = 43.07% [28.30%, 56.16%] | 83/91 = 91.21% [79.66%, 96.83%] |
| smishx / byte_nokd | 3 | fpr_1pct | 0.9557 | label_binary | 36/606 = 5.94% [4.14%, 7.92%] | 277/501 = 55.29% [48.39%, 61.81%] |
| smishx / byte_nokd | 3 | fpr_1pct | 0.9557 | label_scam | 218/881 = 24.74% [20.33%, 29.15%] | 95/226 = 42.04% [32.91%, 52.02%] |
| smishx / byte_nokd | 3 | max_f1 | 0.5495 | label_binary | 66/606 = 10.89% [8.39%, 13.49%] | 392/501 = 78.24% [71.63%, 84.31%] |
| smishx / byte_nokd | 3 | max_f1 | 0.5495 | label_scam | 310/881 = 35.19% [30.26%, 39.85%] | 148/226 = 65.49% [55.79%, 76.38%] |
| groundlm / multilingual_kd | 1 | fpr_1pct | 0.6082 | label_binary | 444/1133 = 39.19% [24.26%, 52.47%] | 49/91 = 53.85% [25.00%, 75.00%] |
| groundlm / multilingual_kd | 1 | fpr_1pct | 0.6082 | label_scam | 444/1133 = 39.19% [24.26%, 52.47%] | 49/91 = 53.85% [25.00%, 75.00%] |
| groundlm / multilingual_kd | 1 | max_f1 | 0.1092 | label_binary | 491/1133 = 43.34% [28.84%, 56.34%] | 65/91 = 71.43% [50.00%, 86.72%] |
| groundlm / multilingual_kd | 1 | max_f1 | 0.1092 | label_scam | 491/1133 = 43.34% [28.84%, 56.34%] | 65/91 = 71.43% [50.00%, 86.72%] |
| smishx / multilingual_kd | 1 | fpr_1pct | 0.6082 | label_binary | 47/606 = 7.76% [5.63%, 9.90%] | 272/501 = 54.29% [46.68%, 61.91%] |
| smishx / multilingual_kd | 1 | fpr_1pct | 0.6082 | label_scam | 221/881 = 25.09% [20.67%, 29.48%] | 98/226 = 43.36% [34.54%, 55.19%] |
| smishx / multilingual_kd | 1 | max_f1 | 0.1092 | label_binary | 63/606 = 10.40% [7.96%, 12.79%] | 387/501 = 77.25% [71.96%, 82.94%] |
| smishx / multilingual_kd | 1 | max_f1 | 0.1092 | label_scam | 293/881 = 33.26% [28.96%, 37.74%] | 157/226 = 69.47% [61.46%, 78.03%] |
| groundlm / multilingual_kd | 2 | fpr_1pct | 0.9077 | label_binary | 395/1133 = 34.86% [20.77%, 47.30%] | 46/91 = 50.55% [26.21%, 70.00%] |
| groundlm / multilingual_kd | 2 | fpr_1pct | 0.9077 | label_scam | 395/1133 = 34.86% [20.77%, 47.30%] | 46/91 = 50.55% [26.21%, 70.00%] |
| groundlm / multilingual_kd | 2 | max_f1 | 0.4759 | label_binary | 462/1133 = 40.78% [25.85%, 53.94%] | 72/91 = 79.12% [60.65%, 88.41%] |
| groundlm / multilingual_kd | 2 | max_f1 | 0.4759 | label_scam | 462/1133 = 40.78% [25.85%, 53.94%] | 72/91 = 79.12% [60.65%, 88.41%] |
| smishx / multilingual_kd | 2 | fpr_1pct | 0.9077 | label_binary | 40/606 = 6.60% [4.64%, 8.73%] | 268/501 = 53.49% [46.24%, 61.07%] |
| smishx / multilingual_kd | 2 | fpr_1pct | 0.9077 | label_scam | 207/881 = 23.50% [19.31%, 28.02%] | 101/226 = 44.69% [34.93%, 56.05%] |
| smishx / multilingual_kd | 2 | max_f1 | 0.4759 | label_binary | 56/606 = 9.24% [6.97%, 11.64%] | 400/501 = 79.84% [74.06%, 85.87%] |
| smishx / multilingual_kd | 2 | max_f1 | 0.4759 | label_scam | 290/881 = 32.92% [28.38%, 37.61%] | 166/226 = 73.45% [64.61%, 82.69%] |
| groundlm / multilingual_kd | 3 | fpr_1pct | 0.7244 | label_binary | 434/1133 = 38.31% [23.45%, 51.63%] | 59/91 = 64.84% [41.89%, 81.51%] |
| groundlm / multilingual_kd | 3 | fpr_1pct | 0.7244 | label_scam | 434/1133 = 38.31% [23.45%, 51.63%] | 59/91 = 64.84% [41.89%, 81.51%] |
| groundlm / multilingual_kd | 3 | max_f1 | 0.1109 | label_binary | 479/1133 = 42.28% [27.59%, 55.35%] | 78/91 = 85.71% [72.50%, 93.01%] |
| groundlm / multilingual_kd | 3 | max_f1 | 0.1109 | label_scam | 479/1133 = 42.28% [27.59%, 55.35%] | 78/91 = 85.71% [72.50%, 93.01%] |
| smishx / multilingual_kd | 3 | fpr_1pct | 0.7244 | label_binary | 51/606 = 8.42% [6.30%, 10.71%] | 310/501 = 61.88% [54.53%, 69.80%] |
| smishx / multilingual_kd | 3 | fpr_1pct | 0.7244 | label_scam | 256/881 = 29.06% [24.74%, 33.70%] | 105/226 = 46.46% [36.94%, 57.89%] |
| smishx / multilingual_kd | 3 | max_f1 | 0.1109 | label_binary | 65/606 = 10.73% [8.28%, 13.37%] | 430/501 = 85.83% [80.85%, 90.32%] |
| smishx / multilingual_kd | 3 | max_f1 | 0.1109 | label_scam | 308/881 = 34.96% [30.28%, 39.63%] | 187/226 = 82.74% [75.00%, 88.84%] |
| groundlm / multilingual_nokd | 1 | fpr_1pct | 0.6450 | label_binary | 427/1133 = 37.69% [22.22%, 51.49%] | 47/91 = 51.65% [27.02%, 72.52%] |
| groundlm / multilingual_nokd | 1 | fpr_1pct | 0.6450 | label_scam | 427/1133 = 37.69% [22.22%, 51.49%] | 47/91 = 51.65% [27.02%, 72.52%] |
| groundlm / multilingual_nokd | 1 | max_f1 | 0.2533 | label_binary | 484/1133 = 42.72% [28.52%, 55.69%] | 64/91 = 70.33% [49.41%, 85.50%] |
| groundlm / multilingual_nokd | 1 | max_f1 | 0.2533 | label_scam | 484/1133 = 42.72% [28.52%, 55.69%] | 64/91 = 70.33% [49.41%, 85.50%] |
| smishx / multilingual_nokd | 1 | fpr_1pct | 0.6450 | label_binary | 43/606 = 7.10% [5.12%, 9.20%] | 260/501 = 51.90% [44.37%, 59.85%] |
| smishx / multilingual_nokd | 1 | fpr_1pct | 0.6450 | label_scam | 214/881 = 24.29% [20.18%, 28.53%] | 89/226 = 39.38% [30.74%, 50.27%] |
| smishx / multilingual_nokd | 1 | max_f1 | 0.2533 | label_binary | 56/606 = 9.24% [6.97%, 11.68%] | 377/501 = 75.25% [68.64%, 81.98%] |
| smishx / multilingual_nokd | 1 | max_f1 | 0.2533 | label_scam | 275/881 = 31.21% [26.78%, 35.71%] | 158/226 = 69.91% [59.89%, 80.79%] |
| groundlm / multilingual_nokd | 2 | fpr_1pct | 0.8759 | label_binary | 361/1133 = 31.86% [18.78%, 43.22%] | 46/91 = 50.55% [25.00%, 70.84%] |
| groundlm / multilingual_nokd | 2 | fpr_1pct | 0.8759 | label_scam | 361/1133 = 31.86% [18.78%, 43.22%] | 46/91 = 50.55% [25.00%, 70.84%] |
| groundlm / multilingual_nokd | 2 | max_f1 | 0.6752 | label_binary | 434/1133 = 38.31% [22.92%, 51.73%] | 65/91 = 71.43% [48.33%, 83.84%] |
| groundlm / multilingual_nokd | 2 | max_f1 | 0.6752 | label_scam | 434/1133 = 38.31% [22.92%, 51.73%] | 65/91 = 71.43% [48.33%, 83.84%] |
| smishx / multilingual_nokd | 2 | fpr_1pct | 0.8759 | label_binary | 43/606 = 7.10% [5.13%, 9.29%] | 290/501 = 57.88% [50.35%, 65.96%] |
| smishx / multilingual_nokd | 2 | fpr_1pct | 0.8759 | label_scam | 233/881 = 26.45% [21.85%, 30.96%] | 100/226 = 44.25% [35.08%, 55.05%] |
| smishx / multilingual_nokd | 2 | max_f1 | 0.6752 | label_binary | 53/606 = 8.75% [6.60%, 11.17%] | 368/501 = 73.45% [67.31%, 79.33%] |
| smishx / multilingual_nokd | 2 | max_f1 | 0.6752 | label_scam | 272/881 = 30.87% [26.32%, 35.32%] | 149/226 = 65.93% [57.59%, 74.62%] |
| groundlm / multilingual_nokd | 3 | fpr_1pct | 0.7394 | label_binary | 417/1133 = 36.80% [22.22%, 49.34%] | 59/91 = 64.84% [41.66%, 82.14%] |
| groundlm / multilingual_nokd | 3 | fpr_1pct | 0.7394 | label_scam | 417/1133 = 36.80% [22.22%, 49.34%] | 59/91 = 64.84% [41.66%, 82.14%] |
| groundlm / multilingual_nokd | 3 | max_f1 | 0.3742 | label_binary | 486/1133 = 42.89% [28.44%, 55.91%] | 75/91 = 82.42% [67.80%, 92.57%] |
| groundlm / multilingual_nokd | 3 | max_f1 | 0.3742 | label_scam | 486/1133 = 42.89% [28.44%, 55.91%] | 75/91 = 82.42% [67.80%, 92.57%] |
| smishx / multilingual_nokd | 3 | fpr_1pct | 0.7394 | label_binary | 48/606 = 7.92% [5.79%, 10.08%] | 317/501 = 63.27% [56.09%, 70.85%] |
| smishx / multilingual_nokd | 3 | fpr_1pct | 0.7394 | label_scam | 256/881 = 29.06% [24.53%, 33.55%] | 109/226 = 48.23% [39.57%, 59.33%] |
| smishx / multilingual_nokd | 3 | max_f1 | 0.3742 | label_binary | 63/606 = 10.40% [7.97%, 12.97%] | 423/501 = 84.43% [79.04%, 89.94%] |
| smishx / multilingual_nokd | 3 | max_f1 | 0.3742 | label_scam | 312/881 = 35.41% [30.70%, 40.15%] | 174/226 = 76.99% [68.88%, 84.69%] |

## Artifact and membership register

Classifier tensors are4,600,490bytes in every hybrid candidate, below8MiB; disk checkpoints include
training-ID provenance and are larger. The saved tokenizer is34,393,406bytes, excluded from the declared
classifier-only cap, as are vocabulary/PCA/audit files. No complete on-device footprint/latency claim.

| Fold | Training rows | Existing synthetic train rows | Vocabulary tokens | Vocabulary SHA256 |
|---|---:|---:|---:|---|
| campaign | 76591 | 3720 | 16382 | dcee90009754f745530957251808fe62b114f0a0c8bda34a51f8eca3d23a0fc7 |
| lang_shift | 58605 | 1357 | 16382 | 22e36d1d0277d80eb62eb3ecd97d06d2dd3ca558d21f021db98bda60a52586ed |
| source_holdout:holdout_mishra_soni_smsphishing | 96867 | 3720 | 16382 | d57201a2c3efb92aae47ba0fe47e803c171568da441610c64fe332060eb73a33 |
| source_holdout:holdout_scamshield | 24391 | 0 | 16382 | 5a08bbb6da182f5d52e4d4cb290234b912a45ac4f73e036cd86665b8be92f3c6 |
| source_holdout:holdout_uci_sms_spam | 92637 | 3720 | 16382 | da6223a0dcdb397e07be77f4c0b8a862cebfa109f246f58c4d7c7e98145c5d13 |
| temporal | 34996 | 0 | 16382 | 027981af697ffac3428ea87a862a48aee07d82bf43fba60ef3780910969a6114 |

Raw metric SHA256: `40f3fe40b4cdcdae1ebb69b2929a4a25b009b85bfd6b67191a64127a2f7588e0`.
Inference source SHA256: `6f25ddecab2e00544a7e7a07c151af798548cc07612933f57c1c0dda27541321`.

All54selected checkpoints,6vocabularies/frozen-ID manifests, tokenizer/modelcard and source/aggregateJSON
are hash-verified for private archival; raw SMS/row-score parquets excluded. HF publish pending.
Private archive maadhavvh/rakshak-multilingual-m3-20261102 version1: owner maadhavvh,
isPrivate=true verified,350,414,915bytes; model manifestSHA
`a1d94a2659dc98b9558c72998f8cf998b5aca883bdbfca7d5353ed0798daf193`. One fullKaggle suite877passed0skipped,
157warnings150.63s; ruffclean. HF publish pending; no shipping approval or diagnostic-arm promotion.

## Failures retained and decision

M1 expected a safetensors file absent from the pinned revision; M2 source-ZIP1970timestamps failed.
Both stopped before SMS access/training and remain UNMEASURABLE, with focused loader/archive regressions.
M3 used the same source/recipe/gates. Original tiny-CUDA unit nondeterminism was fixed with unit-only
cuDNN determinism, retaining bitwise parity assertion and unchanged production numerics.
The original M3 event process was lost; owner authorized one replacement replay, which recovered the
matching completion. One successful fetch followed. No job restart or Kaggle status polling.

Keep KDhybrid as a research candidate under its fixed gate. Do not claim a teacher comparison, independent
Indian improvement, reduced alert FPR or Arm readiness. Next: complete review/private archive and measure
a prospective deployment artifact separately; P1 safety and independent Indian SMS remain unresolved.
