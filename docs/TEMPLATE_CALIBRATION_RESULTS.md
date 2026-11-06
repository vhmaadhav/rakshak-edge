# Last P1 template-family calibration: KILL — 2026-11-02

**All 90 family fits used the exact global fallback; no family-specific map was fitted.**
Every aggregate endpoint, including coverage, equals its matched global control. This is no improvement.
Nine external safety/recall checks fail; every internal recall guard passes by identity.
No candidate successor, deployment wiring, family pruning or additional P1 tuning.

Predeclaration: [TEMPLATE_CALIBRATION.md](TEMPLATE_CALIBRATION.md). Three saved original
clean no-KD seeds/all six folds, no new training or teacher/synthetic extras. Four frozen
cal-campaign stages; family probability AND threshold stages each require 200 negative
and 20 positive examples. All 90 fail those fixed minimum counts; they were not relaxed.

**In-sample intervention selection.** GroundLM/ScamShield informed rule design;
GroundLM/SmishX and prior frozen-fold aggregate feedback informed this experiment.
SmishX is train-independent/non-Indian-specific, not untouched for intervention choice.
Independent consented, scrubbed Indian inbox validation remains OPEN. No held-out gain claim.

## Identity and unchanged gates

Named cohort **template-cal-c1**, newly fitted (not previously shipped). Same experts
within every matched pair; original clean-KD control checkpoint hashes/provenance
and fullcal GPU-to-CPU probability parity<=1e-5 checked. All 18 states/weights/source
manifest frozen before any new test/external decisions. This cohort does **not**
replace frozen v1. Its immutable parent remains
`c310d3ccbc1c1e50ce9436f217c66ee9934f778985bfef487a034cc3070b79fa`.
PyTorch measurement references only; Arm/Android portability UNVERIFIED, ONNX Phase2.

GO required every seed: GroundSCAMFPR<=1%, alertFPR<=10%, neither increase vs matched
global control; SmishX SCAMrecall>=matchedstudent cal1%-FPR recall; no frozen-fold
SCAMrecall loss>controlCIhalf for legacy and known fraud where positive. UCI fraudNA,
legacy mandatory. Known failures KILL; no missing result. Alert recall cannot rescue SCAM guards.

Job `family_calibration-c1-20261102`v1, ok1534.2s, one original event/fetch.
Tree `387fa7f06c56d19df89a42d65421e6b8c8e7d9ff0a12db9efcfa2461fdd25033`.
Manifest `d5b92921f963acab17d728650269829efe01648850055c94ebfeb04fe6e7cfa9`.
Metrics `819ad822565d849d9831b63995f814665a4c09142e98e66a7a5fd8e79c091905`.

Intervals below are campaign-clustered 95% percentiles, 1,000 seed-13 resamples,
conditional on fixed training seeds; external groups are approximate template/domain
clusters. Point caps are audit gates, not population guarantees. FPR includes ALL
eligible endpoint negatives, including UNCERTAIN. Coverage/UNCERTAIN denominators
are endpoint-known rows; legacy table includes all eligible frozen rows. Fraud unknown
labels are excluded and counts shown. Original temporal three train/test campaign overlaps
and 77 legacy positives are retained. No frozen split or external row was changed.

## External known-fraud headline (control and family identical)

| Population | Seed | SCAM FPR [CI]% | Alert FPR [CI]% | SCAM recall [CI]% | Alert recall [CI]% | Student cal1% recall [CI]% |
|---|---|---|---|---|---|---|
| groundlm | 1 | 440/1133=38.83% [24.29,51.69] | 463/1133=40.86% [26.03,54.18] | 40/91=43.96% [24.99,64.29] | 69/91=75.82% [63.89,83.59] | 67/91=73.63% [59.32,86.00] |
| groundlm | 2 | 338/1133=29.83% [17.07,41.28] | 367/1133=32.39% [18.88,44.43] | 28/91=30.77% [17.31,48.94] | 61/91=67.03% [47.22,80.56] | 57/91=62.64% [44.28,78.99] |
| groundlm | 3 | 348/1133=30.71% [18.36,41.42] | 356/1133=31.42% [19.05,42.14] | 66/91=72.53% [55.93,86.52] | 66/91=72.53% [55.93,86.52] | 66/91=72.53% [55.93,86.52] |
| smishx | 1 | 26/881=2.95% [1.56,4.69] | 176/881=19.98% [15.75,24.31] | 20/226=8.85% [5.00,15.03] | 45/226=19.91% [14.47,27.70] | 118/226=52.21% [42.67,63.70] |
| smishx | 2 | 29/881=3.29% [1.63,5.37] | 184/881=20.89% [16.63,24.84] | 29/226=12.83% [7.22,20.48] | 80/226=35.40% [27.01,45.11] | 116/226=51.33% [41.39,61.58] |
| smishx | 3 | 255/881=28.94% [24.23,33.30] | 258/881=29.28% [24.59,33.59] | 106/226=46.90% [37.12,58.85] | 112/226=49.56% [39.74,61.39] | 122/226=53.98% [43.46,66.84] |

Failed checks:

- `1/groundlm_scam_le_1pct`
- `1/groundlm_at_least_uncertain_le_10pct`
- `1/smishx_recall_ge_student`
- `2/groundlm_scam_le_1pct`
- `2/groundlm_at_least_uncertain_le_10pct`
- `2/smishx_recall_ge_student`
- `3/groundlm_scam_le_1pct`
- `3/groundlm_at_least_uncertain_le_10pct`
- `3/smishx_recall_ge_student`

## Every frozen fold: label_binary, both roles / all seeds

| Split | Seed | Role | SCAM FPR | Alert FPR | SCAM recall | Alert recall | Coverage | UNCERTAIN | Unknown excluded |
|---|---|---|---|---|---|---|---|---|---|
| campaign | 1 | baseline | 77/10952=0.70% [0.50,1.01] | 131/10952=1.20% [0.96,1.51] | 9481/11193=84.70% [84.05,85.37] | 10352/11193=92.49% [92.00,92.99] | 21220/22145=95.82% [95.54,96.09] | 925/22145=4.18% [3.91,4.46] | 0 |
| campaign | 1 | family | 77/10952=0.70% [0.50,1.01] | 131/10952=1.20% [0.96,1.51] | 9481/11193=84.70% [84.05,85.37] | 10352/11193=92.49% [92.00,92.99] | 21220/22145=95.82% [95.54,96.09] | 925/22145=4.18% [3.91,4.46] | 0 |
| campaign | 2 | baseline | 100/10952=0.91% [0.69,1.21] | 132/10952=1.21% [0.96,1.51] | 9629/11193=86.03% [85.40,86.69] | 10623/11193=94.91% [94.51,95.34] | 21119/22145=95.37% [95.07,95.65] | 1026/22145=4.63% [4.35,4.93] | 0 |
| campaign | 2 | family | 100/10952=0.91% [0.69,1.21] | 132/10952=1.21% [0.96,1.51] | 9629/11193=86.03% [85.40,86.69] | 10623/11193=94.91% [94.51,95.34] | 21119/22145=95.37% [95.07,95.65] | 1026/22145=4.63% [4.35,4.93] | 0 |
| campaign | 3 | baseline | 111/10952=1.01% [0.79,1.32] | 115/10952=1.05% [0.81,1.34] | 10623/11193=94.91% [94.53,95.29] | 10634/11193=95.01% [94.63,95.40] | 22130/22145=99.93% [99.90,99.96] | 15/22145=0.07% [0.04,0.10] | 0 |
| campaign | 3 | family | 111/10952=1.01% [0.79,1.32] | 115/10952=1.05% [0.81,1.34] | 10623/11193=94.91% [94.53,95.29] | 10634/11193=95.01% [94.63,95.40] | 22130/22145=99.93% [99.90,99.96] | 15/22145=0.07% [0.04,0.10] | 0 |
| lang_shift | 1 | baseline | 524/1320=39.70% [36.81,42.42] | 570/1320=43.18% [40.45,46.10] | 254/357=71.15% [62.97,79.87] | 257/357=71.99% [63.68,81.01] | 1628/1677=97.08% [96.23,97.81] | 49/1677=2.92% [2.19,3.77] | 0 |
| lang_shift | 1 | family | 524/1320=39.70% [36.81,42.42] | 570/1320=43.18% [40.45,46.10] | 254/357=71.15% [62.97,79.87] | 257/357=71.99% [63.68,81.01] | 1628/1677=97.08% [96.23,97.81] | 49/1677=2.92% [2.19,3.77] | 0 |
| lang_shift | 2 | baseline | 38/1320=2.88% [1.54,4.90] | 164/1320=12.42% [10.33,14.78] | 226/357=63.31% [53.89,72.64] | 256/357=71.71% [62.84,80.48] | 1521/1677=90.70% [89.33,92.05] | 156/1677=9.30% [7.95,10.67] | 0 |
| lang_shift | 2 | family | 38/1320=2.88% [1.54,4.90] | 164/1320=12.42% [10.33,14.78] | 226/357=63.31% [53.89,72.64] | 256/357=71.71% [62.84,80.48] | 1521/1677=90.70% [89.33,92.05] | 156/1677=9.30% [7.95,10.67] | 0 |
| lang_shift | 3 | baseline | 64/1320=4.85% [3.74,6.01] | 302/1320=22.88% [20.61,25.23] | 232/357=64.99% [55.50,74.78] | 249/357=69.75% [60.70,78.90] | 1422/1677=84.79% [82.93,86.43] | 255/1677=15.21% [13.57,17.07] | 0 |
| lang_shift | 3 | family | 64/1320=4.85% [3.74,6.01] | 302/1320=22.88% [20.61,25.23] | 232/357=64.99% [55.50,74.78] | 249/357=69.75% [60.70,78.90] | 1422/1677=84.79% [82.93,86.43] | 255/1677=15.21% [13.57,17.07] | 0 |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | baseline | 15/524=2.86% [1.34,4.57] | 15/524=2.86% [1.34,4.57] | 389/417=93.29% [90.35,95.83] | 389/417=93.29% [90.35,95.83] | 941/941=100.00% [100.00,100.00] | 0/941=0.00% [0.00,0.00] | 0 |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | family | 15/524=2.86% [1.34,4.57] | 15/524=2.86% [1.34,4.57] | 389/417=93.29% [90.35,95.83] | 389/417=93.29% [90.35,95.83] | 941/941=100.00% [100.00,100.00] | 0/941=0.00% [0.00,0.00] | 0 |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | baseline | 28/524=5.34% [3.41,7.45] | 29/524=5.53% [3.61,7.68] | 381/417=91.37% [88.12,94.33] | 384/417=92.09% [88.94,94.81] | 937/941=99.57% [99.13,99.90] | 4/941=0.43% [0.10,0.87] | 0 |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | family | 28/524=5.34% [3.41,7.45] | 29/524=5.53% [3.61,7.68] | 381/417=91.37% [88.12,94.33] | 384/417=92.09% [88.94,94.81] | 937/941=99.57% [99.13,99.90] | 4/941=0.43% [0.10,0.87] | 0 |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | baseline | 26/524=4.96% [2.86,7.13] | 26/524=4.96% [2.86,7.13] | 391/417=93.76% [91.07,96.08] | 394/417=94.48% [92.06,96.59] | 938/941=99.68% [99.27,100.00] | 3/941=0.32% [0.00,0.73] | 0 |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | family | 26/524=4.96% [2.86,7.13] | 26/524=4.96% [2.86,7.13] | 391/417=93.76% [91.07,96.08] | 394/417=94.48% [92.06,96.59] | 938/941=99.68% [99.27,100.00] | 3/941=0.32% [0.00,0.73] | 0 |
| source_holdout:holdout_scamshield | 1 | baseline | 6324/46944=13.47% [13.12,13.84] | 7909/46944=16.85% [16.45,17.28] | 4626/29029=15.94% [15.14,16.76] | 7025/29029=24.20% [23.36,25.11] | 71989/75973=94.76% [94.53,94.98] | 3984/75973=5.24% [5.02,5.47] | 0 |
| source_holdout:holdout_scamshield | 1 | family | 6324/46944=13.47% [13.12,13.84] | 7909/46944=16.85% [16.45,17.28] | 4626/29029=15.94% [15.14,16.76] | 7025/29029=24.20% [23.36,25.11] | 71989/75973=94.76% [94.53,94.98] | 3984/75973=5.24% [5.02,5.47] | 0 |
| source_holdout:holdout_scamshield | 2 | baseline | 5024/46944=10.70% [10.39,11.03] | 6984/46944=14.88% [14.47,15.31] | 3612/29029=12.44% [11.68,13.30] | 8785/29029=30.26% [28.98,31.87] | 68840/75973=90.61% [89.99,91.08] | 7133/75973=9.39% [8.92,10.01] | 0 |
| source_holdout:holdout_scamshield | 2 | family | 5024/46944=10.70% [10.39,11.03] | 6984/46944=14.88% [14.47,15.31] | 3612/29029=12.44% [11.68,13.30] | 8785/29029=30.26% [28.98,31.87] | 68840/75973=90.61% [89.99,91.08] | 7133/75973=9.39% [8.92,10.01] | 0 |
| source_holdout:holdout_scamshield | 3 | baseline | 7659/46944=16.32% [15.90,16.83] | 7746/46944=16.50% [16.07,17.02] | 10727/29029=36.95% [35.25,39.01] | 12081/29029=41.62% [39.92,43.59] | 74532/75973=98.10% [97.94,98.26] | 1441/75973=1.90% [1.74,2.06] | 0 |
| source_holdout:holdout_scamshield | 3 | family | 7659/46944=16.32% [15.90,16.83] | 7746/46944=16.50% [16.07,17.02] | 10727/29029=36.95% [35.25,39.01] | 12081/29029=41.62% [39.92,43.59] | 74532/75973=98.10% [97.94,98.26] | 1441/75973=1.90% [1.74,2.06] | 0 |
| source_holdout:holdout_uci_sms_spam | 1 | baseline | 24/4517=0.53% [0.33,0.73] | 32/4517=0.71% [0.49,0.97] | 404/642=62.93% [58.23,67.42] | 448/642=69.78% [65.52,73.82] | 5107/5159=98.99% [98.71,99.26] | 52/5159=1.01% [0.74,1.29] | 0 |
| source_holdout:holdout_uci_sms_spam | 1 | family | 24/4517=0.53% [0.33,0.73] | 32/4517=0.71% [0.49,0.97] | 404/642=62.93% [58.23,67.42] | 448/642=69.78% [65.52,73.82] | 5107/5159=98.99% [98.71,99.26] | 52/5159=1.01% [0.74,1.29] | 0 |
| source_holdout:holdout_uci_sms_spam | 2 | baseline | 37/4517=0.82% [0.55,1.09] | 42/4517=0.93% [0.64,1.20] | 478/642=74.45% [70.35,78.51] | 488/642=76.01% [72.21,79.91] | 5144/5159=99.71% [99.55,99.85] | 15/5159=0.29% [0.15,0.45] | 0 |
| source_holdout:holdout_uci_sms_spam | 2 | family | 37/4517=0.82% [0.55,1.09] | 42/4517=0.93% [0.64,1.20] | 478/642=74.45% [70.35,78.51] | 488/642=76.01% [72.21,79.91] | 5144/5159=99.71% [99.55,99.85] | 15/5159=0.29% [0.15,0.45] | 0 |
| source_holdout:holdout_uci_sms_spam | 3 | baseline | 39/4517=0.86% [0.60,1.13] | 39/4517=0.86% [0.60,1.13] | 500/642=77.88% [73.92,81.65] | 500/642=77.88% [73.92,81.65] | 5159/5159=100.00% [100.00,100.00] | 0/5159=0.00% [0.00,0.00] | 0 |
| source_holdout:holdout_uci_sms_spam | 3 | family | 39/4517=0.86% [0.60,1.13] | 39/4517=0.86% [0.60,1.13] | 500/642=77.88% [73.92,81.65] | 500/642=77.88% [73.92,81.65] | 5159/5159=100.00% [100.00,100.00] | 0/5159=0.00% [0.00,0.00] | 0 |
| temporal | 1 | baseline | 76/10952=0.69% [0.55,0.86] | 3930/10952=35.88% [34.93,36.74] | 68/77=88.31% [80.00,94.67] | 76/77=98.70% [95.89,100.00] | 7167/11029=64.98% [64.03,65.85] | 3862/11029=35.02% [34.15,35.97] | 0 |
| temporal | 1 | family | 76/10952=0.69% [0.55,0.86] | 3930/10952=35.88% [34.93,36.74] | 68/77=88.31% [80.00,94.67] | 76/77=98.70% [95.89,100.00] | 7167/11029=64.98% [64.03,65.85] | 3862/11029=35.02% [34.15,35.97] | 0 |
| temporal | 2 | baseline | 72/10952=0.66% [0.50,0.81] | 4549/10952=41.54% [40.55,42.48] | 66/77=85.71% [76.92,92.50] | 77/77=100.00% [100.00,100.00] | 6541/11029=59.31% [58.37,60.22] | 4488/11029=40.69% [39.78,41.63] | 0 |
| temporal | 2 | family | 72/10952=0.66% [0.50,0.81] | 4549/10952=41.54% [40.55,42.48] | 66/77=85.71% [76.92,92.50] | 77/77=100.00% [100.00,100.00] | 6541/11029=59.31% [58.37,60.22] | 4488/11029=40.69% [39.78,41.63] | 0 |
| temporal | 3 | baseline | 98/10952=0.89% [0.73,1.08] | 3059/10952=27.93% [27.08,28.81] | 67/77=87.01% [78.67,93.75] | 76/77=98.70% [95.95,100.00] | 8059/11029=73.07% [72.18,73.87] | 2970/11029=26.93% [26.13,27.82] | 0 |
| temporal | 3 | family | 98/10952=0.89% [0.73,1.08] | 3059/10952=27.93% [27.08,28.81] | 67/77=87.01% [78.67,93.75] | 76/77=98.70% [95.95,100.00] | 8059/11029=73.07% [72.18,73.87] | 2970/11029=26.93% [26.13,27.82] | 0 |

## Every frozen fold: label_scam, both roles / all seeds

| Split | Seed | Role | SCAM FPR | Alert FPR | SCAM recall | Alert recall | Coverage | UNCERTAIN | Unknown excluded |
|---|---|---|---|---|---|---|---|---|---|
| campaign | 1 | baseline | 101/10986=0.92% [0.69,1.20] | 158/10986=1.44% [1.16,1.75] | 5420/5926=91.46% [90.73,92.20] | 5664/5926=95.58% [95.02,96.09] | 16611/16912=98.22% [98.02,98.42] | 301/16912=1.78% [1.58,1.98] | 5233 |
| campaign | 1 | family | 101/10986=0.92% [0.69,1.20] | 158/10986=1.44% [1.16,1.75] | 5420/5926=91.46% [90.73,92.20] | 5664/5926=95.58% [95.02,96.09] | 16611/16912=98.22% [98.02,98.42] | 301/16912=1.78% [1.58,1.98] | 5233 |
| campaign | 2 | baseline | 124/10986=1.13% [0.89,1.42] | 162/10986=1.47% [1.20,1.78] | 5543/5926=93.54% [92.89,94.15] | 5717/5926=96.47% [95.97,96.95] | 16700/16912=98.75% [98.58,98.91] | 212/16912=1.25% [1.09,1.42] | 5233 |
| campaign | 2 | family | 124/10986=1.13% [0.89,1.42] | 162/10986=1.47% [1.20,1.78] | 5543/5926=93.54% [92.89,94.15] | 5717/5926=96.47% [95.97,96.95] | 16700/16912=98.75% [98.58,98.91] | 212/16912=1.25% [1.09,1.42] | 5233 |
| campaign | 3 | baseline | 141/10986=1.28% [1.01,1.58] | 145/10986=1.32% [1.05,1.61] | 5708/5926=96.32% [95.86,96.84] | 5712/5926=96.39% [95.93,96.88] | 16904/16912=99.95% [99.92,99.98] | 8/16912=0.05% [0.02,0.08] | 5233 |
| campaign | 3 | family | 141/10986=1.28% [1.01,1.58] | 145/10986=1.32% [1.05,1.61] | 5708/5926=96.32% [95.86,96.84] | 5712/5926=96.39% [95.93,96.88] | 16904/16912=99.95% [99.92,99.98] | 8/16912=0.05% [0.02,0.08] | 5233 |
| lang_shift | 1 | baseline | 526/1322=39.79% [36.83,42.59] | 572/1322=43.27% [40.36,46.05] | 76/101=75.25% [61.59,89.77] | 79/101=78.22% [63.86,93.02] | 1374/1423=96.56% [95.51,97.50] | 49/1423=3.44% [2.50,4.49] | 254 |
| lang_shift | 1 | family | 526/1322=39.79% [36.83,42.59] | 572/1322=43.27% [40.36,46.05] | 76/101=75.25% [61.59,89.77] | 79/101=78.22% [63.86,93.02] | 1374/1423=96.56% [95.51,97.50] | 49/1423=3.44% [2.50,4.49] | 254 |
| lang_shift | 2 | baseline | 39/1322=2.95% [1.62,4.86] | 165/1322=12.48% [10.32,14.75] | 65/101=64.36% [49.15,80.24] | 74/101=73.27% [57.50,89.41] | 1288/1423=90.51% [89.00,91.97] | 135/1423=9.49% [8.03,11.00] | 254 |
| lang_shift | 2 | family | 39/1322=2.95% [1.62,4.86] | 165/1322=12.48% [10.32,14.75] | 65/101=64.36% [49.15,80.24] | 74/101=73.27% [57.50,89.41] | 1288/1423=90.51% [89.00,91.97] | 135/1423=9.49% [8.03,11.00] | 254 |
| lang_shift | 3 | baseline | 66/1322=4.99% [3.93,6.23] | 304/1322=23.00% [20.71,25.30] | 70/101=69.31% [54.09,85.39] | 75/101=74.26% [61.60,88.51] | 1180/1423=82.92% [81.06,84.97] | 243/1423=17.08% [15.03,18.94] | 254 |
| lang_shift | 3 | family | 66/1322=4.99% [3.93,6.23] | 304/1322=23.00% [20.71,25.30] | 70/101=69.31% [54.09,85.39] | 75/101=74.26% [61.60,88.51] | 1180/1423=82.92% [81.06,84.97] | 243/1423=17.08% [15.03,18.94] | 254 |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | baseline | 145/668=21.71% [18.04,25.11] | 145/668=21.71% [18.04,25.11] | 259/273=94.87% [91.55,97.64] | 259/273=94.87% [91.55,97.64] | 941/941=100.00% [100.00,100.00] | 0/941=0.00% [0.00,0.00] | 0 |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | family | 145/668=21.71% [18.04,25.11] | 145/668=21.71% [18.04,25.11] | 259/273=94.87% [91.55,97.64] | 259/273=94.87% [91.55,97.64] | 941/941=100.00% [100.00,100.00] | 0/941=0.00% [0.00,0.00] | 0 |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | baseline | 150/668=22.46% [19.00,25.88] | 151/668=22.60% [19.07,26.02] | 259/273=94.87% [91.53,97.33] | 262/273=95.97% [92.96,98.20] | 937/941=99.57% [99.13,99.90] | 4/941=0.43% [0.10,0.87] | 0 |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | family | 150/668=22.46% [19.00,25.88] | 151/668=22.60% [19.07,26.02] | 259/273=94.87% [91.53,97.33] | 262/273=95.97% [92.96,98.20] | 937/941=99.57% [99.13,99.90] | 4/941=0.43% [0.10,0.87] | 0 |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | baseline | 155/668=23.20% [19.76,26.72] | 156/668=23.35% [19.85,26.99] | 262/273=95.97% [93.21,98.23] | 264/273=96.70% [94.37,98.57] | 938/941=99.68% [99.27,100.00] | 3/941=0.32% [0.00,0.73] | 0 |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | family | 155/668=23.20% [19.76,26.72] | 156/668=23.35% [19.85,26.99] | 262/273=95.97% [93.21,98.23] | 264/273=96.70% [94.37,98.57] | 938/941=99.68% [99.27,100.00] | 3/941=0.32% [0.00,0.73] | 0 |
| source_holdout:holdout_scamshield | 1 | baseline | 6324/46944=13.47% [13.12,13.84] | 7909/46944=16.85% [16.45,17.28] | 1286/5568=23.10% [20.56,26.16] | 1722/5568=30.93% [28.02,34.16] | 50491/52512=96.15% [95.93,96.35] | 2021/52512=3.85% [3.65,4.07] | 23461 |
| source_holdout:holdout_scamshield | 1 | family | 6324/46944=13.47% [13.12,13.84] | 7909/46944=16.85% [16.45,17.28] | 1286/5568=23.10% [20.56,26.16] | 1722/5568=30.93% [28.02,34.16] | 50491/52512=96.15% [95.93,96.35] | 2021/52512=3.85% [3.65,4.07] | 23461 |
| source_holdout:holdout_scamshield | 2 | baseline | 5024/46944=10.70% [10.39,11.03] | 6984/46944=14.88% [14.47,15.31] | 1169/5568=20.99% [18.42,23.87] | 2457/5568=44.13% [39.83,48.35] | 49264/52512=93.81% [93.02,94.37] | 3248/52512=6.19% [5.63,6.98] | 23461 |
| source_holdout:holdout_scamshield | 2 | family | 5024/46944=10.70% [10.39,11.03] | 6984/46944=14.88% [14.47,15.31] | 1169/5568=20.99% [18.42,23.87] | 2457/5568=44.13% [39.83,48.35] | 49264/52512=93.81% [93.02,94.37] | 3248/52512=6.19% [5.63,6.98] | 23461 |
| source_holdout:holdout_scamshield | 3 | baseline | 7659/46944=16.32% [15.90,16.83] | 7746/46944=16.50% [16.07,17.02] | 2914/5568=52.33% [46.66,59.00] | 3074/5568=55.21% [49.57,61.77] | 52265/52512=99.53% [99.44,99.61] | 247/52512=0.47% [0.39,0.56] | 23461 |
| source_holdout:holdout_scamshield | 3 | family | 7659/46944=16.32% [15.90,16.83] | 7746/46944=16.50% [16.07,17.02] | 2914/5568=52.33% [46.66,59.00] | 3074/5568=55.21% [49.57,61.77] | 52265/52512=99.53% [99.44,99.61] | 247/52512=0.47% [0.39,0.56] | 23461 |
| source_holdout:holdout_uci_sms_spam | 1 | baseline | 24/4517=0.53% [0.33,0.73] | 32/4517=0.71% [0.49,0.97] | NA (n=0) | NA (n=0) | 4509/4517=99.82% [99.69,99.93] | 8/4517=0.18% [0.07,0.31] | 642 |
| source_holdout:holdout_uci_sms_spam | 1 | family | 24/4517=0.53% [0.33,0.73] | 32/4517=0.71% [0.49,0.97] | NA (n=0) | NA (n=0) | 4509/4517=99.82% [99.69,99.93] | 8/4517=0.18% [0.07,0.31] | 642 |
| source_holdout:holdout_uci_sms_spam | 2 | baseline | 37/4517=0.82% [0.55,1.09] | 42/4517=0.93% [0.64,1.20] | NA (n=0) | NA (n=0) | 4512/4517=99.89% [99.78,99.98] | 5/4517=0.11% [0.02,0.22] | 642 |
| source_holdout:holdout_uci_sms_spam | 2 | family | 37/4517=0.82% [0.55,1.09] | 42/4517=0.93% [0.64,1.20] | NA (n=0) | NA (n=0) | 4512/4517=99.89% [99.78,99.98] | 5/4517=0.11% [0.02,0.22] | 642 |
| source_holdout:holdout_uci_sms_spam | 3 | baseline | 39/4517=0.86% [0.60,1.13] | 39/4517=0.86% [0.60,1.13] | NA (n=0) | NA (n=0) | 4517/4517=100.00% [100.00,100.00] | 0/4517=0.00% [0.00,0.00] | 642 |
| source_holdout:holdout_uci_sms_spam | 3 | family | 39/4517=0.86% [0.60,1.13] | 39/4517=0.86% [0.60,1.13] | NA (n=0) | NA (n=0) | 4517/4517=100.00% [100.00,100.00] | 0/4517=0.00% [0.00,0.00] | 642 |
| temporal | 1 | baseline | 76/10952=0.69% [0.55,0.86] | 3930/10952=35.88% [34.93,36.74] | 68/77=88.31% [80.00,94.67] | 76/77=98.70% [95.89,100.00] | 7167/11029=64.98% [64.03,65.85] | 3862/11029=35.02% [34.15,35.97] | 0 |
| temporal | 1 | family | 76/10952=0.69% [0.55,0.86] | 3930/10952=35.88% [34.93,36.74] | 68/77=88.31% [80.00,94.67] | 76/77=98.70% [95.89,100.00] | 7167/11029=64.98% [64.03,65.85] | 3862/11029=35.02% [34.15,35.97] | 0 |
| temporal | 2 | baseline | 72/10952=0.66% [0.50,0.81] | 4549/10952=41.54% [40.55,42.48] | 66/77=85.71% [76.92,92.50] | 77/77=100.00% [100.00,100.00] | 6541/11029=59.31% [58.37,60.22] | 4488/11029=40.69% [39.78,41.63] | 0 |
| temporal | 2 | family | 72/10952=0.66% [0.50,0.81] | 4549/10952=41.54% [40.55,42.48] | 66/77=85.71% [76.92,92.50] | 77/77=100.00% [100.00,100.00] | 6541/11029=59.31% [58.37,60.22] | 4488/11029=40.69% [39.78,41.63] | 0 |
| temporal | 3 | baseline | 98/10952=0.89% [0.73,1.08] | 3059/10952=27.93% [27.08,28.81] | 67/77=87.01% [78.67,93.75] | 76/77=98.70% [95.95,100.00] | 8059/11029=73.07% [72.18,73.87] | 2970/11029=26.93% [26.13,27.82] | 0 |
| temporal | 3 | family | 98/10952=0.89% [0.73,1.08] | 3059/10952=27.93% [27.08,28.81] | 67/77=87.01% [78.67,93.75] | 76/77=98.70% [95.95,100.00] | 8059/11029=73.07% [72.18,73.87] | 2970/11029=26.93% [26.13,27.82] | 0 |

## Both external populations: label_binary, both roles / all seeds

| Split | Seed | Role | SCAM FPR | Alert FPR | SCAM recall | Alert recall | Coverage | UNCERTAIN | Unknown excluded |
|---|---|---|---|---|---|---|---|---|---|
| groundlm | 1 | baseline | 440/1133=38.83% [24.29,51.69] | 463/1133=40.86% [26.03,54.18] | 40/91=43.96% [24.99,64.29] | 69/91=75.82% [63.89,83.59] | 1172/1224=95.75% [92.31,98.23] | 52/1224=4.25% [1.77,7.69] | 0 |
| groundlm | 1 | family | 440/1133=38.83% [24.29,51.69] | 463/1133=40.86% [26.03,54.18] | 40/91=43.96% [24.99,64.29] | 69/91=75.82% [63.89,83.59] | 1172/1224=95.75% [92.31,98.23] | 52/1224=4.25% [1.77,7.69] | 0 |
| groundlm | 2 | baseline | 338/1133=29.83% [17.07,41.28] | 367/1133=32.39% [18.88,44.43] | 28/91=30.77% [17.31,48.94] | 61/91=67.03% [47.22,80.56] | 1162/1224=94.93% [91.22,97.99] | 62/1224=5.07% [2.01,8.78] | 0 |
| groundlm | 2 | family | 338/1133=29.83% [17.07,41.28] | 367/1133=32.39% [18.88,44.43] | 28/91=30.77% [17.31,48.94] | 61/91=67.03% [47.22,80.56] | 1162/1224=94.93% [91.22,97.99] | 62/1224=5.07% [2.01,8.78] | 0 |
| groundlm | 3 | baseline | 348/1133=30.71% [18.36,41.42] | 356/1133=31.42% [19.05,42.14] | 66/91=72.53% [55.93,86.52] | 66/91=72.53% [55.93,86.52] | 1216/1224=99.35% [98.99,99.72] | 8/1224=0.65% [0.28,1.01] | 0 |
| groundlm | 3 | family | 348/1133=30.71% [18.36,41.42] | 356/1133=31.42% [19.05,42.14] | 66/91=72.53% [55.93,86.52] | 66/91=72.53% [55.93,86.52] | 1216/1224=99.35% [98.99,99.72] | 8/1224=0.65% [0.28,1.01] | 0 |
| smishx | 1 | baseline | 16/606=2.64% [1.48,3.98] | 31/606=5.12% [3.47,7.10] | 30/501=5.99% [3.28,8.98] | 190/501=37.92% [31.12,45.60] | 932/1107=84.19% [81.07,87.20] | 175/1107=15.81% [12.80,18.93] | 0 |
| smishx | 1 | family | 16/606=2.64% [1.48,3.98] | 31/606=5.12% [3.47,7.10] | 30/501=5.99% [3.28,8.98] | 190/501=37.92% [31.12,45.60] | 932/1107=84.19% [81.07,87.20] | 175/1107=15.81% [12.80,18.93] | 0 |
| smishx | 2 | baseline | 13/606=2.15% [0.99,3.46] | 28/606=4.62% [2.97,6.40] | 45/501=8.98% [5.18,13.18] | 236/501=47.11% [40.25,54.67] | 901/1107=81.39% [78.25,84.42] | 206/1107=18.61% [15.58,21.75] | 0 |
| smishx | 2 | family | 13/606=2.15% [0.99,3.46] | 28/606=4.62% [2.97,6.40] | 45/501=8.98% [5.18,13.18] | 236/501=47.11% [40.25,54.67] | 901/1107=81.39% [78.25,84.42] | 206/1107=18.61% [15.58,21.75] | 0 |
| smishx | 3 | baseline | 41/606=6.77% [4.97,8.88] | 41/606=6.77% [4.97,8.88] | 320/501=63.87% [56.03,71.49] | 329/501=65.67% [57.80,72.94] | 1098/1107=99.19% [98.63,99.65] | 9/1107=0.81% [0.35,1.37] | 0 |
| smishx | 3 | family | 41/606=6.77% [4.97,8.88] | 41/606=6.77% [4.97,8.88] | 320/501=63.87% [56.03,71.49] | 329/501=65.67% [57.80,72.94] | 1098/1107=99.19% [98.63,99.65] | 9/1107=0.81% [0.35,1.37] | 0 |

## Both external populations: label_scam, both roles / all seeds

| Split | Seed | Role | SCAM FPR | Alert FPR | SCAM recall | Alert recall | Coverage | UNCERTAIN | Unknown excluded |
|---|---|---|---|---|---|---|---|---|---|
| groundlm | 1 | baseline | 440/1133=38.83% [24.29,51.69] | 463/1133=40.86% [26.03,54.18] | 40/91=43.96% [24.99,64.29] | 69/91=75.82% [63.89,83.59] | 1172/1224=95.75% [92.31,98.23] | 52/1224=4.25% [1.77,7.69] | 0 |
| groundlm | 1 | family | 440/1133=38.83% [24.29,51.69] | 463/1133=40.86% [26.03,54.18] | 40/91=43.96% [24.99,64.29] | 69/91=75.82% [63.89,83.59] | 1172/1224=95.75% [92.31,98.23] | 52/1224=4.25% [1.77,7.69] | 0 |
| groundlm | 2 | baseline | 338/1133=29.83% [17.07,41.28] | 367/1133=32.39% [18.88,44.43] | 28/91=30.77% [17.31,48.94] | 61/91=67.03% [47.22,80.56] | 1162/1224=94.93% [91.22,97.99] | 62/1224=5.07% [2.01,8.78] | 0 |
| groundlm | 2 | family | 338/1133=29.83% [17.07,41.28] | 367/1133=32.39% [18.88,44.43] | 28/91=30.77% [17.31,48.94] | 61/91=67.03% [47.22,80.56] | 1162/1224=94.93% [91.22,97.99] | 62/1224=5.07% [2.01,8.78] | 0 |
| groundlm | 3 | baseline | 348/1133=30.71% [18.36,41.42] | 356/1133=31.42% [19.05,42.14] | 66/91=72.53% [55.93,86.52] | 66/91=72.53% [55.93,86.52] | 1216/1224=99.35% [98.99,99.72] | 8/1224=0.65% [0.28,1.01] | 0 |
| groundlm | 3 | family | 348/1133=30.71% [18.36,41.42] | 356/1133=31.42% [19.05,42.14] | 66/91=72.53% [55.93,86.52] | 66/91=72.53% [55.93,86.52] | 1216/1224=99.35% [98.99,99.72] | 8/1224=0.65% [0.28,1.01] | 0 |
| smishx | 1 | baseline | 26/881=2.95% [1.56,4.69] | 176/881=19.98% [15.75,24.31] | 20/226=8.85% [5.00,15.03] | 45/226=19.91% [14.47,27.70] | 932/1107=84.19% [81.07,87.20] | 175/1107=15.81% [12.80,18.93] | 0 |
| smishx | 1 | family | 26/881=2.95% [1.56,4.69] | 176/881=19.98% [15.75,24.31] | 20/226=8.85% [5.00,15.03] | 45/226=19.91% [14.47,27.70] | 932/1107=84.19% [81.07,87.20] | 175/1107=15.81% [12.80,18.93] | 0 |
| smishx | 2 | baseline | 29/881=3.29% [1.63,5.37] | 184/881=20.89% [16.63,24.84] | 29/226=12.83% [7.22,20.48] | 80/226=35.40% [27.01,45.11] | 901/1107=81.39% [78.25,84.42] | 206/1107=18.61% [15.58,21.75] | 0 |
| smishx | 2 | family | 29/881=3.29% [1.63,5.37] | 184/881=20.89% [16.63,24.84] | 29/226=12.83% [7.22,20.48] | 80/226=35.40% [27.01,45.11] | 901/1107=81.39% [78.25,84.42] | 206/1107=18.61% [15.58,21.75] | 0 |
| smishx | 3 | baseline | 255/881=28.94% [24.23,33.30] | 258/881=29.28% [24.59,33.59] | 106/226=46.90% [37.12,58.85] | 112/226=49.56% [39.74,61.39] | 1098/1107=99.19% [98.63,99.65] | 9/1107=0.81% [0.35,1.37] | 0 |
| smishx | 3 | family | 255/881=28.94% [24.23,33.30] | 258/881=29.28% [24.59,33.59] | 106/226=46.90% [37.12,58.85] | 112/226=49.56% [39.74,61.39] | 1098/1107=99.19% [98.63,99.65] | 9/1107=0.81% [0.35,1.37] | 0 |

## Paired changes on every frozen fold: family minus matched global

Cells are delta percentage points [clusterCI], two-sided null-centred bootstrap p.
All defined differences are identically0, degenerate[0,0],p1 because of fallback;
this is equality of saved outputs, not evidence of equivalence in a new population.

| Split | Seed | Label | SCAM FPR delta | Alert FPR delta | SCAM recall delta | Alert recall delta |
|---|---|---|---|---|---|---|
| campaign | 1 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| campaign | 1 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| campaign | 2 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| campaign | 2 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| campaign | 3 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| campaign | 3 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| lang_shift | 1 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| lang_shift | 1 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| lang_shift | 2 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| lang_shift | 2 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| lang_shift | 3 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| lang_shift | 3 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| source_holdout:holdout_scamshield | 1 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| source_holdout:holdout_scamshield | 1 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| source_holdout:holdout_scamshield | 2 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| source_holdout:holdout_scamshield | 2 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| source_holdout:holdout_scamshield | 3 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| source_holdout:holdout_scamshield | 3 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| source_holdout:holdout_uci_sms_spam | 1 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| source_holdout:holdout_uci_sms_spam | 1 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | NA | NA |
| source_holdout:holdout_uci_sms_spam | 2 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| source_holdout:holdout_uci_sms_spam | 2 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | NA | NA |
| source_holdout:holdout_uci_sms_spam | 3 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| source_holdout:holdout_uci_sms_spam | 3 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | NA | NA |
| temporal | 1 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| temporal | 1 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| temporal | 2 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| temporal | 2 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| temporal | 3 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| temporal | 3 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |

## Paired changes on external sets: family minus matched global

Cells are delta percentage points [clusterCI], two-sided null-centred bootstrap p.
All defined differences are identically0, degenerate[0,0],p1 because of fallback;
this is equality of saved outputs, not evidence of equivalence in a new population.

| Split | Seed | Label | SCAM FPR delta | Alert FPR delta | SCAM recall delta | Alert recall delta |
|---|---|---|---|---|---|---|
| groundlm | 1 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| groundlm | 1 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| groundlm | 2 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| groundlm | 2 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| groundlm | 3 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| groundlm | 3 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| smishx | 1 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| smishx | 1 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| smishx | 2 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| smishx | 2 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| smishx | 3 | label_binary | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |
| smishx | 3 | label_scam | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 | +0.00 [+0.00,+0.00],p=1 |

## Fixed cal eligibility / fallback counts

Counts below are negative/positive in the probability-fit and threshold-fit stages.
All stages are original calibration rows, never test/external/extra synthetic rows.
Other always uses global calibration. No family was dropped after outcomes.

| Fold | Seed | Family | Probability stage neg/pos | Threshold stage neg/pos | Mode / reason |
|---|---|---|---|---|---|
| campaign | 1 | bank | 13/123 | 9/120 | global/sparse_fixed_cal_stages |
| campaign | 1 | delivery | 2/35 | 3/43 | global/sparse_fixed_cal_stages |
| campaign | 1 | otp | 0/14 | 0/8 | global/sparse_fixed_cal_stages |
| campaign | 1 | telecom_utility | 3/23 | 2/21 | global/sparse_fixed_cal_stages |
| campaign | 1 | upi | 1/3 | 0/5 | global/sparse_fixed_cal_stages |
| campaign | 2 | bank | 13/123 | 9/120 | global/sparse_fixed_cal_stages |
| campaign | 2 | delivery | 2/35 | 3/43 | global/sparse_fixed_cal_stages |
| campaign | 2 | otp | 0/14 | 0/8 | global/sparse_fixed_cal_stages |
| campaign | 2 | telecom_utility | 3/23 | 2/21 | global/sparse_fixed_cal_stages |
| campaign | 2 | upi | 1/3 | 0/5 | global/sparse_fixed_cal_stages |
| campaign | 3 | bank | 13/123 | 9/120 | global/sparse_fixed_cal_stages |
| campaign | 3 | delivery | 2/35 | 3/43 | global/sparse_fixed_cal_stages |
| campaign | 3 | otp | 0/14 | 0/8 | global/sparse_fixed_cal_stages |
| campaign | 3 | telecom_utility | 3/23 | 2/21 | global/sparse_fixed_cal_stages |
| campaign | 3 | upi | 1/3 | 0/5 | global/sparse_fixed_cal_stages |
| lang_shift | 1 | bank | 7/122 | 9/123 | global/sparse_fixed_cal_stages |
| lang_shift | 1 | delivery | 1/42 | 1/40 | global/sparse_fixed_cal_stages |
| lang_shift | 1 | otp | 0/4 | 0/15 | global/sparse_fixed_cal_stages |
| lang_shift | 1 | telecom_utility | 1/21 | 3/17 | global/sparse_fixed_cal_stages |
| lang_shift | 1 | upi | 0/2 | 1/3 | global/sparse_fixed_cal_stages |
| lang_shift | 2 | bank | 7/122 | 9/123 | global/sparse_fixed_cal_stages |
| lang_shift | 2 | delivery | 1/42 | 1/40 | global/sparse_fixed_cal_stages |
| lang_shift | 2 | otp | 0/4 | 0/15 | global/sparse_fixed_cal_stages |
| lang_shift | 2 | telecom_utility | 1/21 | 3/17 | global/sparse_fixed_cal_stages |
| lang_shift | 2 | upi | 0/2 | 1/3 | global/sparse_fixed_cal_stages |
| lang_shift | 3 | bank | 7/122 | 9/123 | global/sparse_fixed_cal_stages |
| lang_shift | 3 | delivery | 1/42 | 1/40 | global/sparse_fixed_cal_stages |
| lang_shift | 3 | otp | 0/4 | 0/15 | global/sparse_fixed_cal_stages |
| lang_shift | 3 | telecom_utility | 1/21 | 3/17 | global/sparse_fixed_cal_stages |
| lang_shift | 3 | upi | 0/2 | 1/3 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | bank | 9/115 | 6/121 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | delivery | 0/41 | 2/50 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | otp | 1/5 | 0/9 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | telecom_utility | 2/21 | 3/23 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_mishra_soni_smsphishing | 1 | upi | 0/5 | 0/4 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | bank | 9/115 | 6/121 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | delivery | 0/41 | 2/50 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | otp | 1/5 | 0/9 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | telecom_utility | 2/21 | 3/23 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_mishra_soni_smsphishing | 2 | upi | 0/5 | 0/4 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | bank | 9/115 | 6/121 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | delivery | 0/41 | 2/50 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | otp | 1/5 | 0/9 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | telecom_utility | 2/21 | 3/23 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_mishra_soni_smsphishing | 3 | upi | 0/5 | 0/4 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_scamshield | 1 | bank | 2/112 | 2/119 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_scamshield | 1 | delivery | 0/24 | 0/23 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_scamshield | 1 | otp | 0/10 | 0/5 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_scamshield | 1 | telecom_utility | 0/11 | 1/9 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_scamshield | 1 | upi | 0/0 | 0/2 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_scamshield | 2 | bank | 2/112 | 2/119 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_scamshield | 2 | delivery | 0/24 | 0/23 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_scamshield | 2 | otp | 0/10 | 0/5 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_scamshield | 2 | telecom_utility | 0/11 | 1/9 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_scamshield | 2 | upi | 0/0 | 0/2 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_scamshield | 3 | bank | 2/112 | 2/119 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_scamshield | 3 | delivery | 0/24 | 0/23 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_scamshield | 3 | otp | 0/10 | 0/5 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_scamshield | 3 | telecom_utility | 0/11 | 1/9 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_scamshield | 3 | upi | 0/0 | 0/2 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_uci_sms_spam | 1 | bank | 11/118 | 12/134 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_uci_sms_spam | 1 | delivery | 3/39 | 2/36 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_uci_sms_spam | 1 | otp | 1/13 | 1/9 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_uci_sms_spam | 1 | telecom_utility | 1/17 | 1/20 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_uci_sms_spam | 1 | upi | 0/6 | 0/4 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_uci_sms_spam | 2 | bank | 11/118 | 12/134 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_uci_sms_spam | 2 | delivery | 3/39 | 2/36 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_uci_sms_spam | 2 | otp | 1/13 | 1/9 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_uci_sms_spam | 2 | telecom_utility | 1/17 | 1/20 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_uci_sms_spam | 2 | upi | 0/6 | 0/4 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_uci_sms_spam | 3 | bank | 11/118 | 12/134 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_uci_sms_spam | 3 | delivery | 3/39 | 2/36 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_uci_sms_spam | 3 | otp | 1/13 | 1/9 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_uci_sms_spam | 3 | telecom_utility | 1/17 | 1/20 | global/sparse_fixed_cal_stages |
| source_holdout:holdout_uci_sms_spam | 3 | upi | 0/6 | 0/4 | global/sparse_fixed_cal_stages |
| temporal | 1 | bank | 14/1 | 7/0 | global/sparse_fixed_cal_stages |
| temporal | 1 | delivery | 1/0 | 0/1 | global/sparse_fixed_cal_stages |
| temporal | 1 | otp | 1/0 | 0/0 | global/sparse_fixed_cal_stages |
| temporal | 1 | telecom_utility | 3/1 | 3/0 | global/sparse_fixed_cal_stages |
| temporal | 1 | upi | 1/0 | 0/0 | global/sparse_fixed_cal_stages |
| temporal | 2 | bank | 14/1 | 7/0 | global/sparse_fixed_cal_stages |
| temporal | 2 | delivery | 1/0 | 0/1 | global/sparse_fixed_cal_stages |
| temporal | 2 | otp | 1/0 | 0/0 | global/sparse_fixed_cal_stages |
| temporal | 2 | telecom_utility | 3/1 | 3/0 | global/sparse_fixed_cal_stages |
| temporal | 2 | upi | 1/0 | 0/0 | global/sparse_fixed_cal_stages |
| temporal | 3 | bank | 14/1 | 7/0 | global/sparse_fixed_cal_stages |
| temporal | 3 | delivery | 1/0 | 0/1 | global/sparse_fixed_cal_stages |
| temporal | 3 | otp | 1/0 | 0/0 | global/sparse_fixed_cal_stages |
| temporal | 3 | telecom_utility | 3/1 | 3/0 | global/sparse_fixed_cal_stages |
| temporal | 3 | upi | 1/0 | 0/0 | global/sparse_fixed_cal_stages |

## Validation and publication

Focused family nine tests passed on privateKaggle after an initial fixture-only NumPy
construction failure was corrected. Earlier run/failure is retained in STATUS.
One milestone full invocation on privateKaggle: **882passed,0skipped/errors/failures**
(`unit_validation-c-full-20261102`v1,ok204.0s). Source tree
`4041ccad6be5c0711eea8832f884cebfdaa665cc9855619e936a9a38ec70f6b7`.
No corpus model work ran on the laptop. Exact collection/JUnit inventory verified.
Publisher-only dependency correction after that invocation is covered separately
by focused checks; final output and private owner/isPrivate receipt are in STATUS.
No second full-suite invocation. Ruff output recorded at milestone commit.

Only allowlisted selected weights/URL/source/manifests and aggregate JSON are archived
privately. No SMS texts/row score parquets/secret files in git or publication allowlist.
No shipment approval. **HF publish pending**. Original v1 safetyKILL, policy successor
KILL and historical contaminated KD withdrawal remain unchanged.

Next: predeclared multilingual train-only student; URL three seeds/stronger teacher
and NIDS as capacity allows. No additional P1 template-threshold fishing.

Private archive **maadhavvh/rakshak-template-cal-c1-20261102 v1** verified owner
maadhavvh, **isPrivate=true**, 22,256,698 bytes. Archive SHA256
`11b5b8493aafbee449fd0fdb12db73c5b52c453decf40918e50b4c61c4fca93a`.
Initial immediate metadata403 during processing was followed by one successful
owner read; no re-upload. **HF publish pending**, shipment false.
Post-full publisher-only checks: **3 passed in 1.23s**, requested CPU/thread limits,
Tctl41.0C. Full pytest882pass/0skip/157warnings121.95s; RuffAllcheckspassed.
