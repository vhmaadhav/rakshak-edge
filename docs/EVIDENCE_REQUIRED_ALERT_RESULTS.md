# Evidence-required alert: KILL despite low GroundLM false alarms

**KILL under the owner’s unchanged rules-only recall gate.** GroundLM SCAM0/1133 and alert4/1133=.353% pass the1%/15% caps; SmishX rules-recall guards pass. Ten internal fold/label recall checks fail, with no missing result. No v2, candidate Phase1 wiring or ELA decision change. The preceding evidence-escalation audit remains **UNMEASURABLE**, not a measuredKILL or a pass.

Policy: no qualifying rule/URL evidence alwaysSAFE, even for student-only high risk. Qualifying evidence and fused>=t_scam yieldsSCAM; otherwiseUNCERTAIN. No metric-exempt info alerts. Frozen v1 experts/thresholds, each fold’s own companion. Rules comparator is the unchanged canonicaliser→run_rules→build_package(scam_state=benign): HIGH=SCAM, MEDIUM/UNCERTAIN=alert,LOW=SAFE; no URL model. Student comparator uses the same parent’s frozen cal1%-FPR threshold. Only rules-recall CIhalf is the declared recall guard, not student recall.

**Exposure:** GroundLM/ScamShield informed rules; GroundLM/SmishX aggregate feedback motivated this policy. **In-sample policy selection**, no held-out improvement claim. SmishX is independent of train after overlap removal, non-Indian-specific, but no longer untouched for policy selection. Independent consented scrubbed Indian inbox validation remains OPEN. Frozen memberships/labels/temporal overlaps unchanged.

All-negative FPR denominators include abstentions; unknown label-policy rows excluded. CIs are saved95% campaign-clustered bootstrap intervals (1,000 draws), external campaigns approximate template/domain groups. **Zero events produce a degenerate empirical clustered[0,0] interval, not a zero population-risk guarantee.** GroundLM0/1133 SCAM IIDWilson[0,.34]% is separate and assumes independence; it is not a clustered bound. GroundLM SCAM recall is also0/91 and must not disappear behind the FPR result.

| Population | Label | SCAM FPR[CI] | >=UNCERTAIN FPR[CI] | SCAM recall[CI] | Alert recall[CI] |
|---|---|---|---|---|---|
| campaign | label_binary | 2/10952=0.02% [0.00,0.05] | 70/10952=0.64% [0.49,0.80] | 392/11193=3.50% [3.16,3.86] | 1082/11193=9.67% [9.11,10.23] |
| campaign | label_scam | 2/10986=0.02% [0.00,0.05] | 73/10986=0.66% [0.51,0.82] | 112/5926=1.89% [1.55,2.23] | 240/5926=4.05% [3.55,4.58] |
| lang_shift | label_binary | 3/1320=0.23% [0.00,0.53] | 4/1320=0.30% [0.07,0.63] | 9/357=2.52% [1.07,4.36] | 16/357=4.48% [2.44,7.08] |
| lang_shift | label_scam | 3/1322=0.23% [0.00,0.54] | 5/1322=0.38% [0.08,0.76] | 3/101=2.97% [0.00,7.07] | 3/101=2.97% [0.00,7.07] |
| source_holdout:holdout_mishra_soni_smsphishing | label_binary | 0/524=0.00% [0.00,0.00] degenerate | 0/524=0.00% [0.00,0.00] degenerate | 42/417=10.07% [7.04,13.57] | 46/417=11.03% [7.85,14.45] |
| source_holdout:holdout_mishra_soni_smsphishing | label_scam | 11/668=1.65% [0.61,2.74] | 12/668=1.80% [0.76,2.97] | 31/273=11.36% [7.27,16.07] | 34/273=12.45% [8.18,17.44] |
| source_holdout:holdout_scamshield | label_binary | 8/46944=0.02% [0.01,0.03] | 810/46944=1.73% [1.44,2.07] | 65/29029=0.22% [0.12,0.39] | 6878/29029=23.69% [21.93,25.79] |
| source_holdout:holdout_scamshield | label_scam | 8/46944=0.02% [0.01,0.03] | 810/46944=1.73% [1.44,2.07] | 37/5568=0.66% [0.18,1.52] | 1639/5568=29.44% [23.66,36.36] |
| source_holdout:holdout_uci_sms_spam | label_binary | 0/4517=0.00% [0.00,0.00] degenerate | 4/4517=0.09% [0.02,0.18] | 13/642=2.02% [1.06,3.22] | 21/642=3.27% [1.51,5.69] |
| source_holdout:holdout_uci_sms_spam | label_scam | 0/4517=0.00% [0.00,0.00] degenerate | 4/4517=0.09% [0.02,0.18] | NA(n=0) | NA(n=0) |
| temporal | label_binary | 2/10952=0.02% [0.00,0.05] | 70/10952=0.64% [0.49,0.80] | 1/77=1.30% [0.00,4.05] | 1/77=1.30% [0.00,4.05] |
| temporal | label_scam | 2/10952=0.02% [0.00,0.05] | 70/10952=0.64% [0.49,0.80] | 1/77=1.30% [0.00,4.05] | 1/77=1.30% [0.00,4.05] |
| groundlm | label_binary | 0/1133=0.00% [0.00,0.00] degenerate | 4/1133=0.35% [0.00,1.00] | 0/91=0.00% [0.00,0.00] degenerate | 25/91=27.47% [1.43,57.47] |
| groundlm | label_scam | 0/1133=0.00% [0.00,0.00] degenerate | 4/1133=0.35% [0.00,1.00] | 0/91=0.00% [0.00,0.00] degenerate | 25/91=27.47% [1.43,57.47] |
| smishx | label_binary | 12/606=1.98% [0.83,3.15] | 47/606=7.76% [5.63,9.92] | 158/501=31.54% [24.80,38.85] | 434/501=86.63% [82.48,90.77] |
| smishx | label_scam | 155/881=17.59% [13.63,22.03] | 306/881=34.73% [29.81,39.39] | 15/226=6.64% [3.40,10.95] | 175/226=77.43% [69.56,84.67] |

| Population | Label | Rules-only alert recall[CI] | Student SCAM recall[CI] | Policy-minus-rules alert recall[CI],pairedp | Policy-minus-student[CI],pairedp |
|---|---|---|---|---|---|
| campaign | label_binary | 1463/11193=13.07% [12.43,13.71] | 10676/11193=95.38% [95.01,95.76] | -3.40pp [-4.09,-2.77],p=0.000999001 | -85.71pp [-86.36,-85.04],p=0.000999001 |
| campaign | label_scam | 883/5926=14.90% [14.01,15.85] | 5690/5926=96.02% [95.51,96.51] | -10.85pp [-11.75,-9.99],p=0.000999001 | -91.97pp [-92.64,-91.24],p=0.000999001 |
| lang_shift | label_binary | 51/357=14.29% [9.55,19.36] | 255/357=71.43% [62.59,80.67] | -9.80pp [-14.89,-4.79],p=0.000999001 | -66.95pp [-76.00,-58.27],p=0.000999001 |
| lang_shift | label_scam | 28/101=27.72% [17.31,39.43] | 75/101=74.26% [61.32,87.78] | -24.75pp [-35.96,-14.94],p=0.000999001 | -71.29pp [-85.23,-58.33],p=0.000999001 |
| source_holdout:holdout_mishra_soni_smsphishing | label_binary | 117/417=28.06% [20.19,36.77] | 380/417=91.13% [87.80,94.17] | -17.03pp [-26.72,-7.79],p=0.000999001 | -80.10pp [-84.77,-74.65],p=0.000999001 |
| source_holdout:holdout_mishra_soni_smsphishing | label_scam | 105/273=38.46% [27.30,49.40] | 258/273=94.51% [91.02,97.37] | -26.01pp [-38.27,-12.75],p=0.000999001 | -82.05pp [-87.42,-75.22],p=0.000999001 |
| source_holdout:holdout_scamshield | label_binary | 6201/29029=21.36% [19.40,23.95] | 21269/29029=73.27% [72.06,74.47] | +2.33pp [+0.96,+3.58],p=0.000999001 | -49.57pp [-50.82,-48.11],p=0.000999001 |
| source_holdout:holdout_scamshield | label_scam | 2709/5568=48.65% [42.89,55.74] | 4734/5568=85.02% [82.37,87.76] | -19.22pp [-22.57,-16.08],p=0.000999001 | -55.59pp [-60.28,-50.55],p=0.000999001 |
| source_holdout:holdout_uci_sms_spam | label_binary | 95/642=14.80% [10.75,19.12] | 551/642=85.83% [82.74,88.80] | -11.53pp [-16.48,-6.90],p=0.000999001 | -82.55pp [-86.33,-78.05],p=0.000999001 |
| source_holdout:holdout_uci_sms_spam | label_scam | NA(n=0) | NA(n=0) | NA(n=0) | NA(n=0) |
| temporal | label_binary | 8/77=10.39% [3.85,18.42] | 69/77=89.61% [82.28,95.89] | -9.09pp [-17.28,-1.33],p=0.028971 | -88.31pp [-94.94,-80.26],p=0.000999001 |
| temporal | label_scam | 8/77=10.39% [3.85,18.42] | 69/77=89.61% [82.28,95.89] | -9.09pp [-17.28,-1.33],p=0.028971 | -88.31pp [-94.94,-80.26],p=0.000999001 |
| groundlm | label_binary | 11/91=12.09% [4.90,26.04] | 54/91=59.34% [39.42,79.55] | +15.38pp [-20.84,+49.24],p=0.54046 | -31.87pp [-52.54,-19.00],p=0.00599401 |
| groundlm | label_scam | 11/91=12.09% [4.90,26.04] | 54/91=59.34% [39.42,79.55] | +15.38pp [-20.84,+49.24],p=0.54046 | -31.87pp [-52.54,-19.00],p=0.00599401 |
| smishx | label_binary | 73/501=14.57% [9.45,21.43] | 338/501=67.47% [60.44,74.36] | +72.06pp [+64.66,+78.25],p=0.000999001 | +19.16pp [+11.67,+26.88],p=0.000999001 |
| smishx | label_scam | 39/226=17.26% [10.61,26.00] | 113/226=50.00% [40.81,60.93] | +60.18pp [+48.56,+69.96],p=0.000999001 | +27.43pp [+12.18,+40.73],p=0.000999001 |

| Population | Label | Rules SCAM FPR[CI] | Rules alert FPR[CI] | Rules SCAM recall[CI] | Student SCAM FPR[CI] |
|---|---|---|---|---|---|
| campaign | label_binary | 18/10952=0.16% [0.09,0.25] | 62/10952=0.57% [0.43,0.72] | 225/11193=2.01% [1.73,2.29] | 115/10952=1.05% [0.82,1.38] |
| campaign | label_scam | 19/10986=0.17% [0.10,0.26] | 67/10986=0.61% [0.47,0.76] | 93/5926=1.57% [1.23,1.92] | 145/10986=1.32% [1.06,1.62] |
| lang_shift | label_binary | 6/1320=0.45% [0.15,0.84] | 8/1320=0.61% [0.23,1.04] | 22/357=6.16% [2.47,10.30] | 630/1320=47.73% [44.70,50.66] |
| lang_shift | label_scam | 7/1322=0.53% [0.15,0.98] | 9/1322=0.68% [0.30,1.15] | 9/101=8.91% [1.15,18.46] | 631/1322=47.73% [44.63,50.54] |
| source_holdout:holdout_mishra_soni_smsphishing | label_binary | 1/524=0.19% [0.00,0.73] | 3/524=0.57% [0.00,1.34] | 8/417=1.92% [0.64,3.60] | 18/524=3.44% [1.72,5.37] |
| source_holdout:holdout_mishra_soni_smsphishing | label_scam | 5/668=0.75% [0.00,1.61] | 15/668=2.25% [1.20,3.40] | 4/273=1.47% [0.34,3.08] | 140/668=20.96% [17.36,24.56] |
| source_holdout:holdout_scamshield | label_binary | 87/46944=0.19% [0.13,0.25] | 465/46944=0.99% [0.85,1.17] | 1930/29029=6.65% [5.11,8.76] | 10395/46944=22.14% [21.64,22.66] |
| source_holdout:holdout_scamshield | label_scam | 87/46944=0.19% [0.13,0.25] | 465/46944=0.99% [0.85,1.17] | 951/5568=17.08% [11.15,24.45] | 10395/46944=22.14% [21.64,22.66] |
| source_holdout:holdout_uci_sms_spam | label_binary | 3/4517=0.07% [0.00,0.15] | 9/4517=0.20% [0.09,0.33] | 0/642=0.00% [0.00,0.00] degenerate | 40/4517=0.89% [0.62,1.18] |
| source_holdout:holdout_uci_sms_spam | label_scam | 3/4517=0.07% [0.00,0.15] | 9/4517=0.20% [0.09,0.33] | NA(n=0) | 40/4517=0.89% [0.62,1.18] |
| temporal | label_binary | 18/10952=0.16% [0.09,0.25] | 62/10952=0.57% [0.43,0.72] | 0/77=0.00% [0.00,0.00] degenerate | 92/10952=0.84% [0.68,1.04] |
| temporal | label_scam | 18/10952=0.16% [0.09,0.25] | 62/10952=0.57% [0.43,0.72] | 0/77=0.00% [0.00,0.00] degenerate | 92/10952=0.84% [0.68,1.04] |
| groundlm | label_binary | 1/1133=0.09% [0.00,0.30] | 1/1133=0.09% [0.00,0.30] | 1/91=1.10% [0.00,4.41] | 366/1133=32.30% [17.98,44.65] |
| groundlm | label_scam | 1/1133=0.09% [0.00,0.30] | 1/1133=0.09% [0.00,0.30] | 1/91=1.10% [0.00,4.41] | 366/1133=32.30% [17.98,44.65] |
| smishx | label_binary | 9/606=1.49% [0.66,2.49] | 19/606=3.14% [1.96,4.62] | 6/501=1.20% [0.21,2.52] | 49/606=8.09% [6.08,10.25] |
| smishx | label_scam | 10/881=1.14% [0.56,1.83] | 53/881=6.02% [3.77,9.07] | 5/226=2.21% [0.38,5.13] | 274/881=31.10% [26.41,35.33] |

| Population | Label | Coverage[CI] | UNCERTAIN share[CI] | Conditional decided error[CI] | Saved auto-decided AURC | Saved alert-endpoint AURC |
|---|---|---|---|---|---|---|
| campaign | label_binary | 21387/22145=96.58% [96.33,96.83] | 758/22145=3.42% [3.17,3.67] | 10113/21387=47.29% [46.62,47.97] | 0.258382 | 0.267409 |
| campaign | label_scam | 16713/16912=98.82% [98.66,98.97] | 199/16912=1.18% [1.03,1.34] | 5688/16713=34.03% [33.36,34.71] | 0.183508 | 0.189408 |
| lang_shift | label_binary | 1669/1677=99.52% [99.14,99.82] | 8/1677=0.48% [0.18,0.86] | 344/1669=20.61% [17.73,23.72] | 0.290086 | 0.289804 |
| lang_shift | label_scam | 1421/1423=99.86% [99.65,100.00] | 2/1423=0.14% [0.00,0.35] | 101/1421=7.11% [5.43,9.17] | 0.113450 | 0.113857 |
| source_holdout:holdout_mishra_soni_smsphishing | label_binary | 937/941=99.57% [99.06,99.90] | 4/941=0.43% [0.10,0.94] | 371/937=39.59% [34.58,44.24] | 0.398219 | 0.396658 |
| source_holdout:holdout_mishra_soni_smsphishing | label_scam | 937/941=99.57% [99.06,99.90] | 4/941=0.43% [0.10,0.94] | 250/937=26.68% [21.97,31.70] | 0.355151 | 0.357095 |
| source_holdout:holdout_scamshield | label_binary | 68358/75973=89.98% [88.87,90.79] | 7615/75973=10.02% [9.21,11.13] | 22159/68358=32.42% [31.84,33.02] | 0.396444 | 0.378730 |
| source_holdout:holdout_scamshield | label_scam | 50108/52512=95.42% [94.12,96.26] | 2404/52512=4.58% [3.74,5.88] | 3937/50108=7.86% [7.34,8.43] | 0.158276 | 0.161571 |
| source_holdout:holdout_uci_sms_spam | label_binary | 5147/5159=99.77% [99.54,99.94] | 12/5159=0.23% [0.06,0.46] | 621/5147=12.07% [10.82,13.33] | 0.109582 | 0.110849 |
| source_holdout:holdout_uci_sms_spam | label_scam | 4513/4517=99.91% [99.82,99.98] | 4/4517=0.09% [0.02,0.18] | 0/4513=0.00% [0.00,0.00] degenerate | 0.000000 | 0.002208 |
| temporal | label_binary | 10961/11029=99.38% [99.24,99.52] | 68/11029=0.62% [0.48,0.76] | 78/10961=0.71% [0.56,0.90] | 0.034092 | 0.049025 |
| temporal | label_scam | 10961/11029=99.38% [99.24,99.52] | 68/11029=0.62% [0.48,0.76] | 78/10961=0.71% [0.56,0.90] | 0.034092 | 0.049025 |
| groundlm | label_binary | 1195/1224=97.63% [93.82,99.78] | 29/1224=2.37% [0.22,6.18] | 66/1195=5.52% [2.89,9.29] | 0.030026 | 0.030872 |
| groundlm | label_scam | 1195/1224=97.63% [93.82,99.78] | 29/1224=2.37% [0.22,6.18] | 66/1195=5.52% [2.89,9.29] | 0.030026 | 0.030872 |
| smishx | label_binary | 796/1107=71.91% [67.02,76.79] | 311/1107=28.09% [23.21,32.98] | 79/796=9.92% [6.96,13.20] | 0.092323 | 0.088892 |
| smishx | label_scam | 796/1107=71.91% [67.02,76.79] | 311/1107=28.09% [23.21,32.98] | 206/796=25.88% [21.26,30.88] | 0.116571 | 0.200544 |

Risk–coverage curves rank **already fixed decisions** by savedabs(fused−.5) confidence. Auto-decided curves condition on SAFE/SCAM rows only; their x-axis is coverage within that subset. Alert-endpoint curves cover all eligible rows with positive=SCAM-or-UNCERTAIN. They are descriptive, not alternate deployment thresholds, calibrated risk bounds or evidence of superiority. All101savedpoints per curve remain in private metrics; four eight-population plot figures are rendered in the separate hash-bound policy_curves job without inference, fitting or resampling. No curve was used to alter this policy.

| Failed guard |
|---|
| campaign/label_binary/recall_vs_rules |
| campaign/label_scam/recall_vs_rules |
| lang_shift/label_binary/recall_vs_rules |
| lang_shift/label_scam/recall_vs_rules |
| source_holdout:holdout_mishra_soni_smsphishing/label_binary/recall_vs_rules |
| source_holdout:holdout_mishra_soni_smsphishing/label_scam/recall_vs_rules |
| source_holdout:holdout_scamshield/label_scam/recall_vs_rules |
| source_holdout:holdout_uci_sms_spam/label_binary/recall_vs_rules |
| temporal/label_binary/recall_vs_rules |
| temporal/label_scam/recall_vs_rules |

UCI known-fraud recall isNA(zero mapped positives); its legacy-spam guard remains mandatory and fails. There is no label substitution. SmishX alert recall175/226=77.43% versus rules39/226 and student113/226 is a benefit under this sample, but SmishX fraud alertFPR306/881=34.73% and the internal recall losses remain visible; they cannot be waived by GroundLM’s low FPR.

## Calibration and immutable manifest

Owner-approved **only native URL calibration**, never URL test/benchmark/SMS/external calibration. Original URL kernel selectedcharcnn-wide/s0; seed0/fraction.05 registered-domain grouped carve from official URLtrain partition. Original algorithm/source/defaultargs verified. The native loader filters trainIDs before loadingURL/label columns; benchmark columns excluded. Direct native URL byte/side core scoring preserves all original cal rows, avoiding the SMS extractor’s missing-row behavior. No row dropped or gate changed after outcomes.

- split_id: `url_student/phreshphish/official-train/grouped-cal-seed0-f05`
- partition_file_sha256: `58d8c8c81866491db4f54b1db9a46c70b059b5f261e78be4518d3c5e05640bc5`
- official_train_ids_sha256: `81f55e72f28beb672b1e2ae2b248a46b86e96f556e4b8a8d560427a71dd09e5b`
- cal_ids_sha256: `a93713954509cacb8ea7046982ed37fad42eee4b4a8886441eb721c6b6399be2`
- cal_rows: `18087`
- cal_negative_rows: `11347`
- cal_positive_rows: `6740`
- original_cal_source_sha256: `839fa8fda0a9b0135c87299e67f39c9170ecbde9f0da833b38902dd35536210c`
- url_metrics_sha256: `70b34a58853d1ac2d29ed7a1dadaf506b871c8999a8ba712d1021a7ff4ddec7f`
- Selected native URL threshold: `0.6308487409530461`; calFPR `0.009958579360183309`, calTPR `0.8956973293768546`.

Exact ordered18087calIDs, their hash, threshold, source/URLexpert hashes and all6parent manifests froze **before SMS corpus loading**, not merely before outcome reporting. New tests verify this ordering and exact original carve membership using synthetic fixtures. Parent v1 primary SHA256 `c310d3ccbc1c1e50ce9436f217c66ee9934f778985bfef487a034cc3070b79fa`; named candidate policy SHA256 `c51d8f55102637a67da2786d90b9a7abfc053546a0e8ff18cd771890dd35635a`. Candidate KILL does not approve shipment/Arm portability.

Completed job `evidence_required_alert-native2-20261102`v1, tree `a8cda16cb65a7a166a8ff318cf98b381bbba199f1272fec2587a9e0fd1b96134`, git `6d4669f68c194cafdd9d7827e86d0264d34ba9d3` dirty snapshot,ok443.5s. Args `--v1-tree 0cf9f00c86d115f31a92ad61a4376d501bd36cbed5a396b29fad1857374fe205 --v1-manifest-sha256 c310d3ccbc1c1e50ce9436f217c66ee9934f778985bfef487a034cc3070b79fa --n-boot 1000`. Metrics SHA256 `19130ec61d123af96e739054a36b827ab7451bb4b5ccbf552e7ea32be3291a93`. One original completion subscription and one verified fetch.

Original `evidence_escalation-gate-20261101`v1 failed111.8s because required SMS-fold URLcal lackedbothclasses; **UNMEASURABLE**, no policy/outcome rows. Original evidence-required native1 failed278.5s before threshold freeze because SMS URL extraction could not score every native cal row. The adapter recovery used the original direct URL model interface on identical rows/weights; no native1 SMS/outcome metric exists. Exact failure receipts/tree hashes in STATUS.md.

Private archive maadhavvh/rakshak-policy-audits-20261102v1 verified ownermaadhavvh,isPrivate=true,2229318bytes; manifestSHA394cbd8aec8a859b8e288312aa6abe49df8e2567f0ad521efd3659e821e9808a. Verification and initial processing-time403 in STATUS.md; **HF publish pending**. Row decision parquets/messages are excluded from publication allowlist. No data/weights/secrets in git; no local model/plot/statistical evaluation. Next P1 arm(c) is independently predeclared in TEMPLATE_CALIBRATION.md on its own branch, with all3frozen noKD seed checkpoints and no external fitting.
