# Frozen-rule review audit — 1 November 2026

Rules only. No model weights or fitted thresholds. No held-out improvement claim.

GroundLM and all internal SMS/ScamShield tables: **in-sample: these rows informed rule design**. SmishX alone: **independent, non-Indian-specific**. Independent Indian held-out validation remains **OPEN** (consented, privacy-scrubbed real post-freeze SMS; never tune on it).

## What the audit establishes

- GroundLM: 1,393 released rows, 1,294 policy negatives, 169 hash-only rows, including 161 negatives. All 1,133 scoreable negatives are eval-eligible in this frozen Kaggle data version.
- Exactly **12** negatives have evidence, **11** urgency-only. The urgency downgrade changes warning flags from **12 to 1**; evidence hits remain **12**. HIGH flags remain **1** before and after. The historical 13-total / 12-urgency-only narrative is withdrawn.
- GroundLM warning FPR is **0.0883% in-sample**. Cluster CI: **[0, 0.2971%]**; Wilson score interval: **[0.02%, 0.50%]** (rounded by the job; row-independence assumption).
- Independent SmishX is weak: current warning FPR **53/881 = 6.0159%**, fraud recall **39/226 = 17.2566%**; HIGH FPR **10/881 = 1.1351%**, recall **5/226 = 2.2124%**. The 1% reference FPR gate is not met on SmishX at either cutoff.
- On SmishX, urgency-only downgrading changes warning recall **57/226 to 38/226** and warning FPR **72/881 to 53/881**. Widening then changes recall **38/226 to 39/226**, with no negative warning-count change. Its paired recall delta is **+0.4425 percentage points**, CI **[0, 1.4360] pp**, p=**0.5485**. This does not support an independent improvement claim.

## Protocol and provenance

- Variants: `pre_urgency` = pre-dc223d3 rules + exact pre-f29521a risk ladder; `pre_widening` = the same rules + urgency downgrade; `frozen_current` = current rules and risk ladder.
- `warn`: risk != LOW, including warnings/abstention. `high`: risk == HIGH, the SCAM decision. Every negative is in the FPR denominator, including abstentions.
- Primary labels: known `label_scam` (fraud versus non-fraud). Legacy spam-or-scam results appear separately. Unknown fraud labels are excluded and counted, never filled from a model.
- Original campaign/source_holdout/lang_shift/temporal test memberships are unchanged. The added audit split selects campaigns whose first eight SHA256 bytes for `rule-audit:13:<campaign_id>`, interpreted big-endian, are divisible by five. This selects about 20% of campaigns, not 20% of rows. Both parts remain in-sample.
- Campaign-cluster percentile CIs: 1,000 resamples, seed 13. External-only campaigns use the existing frozen MinHash/domain procedure; approximate grouping is not human campaign annotation. Paired tests resample the same campaigns for both rules, report A−B, and use a two-sided null-centred bootstrap p-value. No multiple-comparison correction; no superiority claim.
- A zero-event bootstrap can yield [0, 0]. The positive Wilson upper bound remains relevant; it is a score interval, not an exact or cluster-aware interval. The GroundLM audit subset has 0/131 warnings: cluster [0, 0], Wilson upper bound 2.85%. It cannot certify a 1% population gate.
- Temporal has only 77 dated positives and known source/campaign caveats. UCI has no known fraud-positive labels, so its fraud recall is undefined.

Private Kaggle run: [rule_review_audit-a23-20261101, kernel version 1](https://www.kaggle.com/code/maadhavvh/rakshak-rule-review-audit-a23-20261101). One event wait, verified fetch, outcome **ok**, elapsed **368.1 s**.

Base: `038bdd33022d` dirty; bundled source tree SHA256 `b1e16e0f1ee9369d0c9811e69ac433490f1da6918649c350bbe14dbca59ccf10`.
Pre-widening rules source: `e2cfa625e3ae48ada2c2358dfee3c3c41ba6dbfe`; SHA256 `c7c3e3185c982b9879f9f45800d6d1ccca06408a646bb584ae62ce7ccd2ce9a3`.

Full machine-readable metrics and ID-only predictions remain in the private Kaggle output. This document only formats saved metrics; all scoring/resampling ran on Kaggle.

## Fraud versus non-fraud (known policy labels)

### Frozen warn threshold

| Cohort | Variant | Negative events/n; FPR [cluster 95% CI] | Positive events/n; recall [cluster 95% CI] | Unknown fraud labels |
|---|---|---|---|---:|
| external_groundlm/eval_eligible | pre_urgency | 12/1133; 1.059% [0.520%, 1.790%] | 11/91; 12.088% [4.895%, 26.038%] | 0 |
| external_groundlm/eval_eligible | pre_widening | 1/1133; 0.088% [0.000%, 0.297%] | 11/91; 12.088% [4.895%, 26.038%] | 0 |
| external_groundlm/eval_eligible | frozen_current | 1/1133; 0.088% [0.000%, 0.297%] | 11/91; 12.088% [4.895%, 26.038%] | 0 |
| external_smishx/eval_eligible | pre_urgency | 72/881; 8.173% [5.176%, 11.734%] | 57/226; 25.221% [15.092%, 36.530%] | 0 |
| external_smishx/eval_eligible | pre_widening | 53/881; 6.016% [3.657%, 8.873%] | 38/226; 16.814% [10.100%, 25.760%] | 0 |
| external_smishx/eval_eligible | frozen_current | 53/881; 6.016% [3.657%, 8.873%] | 39/226; 17.257% [10.612%, 26.001%] | 0 |
| groundlm/all_scoreable | pre_urgency | 12/1133; 1.059% [0.520%, 1.790%] | 0/0; — [—, —] | 0 |
| groundlm/all_scoreable | pre_widening | 1/1133; 0.088% [0.000%, 0.297%] | 0/0; — [—, —] | 0 |
| groundlm/all_scoreable | frozen_current | 1/1133; 0.088% [0.000%, 0.297%] | 0/0; — [—, —] | 0 |
| groundlm/audit | pre_urgency | 3/131; 2.290% [0.000%, 5.038%] | 0/0; — [—, —] | 0 |
| groundlm/audit | pre_widening | 0/131; 0.000% [0.000%, 0.000%] | 0/0; — [—, —] | 0 |
| groundlm/audit | frozen_current | 0/131; 0.000% [0.000%, 0.000%] | 0/0; — [—, —] | 0 |
| groundlm/remainder | pre_urgency | 9/1002; 0.898% [0.344%, 1.712%] | 0/0; — [—, —] | 0 |
| groundlm/remainder | pre_widening | 1/1002; 0.100% [0.000%, 0.377%] | 0/0; — [—, —] | 0 |
| groundlm/remainder | frozen_current | 1/1002; 0.100% [0.000%, 0.377%] | 0/0; — [—, —] | 0 |
| scamshield/all_scoreable | pre_urgency | 883/46944; 1.881% [1.678%, 2.130%] | 2726/5568; 48.958% [43.214%, 55.901%] | 23461 |
| scamshield/all_scoreable | pre_widening | 468/46944; 0.997% [0.852%, 1.183%] | 2580/5568; 46.336% [40.234%, 53.781%] | 23461 |
| scamshield/all_scoreable | frozen_current | 478/46944; 1.018% [0.872%, 1.200%] | 2743/5568; 49.264% [43.396%, 56.344%] | 23461 |
| scamshield/audit | pre_urgency | 176/9248; 1.903% [1.436%, 2.510%] | 447/1012; 44.170% [38.213%, 50.050%] | 4442 |
| scamshield/audit | pre_widening | 81/9248; 0.876% [0.674%, 1.094%] | 419/1012; 41.403% [35.757%, 46.973%] | 4442 |
| scamshield/audit | frozen_current | 82/9248; 0.887% [0.682%, 1.100%] | 458/1012; 45.257% [39.755%, 50.687%] | 4442 |
| scamshield/remainder | pre_urgency | 707/37696; 1.876% [1.661%, 2.113%] | 2279/4556; 50.022% [43.098%, 57.548%] | 19019 |
| scamshield/remainder | pre_widening | 387/37696; 1.027% [0.852%, 1.227%] | 2161/4556; 47.432% [40.296%, 55.193%] | 19019 |
| scamshield/remainder | frozen_current | 396/37696; 1.051% [0.875%, 1.252%] | 2285/4556; 50.154% [43.310%, 57.505%] | 19019 |
| split/campaign/test | pre_urgency | 174/10986; 1.584% [1.367%, 1.821%] | 997/5926; 16.824% [15.875%, 17.854%] | 5233 |
| split/campaign/test | pre_widening | 68/10986; 0.619% [0.474%, 0.767%] | 823/5926; 13.888% [12.994%, 14.774%] | 5233 |
| split/campaign/test | frozen_current | 69/10986; 0.628% [0.483%, 0.782%] | 894/5926; 15.086% [14.211%, 15.998%] | 5233 |
| split/lang_shift/test | pre_urgency | 43/1322; 3.253% [2.361%, 4.331%] | 41/101; 40.594% [27.825%, 53.263%] | 254 |
| split/lang_shift/test | pre_widening | 9/1322; 0.681% [0.296%, 1.155%] | 31/101; 30.693% [19.128%, 42.595%] | 254 |
| split/lang_shift/test | frozen_current | 9/1322; 0.681% [0.296%, 1.155%] | 31/101; 30.693% [19.128%, 42.595%] | 254 |
| split/source_holdout:holdout_mishra_soni_smsphishing/test | pre_urgency | 41/668; 6.138% [4.160%, 8.432%] | 128/273; 46.886% [35.144%, 57.277%] | 0 |
| split/source_holdout:holdout_mishra_soni_smsphishing/test | pre_widening | 15/668; 2.246% [1.046%, 3.687%] | 105/273; 38.462% [27.303%, 49.399%] | 0 |
| split/source_holdout:holdout_mishra_soni_smsphishing/test | frozen_current | 17/668; 2.545% [1.312%, 4.001%] | 105/273; 38.462% [27.303%, 49.399%] | 0 |
| split/source_holdout:holdout_scamshield/test | pre_urgency | 883/46944; 1.881% [1.678%, 2.130%] | 2726/5568; 48.958% [43.214%, 55.901%] | 23461 |
| split/source_holdout:holdout_scamshield/test | pre_widening | 468/46944; 0.997% [0.852%, 1.183%] | 2580/5568; 46.336% [40.234%, 53.781%] | 23461 |
| split/source_holdout:holdout_scamshield/test | frozen_current | 478/46944; 1.018% [0.872%, 1.200%] | 2743/5568; 49.264% [43.396%, 56.344%] | 23461 |
| split/source_holdout:holdout_uci_sms_spam/test | pre_urgency | 55/4517; 1.218% [0.908%, 1.551%] | 0/0; — [—, —] | 642 |
| split/source_holdout:holdout_uci_sms_spam/test | pre_widening | 10/4517; 0.221% [0.110%, 0.376%] | 0/0; — [—, —] | 642 |
| split/source_holdout:holdout_uci_sms_spam/test | frozen_current | 10/4517; 0.221% [0.110%, 0.376%] | 0/0; — [—, —] | 642 |
| split/temporal/test | pre_urgency | 169/10952; 1.543% [1.324%, 1.777%] | 9/77; 11.688% [4.110%, 19.719%] | 0 |
| split/temporal/test | pre_widening | 64/10952; 0.584% [0.447%, 0.740%] | 8/77; 10.390% [3.846%, 18.421%] | 0 |
| split/temporal/test | frozen_current | 64/10952; 0.584% [0.447%, 0.740%] | 8/77; 10.390% [3.846%, 18.421%] | 0 |

### Frozen high threshold

| Cohort | Variant | Negative events/n; FPR [cluster 95% CI] | Positive events/n; recall [cluster 95% CI] | Unknown fraud labels |
|---|---|---|---|---:|
| external_groundlm/eval_eligible | pre_urgency | 1/1133; 0.088% [0.000%, 0.297%] | 1/91; 1.099% [0.000%, 4.413%] | 0 |
| external_groundlm/eval_eligible | pre_widening | 1/1133; 0.088% [0.000%, 0.297%] | 1/91; 1.099% [0.000%, 4.413%] | 0 |
| external_groundlm/eval_eligible | frozen_current | 1/1133; 0.088% [0.000%, 0.297%] | 1/91; 1.099% [0.000%, 4.413%] | 0 |
| external_smishx/eval_eligible | pre_urgency | 10/881; 1.135% [0.534%, 1.878%] | 5/226; 2.212% [0.377%, 5.128%] | 0 |
| external_smishx/eval_eligible | pre_widening | 10/881; 1.135% [0.534%, 1.878%] | 5/226; 2.212% [0.377%, 5.128%] | 0 |
| external_smishx/eval_eligible | frozen_current | 10/881; 1.135% [0.534%, 1.878%] | 5/226; 2.212% [0.377%, 5.128%] | 0 |
| groundlm/all_scoreable | pre_urgency | 1/1133; 0.088% [0.000%, 0.297%] | 0/0; — [—, —] | 0 |
| groundlm/all_scoreable | pre_widening | 1/1133; 0.088% [0.000%, 0.297%] | 0/0; — [—, —] | 0 |
| groundlm/all_scoreable | frozen_current | 1/1133; 0.088% [0.000%, 0.297%] | 0/0; — [—, —] | 0 |
| groundlm/audit | pre_urgency | 0/131; 0.000% [0.000%, 0.000%] | 0/0; — [—, —] | 0 |
| groundlm/audit | pre_widening | 0/131; 0.000% [0.000%, 0.000%] | 0/0; — [—, —] | 0 |
| groundlm/audit | frozen_current | 0/131; 0.000% [0.000%, 0.000%] | 0/0; — [—, —] | 0 |
| groundlm/remainder | pre_urgency | 1/1002; 0.100% [0.000%, 0.377%] | 0/0; — [—, —] | 0 |
| groundlm/remainder | pre_widening | 1/1002; 0.100% [0.000%, 0.377%] | 0/0; — [—, —] | 0 |
| groundlm/remainder | frozen_current | 1/1002; 0.100% [0.000%, 0.377%] | 0/0; — [—, —] | 0 |
| scamshield/all_scoreable | pre_urgency | 87/46944; 0.185% [0.130%, 0.246%] | 951/5568; 17.080% [11.148%, 24.454%] | 23461 |
| scamshield/all_scoreable | pre_widening | 87/46944; 0.185% [0.130%, 0.246%] | 951/5568; 17.080% [11.148%, 24.454%] | 23461 |
| scamshield/all_scoreable | frozen_current | 87/46944; 0.185% [0.130%, 0.246%] | 951/5568; 17.080% [11.148%, 24.454%] | 23461 |
| scamshield/audit | pre_urgency | 16/9248; 0.173% [0.087%, 0.262%] | 103/1012; 10.178% [6.256%, 14.827%] | 4442 |
| scamshield/audit | pre_widening | 16/9248; 0.173% [0.087%, 0.262%] | 103/1012; 10.178% [6.256%, 14.827%] | 4442 |
| scamshield/audit | frozen_current | 16/9248; 0.173% [0.087%, 0.262%] | 103/1012; 10.178% [6.256%, 14.827%] | 4442 |
| scamshield/remainder | pre_urgency | 71/37696; 0.188% [0.122%, 0.272%] | 848/4556; 18.613% [11.366%, 26.874%] | 19019 |
| scamshield/remainder | pre_widening | 71/37696; 0.188% [0.122%, 0.272%] | 848/4556; 18.613% [11.366%, 26.874%] | 19019 |
| scamshield/remainder | frozen_current | 71/37696; 0.188% [0.122%, 0.272%] | 848/4556; 18.613% [11.366%, 26.874%] | 19019 |
| split/campaign/test | pre_urgency | 19/10986; 0.173% [0.100%, 0.263%] | 93/5926; 1.569% [1.231%, 1.925%] | 5233 |
| split/campaign/test | pre_widening | 19/10986; 0.173% [0.100%, 0.263%] | 93/5926; 1.569% [1.231%, 1.925%] | 5233 |
| split/campaign/test | frozen_current | 19/10986; 0.173% [0.100%, 0.263%] | 93/5926; 1.569% [1.231%, 1.925%] | 5233 |
| split/lang_shift/test | pre_urgency | 7/1322; 0.530% [0.151%, 0.979%] | 9/101; 8.911% [1.148%, 18.456%] | 254 |
| split/lang_shift/test | pre_widening | 7/1322; 0.530% [0.151%, 0.979%] | 9/101; 8.911% [1.148%, 18.456%] | 254 |
| split/lang_shift/test | frozen_current | 7/1322; 0.530% [0.151%, 0.979%] | 9/101; 8.911% [1.148%, 18.456%] | 254 |
| split/source_holdout:holdout_mishra_soni_smsphishing/test | pre_urgency | 5/668; 0.749% [0.000%, 1.613%] | 4/273; 1.465% [0.340%, 3.077%] | 0 |
| split/source_holdout:holdout_mishra_soni_smsphishing/test | pre_widening | 5/668; 0.749% [0.000%, 1.613%] | 4/273; 1.465% [0.340%, 3.077%] | 0 |
| split/source_holdout:holdout_mishra_soni_smsphishing/test | frozen_current | 5/668; 0.749% [0.000%, 1.613%] | 4/273; 1.465% [0.340%, 3.077%] | 0 |
| split/source_holdout:holdout_scamshield/test | pre_urgency | 87/46944; 0.185% [0.130%, 0.246%] | 951/5568; 17.080% [11.148%, 24.454%] | 23461 |
| split/source_holdout:holdout_scamshield/test | pre_widening | 87/46944; 0.185% [0.130%, 0.246%] | 951/5568; 17.080% [11.148%, 24.454%] | 23461 |
| split/source_holdout:holdout_scamshield/test | frozen_current | 87/46944; 0.185% [0.130%, 0.246%] | 951/5568; 17.080% [11.148%, 24.454%] | 23461 |
| split/source_holdout:holdout_uci_sms_spam/test | pre_urgency | 3/4517; 0.066% [0.000%, 0.155%] | 0/0; — [—, —] | 642 |
| split/source_holdout:holdout_uci_sms_spam/test | pre_widening | 3/4517; 0.066% [0.000%, 0.155%] | 0/0; — [—, —] | 642 |
| split/source_holdout:holdout_uci_sms_spam/test | frozen_current | 3/4517; 0.066% [0.000%, 0.155%] | 0/0; — [—, —] | 642 |
| split/temporal/test | pre_urgency | 18/10952; 0.164% [0.091%, 0.247%] | 0/77; 0.000% [0.000%, 0.000%] | 0 |
| split/temporal/test | pre_widening | 18/10952; 0.164% [0.091%, 0.247%] | 0/77; 0.000% [0.000%, 0.000%] | 0 |
| split/temporal/test | frozen_current | 18/10952; 0.164% [0.091%, 0.247%] | 0/77; 0.000% [0.000%, 0.000%] | 0 |

## Legacy spam-or-scam labels (not fraud performance)

### Frozen warn threshold

| Cohort | Variant | Negative events/n; FPR [cluster 95% CI] | Positive events/n; recall [cluster 95% CI] | Unknown fraud labels |
|---|---|---|---|---:|
| external_groundlm/eval_eligible | pre_urgency | 12/1133; 1.059% [0.520%, 1.790%] | 11/91; 12.088% [4.895%, 26.038%] | 0 |
| external_groundlm/eval_eligible | pre_widening | 1/1133; 0.088% [0.000%, 0.297%] | 11/91; 12.088% [4.895%, 26.038%] | 0 |
| external_groundlm/eval_eligible | frozen_current | 1/1133; 0.088% [0.000%, 0.297%] | 11/91; 12.088% [4.895%, 26.038%] | 0 |
| external_smishx/eval_eligible | pre_urgency | 30/606; 4.950% [3.311%, 6.732%] | 99/501; 19.760% [12.951%, 26.959%] | 0 |
| external_smishx/eval_eligible | pre_widening | 19/606; 3.135% [1.821%, 4.488%] | 72/501; 14.371% [9.248%, 21.256%] | 0 |
| external_smishx/eval_eligible | frozen_current | 19/606; 3.135% [1.821%, 4.488%] | 73/501; 14.571% [9.445%, 21.429%] | 0 |
| groundlm/all_scoreable | pre_urgency | 12/1133; 1.059% [0.520%, 1.790%] | 0/0; — [—, —] | 0 |
| groundlm/all_scoreable | pre_widening | 1/1133; 0.088% [0.000%, 0.297%] | 0/0; — [—, —] | 0 |
| groundlm/all_scoreable | frozen_current | 1/1133; 0.088% [0.000%, 0.297%] | 0/0; — [—, —] | 0 |
| groundlm/audit | pre_urgency | 3/131; 2.290% [0.000%, 5.038%] | 0/0; — [—, —] | 0 |
| groundlm/audit | pre_widening | 0/131; 0.000% [0.000%, 0.000%] | 0/0; — [—, —] | 0 |
| groundlm/audit | frozen_current | 0/131; 0.000% [0.000%, 0.000%] | 0/0; — [—, —] | 0 |
| groundlm/remainder | pre_urgency | 9/1002; 0.898% [0.344%, 1.712%] | 0/0; — [—, —] | 0 |
| groundlm/remainder | pre_widening | 1/1002; 0.100% [0.000%, 0.377%] | 0/0; — [—, —] | 0 |
| groundlm/remainder | frozen_current | 1/1002; 0.100% [0.000%, 0.377%] | 0/0; — [—, —] | 0 |
| scamshield/all_scoreable | pre_urgency | 883/46944; 1.881% [1.678%, 2.130%] | 7019/29029; 24.179% [22.254%, 26.650%] | 23461 |
| scamshield/all_scoreable | pre_widening | 468/46944; 0.997% [0.852%, 1.183%] | 6024/29029; 20.752% [18.807%, 23.319%] | 23461 |
| scamshield/all_scoreable | frozen_current | 478/46944; 1.018% [0.872%, 1.200%] | 6335/29029; 21.823% [19.862%, 24.340%] | 23461 |
| scamshield/audit | pre_urgency | 176/9248; 1.903% [1.436%, 2.510%] | 1186/5454; 21.746% [19.988%, 23.732%] | 4442 |
| scamshield/audit | pre_widening | 81/9248; 0.876% [0.674%, 1.094%] | 974/5454; 17.858% [16.274%, 19.699%] | 4442 |
| scamshield/audit | frozen_current | 82/9248; 0.887% [0.682%, 1.100%] | 1034/5454; 18.959% [17.392%, 20.741%] | 4442 |
| scamshield/remainder | pre_urgency | 707/37696; 1.876% [1.661%, 2.113%] | 5833/23575; 24.742% [22.394%, 27.904%] | 19019 |
| scamshield/remainder | pre_widening | 387/37696; 1.027% [0.852%, 1.227%] | 5050/23575; 21.421% [19.105%, 24.548%] | 19019 |
| scamshield/remainder | frozen_current | 396/37696; 1.051% [0.875%, 1.252%] | 5301/23575; 22.486% [20.178%, 25.521%] | 19019 |
| split/campaign/test | pre_urgency | 169/10952; 1.543% [1.324%, 1.777%] | 1724/11193; 15.402% [14.685%, 16.109%] | 5233 |
| split/campaign/test | pre_widening | 64/10952; 0.584% [0.447%, 0.740%] | 1390/11193; 12.418% [11.770%, 13.066%] | 5233 |
| split/campaign/test | frozen_current | 64/10952; 0.584% [0.447%, 0.740%] | 1489/11193; 13.303% [12.659%, 13.945%] | 5233 |
| split/lang_shift/test | pre_urgency | 42/1320; 3.182% [2.288%, 4.183%] | 106/357; 29.692% [22.713%, 37.116%] | 254 |
| split/lang_shift/test | pre_widening | 8/1320; 0.606% [0.227%, 1.041%] | 64/357; 17.927% [12.021%, 24.063%] | 254 |
| split/lang_shift/test | frozen_current | 8/1320; 0.606% [0.227%, 1.041%] | 64/357; 17.927% [12.021%, 24.063%] | 254 |
| split/source_holdout:holdout_mishra_soni_smsphishing/test | pre_urgency | 30/524; 5.725% [3.598%, 8.286%] | 139/417; 33.333% [25.557%, 41.818%] | 0 |
| split/source_holdout:holdout_mishra_soni_smsphishing/test | pre_widening | 5/524; 0.954% [0.000%, 2.486%] | 115/417; 27.578% [19.849%, 36.106%] | 0 |
| split/source_holdout:holdout_mishra_soni_smsphishing/test | frozen_current | 5/524; 0.954% [0.000%, 2.486%] | 117/417; 28.058% [20.194%, 36.765%] | 0 |
| split/source_holdout:holdout_scamshield/test | pre_urgency | 883/46944; 1.881% [1.678%, 2.130%] | 7019/29029; 24.179% [22.254%, 26.650%] | 23461 |
| split/source_holdout:holdout_scamshield/test | pre_widening | 468/46944; 0.997% [0.852%, 1.183%] | 6024/29029; 20.752% [18.807%, 23.319%] | 23461 |
| split/source_holdout:holdout_scamshield/test | frozen_current | 478/46944; 1.018% [0.872%, 1.200%] | 6335/29029; 21.823% [19.862%, 24.340%] | 23461 |
| split/source_holdout:holdout_uci_sms_spam/test | pre_urgency | 55/4517; 1.218% [0.908%, 1.551%] | 129/642; 20.093% [15.835%, 24.731%] | 642 |
| split/source_holdout:holdout_uci_sms_spam/test | pre_widening | 10/4517; 0.221% [0.110%, 0.376%] | 94/642; 14.642% [10.673%, 18.941%] | 642 |
| split/source_holdout:holdout_uci_sms_spam/test | frozen_current | 10/4517; 0.221% [0.110%, 0.376%] | 96/642; 14.953% [10.912%, 19.283%] | 642 |
| split/temporal/test | pre_urgency | 169/10952; 1.543% [1.324%, 1.777%] | 9/77; 11.688% [4.110%, 19.719%] | 0 |
| split/temporal/test | pre_widening | 64/10952; 0.584% [0.447%, 0.740%] | 8/77; 10.390% [3.846%, 18.421%] | 0 |
| split/temporal/test | frozen_current | 64/10952; 0.584% [0.447%, 0.740%] | 8/77; 10.390% [3.846%, 18.421%] | 0 |

### Frozen high threshold

| Cohort | Variant | Negative events/n; FPR [cluster 95% CI] | Positive events/n; recall [cluster 95% CI] | Unknown fraud labels |
|---|---|---|---|---:|
| external_groundlm/eval_eligible | pre_urgency | 1/1133; 0.088% [0.000%, 0.297%] | 1/91; 1.099% [0.000%, 4.413%] | 0 |
| external_groundlm/eval_eligible | pre_widening | 1/1133; 0.088% [0.000%, 0.297%] | 1/91; 1.099% [0.000%, 4.413%] | 0 |
| external_groundlm/eval_eligible | frozen_current | 1/1133; 0.088% [0.000%, 0.297%] | 1/91; 1.099% [0.000%, 4.413%] | 0 |
| external_smishx/eval_eligible | pre_urgency | 9/606; 1.485% [0.658%, 2.488%] | 6/501; 1.198% [0.209%, 2.516%] | 0 |
| external_smishx/eval_eligible | pre_widening | 9/606; 1.485% [0.658%, 2.488%] | 6/501; 1.198% [0.209%, 2.516%] | 0 |
| external_smishx/eval_eligible | frozen_current | 9/606; 1.485% [0.658%, 2.488%] | 6/501; 1.198% [0.209%, 2.516%] | 0 |
| groundlm/all_scoreable | pre_urgency | 1/1133; 0.088% [0.000%, 0.297%] | 0/0; — [—, —] | 0 |
| groundlm/all_scoreable | pre_widening | 1/1133; 0.088% [0.000%, 0.297%] | 0/0; — [—, —] | 0 |
| groundlm/all_scoreable | frozen_current | 1/1133; 0.088% [0.000%, 0.297%] | 0/0; — [—, —] | 0 |
| groundlm/audit | pre_urgency | 0/131; 0.000% [0.000%, 0.000%] | 0/0; — [—, —] | 0 |
| groundlm/audit | pre_widening | 0/131; 0.000% [0.000%, 0.000%] | 0/0; — [—, —] | 0 |
| groundlm/audit | frozen_current | 0/131; 0.000% [0.000%, 0.000%] | 0/0; — [—, —] | 0 |
| groundlm/remainder | pre_urgency | 1/1002; 0.100% [0.000%, 0.377%] | 0/0; — [—, —] | 0 |
| groundlm/remainder | pre_widening | 1/1002; 0.100% [0.000%, 0.377%] | 0/0; — [—, —] | 0 |
| groundlm/remainder | frozen_current | 1/1002; 0.100% [0.000%, 0.377%] | 0/0; — [—, —] | 0 |
| scamshield/all_scoreable | pre_urgency | 87/46944; 0.185% [0.130%, 0.246%] | 1930/29029; 6.649% [5.107%, 8.760%] | 23461 |
| scamshield/all_scoreable | pre_widening | 87/46944; 0.185% [0.130%, 0.246%] | 1930/29029; 6.649% [5.107%, 8.760%] | 23461 |
| scamshield/all_scoreable | frozen_current | 87/46944; 0.185% [0.130%, 0.246%] | 1930/29029; 6.649% [5.107%, 8.760%] | 23461 |
| scamshield/audit | pre_urgency | 16/9248; 0.173% [0.087%, 0.262%] | 267/5454; 4.895% [3.935%, 6.077%] | 4442 |
| scamshield/audit | pre_widening | 16/9248; 0.173% [0.087%, 0.262%] | 267/5454; 4.895% [3.935%, 6.077%] | 4442 |
| scamshield/audit | frozen_current | 16/9248; 0.173% [0.087%, 0.262%] | 267/5454; 4.895% [3.935%, 6.077%] | 4442 |
| scamshield/remainder | pre_urgency | 71/37696; 0.188% [0.122%, 0.272%] | 1663/23575; 7.054% [5.238%, 9.615%] | 19019 |
| scamshield/remainder | pre_widening | 71/37696; 0.188% [0.122%, 0.272%] | 1663/23575; 7.054% [5.238%, 9.615%] | 19019 |
| scamshield/remainder | frozen_current | 71/37696; 0.188% [0.122%, 0.272%] | 1663/23575; 7.054% [5.238%, 9.615%] | 19019 |
| split/campaign/test | pre_urgency | 18/10952; 0.164% [0.091%, 0.247%] | 225/11193; 2.010% [1.735%, 2.292%] | 5233 |
| split/campaign/test | pre_widening | 18/10952; 0.164% [0.091%, 0.247%] | 225/11193; 2.010% [1.735%, 2.292%] | 5233 |
| split/campaign/test | frozen_current | 18/10952; 0.164% [0.091%, 0.247%] | 225/11193; 2.010% [1.735%, 2.292%] | 5233 |
| split/lang_shift/test | pre_urgency | 6/1320; 0.455% [0.150%, 0.837%] | 22/357; 6.162% [2.467%, 10.303%] | 254 |
| split/lang_shift/test | pre_widening | 6/1320; 0.455% [0.150%, 0.837%] | 22/357; 6.162% [2.467%, 10.303%] | 254 |
| split/lang_shift/test | frozen_current | 6/1320; 0.455% [0.150%, 0.837%] | 22/357; 6.162% [2.467%, 10.303%] | 254 |
| split/source_holdout:holdout_mishra_soni_smsphishing/test | pre_urgency | 1/524; 0.191% [0.000%, 0.731%] | 8/417; 1.918% [0.638%, 3.601%] | 0 |
| split/source_holdout:holdout_mishra_soni_smsphishing/test | pre_widening | 1/524; 0.191% [0.000%, 0.731%] | 8/417; 1.918% [0.638%, 3.601%] | 0 |
| split/source_holdout:holdout_mishra_soni_smsphishing/test | frozen_current | 1/524; 0.191% [0.000%, 0.731%] | 8/417; 1.918% [0.638%, 3.601%] | 0 |
| split/source_holdout:holdout_scamshield/test | pre_urgency | 87/46944; 0.185% [0.130%, 0.246%] | 1930/29029; 6.649% [5.107%, 8.760%] | 23461 |
| split/source_holdout:holdout_scamshield/test | pre_widening | 87/46944; 0.185% [0.130%, 0.246%] | 1930/29029; 6.649% [5.107%, 8.760%] | 23461 |
| split/source_holdout:holdout_scamshield/test | frozen_current | 87/46944; 0.185% [0.130%, 0.246%] | 1930/29029; 6.649% [5.107%, 8.760%] | 23461 |
| split/source_holdout:holdout_uci_sms_spam/test | pre_urgency | 3/4517; 0.066% [0.000%, 0.155%] | 0/642; 0.000% [0.000%, 0.000%] | 642 |
| split/source_holdout:holdout_uci_sms_spam/test | pre_widening | 3/4517; 0.066% [0.000%, 0.155%] | 0/642; 0.000% [0.000%, 0.000%] | 642 |
| split/source_holdout:holdout_uci_sms_spam/test | frozen_current | 3/4517; 0.066% [0.000%, 0.155%] | 0/642; 0.000% [0.000%, 0.000%] | 642 |
| split/temporal/test | pre_urgency | 18/10952; 0.164% [0.091%, 0.247%] | 0/77; 0.000% [0.000%, 0.000%] | 0 |
| split/temporal/test | pre_widening | 18/10952; 0.164% [0.091%, 0.247%] | 0/77; 0.000% [0.000%, 0.000%] | 0 |
| split/temporal/test | frozen_current | 18/10952; 0.164% [0.091%, 0.247%] | 0/77; 0.000% [0.000%, 0.000%] | 0 |

## Paired warning-rate changes (known fraud policy labels)

All deltas are percentage points, after minus before. Negative-side deltas describe FPR; positive-side deltas describe recall. Exposure caveats above apply to every internal/GroundLM row.

| Cohort | Change | FPR delta [95% CI]; p | Recall delta [95% CI]; p |
|---|---|---|---|
| external_groundlm/eval_eligible | frozen_current_minus_pre_widening | 0.000 pp [0.000 pp, 0.000 pp]; 1 | 0.000 pp [0.000 pp, 0.000 pp]; 1 |
| external_groundlm/eval_eligible | pre_widening_minus_pre_urgency | -0.971 pp [-1.653 pp, -0.451 pp]; 0.004995 | 0.000 pp [0.000 pp, 0.000 pp]; 1 |
| external_smishx/eval_eligible | frozen_current_minus_pre_widening | 0.000 pp [0.000 pp, 0.000 pp]; 1 | 0.442 pp [0.000 pp, 1.436 pp]; 0.5485 |
| external_smishx/eval_eligible | pre_widening_minus_pre_urgency | -2.157 pp [-3.648 pp, -0.923 pp]; 0.004995 | -8.407 pp [-17.158 pp, -1.374 pp]; 0.03097 |
| groundlm/all_scoreable | frozen_current_minus_pre_widening | 0.000 pp [0.000 pp, 0.000 pp]; 1 | — [—, —]; — |
| groundlm/all_scoreable | pre_widening_minus_pre_urgency | -0.971 pp [-1.653 pp, -0.451 pp]; 0.004995 | — [—, —]; — |
| groundlm/audit | frozen_current_minus_pre_widening | 0.000 pp [0.000 pp, 0.000 pp]; 1 | — [—, —]; — |
| groundlm/audit | pre_widening_minus_pre_urgency | -2.290 pp [-5.038 pp, 0.000 pp]; 0.0979 | — [—, —]; — |
| groundlm/remainder | frozen_current_minus_pre_widening | 0.000 pp [0.000 pp, 0.000 pp]; 1 | — [—, —]; — |
| groundlm/remainder | pre_widening_minus_pre_urgency | -0.798 pp [-1.544 pp, -0.296 pp]; 0.01898 | — [—, —]; — |
| scamshield/all_scoreable | frozen_current_minus_pre_widening | 0.021 pp [0.009 pp, 0.034 pp]; 0.002997 | 2.927 pp [2.271 pp, 3.634 pp]; 0.000999 |
| scamshield/all_scoreable | pre_widening_minus_pre_urgency | -0.884 pp [-1.040 pp, -0.765 pp]; 0.000999 | -2.622 pp [-3.643 pp, -1.807 pp]; 0.000999 |
| scamshield/audit | frozen_current_minus_pre_widening | 0.011 pp [0.000 pp, 0.033 pp]; 0.5405 | 3.854 pp [2.375 pp, 5.766 pp]; 0.000999 |
| scamshield/audit | pre_widening_minus_pre_urgency | -1.027 pp [-1.625 pp, -0.642 pp]; 0.001998 | -2.767 pp [-4.450 pp, -1.395 pp]; 0.000999 |
| scamshield/remainder | frozen_current_minus_pre_widening | 0.024 pp [0.011 pp, 0.040 pp]; 0.004995 | 2.722 pp [2.076 pp, 3.503 pp]; 0.000999 |
| scamshield/remainder | pre_widening_minus_pre_urgency | -0.849 pp [-0.952 pp, -0.746 pp]; 0.000999 | -2.590 pp [-3.714 pp, -1.714 pp]; 0.000999 |
| split/campaign/test | frozen_current_minus_pre_widening | 0.009 pp [0.000 pp, 0.027 pp]; 0.5415 | 1.198 pp [0.912 pp, 1.472 pp]; 0.000999 |
| split/campaign/test | pre_widening_minus_pre_urgency | -0.965 pp [-1.158 pp, -0.791 pp]; 0.000999 | -2.936 pp [-3.356 pp, -2.532 pp]; 0.000999 |
| split/lang_shift/test | frozen_current_minus_pre_widening | 0.000 pp [0.000 pp, 0.000 pp]; 1 | 0.000 pp [0.000 pp, 0.000 pp]; 1 |
| split/lang_shift/test | pre_widening_minus_pre_urgency | -2.572 pp [-3.433 pp, -1.757 pp]; 0.000999 | -9.901 pp [-16.982 pp, -4.299 pp]; 0.003996 |
| split/source_holdout:holdout_mishra_soni_smsphishing/test | frozen_current_minus_pre_widening | 0.299 pp [0.000 pp, 0.753 pp]; 0.2198 | 0.000 pp [0.000 pp, 0.000 pp]; 1 |
| split/source_holdout:holdout_mishra_soni_smsphishing/test | pre_widening_minus_pre_urgency | -3.892 pp [-5.498 pp, -2.362 pp]; 0.000999 | -8.425 pp [-13.799 pp, -3.788 pp]; 0.004995 |
| split/source_holdout:holdout_scamshield/test | frozen_current_minus_pre_widening | 0.021 pp [0.009 pp, 0.034 pp]; 0.002997 | 2.927 pp [2.271 pp, 3.634 pp]; 0.000999 |
| split/source_holdout:holdout_scamshield/test | pre_widening_minus_pre_urgency | -0.884 pp [-1.040 pp, -0.765 pp]; 0.000999 | -2.622 pp [-3.643 pp, -1.807 pp]; 0.000999 |
| split/source_holdout:holdout_uci_sms_spam/test | frozen_current_minus_pre_widening | 0.000 pp [0.000 pp, 0.000 pp]; 1 | — [—, —]; — |
| split/source_holdout:holdout_uci_sms_spam/test | pre_widening_minus_pre_urgency | -0.996 pp [-1.283 pp, -0.708 pp]; 0.000999 | — [—, —]; — |
| split/temporal/test | frozen_current_minus_pre_widening | 0.000 pp [0.000 pp, 0.000 pp]; 1 | 0.000 pp [0.000 pp, 0.000 pp]; 1 |
| split/temporal/test | pre_widening_minus_pre_urgency | -0.959 pp [-1.141 pp, -0.782 pp]; 0.000999 | -1.299 pp [-4.000 pp, 0.000 pp]; 0.5584 |

## Widening: fraud evidence coverage and warning tradeoff

All entries below are **in-sample: these rows informed rule design** except SmishX (**independent, non-Indian-specific**). Evidence coverage is not detector recall.

| Cohort | Pre-widening fraud evidence [95% CI] | Current fraud evidence [95% CI] | Paired coverage delta [95% CI]; p |
|---|---|---|---|
| external_groundlm/eval_eligible | 11/91; 12.088% [4.895%, 26.038%] | 11/91; 12.088% [4.895%, 26.038%] | 0.000 pp [0.000 pp, 0.000 pp]; 1 |
| external_smishx/eval_eligible | 57/226; 25.221% [15.092%, 36.530%] | 58/226; 25.664% [15.625%, 36.714%] | 0.442 pp [0.000 pp, 1.436 pp]; 0.5485 |
| groundlm/all_scoreable | 0/0; — [—, —] | 0/0; — [—, —] | — [—, —]; — |
| groundlm/audit | 0/0; — [—, —] | 0/0; — [—, —] | — [—, —]; — |
| groundlm/remainder | 0/0; — [—, —] | 0/0; — [—, —] | — [—, —]; — |
| scamshield/all_scoreable | 2726/5568; 48.958% [43.214%, 55.901%] | 2877/5568; 51.670% [46.212%, 58.514%] | 2.712 pp [2.082 pp, 3.366 pp]; 0.000999 |
| scamshield/audit | 447/1012; 44.170% [38.213%, 50.050%] | 485/1012; 47.925% [42.467%, 53.471%] | 3.755 pp [2.272 pp, 5.631 pp]; 0.000999 |
| scamshield/remainder | 2279/4556; 50.022% [43.098%, 57.548%] | 2392/4556; 52.502% [45.993%, 59.702%] | 2.480 pp [1.870 pp, 3.206 pp]; 0.000999 |
| split/campaign/test | 997/5926; 16.824% [15.875%, 17.854%] | 1063/5926; 17.938% [16.951%, 18.985%] | 1.114 pp [0.844 pp, 1.384 pp]; 0.000999 |
| split/lang_shift/test | 41/101; 40.594% [27.825%, 53.263%] | 41/101; 40.594% [27.825%, 53.263%] | 0.000 pp [0.000 pp, 0.000 pp]; 1 |
| split/source_holdout:holdout_mishra_soni_smsphishing/test | 128/273; 46.886% [35.144%, 57.277%] | 128/273; 46.886% [35.144%, 57.277%] | 0.000 pp [0.000 pp, 0.000 pp]; 1 |
| split/source_holdout:holdout_scamshield/test | 2726/5568; 48.958% [43.214%, 55.901%] | 2877/5568; 51.670% [46.212%, 58.514%] | 2.712 pp [2.082 pp, 3.366 pp]; 0.000999 |
| split/source_holdout:holdout_uci_sms_spam/test | 0/0; — [—, —] | 0/0; — [—, —] | — [—, —]; — |
| split/temporal/test | 9/77; 11.688% [4.110%, 19.719%] | 9/77; 11.688% [4.110%, 19.719%] | 0.000 pp [0.000 pp, 0.000 pp]; 1 |

## Input fingerprints

| Input | SHA256 |
|---|---|
| external/external_groundlm.parquet | `979aaca4c6a922912d223515ec6a3373276a7f5308aaa8d72e17b9f092850f73` |
| external/external_smishx.parquet | `872d08e8885e101b33ac83732ec776e36d8cf183a669733baaf15165d3ec7780` |
| sms.parquet | `430ded382b22ea1c3c14469c80ff9b34a3214ab82257e89be20d67439053e135` |
| splits/campaign.json | `3e911c50c071a5a4f8b3749ee62cd9176f750cf69dbf60e174f8638663491f36` |
| splits/lang_shift.json | `3009c13da634f394438f47d0f79eb3aa33fde6eee0f70a7d727ffb282f43f15f` |
| splits/legacy_mishra_soni.json | `5c3ac7adabd4e1cd96f7725a7214b962f453f3e87ab528c56a37c9386fa158fc` |
| splits/legacy_scamshield.json | `bb5b98207ee9ba8f86d1a328be7efe0964c0e72fd778249693188a6639598acf` |
| splits/legacy_uci_sms_spam.json | `33945e70a740f425398bd40ab372b18909f7f13b2514631f4c84fb17fefb8143` |
| splits/source_holdout.json | `019d428d1e8085ad6e4f8dba243b8715a2457e07d23906a332fd3414dde58ff2` |
| splits/temporal.json | `841e3dedf7ad9b3f0cba066b7907e1ecb7c3030eddeda2f49817758bf8e7a0c3` |
| splits/url_domain.json | `170a810a0071d65edf2718607b0e88e94ef81ab9aa06a541cf1feecd164f5a47` |
| splits/url_phreshphish.json | `58d8c8c81866491db4f54b1db9a46c70b059b5f261e78be4518d3c5e05640bc5` |

## Decision

Keep measurement/provenance corrections and publish all negative tradeoffs. The in-sample GroundLM warning endpoint remains below 1%; it is not independent safety evidence. SmishX does not support a low-FPR/high-recall rules-only claim, and widening has no established independent recall benefit. Independent Indian validation and model-side GroundLM remain OPEN. PhreshPhish is unchanged and was not rerun: this milestone modifies no URL detector.
