# Transactional student false-alarm diagnosis — 2026-11-01

**Keep the measurement; the causal real-Indian-data-gap hypothesis remains UNVERIFIED.** These are overlapping lexical/structure proxies, not human transaction-family labels or proof of authentic Indian origin. No message examples were inspected or emitted. No model, threshold or label changed. The next cal-only fusion candidate was fixed before these diagnostic outcomes. This is not a fused-system measurement.

Private `sms_transactional_audit-p1-diag-20261101` v1, tree `c16875190b4cf0251769663a12f35e80034554bc9134664a50ea3f44b787d504`, ok 81.8s, `--n-boot 1000`; one wait/fetch. Source model scores tree `62a788d27aa9db20b84964f00b8208d3025c3be8f5128352d5c76c53b5ae589e`, source metrics SHA `db3811b2ede62d577e3dad40684276d0ec0580d231d0505cafe3d5d54fb46538`. Diagnostic metrics SHA `79774990ad4f6ab3e63d36d4596f1985b770bbe6bd6a3a7373e5438f6554d645`.

GroundLM is in-sample for rule-feature design. GroundLM/SmishX aggregate FPR feedback motivated this new task/intervention selection, so subsequent feedback-guided changes cannot claim untouched validation on either set. SmishX is non-Indian-specific. Independent Indian validation remains OPEN. Every exact eligible saved-score ID and original cal-1%-FPR threshold was checked; the calibration target is not an external guarantee.

## Frozen campaign training population

Rows 76,591; known fraud negatives 37,645; unknown-policy rows 16,367. Counts are a fixed training census, not sampled CIs.

| Source | Known fraud negatives |
|---|---:|
| anmol_scam_ham_india | 69 |
| bolewara_hinglish_scam | 40 |
| mishra_soni_smsphishing | 388 |
| scamshield | 34,421 |
| uci_sms_spam | 2,727 |

Source names do not certify country or real-world provenance. No claim that Indian-named sources are independently verified real benign messages.

| Training proxy | Known fraud negatives | Share of known negatives |
|---|---:|---:|
| bank | 1,554 | 4.13% |
| delivery | 125 | 0.33% |
| digits:<10pct | 6,637 | 17.63% |
| digits:>=10pct | 2,648 | 7.03% |
| digits:none | 28,360 | 75.34% |
| do_not_share | 321 | 0.85% |
| length:80-159 | 11,100 | 29.49% |
| length:<80 | 24,429 | 64.89% |
| length:>=160 | 2,116 | 5.62% |
| otp | 641 | 1.70% |
| otp_protective_no_url | 315 | 0.84% |
| sender:missing | 37,645 | 100.00% |
| telecom_utility | 345 | 0.92% |
| upi | 317 | 0.84% |
| urls:absent | 36,720 | 97.54% |
| urls:present | 925 | 2.46% |

Sender metadata is missing for every retained training and external message. No sender-style conclusion is supported. Categories overlap; never add their counts. Keyword absence is not evidence that a message is non-transactional.

## All external seeds and proxy cohorts

Rates below are false alarms among **all known fraud negatives in that proxy cohort**, including promotions. CIs resample approximate external campaign groups,1000 draws/seed13. Small group counts can make CIs fragile. All-zero/one bootstrap CIs can collapse; Wilson references assume IID messages and do not certify clustered population risk. An absent category is unmeasured, not a zero false-alarm finding.

### groundlm

| Seed | Frozen threshold | All-negative false alarms |
|---|---:|---:|
| 1 | 0.944833279 | 407/1133 |
| 2 | 0.930649340 | 415/1133 |
| 3 | 0.962241292 | 406/1133 |

#### Seed 1

| Proxy | FP / negatives | FPR | Campaign95% CI | Campaigns | IID Wilson95% |
|---|---:|---:|---|---:|---|
| bank | 195/201 | 97.01% | [95.18%, 98.92%] | 7 | [93.64%, 98.62%] |
| delivery | 51/71 | 71.83% | [2.08%, 95.33%] | 9 | [60.46%, 80.96%] |
| digits:<10pct | 174/326 | 53.37% | [19.19%, 73.86%] | 123 | [47.95%, 58.72%] |
| digits:>=10pct | 231/256 | 90.23% | [76.63%, 96.36%] | 15 | [85.98%, 93.30%] |
| digits:none | 2/551 | 0.36% | [0.00%, 0.96%] | 513 | [0.10%, 1.31%] |
| length:80-159 | 318/467 | 68.09% | [48.56%, 79.02%] | 153 | [63.73%, 72.16%] |
| length:<80 | 89/635 | 14.02% | [2.61%, 26.75%] | 468 | [11.53%, 16.93%] |
| length:>=160 | 0/31 | 0.00% | [0.00%, 0.00%] | 31 | [0.00%, 11.03%] |
| otp | 0/1 | 0.00% | [0.00%, 0.00%] | 1 | [0.00%, 79.35%] |
| sender:missing | 407/1133 | 35.92% | [21.80%, 48.54%] | 650 | [33.18%, 38.76%] |
| telecom_utility | 87/93 | 93.55% | [69.23%, 100.00%] | 9 | [86.63%, 97.01%] |
| upi | 70/74 | 94.59% | [91.18%, 97.50%] | 2 | [86.91%, 97.88%] |
| urls:absent | 406/1111 | 36.54% | [20.98%, 48.89%] | 643 | [33.76%, 39.42%] |
| urls:present | 1/22 | 4.55% | [0.00%, 23.08%] | 7 | [0.81%, 21.80%] |

#### Seed 2

| Proxy | FP / negatives | FPR | Campaign95% CI | Campaigns | IID Wilson95% |
|---|---:|---:|---|---:|---|
| bank | 170/201 | 84.58% | [65.89%, 97.10%] | 7 | [78.94%, 88.92%] |
| delivery | 49/71 | 69.01% | [8.33%, 89.72%] | 9 | [57.52%, 78.56%] |
| digits:<10pct | 201/326 | 61.66% | [32.83%, 76.52%] | 123 | [56.27%, 66.77%] |
| digits:>=10pct | 212/256 | 82.81% | [68.35%, 93.37%] | 15 | [77.71%, 86.94%] |
| digits:none | 2/551 | 0.36% | [0.00%, 0.96%] | 513 | [0.10%, 1.31%] |
| length:80-159 | 312/467 | 66.81% | [47.49%, 77.24%] | 153 | [62.42%, 70.93%] |
| length:<80 | 103/635 | 16.22% | [2.75%, 29.76%] | 468 | [13.56%, 19.29%] |
| length:>=160 | 0/31 | 0.00% | [0.00%, 0.00%] | 31 | [0.00%, 11.03%] |
| otp | 0/1 | 0.00% | [0.00%, 0.00%] | 1 | [0.00%, 79.35%] |
| sender:missing | 415/1133 | 36.63% | [23.19%, 48.65%] | 650 | [33.87%, 39.47%] |
| telecom_utility | 84/93 | 90.32% | [86.36%, 100.00%] | 9 | [82.62%, 94.82%] |
| upi | 47/74 | 63.51% | [45.00%, 85.29%] | 2 | [52.13%, 73.56%] |
| urls:absent | 411/1111 | 36.99% | [21.65%, 48.85%] | 643 | [34.20%, 39.87%] |
| urls:present | 4/22 | 18.18% | [0.00%, 69.23%] | 7 | [7.31%, 38.52%] |

#### Seed 3

| Proxy | FP / negatives | FPR | Campaign95% CI | Campaigns | IID Wilson95% |
|---|---:|---:|---|---:|---|
| bank | 174/201 | 86.57% | [63.23%, 100.00%] | 7 | [81.16%, 90.60%] |
| delivery | 31/71 | 43.66% | [0.00%, 55.77%] | 9 | [32.75%, 55.23%] |
| digits:<10pct | 192/326 | 58.90% | [27.58%, 75.00%] | 123 | [53.48%, 64.10%] |
| digits:>=10pct | 212/256 | 82.81% | [64.64%, 97.48%] | 15 | [77.71%, 86.94%] |
| digits:none | 2/551 | 0.36% | [0.00%, 0.96%] | 513 | [0.10%, 1.31%] |
| length:80-159 | 289/467 | 61.88% | [40.86%, 73.78%] | 153 | [57.40%, 66.18%] |
| length:<80 | 117/635 | 18.43% | [2.17%, 34.71%] | 468 | [15.60%, 21.63%] |
| length:>=160 | 0/31 | 0.00% | [0.00%, 0.00%] | 31 | [0.00%, 11.03%] |
| otp | 0/1 | 0.00% | [0.00%, 0.00%] | 1 | [0.00%, 79.35%] |
| sender:missing | 406/1133 | 35.83% | [21.47%, 48.00%] | 650 | [33.09%, 38.67%] |
| telecom_utility | 58/93 | 62.37% | [56.90%, 81.89%] | 9 | [52.21%, 71.54%] |
| upi | 37/74 | 50.00% | [37.50%, 64.71%] | 2 | [38.89%, 61.11%] |
| urls:absent | 401/1111 | 36.09% | [20.27%, 48.26%] | 643 | [33.32%, 38.96%] |
| urls:present | 5/22 | 22.73% | [0.00%, 69.23%] | 7 | [10.12%, 43.44%] |
### smishx

| Seed | Frozen threshold | All-negative false alarms |
|---|---:|---:|
| 1 | 0.944833279 | 267/881 |
| 2 | 0.930649340 | 282/881 |
| 3 | 0.962241292 | 296/881 |

#### Seed 1

| Proxy | FP / negatives | FPR | Campaign95% CI | Campaigns | IID Wilson95% |
|---|---:|---:|---|---:|---|
| bank | 17/22 | 77.27% | [59.09%, 95.45%] | 22 | [56.56%, 89.88%] |
| delivery | 4/8 | 50.00% | [14.29%, 81.82%] | 7 | [21.52%, 78.48%] |
| digits:<10pct | 206/349 | 59.03% | [51.53%, 66.41%] | 249 | [53.80%, 64.06%] |
| digits:>=10pct | 42/55 | 76.36% | [60.37%, 88.71%] | 33 | [63.65%, 85.63%] |
| digits:none | 19/477 | 3.98% | [1.90%, 6.42%] | 473 | [2.56%, 6.14%] |
| do_not_share | 3/4 | 75.00% | [25.00%, 100.00%] | 4 | [30.06%, 95.44%] |
| length:80-159 | 205/408 | 50.25% | [43.07%, 56.47%] | 303 | [45.41%, 55.07%] |
| length:<80 | 9/382 | 2.36% | [0.53%, 5.08%] | 376 | [1.24%, 4.42%] |
| length:>=160 | 53/91 | 58.24% | [45.24%, 68.94%] | 77 | [47.98%, 67.84%] |
| otp | 6/6 | 100.00% | [100.00%, 100.00%] | 6 | [60.97%, 100.00%] |
| otp_protective_no_url | 2/2 | 100.00% | [100.00%, 100.00%] | 2 | [34.24%, 100.00%] |
| sender:missing | 267/881 | 30.31% | [25.81%, 34.67%] | 753 | [27.36%, 33.42%] |
| telecom_utility | 6/7 | 85.71% | [50.00%, 100.00%] | 6 | [48.69%, 97.43%] |
| urls:absent | 16/551 | 2.90% | [1.63%, 4.36%] | 550 | [1.80%, 4.66%] |
| urls:present | 251/330 | 76.06% | [69.58%, 82.34%] | 203 | [71.17%, 80.35%] |

#### Seed 2

| Proxy | FP / negatives | FPR | Campaign95% CI | Campaigns | IID Wilson95% |
|---|---:|---:|---|---:|---|
| bank | 17/22 | 77.27% | [59.09%, 95.45%] | 22 | [56.56%, 89.88%] |
| delivery | 3/8 | 37.50% | [10.00%, 85.71%] | 7 | [13.68%, 69.43%] |
| digits:<10pct | 222/349 | 63.61% | [56.77%, 70.06%] | 249 | [58.44%, 68.48%] |
| digits:>=10pct | 45/55 | 81.82% | [66.67%, 93.34%] | 33 | [69.67%, 89.81%] |
| digits:none | 15/477 | 3.14% | [1.27%, 5.60%] | 473 | [1.91%, 5.12%] |
| do_not_share | 3/4 | 75.00% | [25.00%, 100.00%] | 4 | [30.06%, 95.44%] |
| length:80-159 | 220/408 | 53.92% | [47.24%, 60.23%] | 303 | [49.07%, 58.70%] |
| length:<80 | 11/382 | 2.88% | [0.80%, 5.67%] | 376 | [1.62%, 5.08%] |
| length:>=160 | 51/91 | 56.04% | [43.67%, 68.04%] | 77 | [45.81%, 65.79%] |
| otp | 5/6 | 83.33% | [50.00%, 100.00%] | 6 | [43.65%, 96.99%] |
| otp_protective_no_url | 2/2 | 100.00% | [100.00%, 100.00%] | 2 | [34.24%, 100.00%] |
| sender:missing | 282/881 | 32.01% | [27.15%, 36.38%] | 753 | [29.01%, 35.16%] |
| telecom_utility | 5/7 | 71.43% | [33.33%, 100.00%] | 6 | [35.89%, 91.78%] |
| urls:absent | 12/551 | 2.18% | [1.09%, 3.62%] | 550 | [1.25%, 3.77%] |
| urls:present | 270/330 | 81.82% | [75.65%, 87.43%] | 203 | [77.30%, 85.61%] |

#### Seed 3

| Proxy | FP / negatives | FPR | Campaign95% CI | Campaigns | IID Wilson95% |
|---|---:|---:|---|---:|---|
| bank | 18/22 | 81.82% | [63.64%, 95.45%] | 22 | [61.48%, 92.69%] |
| delivery | 6/8 | 75.00% | [50.00%, 100.00%] | 7 | [40.93%, 92.85%] |
| digits:<10pct | 228/349 | 65.33% | [58.86%, 71.66%] | 249 | [60.19%, 70.13%] |
| digits:>=10pct | 49/55 | 89.09% | [78.72%, 96.55%] | 33 | [78.17%, 94.90%] |
| digits:none | 19/477 | 3.98% | [2.11%, 6.09%] | 473 | [2.56%, 6.14%] |
| do_not_share | 3/4 | 75.00% | [25.00%, 100.00%] | 4 | [30.06%, 95.44%] |
| length:80-159 | 237/408 | 58.09% | [51.41%, 64.17%] | 303 | [53.25%, 62.78%] |
| length:<80 | 10/382 | 2.62% | [0.53%, 5.84%] | 376 | [1.43%, 4.75%] |
| length:>=160 | 49/91 | 53.85% | [40.50%, 65.10%] | 77 | [43.66%, 63.72%] |
| otp | 6/6 | 100.00% | [100.00%, 100.00%] | 6 | [60.97%, 100.00%] |
| otp_protective_no_url | 2/2 | 100.00% | [100.00%, 100.00%] | 2 | [34.24%, 100.00%] |
| sender:missing | 296/881 | 33.60% | [28.65%, 38.08%] | 753 | [30.56%, 36.78%] |
| telecom_utility | 5/7 | 71.43% | [33.33%, 100.00%] | 6 | [35.89%, 91.78%] |
| urls:absent | 12/551 | 2.18% | [1.09%, 3.45%] | 550 | [1.25%, 3.77%] |
| urls:present | 284/330 | 86.06% | [81.42%, 90.32%] | 203 | [81.91%, 89.38%] |

The cal-only baseline/candidate and deployed-state audit are separate. No shipped fusion model/threshold has yet been located. No benchmark row was copied into synthetic templates. Future training augmentation may use independently sourced semantics only and remains training-only.
