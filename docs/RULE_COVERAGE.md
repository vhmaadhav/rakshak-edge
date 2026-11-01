# Rule coverage on ScamShield

**in-sample: these rows informed rule design.** dc223d3 was designed using the sampled
and full ScamShield corpus, including official test rows. The historical “independently
verified” measurements below mean rerunning the implementation, not independent validation.
The 2026-11-01 frozen-rule audit compares old/new on campaign-disjoint retrospective
subsets without changing the original splits. SmishX is the sole **independent,
non-Indian-specific** validation; no held-out improvement is claimed. The exposure register
and the OPEN post-freeze Indian data collection plan are in `docs/GROUNDLM_FPR.md`.

Frozen Kaggle results (2026-11-01; full tables/CIs/paired tests in `docs/REVIEW_AUDIT.md`):

| Population | Before widening | After widening | Interpretation |
|---|---|---|---|
| ScamShield audit, known fraud with evidence | 447/1,012 (44.170%) | 485/1,012 (47.925%) | In-sample |
| ScamShield audit, warning recall | 419/1,012 (41.403%) | 458/1,012 (45.257%) | In-sample |
| ScamShield audit, negative warning flags | 81/9,248 (0.876%) | 82/9,248 (0.887%) | In-sample false-positive cost |
| SmishX, fraud warning recall | 38/226 (16.814%) | 39/226 (17.257%) | Independent, non-Indian-specific; delta CI spans zero |
| SmishX, negative warning flags | 53/881 (6.016%) | 53/881 (6.016%) | Independent, non-Indian-specific; high FPR |

The ScamShield audit is eval-eligible only, so it excludes duplicates and known synthetic
rows. It is not comparable to the historical full raw-corpus denominator below. HIGH/SCAM
decisions are unchanged by widening. No independent improvement is established.

How often the deterministic rules in `src/rakshak/rules.py` stay silent on the real ScamShield
data that ships in `data/raw/hf_sidzzz07_scamshield/` (85,680 rows: 54,598 benign, 24,312
promotion, 6,770 genuine fraud).

Reproduce with:

```bash
.venv/bin/python scripts/rule_coverage.py                 # 3,000-row sample, seed 13
.venv/bin/python scripts/rule_coverage.py --n 85680       # whole corpus
.venv/bin/python scripts/rule_coverage.py --compare-git HEAD   # working tree vs HEAD
```

## Slices and metrics

* `fraud` - `is_scam == 1` and intent is not "General Spam / Telemarketing".
* `promotion` - `is_scam == 1` and intent is "General Spam / Telemarketing". A promotion that
  fires no rule is correct: spam is not scam.
* `benign` - `is_scam == 0`.

"Firing a rule" and "the detector decides" are different, so three metrics are reported:

1. **no evidence** - the old coverage metric: at least one `Evidence` item, or none.
2. **>=UNCERTAIN** - `package.risk` is UNCERTAIN, MEDIUM or HIGH. The package is built with
   `scam_state=None`, so `rakshak.ela_eval.proxy_scam_state` supplies the label-free fusion
   stand-in (a weight >= 0.6 hit gives "scam", any weaker hit "uncertain", no hit "benign").
3. **>=MEDIUM** - `package.risk` is MEDIUM or HIGH.

`urgency`-only messages are LOW risk (see `evidence._risk`), so they count under metric 1 but not
under metrics 2 and 3. That is why benign metric 1 (2.43%) and metric 2 (1.38%) differ.

Label noise to keep in mind when reading the fraud rows: "Loan Scam" (84.7% empty, 12.0%
>=MEDIUM) and "UPI Fraud" (80.1% empty) are mostly genuine bank/cashback/payment promotions that
the source mislabelled. Their emptiness is not a rule defect, and widening rules to catch them
would fire on legitimate bank SMS.

## BEFORE baseline: 3,000-row sample (seed 13)

| slice | n | no evidence | rate | >=UNCERTAIN | rate | >=MEDIUM | rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fraud | 266 | 104 | 39.10% | 147 | 55.26% | 145 | 54.51% |
| promotion | 843 | 679 | 80.55% | 121 | 14.35% | 120 | 14.24% |
| benign | 1891 | 1845 | 97.57% | 26 | 1.38% | 26 | 1.38% |

### Baseline by fraud intent (3,000-row sample)

| fraud intent | n | no evidence | empty rate | >=UNCERTAIN | >=MEDIUM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bank Account Freeze | 9 | 1 | 11.11% | 88.89% | 66.67% |
| Courier Scam | 36 | 22 | 61.11% | 25.00% | 25.00% |
| Electricity Disconnection | 12 | 2 | 16.67% | 58.33% | 58.33% |
| Fake KYC | 69 | 2 | 2.90% | 95.65% | 95.65% |
| Loan Scam | 13 | 12 | 92.31% | 7.69% | 7.69% |
| Lottery / Prize | 37 | 7 | 18.92% | 81.08% | 81.08% |
| OTP / Credential Theft | 2 | 2 | 100.00% | 0.00% | 0.00% |
| Telegram Job | 17 | 6 | 35.29% | 52.94% | 52.94% |
| Traffic e-Challan | 6 | 1 | 16.67% | 83.33% | 83.33% |
| UPI Fraud | 65 | 49 | 75.39% | 18.46% | 18.46% |

## BEFORE baseline: whole corpus (n = 85,680)

| slice | n | no evidence | rate | >=UNCERTAIN | rate | >=MEDIUM | rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fraud | 6770 | 2955 | 43.65% | 3530 | 52.14% | 3470 | 51.26% |
| promotion | 24312 | 19919 | 81.93% | 3501 | 14.40% | 3398 | 13.98% |
| benign | 54598 | 53445 | 97.89% | 672 | 1.23% | 655 | 1.20% |

### Baseline by fraud intent (whole corpus)

| fraud intent | n | no evidence | empty rate | >=UNCERTAIN | >=MEDIUM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bank Account Freeze | 239 | 41 | 17.16% | 81.59% | 63.18% |
| Courier Scam | 1072 | 681 | 63.53% | 29.01% | 29.01% |
| Electricity Disconnection | 354 | 49 | 13.84% | 69.21% | 69.21% |
| Fake KYC | 1474 | 124 | 8.41% | 88.06% | 87.31% |
| Loan Scam | 465 | 394 | 84.73% | 12.04% | 12.04% |
| Lottery / Prize | 896 | 89 | 9.93% | 89.73% | 89.73% |
| OTP / Credential Theft | 103 | 82 | 79.61% | 17.48% | 16.50% |
| Telegram Job | 371 | 156 | 42.05% | 48.25% | 48.25% |
| Traffic e-Challan | 189 | 51 | 26.98% | 72.49% | 70.37% |
| UPI Fraud | 1607 | 1288 | 80.15% | 17.86% | 17.86% |

## Notes on the baseline

* The 3,000-row sample reproduces the numbers quoted in `docs/RANK_ARM_AUDIT.md` exactly:
  39.1% of genuine fraud produces no evidence, and 2.43% of benign rows fire some rule.
* The sample is small where it matters: Courier Scam n=36, Telegram Job n=17, Lottery n=37.
  A widening that helps 2-3 sampled rows can be worth 30-40 rows corpus-wide, so every change in
  this document is also checked on the whole corpus before it is kept.
* The rules already fire on 30 corpus-wide benign rows with `parcel_customs`, all of them
  USPS/DHL/parcel-theme phishing ("your parcel is on hold, confirm your zip code"). They sit in
  the benign bucket because the source followed the link-free wording, not because the rule is
  wrong.

## AFTER: what changed in `src/rakshak/rules.py`

Three deterministic, regex-only changes. Nothing else in the rule set moved.

1. **`parcel_customs` - sentence-scoped pairing.** The flat 50-char window missed the commonest
   real phrasing, where the parcel and the money demand are in one sentence but far apart:
   > Post Office Your parcel have be redirect to your local Post Office branch due to an unpaid
   > shipping fee To reschedule a delivery please visit http://postoffice-tracking updates.com

   The rule now additionally pairs a `subject` (parcel/package/courier/shipment/dhl/fedex/usps/...)
   with a `trigger` (unpaid/outstanding/fee/charges/duty/on hold/held/clearance/redeliver/...) when
   both sit in the **same sentence**, at most 120 chars apart, and no `guard` phrase is present
   (`amount payable`, `amount due`, `total payable`, `cash on delivery`, `cod`, `paid`,
   `payment received/successful`). The guard is what keeps an ordinary order confirmation
   ("Amount payable is Rs 439") clean.
2. **`job_task_scam` - same sentence-scoped pairing.** Subject `job|hiring|recruit|part-time|work
   from home|...` paired with `earn|salary|income|kamai|...` in one sentence, 80 chars apart. This
   is the Telegram-job signature ("Part-time daily salary of 3,000 rupee ... wa.me/...").
3. **`prize_lottery` - two unambiguous additions.** `\bprize\b` became `\bprizes?\b` (the plural
   was a straight miss), and a new `win\w*` + `prize|gold|voucher|jackpot` phrase inside one
   sentence ("win exciting prizes", "win gold vouchers").

`sentences` are split on `.` `!` `?` `;` `|` newlines and `।`, with a digit look-behind so
"Rs 4.50" and "5.50-7.50" are not split mid-amount, and sentences longer than 400 chars are
skipped.

The `lookalike_domain` evidence type was **not** added. Adding a type to `EVIDENCE_TYPES` changes
the width of the fusion feature vector (`rakshak.fusion.fuse.FEATURE_NAMES`, which includes every
evidence type) and of `rakshak.models.encoding.N_RULES`, and a saved fusion model rejects a
feature-width mismatch (`fuse.py` checks `x.shape[1] != len(FEATURE_NAMES)`). Widening the schema
for a handful of rows was not worth invalidating those artifacts, so it was skipped as the brief
allowed.

## AFTER: 3,000-row sample (seed 13)

| slice | n | no evidence | rate | >=UNCERTAIN | rate | >=MEDIUM | rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fraud | 266 | 96 | 36.09% | 155 | 58.27% | 153 | 57.52% |
| promotion | 843 | 675 | 80.07% | 125 | 14.83% | 124 | 14.71% |
| benign | 1891 | 1845 | 97.57% | 26 | 1.38% | 26 | 1.38% |

### AFTER by fraud intent (3,000-row sample)

| fraud intent | n | no evidence | empty rate | >=UNCERTAIN | >=MEDIUM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bank Account Freeze | 9 | 1 | 11.11% | 88.89% | 66.67% |
| Courier Scam | 36 | 18 | 50.00% | 36.11% | 36.11% |
| Electricity Disconnection | 12 | 2 | 16.67% | 58.33% | 58.33% |
| Fake KYC | 69 | 2 | 2.90% | 95.65% | 95.65% |
| Loan Scam | 13 | 12 | 92.31% | 7.69% | 7.69% |
| Lottery / Prize | 37 | 3 | 8.11% | 91.89% | 91.89% |
| OTP / Credential Theft | 2 | 2 | 100.00% | 0.00% | 0.00% |
| Telegram Job | 17 | 6 | 35.29% | 52.94% | 52.94% |
| Traffic e-Challan | 6 | 1 | 16.67% | 83.33% | 83.33% |
| UPI Fraud | 65 | 49 | 75.39% | 18.46% | 18.46% |

## AFTER: whole corpus (n = 85,680)

| slice | n | no evidence | rate | >=UNCERTAIN | rate | >=MEDIUM | rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fraud | 6770 | 2794 | 41.27% | 3703 | 54.70% | 3643 | 53.81% |
| promotion | 24312 | 19772 | 81.33% | 3651 | 15.02% | 3548 | 14.59% |
| benign | 54598 | 53435 | 97.87% | 682 | 1.25% | 666 | 1.22% |

### AFTER by fraud intent (whole corpus)

| fraud intent | n | no evidence | empty rate | >=UNCERTAIN | >=MEDIUM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bank Account Freeze | 239 | 41 | 17.16% | 81.59% | 63.18% |
| Courier Scam | 1072 | 606 | 56.53% | 36.47% | 36.47% |
| Electricity Disconnection | 354 | 49 | 13.84% | 69.21% | 69.21% |
| Fake KYC | 1474 | 124 | 8.41% | 88.06% | 87.31% |
| Loan Scam | 465 | 394 | 84.73% | 12.04% | 12.04% |
| Lottery / Prize | 896 | 37 | 4.13% | 95.65% | 95.65% |
| OTP / Credential Theft | 103 | 82 | 79.61% | 17.48% | 16.50% |
| Telegram Job | 371 | 126 | 33.96% | 57.95% | 57.95% |
| Traffic e-Challan | 189 | 51 | 26.98% | 72.49% | 70.37% |
| UPI Fraud | 1607 | 1284 | 79.90% | 18.11% | 18.11% |

## BEFORE -> AFTER

### 3,000-row sample (seed 13)

| slice | n | metric | before | after | delta |
| --- | ---: | --- | ---: | ---: | ---: |
| --- | ---: | --- | ---: | ---: | ---: |
| fraud | 266 | no evidence | 39.10% | 36.09% | -3.01 pp |
| fraud | 266 | >=UNCERTAIN | 55.26% | 58.27% | +3.01 pp |
| fraud | 266 | >=MEDIUM | 54.51% | 57.52% | +3.01 pp |
| promotion | 843 | no evidence | 80.55% | 80.07% | -0.47 pp |
| promotion | 843 | >=UNCERTAIN | 14.35% | 14.83% | +0.48 pp |
| promotion | 843 | >=MEDIUM | 14.24% | 14.71% | +0.47 pp |
| benign | 1891 | no evidence | 97.57% | 97.57% | +0.00 pp |
| benign | 1891 | >=UNCERTAIN | 1.38% | 1.38% | +0.00 pp |
| benign | 1891 | >=MEDIUM | 1.38% | 1.38% | +0.00 pp |

### Whole corpus (n = 85,680)

| slice | n | metric | before | after | delta |
| --- | ---: | --- | ---: | ---: | ---: |
| --- | ---: | --- | ---: | ---: | ---: |
| fraud | 6770 | no evidence | 43.65% | 41.27% | -2.38 pp |
| fraud | 6770 | >=UNCERTAIN | 52.14% | 54.70% | +2.55 pp |
| fraud | 6770 | >=MEDIUM | 51.26% | 53.81% | +2.55 pp |
| promotion | 24312 | no evidence | 81.93% | 81.33% | -0.60 pp |
| promotion | 24312 | >=UNCERTAIN | 14.40% | 15.02% | +0.62 pp |
| promotion | 24312 | >=MEDIUM | 13.98% | 14.59% | +0.62 pp |
| benign | 54598 | no evidence | 97.89% | 97.87% | -0.02 pp |
| benign | 54598 | >=UNCERTAIN | 1.23% | 1.25% | +0.02 pp |
| benign | 54598 | >=MEDIUM | 1.20% | 1.22% | +0.02 pp |

## Reading the result honestly

* **Genuine fraud that produces no evidence: 39.10% -> 36.09% on the 3,000-row sample**
  (104/266 -> 96/266, i.e. 8 more fraud rows caught) and **43.65% -> 41.27% on the whole corpus**
  (2,955/6,770 -> 2,794/6,770, 161 more fraud rows caught).
* **Benign that fires any rule: 2.43% -> 2.43% on the sample, 2.11% -> 2.13% corpus-wide.**
  On the binding metric, benign at `>=UNCERTAIN` risk: **1.38% -> 1.38%** on the sample and
  **1.23% -> 1.25%** corpus-wide. The gate (`<=1.5%`) holds with margin; the corpus-wide rise is
  6 rows out of 54,598, and they are USPS/parcel-fixture SMS ("A redelivery fee may apply") that
  sit in the benign bucket despite being smishing-shaped text.
* **The sample is small where it matters.** Courier Scam is n=36 and Telegram Job n=17 in the
  sample, so the sample understates the parcel/job work: the same changes move 75 Courier Scam
  rows and 30 Telegram Job rows corpus-wide. Both numbers are reported above for that reason.
* **Promotion also rises slightly** (fires 19.45% -> 19.93% on the sample). That is expected and
  harmless: a promotion firing a scam rule is a precision question, not a recall one, and it is
  not counted against the benign gate.
* **Where the remaining fraud misses are.** Corpus-wide, of the 2,794 still-empty genuine fraud
  rows: 1,284 are "UPI Fraud" and 394 are "Loan Scam", the two buckets the brief says are mostly
  mislabelled bank/cashback promotions (79.9% and 84.7% empty). 1,231 of the UPI rows and 294 of
  the Loan rows contain **no URL at all**. Another large block of the remaining "Courier Scam"
  misses are grocery/pizza "free home delivery" promotions filed under courier. Those are label
  noise, not rule gaps, and chasing them would fire on real order and delivery SMS.

### Ideas that were measured and rejected

| candidate | fraud gained (corpus) | benign gained (corpus) | verdict |
| --- | ---: | ---: | --- |
| parcel subject + any `pay*` verb | 69 new | 42 new | **rejected** - "cannot be delivered due to incomplete address" is the modal benign parcel SMS |
| parcel + `pending` / `held` / `on hold` as the only trigger | 3-4 | 2 | **rejected** - dominated by benign fixture text |
| parcel + `not delivered` | 10 | **36** | **rejected** - pure false-positive generation |
| job subject + any money word, flat 120-char window | 38 | 5 | **rejected** - catches personal chatter ("I got this temp job ... pay 5.50") |
| job + `earn ... per day/week` | 5 | 7 | **rejected** - benign rows outnumber fraud |
| `win` alone, or `win + amount` | 124 / 27 | 89 / 16 | **rejected** - "our group can win", betting chatter |
| `lucky day` without a link | 5 | 0 | **rejected** - "I swear today is my lucky day" is benign; kept only when a link/log-on phrase is present (already covered by the URL rules) |
| `chance to win` | 46 | 5 | **rejected** - also 229 promotions; fires on legitimate surveys |

## Reproducing every number in this file

```bash
.venv/bin/python scripts/rule_coverage.py                    # AFTER, 3,000 rows
.venv/bin/python scripts/rule_coverage.py --n 85680          # AFTER, whole corpus
.venv/bin/python scripts/rule_coverage.py --compare-git HEAD # BEFORE vs working tree
.venv/bin/python scripts/rule_coverage.py --json             # stable blob to diff
```

`--compare-git REV` copies the tree to a temp directory, restores **only**
`src/rakshak/rules.py` (or whatever `--compare-path` names) from `REV`, and re-runs the same
measurement with the same interpreter. Every other module stays at its working-tree state, so an
unrelated in-flight edit elsewhere in the package does not contaminate the diff.
