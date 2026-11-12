# Surrogate fill for placeholder tags, then re-run the inbox arms (pre-registration draft)

Drafted 2026-11-07 after `docs/INBOX_RETRAIN_RESULTS.md`. **Approved 2026-11-11 by Claude under Maadhav's delegation of plan decisions that day (his words: "you're free to redesign the plan or architecture whichever u think is best"). Maadhav has not reviewed this file line by line.** Frozen as of 2026-11-11. Once approved, the hypotheses, arms, gates, stop rules
and threshold rule are frozen, and any later change goes in a dated amendment.

## 1. Why

`<TAG>` placeholders are a scam shortcut, and they come from two places:

| Where | Tags | Share of rows tagged | Label mix |
|---|---|---|---|
| `sms.parquet`, source `imc25_fishing_for_smishing` (33,869 rows) | Presidio entity types: `<NAMED_ENTITY>` 17,270, `<URL>` 15,943, `<DATE_TIME>` 13,713, `<LOCATION>` 3,541, `<PHONE_NUMBER>` 3,228, `<US_DRIVER_LICENSE>` 2,182, `<US_BANK_NUMBER>` 1,067, `<NRP>` 782, `<EMAIL_ADDRESS>` 434, and a long tail | 82.4% | 100% scam |
| `sms.parquet`, `scamshield` (85,680 rows) | `<DECIMAL>` 403, `<TIME>` 89, `<EMAIL>` 53, a few others | 0.6% | 36% scam |
| Inbox export (8,040 rows) | `<NUMBER>` 15,385, `<PATH>` 5,383, `<AMOUNT>` 4,706, `<ACCOUNT_CARD>` 2,378, `<UPI_ID>` 1,688, `<PHONE>` 599, `<NAME>` 569, `<QUERY>` 147, `<OTP>` 126, `<EMAIL>` 17, `<FRAGMENT>` 4 | 95–99% | 2% scam |

In training, 40.8% of scam rows carry a tag and 0.7% of legit rows do. The inbox scrubber also
removes every digit: 0% of inbox rows contain one, against 31.8% of training rows. GroundLM, SmishX and
the other evaluation files have no tags (all under 0.1%), so no other set needs filling.

IMC25 was published already anonymised by its authors, so its original values do not exist. The
inbox's original values must never leave the donor's phone. In both cases the only fix is to fill in
**surrogates**: realistic fake values of the right type. Clinical de-identification uses the same
approach. Yeniterzi et al. (JAMIA 2010, "Effects of personal identifier resynthesis on clinical text
de-identification") found that models trained on tagged text transfer badly to text with real-looking
values, and that resynthesising surrogates removes most of that gap. Presidio, the tool IMC25 was
tagged with, supports the same operation through custom replace operators.

## 2. Hypotheses (frozen)

- **H1 (the confound is the main cause):** the shipped v0 model, scored on phone 1 with tags filled,
  has legit FPR at least 10× lower than on scrubbed phone 1. Before: 98.6% at the R0 thresholds.
- **H2 (filling fixes the shortcut):** S0, the v0 recipe trained on filled `sms.parquet`, passes
  G1–G4 at least as often as v0 did, and its SmishX recall does not drop below v0's 157/226.
- **H3 (inbox data helps once filled):** S1 lowers phone-1 legit FPR against S0 (paired bootstrap
  over sender groups, CI excludes 0) without failing G2.

## 3. Surrogate generator (`src/rakshak/data/surrogates.py`)

`fill(text, rng) -> str` replaces each tag with a value drawn from a type-specific pool:

| Tag(s) | Surrogate |
|---|---|
| `<AMOUNT>` | `Rs.`, `Rs `, `INR ` or `₹` plus an amount from a log-uniform 1–2,00,000, with 0 or 2 decimals and Indian or Western grouping |
| `<NUMBER>`, `<DECIMAL>` | Digit string, length drawn from 1–8 (mode 4); `<DECIMAL>` gets a fraction |
| `<OTP>` | 4 or 6 digits |
| `<ACCOUNT_CARD>`, `<US_BANK_NUMBER>`, `<CREDIT_CARD>`, `<IBAN_CODE>` | Masked form: `XX1234`, `XXXXXXXX1234` or `**** 1234` |
| `<PHONE>`, `<PHONE_NUMBER>` | Valid-format number: Indian mobile `[6-9]\d{9}`, optional `+91`; for IMC25 non-Indian rows, a country-plausible format |
| `<NAME>`, `<NAMED_ENTITY>` | Name sampled from a mixed Indian and international given/family list |
| `<UPI_ID>` | `<name><digits>@<real PSP handle>`, e.g. `@okhdfcbank`, `@ybl`, `@paytm` |
| `<EMAIL>`, `<EMAIL_ADDRESS>` | `<name><digits>@<common webmail domain>` |
| `<PATH>`, `<QUERY>`, `<FRAGMENT>` | Random short path, `k=v` query or fragment from a fixed alphabet |
| `<URL>` (IMC25 only: the whole URL is gone) | Synthetic URL. Domain from a pool mixing ~40% brand-lookalike, ~30% shortener and ~30% random-word domains, then a path |
| `<DATE_TIME>`, `<TIME>`, `<DATE>` | A date or time in a random common Indian or US format |
| `<LOCATION>` | City from a mixed Indian and international list |
| `<US_DRIVER_LICENSE>`, `<US_PASSPORT>`, `<US_SSN>`, `<US_ITIN>`, `<UK_NHS>`, `<MEDICAL_LICENSE>`, `<IP_ADDRESS>`, `<NRP>`, `<CRYPTO>` | Type-valid fake ID; `<NRP>` (nationality/religion/political group) becomes a nationality word |
| Anything else matching `<[A-Z_]+>` | A neutral word from a fixed list. The tail has under 30 occurrences in total |

Rules:
1. **Independent of the label:** the generator never sees the label or the source, so a surrogate
   cannot carry label information. Unit test: identical outputs for the same text and seed whatever
   the label.
2. **Deterministic:** each row's RNG is seeded from `hash(row id, SURROGATE_SEED = 20261107)`, so
   every arm sees the same filled text. One fixed realisation, with no per-epoch resampling.
3. **Applied at load, in memory, inside the Kaggle job.** No dataset is rewritten, and the private
   inbox dataset stays exactly as uploaded.
4. **Applied to every row of `sms.parquet` (all splits) and every inbox row (train and test donor).**
   GroundLM, SmishX and the shift sets are left as they are (no tags).
5. **Assertion:** after filling, zero rows in any set match `<[A-Z_]+>`. Otherwise the job stops.
6. **Brand-lookalike URL pool:** built from brand names already in `src/rakshak` URL features, so no
   new brand list is invented. The pool is the same for scam and legit, because rule 1 applies.

## 4. Diagnostic D0 (runs first, cheap, no training)

Score the shipped v0 ONNX (sha 5341e14c) on scrubbed and on filled versions of: phone 1, the phone 2+3
held-out legit set, and IMC25's test split. Report legit FPR at v0's shipped τ and at the R0 τ's,
plus the score distributions.

**Stop rule 1:** if H1 fails (filled phone-1 FPR is not at least 10× lower), the confound is not the
main cause. Stop and report; S0–S2 do not run.

## 5. Arms (3 seeds each, 1 / 2 / 3; same dual-base recipe and data seed as v0)

| Arm | Training data | Sender feature |
|---|---|---|
| S0 | v0's data, filled | no |
| S1 | S0 + inbox train donor (phones 2+3), filled | no |
| S2 | S1 | yes (side width 21) |

Inbox splits, the test donor (phone 1), caps and the sender-ID split are the same as in
`docs/INBOX_RETRAIN_PLAN.md` addendum 3. S2 runs after S1, with S1's `probs.parquet` attached for the
paired comparison.

## 6. Threshold rule

Unchanged (`rakshak.data.retrain_prep.threshold_rule`): τ is picked on the inbox held-out legit set
(filled) plus campaign-cal, before any test set is scored.

## 7. Gates (an arm passes if each gate holds on the seed median and on at least 2 of 3 seeds)

Unchanged: G1 GroundLM FP ≤ 11/1,133 · G2 SmishX TP ≥ 157/226 · G3 campaign F1 ≥ 0.971 · G4 ONNX ≤ 10 MB ·
G5 test-donor legit FPR ≤ 1%. G3 is now scored on the **filled** campaign test split. v0's G3 on the
filled split is reported next to it, so the gate isn't silently easier or harder.

Also reported, without a gate: test-donor scam recall (n = 14, with an exact binomial CI), the
template-novel subset, and a source-identity probe (`rakshak.datasets.confound`) before and after
filling, to check that IMC25 is harder to tell apart from the other scam sources.

**Stop rule 2:** if S0 fails G2 on all 3 seeds, the fill has damaged scam recall. Stop before S1.

## 8. Compute

Kaggle T4 only; D0 runs on CPU in a Kaggle kernel. About 9 training runs, the same as the last round.
No Modal. Nothing goes to the Pi unless an arm passes every gate; Pi latency is then checked for the
winner only.

## 9. Limits

- Surrogates are fake. Their realism, in amounts, URL shapes and how often names appear, is a modelling
  choice. The source-identity probe is the only check that it hides IMC25's style.
- Test and train still come from one person (addendum 3), and the labels are Codex's, not human ones.
- One surrogate realisation. Variance from the fill itself is not measured; that would need a second
  `SURROGATE_SEED` run of S1, which isn't budgeted here.

## Amendment 1 (2026-11-11, before any run)

Decisions the plan left open, made while building sections 3 and 4 (D0 only; no arm has run):

- H1 verdict over three R0 taus: the verdict is taken at the median R0 tau by value; every tau is still reported, and H1 holds iff it holds at that tau.
- H1 per tau: holds iff the scrubbed phone-1 legit FPR is above 0 and filled FPR is at most one tenth of it (exact fractions, boundary inclusive). A scrubbed FPR of 0 does not hold; a filled FPR of 0 with scrubbed above 0 holds.
- R0 taus: the three seed thresholds of `kaggle/out/sms_inbox_retrain-r0/output/thresholds.json` (0.5127761960029602, 0.9999979734420776, 0.7533388137817383), applied to v0 as section 4 states, although they were calibrated on R0's own scores.
- Row id for seeding: the `id` column of `sms.parquet` (campaign and IMC25 rows) and of the inbox files; inbox ids and sms ids share one seed namespace, so a collision could only correlate two realisations.
- Held-out legit set (phones 2+3): `build_inbox_pools` `val_legit` with the same test-only text list as `sms_inbox_retrain`; because that function drops ids, ids are recovered by exact (text, sender) matching in memory.
- Phone-1 set: every `label_inbox` row of `test_donor.parquet` (labels 0 and 1; unclear and personal rows dropped as in R0).
- IMC25 set: the campaign test partition after the eval mask, rows with source `imc25_fishing_for_smishing`.
- Phone tags: Indian mobile format for every `<PHONE>` and `<PHONE_NUMBER>` row, not a country-plausible format for non-Indian IMC25 rows, as the build specification requires.
- Masked-account forms (`XX1234`, `XXXXXXXX1234`, `**** 1234`) are used for ACCOUNT_CARD, US_BANK_NUMBER, CREDIT_CARD and IBAN_CODE.
- Type-valid fake IDs use reserved or documentation ranges where they exist (SSN area 9xx and group 00, UK NHS prefix 999, IP addresses in RFC 5737 ranges); ITIN and driver-licence formats are format-only.
- Lookalike domains: the first host alias of each brand, with the templates and TLDs in `surrogates.py`; shortener and random-word lists are fixed in `surrogates.py`.
- Wilson 95 % intervals use `rakshak.ela_eval.wilson_interval` (rounded to 4 places).
