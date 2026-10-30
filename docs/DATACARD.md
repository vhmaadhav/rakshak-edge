# Datacard: rakshak datasets

Built by `python -m rakshak.datasets.build` (about 3 minutes on the laptop, seed 13). Outputs are git-ignored under `data/processed/`:

- `sms.parquet`: one row per raw SMS message (137,153 rows; duplicates kept with `dup_of` set).
- `urls.parquet`: one row per kept URL (154,133 rows; `url_features` columns included).
- `splits/*.json`: id lists only. No text is stored twice.
- `build_report.json`: all numbers in this card, plus top-20 campaign examples.

Every figure below is read from `build_report.json`. Nothing is estimated by hand.

## 1. Sources, provenance and licences

| Source | Used as | Licence (as stated by source) | Notes |
|---|---|---|---|
| UCI SMS Spam Collection (`uci_sms_spam`) | SMS | CC BY 4.0 (research notes) | 5,574 rows; no provenance or synthetic flag |
| Mishra & Soni, Mendeley f45bkkt8pr v1 (`mishra_soni_smsphishing`) | SMS | Not stated (API null) | 5,971 rows; mostly a copy of UCI (see overlap) |
| IMC25 Fishing for Smishing (`imc25_fishing_for_smishing`) | SMS | CC BY 4.0 (LICENSE.txt) | User-reported messages; all smishing; `time` is a date for only 428 rows |
| ScamShield, HF `sidzzz07/scamshield-dataset` (`scamshield`) | SMS | MIT (README) | train/val/test as released; `source_dataset` kept as `subsource` |
| anmol scam-ham-india, HF (`anmol_scam_ham_india`) | SMS, augmentation only | Apache-2.0 (card) | README says it mixes synthetic examples without a per-row flag |
| bolewara hinglish-scam, HF (`bolewara_hinglish_scam`) | SMS, augmentation only | CC BY 4.0 (README) | English rows are UCI copies; 63 Hinglish rows are hand-written |
| PhishTank verified-online (`phishtank_online_valid`) | URL, phish | PhishTank terms (not verified here) | 53,457 entries fetched 2026-10-30 |
| OpenPhish community feed (`openphish_community_feed`) | URL, phish | Non-commercial per secondary sources (unverified) | 300 URLs, snapshot 2026-10-30T04:03:39Z |
| LegitPhish, Mendeley hx4m73v2sf (`legitphish_mendeley`) | URL | Not stated (API null) | 101,219 rows |
| PhreshPhish, HF `phreshphish/phreshphish` (`phreshphish`) | URL | CC BY 4.0 | Local data: only the `train_0000` and `test_0000` shards (2,000 rows) |

Not used: Nahapetyan et al. S&P 2024 (no data released to us, pending); Hindi IEEE DataPort set and M-Arjun SpamShield (access needed); Smishtank (no verified download).

## 2. Label mappings

- **UCI:** ham to `ham` (0); spam to `spam` (1).
- **Mishra & Soni:** ham to `ham`; `spam`/`Spam` to `spam`; `smishing`/`Smishing` to `smishing`. Positive = spam or smishing.
- **IMC25:** every row is `smishing` (1). Its `scam_type` (banking, others, delivery, government, telecom, spam, wrong number, hey mum/dad) is kept as `subsource`.
- **ScamShield:** `is_scam` 0 to `ham`. For `is_scam` 1: intent "General Spam / Telemarketing" to `spam`; any other intent to `scam`. This split is heuristic.
- **anmol:** ham to `ham`; spam to `spam`.
- **bolewara:** 0 to `ham`; 1 to `scam`.
- **LegitPhish:** `ClassLabel` 0 = phishing, 1 = legitimate (mapped to URL label 1 = phish). Checked against the data: `has_ip_address` is 0.77 for ClassLabel 0 and 0.00 for ClassLabel 1; HTTPS is 0.06 versus 1.00.
- **PhreshPhish:** `phish` to 1, `benign` to 0.
- **PhishTank, OpenPhish:** all label 1 (no benign side).

`synthetic` is True only for ScamShield `Synthetic_Tier_C` (3,720 rows, all in train). It is False for ScamShield `real_*` IDs, which is the dataset's own naming and not independently verified. It is True for IMC25 user reports. It is unknown (null) for the other SMS sources.

`eval_ok` marks rows allowed in a test or cal set. It is False for duplicates, known-synthetic rows, augmentation-only sources (anmol, bolewara) and rows in label-conflicting text groups (13 rows). Duplicates are never in any partition.

## 3. Language

`lang` is set by a deterministic detector in `rakshak/datasets/lang.py` with four buckets: `hi_deva` (at least 30% Devanagari letters), `hinglish` (at least 2 romanised-Hindi lexicon hits and at least 12% of tokens), `en` (English function-word hits), and `other`.

Kept-row buckets: ScamShield en 58,122 / other 13,756 / hi_deva 6,301 / hinglish 1,515; IMC25 en 16,212 / other 8,360 / hi_deva 94 / hinglish 48; UCI en 4,066 / other 1,091.

Agreement with the source's own labels:
- IMC25 English rows: detector recall for `en` is 0.969 (15,486 rows).
- IMC25 non-English rows: 12.9% are flagged `en`.
- ScamShield: overall agreement 0.806. Hindi (Devanagari) agrees for 6,301 of 6,352 rows. Hinglish agrees for 1,371 of 3,000 kept rows; 1,564 are called `en`. **Hinglish is under-detected.** Treat `lang_shift` results as a lower bound on the shift.

## 4. Deduplication

**Exact duplicates.** Key = `text_canon` lower-cased with whitespace collapsed. The representative is the first row by source priority (UCI, Mishra, IMC25, ScamShield, anmol, bolewara), then non-synthetic before synthetic, then file order.

| Source | Raw | Kept | Exact dups removed | Eval-eligible |
|---|---:|---:|---:|---:|
| uci_sms_spam | 5,574 | 5,159 | 415 | 5,159 |
| mishra_soni_smsphishing | 5,971 | 945 | 5,026 | 941 |
| imc25_fishing_for_smishing | 33,869 | 24,714 | 9,155 | 24,706 |
| scamshield | 85,680 | 79,694 | 5,986 | 75,973 |
| anmol_scam_ham_india | 2,272 | 228 | 2,044 | 0 |
| bolewara_hinglish_scam | 3,787 | 87 | 3,700 | 0 |
| **Total** | **137,153** | **110,827** | **26,326** | **106,779** |

**Cross-source overlap** (pre-dedup, distinct keys, row counts in the report):
- Mishra and UCI: 4,886 of Mishra's 5,831 distinct keys are in UCI (84%).
- ScamShield and UCI: 5,154 of ScamShield's 85,595 distinct keys are in UCI.
- ScamShield and Mishra: 5,616 shared keys.
- bolewara and UCI: 3,471 of 3,558 keys shared. bolewara and Mishra: 3,279. bolewara and ScamShield: 3,471.
- anmol and ScamShield: 2,027 of 2,257 keys shared.
- IMC25 and ScamShield: 14 shared keys. IMC25 shares almost nothing with the other SMS sources.

Consequence: the Mishra and bolewara sources contribute little new text. Their kept rows are mostly from Mishra's own internal variants and from bolewara's hand-written Hinglish rows.

**Label conflicts.** 13 kept rows sit in text groups with both labels. They are marked `label_conflict` and set `eval_ok` = False.

**Near-duplicates and campaigns.**
- Template key: URLs to `<URL>`, phones to `<PHONE>`, amounts to `<AMT>`, other numbers to `<NUM>`. Months and weekdays are kept.
- Near-duplicate edges: MinHash (128 permutations, seed 13) over char 5-gram shingles of the template key, with datasketch LSH at threshold 0.7. Each LSH candidate is verified by its MinHash Jaccard estimate. 492,952 near-duplicate pairs.
- Shared-domain edges: scam messages sharing a registered domain join a campaign if the domain is not a shortener or a shared platform, and is not shared by more than 200 scam messages. 694 domains joined. The fan-out cap excluded `jazzmobilemag.pk` (460 scam messages) and `airtel.in` (214).
- Campaign = connected component (union-find). Ham takes part in near-duplicate edges, so near-identical bank and telecom alerts stay in one group.

**Threshold 0.7.** Chosen by sensitivity. Near-duplicate components at thresholds 0.5, 0.7 and 0.9 are 80,285, 89,725 and 101,047. At 0.5, distinct lures that share only boilerplate ("your account has been ...") merge into one group. At 0.9, variants of one template split apart and leak across partitions. 0.7 means roughly 70% of the template's 5-grams are shared, so the messages are the same template with a few slot words changed. This is a judgement, not validated against human-labelled campaigns.

**Campaign statistics (kept rows).**
- 88,401 campaigns over 110,827 rows. 81,637 are singletons (92.4%).
- Size histogram: 1: 81,637; 2: 4,231; 3-5: 1,818; 6-10: 426; 11-50: 254; 51-200: 25; 201+: 10. Largest: 1,236 rows (an SBI YONO suspension template spanning IMC25 and ScamShield).
- 1,615 campaigns span more than one source; 446 contain both labels.
- Top-20 campaigns are listed with examples in `build_report.json` (`campaigns.top`).

## 5. Confound findings (critical)

Measured in `build_report.json` on the kept SMS rows.

- **Label balance per source (kept rows):**
  - UCI: 4,517 ham / 642 spam.
  - Mishra: 524 / 421.
  - IMC25: 0 / 24,714. **Single-class: every IMC25 row is smishing.**
  - ScamShield: 49,660 / 30,034.
  - anmol: 69 / 159 (augmentation only).
  - bolewara: 40 / 47 (augmentation only).
- **Source-identity probe.** Char TF-IDF + logistic regression predicting source, capped at 2,000 rows per source, 80/20 stratified: **accuracy 0.738** against a majority baseline of 0.275. Per-source recall: IMC25 0.92, ScamShield 0.73, UCI 0.76, Mishra 0.43, anmol 0.61, bolewara 0.00. Top confusions: ScamShield to UCI (96), Mishra to UCI (76), UCI to ScamShield (70).
- **Label-from-source baseline.** Predicting each message's label as its source's majority label gives **accuracy 0.718**, against a pooled majority baseline of 0.505. Source identity alone is a strong shortcut for the binary label.

**Which evaluations are confounded:**
- Every SMS partition: positives are dominated by IMC25 and ScamShield, and the negatives come mostly from ScamShield and UCI. The campaign split removes campaign leakage but not source-style shortcuts.
- `temporal`: the test scam class is IMC25 alone (77 rows). The ham class comes from other sources. Any result there mostly measures source discrimination.
- `source_holdout` for UCI or Mishra: the held-out test is one source with a fixed label mix, so it probes transfer between sources, not between campaigns.
- `legacy_*`: random splits on the dedup pool. They are comparable to published numbers only in protocol, not in data (see section 7).
- `url_temporal` test: single-class (all phish).

## 6. Splits

All partitions use seed 13. Group-based splits are asserted to have zero group overlap between splits, and test/cal sets are asserted to contain no duplicates, synthetic rows or ineligible rows.

**campaign** (groups = `campaign_id`; forced-train groups hold any synthetic or ineligible row):

| Split | Rows | Positive share | ScamShield | IMC25 | UCI | Mishra | Hi-Deva | Hinglish | Campaigns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 77,609 | 0.505 | 55,728 | 17,294 | 3,611 | 661 | 4,477 | 1,122 | 55,456 |
| cal | 11,073 | 0.505 | 7,988 | 2,474 | 516 | 95 | 639 | 160 | 11,030 |
| test | 22,145 | 0.505 | 15,978 | 4,946 | 1,032 | 189 | 1,279 | 321 | 21,915 |

Group overlap between splits: 0 for all pairs. Split fractions are 70.0%, 10.0% and 20.0%. Train includes all 3,720 synthetic rows.

**temporal** (IMC25 smishing by date; ham paired from the campaign split):

| Split | Rows | Smishing (IMC25) | Ham |
|---|---:|---:|---:|
| train | 35,828 | 271 | 35,557 |
| cal | 5,515 | 39 | 5,476 |
| test | 11,029 | 77 | 10,952 |

Only 428 of 33,869 IMC25 rows have a date, and 387 of them survive dedup. Ham has no dates. Campaigns shared between temporal train and test: 3 (train-cal 4, cal-test 1), reported and not enforced.

**lang_shift** (train and cal English; test Hinglish and Hi-Deva, eval-eligible, campaign-disjoint from train):

| Split | Rows | Positive share | Languages |
|---|---:|---:|---|
| train | 55,667 | 0.556 | en |
| cal | 7,940 | 0.572 | en |
| test | 1,600 | 0.209 | hi_deva 1,279; hinglish 321 |

Test is small and mostly ScamShield. The test positive share is low (0.209), so use PR-AUC and calibration, not accuracy.

**source_holdout** (leave one source out; train and cal exclude any campaign that also occurs in the held-out test):

| Held-out source | Test rows (positive) | Train | Cal |
|---|---|---:|---:|
| uci_sms_spam | 5,159 (642) | 93,241 | 10,350 |
| mishra_soni_smsphishing | 941 (417) | 97,988 | 10,876 |
| scamshield | 75,973 (29,029) | 24,895 | 2,755 |

For ScamShield, most other-source campaigns are excluded (those shared with ScamShield's test), so train is small.

**legacy** (published-style protocols on the dedup pool, for comparability only):
- `legacy_uci_sms_spam`: stratified 80/20 holdout (train 4,127, test 1,032) and 5-fold CV. Base dedup pool is 5,159 rows, not the published 5,574.
- `legacy_mishra_soni`: stratified 80/20 (train 752, test 189) and 5-fold CV. Pool is 941 rows.
- `legacy_scamshield`: official train (64,475), val as cal (7,582), official test (7,637). Official splits share campaigns across train and test (844 and 852 shared campaigns); this is expected and recorded.
- No legacy split for IMC25 (single-class) or anmol/bolewara (no eval-eligible rows).

**URL** (`urls.parquet`):

| Split | Rows | Positive share | Notes |
|---|---:|---:|---|
| url_domain train | 107,893 | 0.750 | grouped by registered domain; 0 domain overlap |
| url_domain cal | 15,413 | 0.750 | |
| url_domain test | 30,827 | 0.750 | |
| url_temporal train | 38,248 | 0.970 | 1,141 benign (PhreshPhish only) |
| url_temporal cal | 5,464 | 1.000 | |
| url_temporal test | 10,928 | 1.000 | **single-class** |

URL temporal shares 134 train-test registered domains (reported, not enforced).

URL kept counts: PhishTank 52,665 (raw 53,457); LegitPhish 99,204 (raw 101,218); PhreshPhish 1,975 (local shards, raw 2,000); OpenPhish 289 (raw 300). Label conflicts dropped: 2 PhishTank and 2 LegitPhish rows. Duplicates dropped by normalised URL: 790 PhishTank, 2,012 LegitPhish, 25 PhreshPhish, 11 OpenPhish.

## 7. Known limitations

1. **No real Indian test set with dates.** IMC25 dates cover 428 rows (2023) and the temporal test has 77 smishing rows. Hinglish and Hindi test rows come only from ScamShield, which is mostly synthetic-heavy or of unknown provenance.
2. **Source confound** (section 5). Single-class IMC25 and the label-from-source accuracy of 0.718 mean every SMS evaluation needs a source-held-out or source-balanced companion.
3. **Temporal URL test is single-class.** Only the local PhreshPhish shards carry benign examples with dates. The full PhreshPhish set is needed.
4. **PhreshPhish is partial.** Locally there are two 1,000-row shards. The full dataset (~145 GB train, ~48 GB test) is not built. `load_phreshphish` takes a local parquet path or an HF id; the HF path streams the four columns (`url`, `label`, `date`, `target`) through the `datasets` library, which is not a project dependency, so the Kaggle job must install it. Run with `--phreshphish phreshphish/phreshphish`.
5. **Nahapetyan et al. (S&P 2024) data not available.** No campaign IDs or timestamps from it are used.
6. **ScamShield provenance is uncertain.** The `Smishing_Dataset` subsource (55,475 rows) has no documented origin. `synthetic` for real rows relies on the dataset's own `real_` naming.
7. **Augmentation-only sources.** anmol and bolewara are kept for training only. They are in the campaign pool but have no eval-eligible rows.
8. **Language detector.** Hinglish is under-detected (about 46% of ScamShield Hinglish rows agree with the source label). Hindi and English are reliable.
9. **Campaigns are approximate.** The threshold was picked by sensitivity, not by human-labelled campaigns. LSH recall near 0.7 is probabilistic.
10. **Scam sub-labels are heuristic.** The spam-versus-scam split in ScamShield follows an intent label.
11. **Legacy protocols** use the dedup pool, so counts differ from the published protocols and their numbers are not directly comparable.
12. **Licences unverified** for Mishra & Soni, LegitPhish, PhishTank terms and the OpenPhish feed (see section 1).
