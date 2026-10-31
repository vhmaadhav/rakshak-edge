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
| PhreshPhish, HF `phreshphish/phreshphish` (`phreshphish`) | URL, own split family `url_phreshphish` | CC BY 4.0 | Full official release (666,315 rows: train 498,255, test 168,060), extracted by Kaggle kernel `maadhavvh/rakshak-phreshphish-extract` into `data/raw/phreshphish_full/`. Never mixed into the general URL pool |
| SmishX, SOUPS 2025 (`smishx_soups2025`) | SMS, **eval only** | MIT (LICENSE.txt) | 1,200 manually relabelled messages (legitimate / spam / smishing). Feeds `external_smishx` |
| GroundLM 2026 cleaned Indian financial SMS test set (`groundlm_zenodo_20480151`) | SMS, **eval only** | Apache-2.0 for code and redistributable data (Zenodo 20480151 v4 supplement) | Member `data/cleaned_test_1393.csv` of the v4 archive, read in memory. 1,393 rows, 1,294 policy-negatives. 169 IIITD rows are hash-only in the release, so their text is unavailable. Feeds `external_groundlm` |

Not used: Nahapetyan et al. S&P 2024 (no data released to us, pending); Hindi IEEE DataPort set and M-Arjun SpamShield (access needed); Smishtank (no verified download).

## 2. Label mappings

- **UCI:** ham to `ham` (0); spam to `spam` (1).
- **Mishra & Soni:** ham to `ham`; `spam`/`Spam` to `spam`; `smishing`/`Smishing` to `smishing`. Positive = spam or smishing.
- **IMC25:** every row is `smishing` (1). Its `scam_type` (banking, others, delivery, government, telecom, spam, wrong number, hey mum/dad) is kept as `subsource`.
- **ScamShield:** `is_scam` 0 to `ham`. For `is_scam` 1: intent "General Spam / Telemarketing" to `spam`; any other intent to `scam`. This split is heuristic.
- **anmol:** ham to `ham`; spam to `spam`.
- **bolewara:** 0 to `ham`; 1 to `scam`.
- **LegitPhish:** `ClassLabel` 0 = phishing, 1 = legitimate (mapped to URL label 1 = phish). Checked against the data: `has_ip_address` is 0.77 for ClassLabel 0 and 0.00 for ClassLabel 1; HTTPS is 0.06 versus 1.00.
- **PhreshPhish:** `phish` to 1, `benign` to 0. Official `split` (train or test) kept on every row.
- **SmishX:** `legitimate` to `ham`-like benign (`label_binary` 0); `spam` and `smishing` to 1 (legacy spam + scam).
- **GroundLM:** `label_cleaned` 1 (fraud) to `label_binary` 1; 0 (safe) to 0. Policy-negatives are the rows with `label_cleaned` 0.

### 2a. Label taxonomy (`label_policy`, `label_scam`)

`label_binary` (legacy: spam or scam = 1) is unchanged. The new columns separate fraud from marketing (`src/rakshak/datasets/taxonomy.py`):

- `label_policy` in {`benign_transactional`, `benign_personal`, `promotion`, `scam`, `unclear`, null}. Filled only where the source's own label determines it:
  - Mishra & Soni smishing, IMC25 smishing, SmishX smishing, ScamShield non-spam scam intents: `scam`.
  - Mishra & Soni spam, SmishX spam: `promotion`.
  - Legitimate or ham rows from any source: null (the benign subtype is not in the source label).
  - UCI spam, ScamShield spam, anmol spam, bolewara positives: null (needs relabel; a generic "spam" may be fraud or marketing).
  - Rows whose duplicate group carries conflicting labels: `unclear`.
  - `benign_transactional` and `benign_personal` are reserved for a relabel pass and are not set by the build.
- `label_scam`: 1 = scam, 0 = benign or promotion, null = unknown (the spam rows above that need relabelling).

Coverage is reported in section 3a of this card after each build.
- **PhishTank, OpenPhish:** all label 1 (no benign side).

`synthetic` is True only for ScamShield `Synthetic_Tier_C` (3,720 rows, all in train). It is False for ScamShield `real_*` IDs, which is the dataset's own naming and not independently verified. It is True for IMC25 user reports. It is unknown (null) for the other SMS sources.

`eval_ok` marks rows allowed in a test or cal set. It is False for duplicates, known-synthetic rows, augmentation-only sources (anmol, bolewara) and rows in label-conflicting text groups (13 rows). Duplicates are never in any partition.

## 3. Language

`lang` is set by a deterministic detector in `rakshak/datasets/lang.py` with four buckets: `hi_deva` (at least 30% Devanagari letters), `hinglish` (at least 2 romanised-Hindi lexicon hits and at least 12% of tokens), `en` (English function-word hits), and `other`.

Kept-row buckets (rebuild 2026-10-30, after the Hinglish fix): ScamShield en 62,624 / other 8,902 / hi_deva 6,301 / hinglish 1,867; IMC25 en 16,458 / other 8,079 / hi_deva 94 / hinglish 83; UCI en 4,434 / other 724 / hinglish 1; Mishra en 858 / other 85 / hinglish 2; anmol en 209 / other 18 / hinglish 1; bolewara en 51 / hinglish 36.

Agreement with the source's own labels:
- IMC25 English rows: detector recall for `en` is 0.969 (15,486 rows).
- IMC25 non-English rows: 12.9% are flagged `en`.
- ScamShield: overall agreement 0.866 (was 0.806 before the fix). Hindi (Devanagari) agrees for 6,301 of 6,352 rows. Hinglish agrees for 1,525 of 3,000 source-labelled rows; 1,454 are called `en`. **Hinglish is still under-detected** (about 51% agreement). Treat `lang_shift` results as a lower bound on the shift.
- IMC25 English rows: detector recall for `en` is 0.975 (15,486 rows); 14.6% of non-English rows are flagged `en` (was 0.969 and 12.9%).

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

### 3a. Label taxonomy coverage (kept SMS rows, 110,827)

| label_policy | rows |
|---|---:|
| scam | 31,551 |
| promotion | 144 |
| unclear (label-conflict groups) | 13 |
| null (benign subtype or needs relabel) | 79,119 |

`label_scam`: 1 = 31,551; 0 = 54,954; null = 24,322 (the spam rows from UCI, ScamShield, anmol and bolewara positives). Rows with a policy value are a minority: the benign subtype (`benign_transactional`, `benign_personal`) is not in any source label and is not set. A relabel pass is required before any fraud-versus-marketing claim. Per source (all rows, duplicates included): IMC25 33,815 scam, 54 unclear; Mishra & Soni 634 scam, 489 promotion, 4 unclear; ScamShield 6,770 scam (non-spam intents, heuristic). Of the promotion rows, 144 are kept (the rest are duplicates).

## 5a. External evaluation sets (eval only)

`external/external_smishx.parquet` and `external/external_groundlm.parquet` are written by the build. Their rows never enter a training partition. Each external row is compared with the training pool (kept SMS rows, after canonicalisation):

- exact: the same dedup key (canonical text, lower-cased, whitespace collapsed);
- near-duplicate: MinHash (128 permutations, seed 13) over template keys, estimated Jaccard at least 0.7 (the campaign threshold).

Pool rows that overlap any external row are removed from **every TRAIN partition** (`train_rows_removed_external_overlap` per family in `build_report.json`). The external rows are never removed for overlap. Within one external set, later copies of a text are not eval-eligible (`dup_within_set`), and texts with both labels are not eval-eligible (`label_conflict_in_set`). Rows without text (`text_available` false) keep their labels but cannot be scored.

Numbers: section 6a.

### 6a. External sets: overlap with the training pool (build report `external`)

| Set | Rows | Text available | Eval-eligible | Within-set duplicates | Pool exact | Pool near-only | Pool any |
|---|---:|---:|---:|---:|---:|---:|---:|
| external_smishx (SmishX) | 1,200 | 1,200 | 1,107 | 93 | 610 | 153 | 634 |
| external_groundlm (GroundLM policy set) | 1,393 | 1,224 | 1,224 | 0 | 442 | 2 | 442 |

Pool = 110,827 kept SMS rows. Cross-set exact overlap between SmishX and GroundLM: 0. **Pool rows dropped from every TRAIN partition: 1,410** (train sizes shrink by the per-family count in `build_report.json`, e.g. campaign -1,019, legacy_uci -2,633). GroundLM: 169 IIITD rows are hash-only, and 161 of the 1,294 policy-negatives among them cannot be scored. So 1,133 of the 1,294 policy-negatives are scoreable; the FPR denominator is 1,133, not 1,294. The GroundLM overlap is high (442 exact), so any model trained on the pool would have seen those texts. The build drops them from TRAIN, which is the only way the external FPR is clean.

## 6. Splits

All partitions use seed 13. Group-based splits are asserted to have zero group overlap between splits, and test/cal sets are asserted to contain no duplicates, synthetic rows or ineligible rows.

**campaign** (groups = `campaign_id`; forced-train groups hold any synthetic or ineligible row):

| Split | Rows | Positive share | ScamShield | IMC25 | UCI | Mishra | Hi-Deva | Hinglish | Campaigns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 76,591 | 0.510 | 4,236 | 1,393 |
| cal | 11,072 | 0.505 | 639 | 199 |
| test | 22,145 | 0.505 | 1,279 | 398 |

Rebuilt 2026-10-30. Campaign overlap between splits: 0. The train size is after the external-overlap drop (1,019 rows). Source and label columns per split are in `build_report.json` (`sms_splits.campaign.stats`).

Group overlap between splits: 0 for all pairs. Split fractions are 70.0%, 10.0% and 20.0%. Train includes all 3,720 synthetic rows.

**temporal** (IMC25 smishing by date; ham paired from the campaign split):

| Split | Rows | Smishing (IMC25) | Ham |
|---|---:|---:|---:|
| train | 34,996 | 271 | 34,725 |
| cal | 5,515 | 39 | 5,476 |
| test | 11,029 | 77 | 10,952 |

Only 428 of 33,869 IMC25 rows have a date, and 387 of them survive dedup. Ham has no dates. Campaigns shared between temporal train and test: 3 (train-cal 4, cal-test 1), reported and not enforced.

**lang_shift** (train and cal English; test Hinglish and Hi-Deva, eval-eligible, campaign-disjoint from train):

| Split | Rows | Positive share | Languages |
|---|---:|---:|---|
| train | 58,605 | 0.536 | en |
| cal | 8,454 | 0.569 | en |
| test | 1,677 | 0.213 | hi_deva 1,279; hinglish 398 |

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

**url_phreshphish** (own family, not mixed with any other URL source; `urls_phreshphish.parquet`): the official PhreshPhish train and test splits, as released. One partition `official` with train and test id lists and no cal split (the PhreshPhish protocol takes no threshold). Benchmark membership is kept on each row (`bench_benign`, `bench_full`, `bench_subsets`), and the 975 benchmark subsets are the protocol of docs/SOTA_TARGETS.md section 5. Numbers: section 6b.

**URL** (`urls.parquet`, PhishTank, OpenPhish and LegitPhish only):

| Split | Rows | Positive share | Notes |
|---|---:|---:|---|
| url_domain train | 106,512 | 0.754 | grouped by registered domain; 0 domain overlap |
| url_domain cal | 15,215 | 0.754 | |
| url_domain test | 30,432 | 0.754 | |

`url_temporal` is **not written**: the general pool has no benign class with dates (PhishTank rows are all phish), so the build skips it (`url_splits_skipped`).

**url_phreshphish** (own family, `urls_phreshphish.parquet`, `splits/url_phreshphish.json`): official release splits, no cal.

| Split | Rows | Phish | Benign | Date range |
|---|---:|---:|---:|---|
| train | 498,255 | 221,526 | 276,729 | 2024-07-02 to 2025-09-08 |
| test | 168,060 | 76,800 | 91,260 | 2025-09-08 to 2025-12-16 |

Benchmark membership: 90,262 benign rows and 28,476 phish rows in the full (24%) benchmark (118,738 rows, all in test); 975 subsets (0.05%: 586, 0.1%: 296, 0.5%: 59, 1%: 29, 5%: 5). Overlap train-test: 15 normalised URLs, 17,450 registered domains (reported, not enforced; the paper's leakage filter is not reproduced here). 12,272 normalised URLs repeat inside the release; they are not deduplicated because the benchmark subsets are defined on the released rows.

URL temporal shares 134 train-test registered domains (reported, not enforced). Since PhreshPhish left the general pool, `url_temporal` has no benign class (PhishTank rows are all phish), so the build records it as skipped (`url_splits_skipped`) and does not write `splits/url_temporal.json`.

General pool kept counts (rebuild 2026-10-30): PhishTank 52,665 (raw 53,457); LegitPhish 99,205 (raw 101,218); OpenPhish 289 (raw 300). Label conflicts dropped: 2 PhishTank and 2 LegitPhish rows. Duplicates dropped by normalised URL: 790 PhishTank, 2,011 LegitPhish, 11 OpenPhish. PhreshPhish is not in this pool any more (see url_phreshphish).

## 7. Known limitations

1. **No real Indian test set with dates.** IMC25 dates cover 428 rows (2023) and the temporal test has 77 smishing rows. Hinglish and Hindi test rows come only from ScamShield, which is mostly synthetic-heavy or of unknown provenance.
2. **Source confound** (section 5). Single-class IMC25 and the label-from-source accuracy of 0.718 mean every SMS evaluation needs a source-held-out or source-balanced companion.
3. **No dated benign URL in the general pool.** `url_temporal` is skipped. Benign dated URLs exist only in PhreshPhish, which is evaluated as its own family (`url_phreshphish`).
4. **PhreshPhish protocol deviations.** The full release (666,315 rows) is now used, extracted from the HF parquet by a Kaggle kernel into `data/raw/phreshphish_full/`. The 975 benchmark subsets are taken as released; we do not regenerate the paper's temporal test split, leakage filter or difficulty filter. The PhreshPhish protocol takes no cal split.
5. **Nahapetyan et al. (S&P 2024) data not available.** No campaign IDs or timestamps from it are used.
6. **ScamShield provenance is uncertain.** The `Smishing_Dataset` subsource (55,475 rows) has no documented origin. `synthetic` for real rows relies on the dataset's own `real_` naming.
7. **Augmentation-only sources.** anmol and bolewara are kept for training only. They are in the campaign pool but have no eval-eligible rows.
8. **Language detector.** Hinglish is under-detected (about 51% of ScamShield Hinglish rows agree with the source label). Hindi and English are reliable.
9. **Campaigns are approximate.** The threshold was picked by sensitivity, not by human-labelled campaigns. LSH recall near 0.7 is probabilistic.
10. **Scam sub-labels are heuristic.** The spam-versus-scam split in ScamShield follows an intent label.
11. **Legacy protocols** use the dedup pool, so counts differ from the published protocols and their numbers are not directly comparable.
12. **Licences unverified** for Mishra & Soni, LegitPhish, PhishTank terms and the OpenPhish feed (see section 1).

## Data store

A private copy of the processed data, the teacher logits and the licence-cleared raw files lives at `hf://datasets/vhmaadhav/rakshak-data` (private; not public). It is being moved to the HF org `highestintheroom` as `highestintheroom/rakshak-data` (move pending, so the vhmaadhav path is current until then). It holds `processed/`, `teacher_logits/mmbert-base-s0/` (no checkpoints), `raw/` (only sources whose licence allows it) and the dataset card. Pull it with:

```bash
hf download vhmaadhav/rakshak-data --repo-type dataset --local-dir data/hf   # after the org move: highestintheroom/rakshak-data
```

Requires a Hugging Face login with access to the private repo. Do not make the repo public without a per-source licence review.
