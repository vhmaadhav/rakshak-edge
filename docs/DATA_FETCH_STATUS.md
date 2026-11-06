# Raw data fetch status

Snapshot taken 2026-10-31 (Asia/Calcutta) on the development Mac. Every row count below was
produced by calling the actual loader from `src/rakshak/datasets/` against `data/raw/`, not by
inspecting files by hand. No `.py` file was modified to make anything parse.

## Table

| Source | Path | Status | Rows parsed | Size | Notes / blocker |
|---|---|---|---|---|---|
| UCI SMS Spam Collection | `data/raw/uci_sms_spam/` | present (pre-existing) | 5,574 | 0.68 MB | Already in repo. `load_uci` OK. |
| ScamShield (sidzzz07) | `data/raw/hf_sidzzz07_scamshield/` | present (pre-existing) | 85,680 | 44 MB | Already in repo. `load_scamshield` OK. |
| Scam/ham India (anmolshrivastav) | `data/raw/hf_anmolshrivastav_scam_ham_india/` | present (pre-existing) | 2,272 | 0.25 MB | Already in repo. `load_anmol` OK. |
| **Mishra & Soni SMS Phishing** | `data/raw/mishra_soni_mendeley_smsphishing/Dataset_5971.zip` | **downloaded** | **5,971** | 0.24 MB | Mendeley `f45bkkt8pr` v1, `public-api` → `public-files` download. `load_mishra` OK. Labels: ham 4,844 / smishing 638 / spam 489, matching the record's stated composition exactly. |
| **IMC25 Fishing for Smishing** | `data/raw/imc25_fishing_for_smishing/final_dataset_output.csv` | **downloaded** | **33,869** | 9.02 MB | `github.com/reportsmishing/Smishing-Dataset-IMC25`, CC-BY-4.0, `dataset/final_dataset_output.csv` on `main`. Not gated, no request form. `load_imc25` OK. |
| **Bolewara Hinglish scam** | `data/raw/hf_bolewara_hinglish_scam/data.json` | **downloaded** | **3,787** | 0.45 MB | HF `bolewara/hinglish-scam-text-dataset`, CC-BY-4.0, not gated. `load_bolewara` OK (3,040 ham / 747 scam). |
| **LegitPhish** | `data/raw/legitphish_mendeley/url_features_extracted1.csv` | **downloaded** | **101,218** | 8.55 MB | Mendeley `hx4m73v2sf` v2, 18 columns incl. `URL` + `ClassLabel`. `load_legitphish` OK. |
| **PhishTank online-valid** | `data/raw/phishtank_online_valid/online-valid.json.bz2` | **downloaded (mirror)** | **52,838** | 2.04 MB | See blocker below: `data.phishtank.com` returned HTTP 429. Served from the auto-updating mirror `ProKn1fe/phishtank-database`, refreshed 2026-10-31T05:03Z, checksum verified against its published `.sha256`. `load_phishtank` OK, dates 2011-02-18 → 2026-10-31, 100% `target` populated. |
| **OpenPhish community feed** | `data/raw/openphish_community_feed/feed.txt` | **downloaded** | **300** | 0.016 MB | `https://openphish.com/feed.txt` now HTTP 302-redirects to `raw.githubusercontent.com/openphish/public_feed` (owner `openphish`, the official community feed). `load_openphish` OK. |
| PhreshPhish | `data/raw/phreshphish_full/` | **metadata only — not downloaded** | n/a (train 498,255 / test 168,060 per HF API) | **5.25 GB** on the Hub | Deliberately not fetched: larger than free disk. See "Still blocked". |
| SmishX (SOUPS 2025) | `data/raw/smishx_soups2025/dataset.csv` | **downloaded** | **1,200** | 0.16 MB | `github.com/yizhu-joy/SmishX`, MIT, `data/dataset.csv`. `load_smishx` OK: legitimate 622 / spam 319 / smishing 259. |
| GroundLM (Zenodo 20480151 v4) | `data/raw/zenodo_20480151_scamshield_v4/scamshield_emnlp_artifact_v4_2026-05-31.zip` | **extracted member (RESOLVED — see note)** | **1,393** | 0.05 MB | Not the 1.66 GB archive: only the one needed member was pulled out of the *real* upstream archive with HTTP range requests and repacked. Verified byte-identical by CRC32 against upstream's central directory (`e42764c9`, 205,858 bytes). 1,294 policy negatives / 169 hash-only / 1,133 scoreable — matches the datacard exactly. See `docs/GROUNDLM_FPR.md`. |

Total raw corpus on disk: ~64 MB (excluding the 44 MB ScamShield set already present).

## Verified loader runs

All nine loaders were called against `data/raw/` in one pass; every one returned the expected
column set (11 columns for SMS loaders, 6 for URL loaders):

```
uci          OK  rows=   5574      imc25        OK  rows=  33869      openphish   OK  rows=    300
mishra       OK  rows=   5971      scamshield   OK  rows=  85680      legitphish  OK  rows= 101218
anmol        OK  rows=   2272      bolewara     OK  rows=   3787      phishtank   OK  rows=  52838
```

## Build run

`python -m rakshak.datasets.build --skip-urls` completes cleanly end to end (221 s) and writes
`data/processed/sms.parquet`, `data/processed/build_report.json`, `external/`, and eight split
families. The URL half needs its own invocation (see "What this unlocks"):

```
SMS raw 137,153 → kept after exact dedup 110,827 (26,326 duplicates) → 88,401 campaigns
split families written: campaign, temporal, lang_shift, source_holdout,
                        legacy_uci_sms_spam, legacy_mishra_soni, legacy_scamshield,
                        url_domain
```

## What this unlocks

**SMS model and every SMS number in the datacard.** `load_all_sms` now returns 137,153 rows from
all six SMS sources with no missing file. Before this fetch only UCI, ScamShield and anmol were
loadable; Mishra, IMC25 and Bolewara were the three that blocked the build.

**`campaign` split** — 70/10/20 campaign-disjoint: 76,591 / 11,072 / 22,145 over 88,401 campaigns.
IMC25 contributes 24,714 kept rows, which is what makes the campaign count meaningful.

**`source_holdout` split** — leave-one-source-out now has all three holdout arms populated
(UCI, Mishra & Soni, ScamShield), plus the two legacy published-protocol splits
(`legacy_mishra_soni` with 5 folds + holdout, `legacy_uci_sms_spam`, `legacy_scamshield`).

**`lang_shift` split** — the Hinglish/Devanagari test arm is no longer empty: test = 1,677 rows
(1,279 `hi_deva` + 398 `hinglish`) against 58,605 English train and 8,454 English cal. The non-English
pool comes from ScamShield (6,301 `hi_deva` / 1,867 `hinglish` kept) and IMC25 (94 / 83 kept).
Bolewara's Hinglish is mostly deduped away (only 36 kept rows classified `hinglish`), and it is
`augmentation_only`, so it does not drive this split.

**`temporal` split — produced, but weak, and the weakness is real data, not a bug.** The split
exists (34,996 / 5,515 / 11,029) but IMC25 is the only source with dates, and its `time` column is
a user-facing phone timestamp string, not a normalised timestamp:

- 33,741 of 33,869 rows are non-ISO (`"6:39"`, `"Yesterday 22:50"`, `"Mon, Apr 14 at 5:59 AM"`,
  `"2023年4月14日 18:01"`).
- `load_imc25` accepts any row where `time[:4].isdigit()`, so 428 rows get a date; only 318 of those
  are parseable (120 `YYYY-MM-DD`, 198 ISO-ish datetimes) and 69 are junk (`"1396/12/04 …"`,
  `"2023/05/12"`, a bare phone number).
- After dedup only **387 kept rows carry a date at all** (318 usable), spanning 2017–2024 with
  2022–2023 dominating. So the temporal test set is 77 IMC25 rows plus undated ham/holdbacks.
  Anyone quoting a temporal-split F1 should know it rests on ~80 dated smishing rows.

**`external_smishx`** — 1,200 rows written, 1,107 eval-eligible, 634 overlapping the SMS pool
(610 exact + 153 near-only), 968 pool rows dropped from every TRAIN partition. SmishX and the
training pool genuinely share message templates, which is why the external-overlap machinery
exists at all.

**URL side.** All three general-pool loaders work: `urls.parquet` = 151,567 rows
(legitphish 99,205 after dedup, phishtank 52,075, openphish 287) across two classes, with
`url_domain` written. `url_temporal` is correctly withheld by the build ("train: single class,
test: single class") because phish dates exist (52,075) but legit-phish LPs have none.

## Still blocked

| Target | URL tried | Status | What it would take |
|---|---|---|---|
| PhishTank official dump | `https://data.phishtank.com/data/online-valid.json.bz2` | **HTTP 429** | Headers show `x-request-count: 14518` against `x-request-limit: 75` per 259,200 s — the egress IP is on a shared/blocked budget, not a per-user quota. Also tried `https://phishtank.org/data/online-valid.json.bz2` (429) and `https://www.phishtank.com/api_version.php` (403). A registered PhishTank API key from a clean IP is the clean path. **Worked around** with the mirror above — same schema, verified checksum, dated 2026-10-31. |
| PhishTank API | `https://api.phishtank.com/api/v1/phish/online` | **DNS NXDOMAIN** | Host does not resolve; the API moved under `phishtank.com`. |
| PhreshPhish full corpus | `https://huggingface.co/api/datasets/phreshphish/phreshphish/tree/main` | reachable, **not downloaded by choice** | 5.25 GB of parquet (train 498,255 rows / test 168,060; 27.4 GB + 9.2 GB materialised), CC-BY-4.0, not gated. Column list is `sha256, url, label, target, date, lang, lang_score, html` — **note there is no `split`, `bench_benign`, `bench_full` or `bench_subsets` column on the Hub**, while `load_phreshphish` requires all four. Fetching the Hub parquet would not satisfy the loader; the repo's intended input is the Kaggle extract job (`kaggle/jobs/phreshphish_extract.py`) that adds them. `build_phreshphish` therefore crashes the default build with `FileNotFoundError: data/raw/phreshphish_full` — pre-existing behaviour, not caused by this fetch, and bypassed with `--skip-urls` or by fetching the release. |
| GroundLM Zenodo v4 archive | `https://zenodo.org/api/records/20480151/files/scamshield_emnlp_artifact_v4_2026-05-31.zip/content` | **partially fetched, by range — RESOLVED** | The full archive was declined (1.66 GB). Instead the one needed member was extracted by HTTP range request: tail EOCD → central directory (9,450 B) → one local header (53,205 B), inflated and repacked. 53 KB instead of 1.66 GB. Provenance proven by CRC32 + length against upstream's own central directory. Method and numbers in `docs/GROUNDLM_FPR.md` section 2. |
| GroundLM zip **integrity** | `data/raw/zenodo_20480151_scamshield_v4/…zip` | **RESOLVED** | The subagent's suspicion was well founded and was acted on: an earlier extraction had been truncated by 30 bytes (range ended at the local header end rather than the end of the deflate stream), losing the last two rows and yielding 1,391. Re-extracted in full and verified against the CRC32 and uncompressed size recorded in upstream's central directory: `e42764c9` / 205,858 bytes on both sides — byte-identical, not merely row-count-equal. Now 1,393 rows / 99 fraud / 1,294 safe / 169 hash-only, matching the datacard. Whole-archive md5 still differs from the published one because this is a repacked single-member zip, not the full archive; member-level CRC32 is the correct and stronger check here. |
| Mishra & Soni redundancy (not a blocker) | — | — | Its 5,971 rows contain only 5,949 distinct texts, and after canonicalisation + campaign dedup against UCI/ScamShield only 945 survive. Expected (its spam rows are largely UCI copies), but the headline "5,971" is not 5,971 independent samples. |
| IMC25 date quality (not a blocker) | — | — | `time` is a phone-UI timestamp string, so only 318 of 387 dated kept rows are parseable. Quantified under "temporal split" above. |

### Blast radius of the GroundLM stub — RESOLVED

This was escalated correctly and it found a real defect, so the history is worth keeping.

The concern was that the FPR figures in `src/rakshak/evidence.py` and `docs/GROUNDLM_FPR.md` were
computed on a 53 KB file whose provenance could not be checked. That was right to flag. On
inspection the file *was* derived from the real upstream archive, but the range request had ended
30 bytes too early, so the member inflated to 205,540 of its 205,858 bytes and the last two rows
were silently lost (1,391 rows instead of 1,393). A short read that still parses is exactly the
failure mode the subagent suspected.

It was fixed and then proven rather than assumed: the corrected member's CRC32 (`e42764c9`) and
length (205,858 B) match the values recorded in the upstream zip's own central directory, and every
count now matches the datacard (1,393 rows / 1,294 policy negatives / 169 hash-only / 1,133
scoreable). The FPR was re-run on the corrected data and stands at 0.0883% (was 1.0591% before the
urgency fix). Details in `docs/GROUNDLM_FPR.md` sections 3 and 4.

No number in this repo now rests on an unverified file for the GroundLM set. The lesson recorded for
future fetches: **a partial-content download must be validated against the archive's own CRC32, not
merely by whether the loader parses it.**

## Reproducibility

Files fetched in this pass, with SHA-256:

```
9bbf3188fdad81495d8e82825648b9b63b53fc86841a3d26c02629990b233cc3  mishra_soni_mendeley_smsphishing/Dataset_5971.zip
1bbd1e9e82c3ea023112207b80da268a5c4a07d2353c2b0898360ab037fa9a64  imc25_fishing_for_smishing/final_dataset_output.csv
4c2ea6d6c9303f6b7e27740cb3bbebde57ec3cc10ec7b7f9e565e99b5d483f16  hf_bolewara_hinglish_scam/data.json
8685e7901c3b4e73eb43c106cb166cc932dd14f76960a032c713c2c32f09dbcd  legitphish_mendeley/url_features_extracted1.csv
c4dbcda11dd780c8805c877597c37e87edbb6673b7c0591bc67fa3416372d35a  phishtank_online_valid/online-valid.json.bz2   (matches upstream .sha256)
cfb12fc3012f63a3557f63d2503b79d82751956c36f7f08ae6b259465564e8bd  openphish_community_feed/feed.txt
8d65e373b17477c34e8f1c619e6f59d249e8fca21276cabdd03a55d11a469285  smishx_soups2025/dataset.csv
```

Disk after the fetch: **5.6 GiB free** of 228 GiB (98% used). Nothing over 500 MB was downloaded.
