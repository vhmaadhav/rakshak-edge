# URL general student: PhreshPhish plus url_domain training (pre-registration)

Written 2026-11-12, before any model in this arm is trained or any threshold for it is computed. It is
the W2 step of `docs/SOTA_PUSH_PLAN.md`. Frozen when the job is launched. Changes after that need a new
pre-registration. Maadhav delegated plan decisions to Claude on 2026-11-11; no result from this arm exists yet.

Current model (the comparator, not an arm): `kaggle/out/url_student` `best_url_student.onnx` (sha256
`1bbd2c57…`), scaler `url_side_scaler.json` (sha `c6b3fc96…`), trained on PhreshPhish train only. Its τ_url is
0.6308487409530461 (`docs/URL_SERVICE_PLAN.md`, `docs/URL_SERVICE_RESULTS.md`). It is scored here on the same test rows,
at its own τ_url, for the paired comparison only. Its results are not re-decided.

## Data

- **PhreshPhish (PP)**: `urls_phreshphish.parquet`, official split `official.train` / `official.test`
  (`splits/url_phreshphish.json`). Cal carve: `rakshak.eval.phreshphish.grouped_cal` on `official.train`, 5 %,
  seed 0, groups = registered domain (`reg_domain`, else `url_norm`), as in `url_student.py`. Sizes must be
  cal 18 087 and fit 480 168 (checked in code).
- **url_domain (UD)**: `urls.parquet`, partitions `train` (106 512), `cal` (15 215) and `test` (30 432) in
  `splits/url_domain.json`, grouped by registered domain by its own split file.
- Side features: the same 29 `rakshak.url_features` columns, in PhreshPhish's order, from both tables. The job
  checks that UD has all 29 and that the names match the current scaler's `columns`.

## Sets

- **Train** = (PP cal-carve fit rows) ∪ (UD train), **minus every row whose registered domain appears in
  either test set** (PP test or UD test). Rows with an empty `reg_domain` have no registered domain, so they
  are kept and never matched.
- **Cal** = PP carve cal ∪ UD cal. Used for early stopping and for τ. No row is dropped from cal.
- **Test** = PP official test (all rows, protocol and FPR) and UD test (all rows). Never used for any choice.
- **Overlap report** (from the job, before any scoring): for every pair of the six cells (train, cal, test ×
  PP, UD), the distinct shared registered domains and the rows on each side. Also the number of train rows
  dropped, and the exact URL strings shared between train and each test set after the drop (added to the
job before launch; report only). These are reported, not gated.

## Model and training

- Architecture and side input as the current student: `rakshak.models.charcnn` config `wide`, one raw view,
  max_len 200, the 29 side features. ONNX contract unchanged: input `raw` int64[batch,200], `side`
  float32[batch,29], output `logit`.
- Training as `kaggle/jobs/url_student.py` defaults: class-weighted BCE (pos_weight = n_neg / n_pos on the train
  rows), AdamW lr 2e-3, weight decay 1e-4, batch 512, max 8 epochs, early stop on cal PR-AUC, patience 3, best
  cal checkpoint kept. No distillation.
- Scaler refit on the train rows only (mean, std; clip 10; NaN as 0). The shipped scaler is written per seed.
- Seeds: 0, 1, 2 (torch seed). The cal carve is fixed (seed 0) for all seeds.

## Threshold rule (per seed, fixed before any test scoring)

- Cal rows are scored on the runtime path: `rakshak.url_features.url_features`, then that seed's scaler, then
  that seed's ONNX (`kaggle/jobs/url_threshold.score_url_strings`). That is the service path.
- **τ_s = the smallest observed cal score whose benign FPR is at most 1 % on the COMBINED cal benign rows**
  (PP carve benign and UD cal benign together), using `url_threshold.threshold_rule`. The job also reports the
  per-source cal FPR at τ_s (PP only, UD only). A flag is score ≥ τ_s.
- τ_s is written to `thresholds_s<seed>.json` before any other set is scored (`ThresholdGate` enforces the order).

## Gates

Evaluated per seed, at that seed's τ_s:

- **G1**: PP official test FPR ≤ 1.5 % and recall ≥ 0.85.
- **G2**: UD test FPR ≤ 5 %.
- **G3**: PP base-rate AP at 1 % (mean over the 29 benchmark subsets, `rakshak.eval.phreshphish`) ≥ 0.8647 − 0.02
  = **0.8447**.

Arm rule (the seed median plus at least 2 of 3 seeds): the arm **passes** when
(a) the median over the 3 seeds meets each gate's metric (median FPR_G1 ≤ 1.5 %, median recall_G1 ≥ 0.85,
median FPR_G2 ≤ 5 %, median AP_1% ≥ 0.8447), **and** (b) at least 2 of the 3 seeds meet G1, G2 and G3 together.

The ship decision is not made by this job. If the arm passes, a separate step decides whether it replaces the
current URL student, under its own pre-registration.

## Reported, not gated

- UD test recall at τ_s, UD test AP and ROC-AUC, per seed and median.
- PP protocol AP and P@R=0.90 at all base rates (0.05, 0.1, 0.5, 1, 5 %), the 24 % full benchmark, and the
  full test split.
- GroundLM (scoreable policy negatives) and SmishX (eval_ok) URL-message counts at τ_s, as in
  `url_threshold.py` (`_groundlm_table`, `_smishx_table`). Output `url_scores.parquet` holds one score column per
  seed for GroundLM and SmishX rows.
- Paired comparison with the current ONNX on the same test rows: its flags at its own τ_url versus the new
  arm's flags at τ_s, on PP test and UD test, with the discordant counts (new-only, current-only) and the exact
  two-sided McNemar p. AP per base rate, new minus current.
- ONNX parity per seed: max |Δlogit| between torch and onnxruntime on 2 000 **cal** rows (cal, not test, so
  parity does not touch test). Gate for the export: ≤ 1e-3. A failed parity fails that seed.

## Compute

- Kaggle T4 job `url_general` (build with `kaggle/launch.sh url_general --gpu t4 --internet`, data
  `maadhavvh/rakshak-data`, kernel output `maadhavvh/rakshak-url-student` for the comparator ONNX). No Modal.
- The laptop runs nothing except the offline pytest and the build dry run.

## Pre-launch amendment (2026-11-12, before any training or scoring of this arm)

Requested by the coordinator. Nothing has been trained or scored under the earlier text. The amendment
supersedes the drop rule in "Sets" and adds one reported result. Gates G1 to G3, the threshold rule and the
seed rule are unchanged.

1. **PhreshPhish split kept as published.** Its official train and test share registered domains by design.
   Dropping them removed 62 % of the fit rows (296 952 of 480 168) and would break comparability with the
   PhreshPhish protocol and with the current model. PhreshPhish official.train and official.test are used exactly as
   published, minus only the 5 % grouped cal carve (unchanged).
2. **Drop rule.** A training row is dropped only when its non-empty registered domain appears in the
   **url_domain test** set. This applies to every training row, so it removes PhreshPhish train rows whose domain is
   also in url_domain test. url_domain test is therefore domain-disjoint from training. url_domain train and test
   share no domain by its own split, so the rule removes no url_domain train rows in practice.
3. **url_domain cal.** It is reported for overlap with training (the overlap table), not dropped. This is my
   reading of "keep url_domain cal and test domain-disjoint from training as before". The drop rule above is
   the only enforced disjointness. If the coordinator meant cal too, that is a separate decision before launch.
4. **Domain-novel subset (reported, not gated).** The PhreshPhish official.test rows whose non-empty registered
   domain is not in the final training set (the domains of the training rows after the drop). Rows with an empty
   registered domain are excluded from this subset. For each seed, and for the current ONNX at its own τ_url:
   n, positives and negatives, AP, FPR and recall at τ, and the paired discordant counts with the exact McNemar p.
