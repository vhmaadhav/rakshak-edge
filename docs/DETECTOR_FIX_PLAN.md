# Detector fix plan: bank, UPI and telecom false alarms (2026-11-06)

Status: **plan, protocol fixed before training**. Training and evaluation of the candidate are in
`docs/STATUS.md` (2026-11-06 entry). Nothing here replaces the shipped student or the Pi registry default.

## 1. Problem

The shipped student (`sms_student`, dual byte char-CNN, base size, seed 1, threshold 0.35127517580986023,
`registry.toml`) flags 471 of 1,133 GroundLM policy negatives at its threshold (41.6%). In verify mode the
user-visible rate is 41.66% (`docs/GROUNDLM_EDGE_FPR.md`). Nearly all of these are bank, UPI and telecom
notices with score at or above 0.8. A threshold cannot fix it (smallest threshold for GroundLM FPR at most 1%
is 0.998, which cuts SmishX scam recall to 5.8%).

## 2. Diagnosis (data inventory, `data/processed/sms.parquet`, 137,153 rows)

Counts are over `text_canon` with a bank keyword (account, bank, debited, credited, balance, NEFT, IMPS, UPI)
or a rupee amount (`Rs`/`INR`/`₹` followed by a digit).

| Subset | Real (not synthetic) | Synthetic (ScamShield `Synthetic_Tier_C`) |
|---|---:|---:|
| Scam (label 1) with bank keyword | 13,212 | 148 |
| Ham (label 0) with bank keyword | **613** | 817 |
| Scam with rupee amount | 3,237 | 464 |
| Ham with rupee amount | **61** | **1,771** |

1. **Missing real Indian transactional negatives.** Real bank-keyword ham is 613 rows against 13,212 real
   bank-keyword scams, about 1:22. Ham is mostly personal chat (UCI, Mishra, ScamShield ham), so the
   pool has almost no real-world notice negatives.
2. **Generator shortcut.** Of the rupee-amount ham rows, 97% (1,771 of 1,832) are synthetic
   `Synthetic_Tier_C` rows. The model's only dense supply of "benign transactional" text is generator
   output, so it can learn generator style and the bank/rupee format as scam evidence, rather than
   notice-versus-scam. Real GroundLM notices look like the real scams, so they are flagged. This
   is consistent with the counts. It was not measured directly (no ablation was run).
3. **Source and label shortcut.** Bank and rupee tokens are much more common among scams than ham, so
   their presence alone is a strong scam cue in training. This explains why every bank (161/161) and UPI
   (146/146) GroundLM negative is flagged. It is a shortcut, not a calibration problem.
4. **Synthetic hard-negative augmentation already failed.** PR #10 (`docs/TRANSACTIONAL_HARDNEG_FUSED_RESULTS.md`)
   added 25% synthetic deterministic notices (six families, EN and Hinglish). It lowered GroundLM FPR
   (35 to 9-17%) but not to 1%. It also collapsed `lang_shift` (student FPR to 57-78% on one seed) and
   failed the SmishX recall gate. Synthetic notices are not a fix. Real notices are needed.

**Root cause (working):** training lacks real Indian transactional negatives, and the bank and rupee
features are learned as scam evidence. The fix must add real notice negatives with an honest held-out split.

## 3. Evaluation protocol (fixed before training)

**Held-out population.** The 1,133 GroundLM policy negatives with text and `eval_ok` (the scoreable set in
`docs/GROUNDLM_EDGE_FPR.md`), used with the 91 GroundLM fraud rows for recall.

**Group-disjoint split.** Each negative is clustered with union-find: two rows are linked if they have the
same `template_key` or the char-5-gram Jaccard of their `template_key` is at least 0.7 (the repo's near-duplicate
threshold, `docs/DATACARD.md` section 4). That gives 653 clusters (largest 59, 629 singletons). Clusters are
assigned to two folds greedily, largest first, to the smaller fold, so no near-duplicate template crosses the
fold boundary. Rows with the same campaign family but different wording stay in one fold.

**Cross-fit.** Model `M_k` is trained with GroundLM fold-`k` negatives added to the training pool, then
scores fold `1-k`. The pooled fold `0` and `1` scores cover all 1,133 negatives, each scored by a model that
never saw it. This is an estimate for held-out GroundLM-style notices. It is not the in-sample GroundLM
number.

**Deployment candidate.** `M_full` is trained with all GroundLM negatives added. It is the model that would be
registered. Its GroundLM rows are in-sample, so it is never scored on GroundLM. Its threshold is set on the
campaign cal split only.

**Unchanged evaluation sets.** Campaign partition (`splits/campaign.json`, `test` and `cal`), SmishX eval
rows (`eval_ok`, n = 1,107; smishing 226, legitimate 606, spam 275), Hinglish subset of the campaign test.
GroundLM and SmishX are not used for threshold choice, early stopping or model selection.

**Training recipe.** Same as the shipped `dual-base` seed 1: AdamW lr 2e-3, weight decay 1e-4, batch 128,
max 15 epochs, patience 3 on cal PR-AUC, rule side features on, same train split and overlap exclusions. The
only change is the added GroundLM negatives (label 0, `label_policy = benign_transactional`), each repeated
R = 5 times. R = 5 is fixed in advance and is not tuned: it lifts real notice negatives from 613 to about
3,400 rows against 13,212 real bank scams.

**Threshold.** Max-F1 on campaign cal, as the shipped student. One threshold applies to every decision
(cross-fit and deployment).

**Decision.** Verify mode, `decide()` and `apply_mode` from `deploy/pi/service/eval_groundlm.py` (the same
decision as `edge_decision.decide_edge`). The user-visible rate is `action != IGNORE`.

**Statistics.** Wilson 95% CI for rates. Paired bootstrap over campaigns for the campaign test (1,000 draws,
seed 13). Each count is shown as k/n.

## 4. Frozen gates (kill rule, decided now)

| Gate | Requirement | Shipped (reference) |
|---|---|---|
| G1 | Campaign test F1 at the candidate's cal threshold >= 0.97 | 0.980 |
| G2 | SmishX smishing recall (n = 226) at the candidate's threshold >= shipped at its threshold | 69.47% (score-only), 76.55% (verify visible) |
| G3 | Hinglish campaign test F1 >= shipped Hinglish F1 - 0.01 | from cache |
| G4 | Cross-fit GroundLM user-visible FPR (verify, pooled 1,133) point estimate <= 1% (Wilson CI reported) | 41.66% |
| G5 | ONNX parity on sampled rows: max abs prob difference <= 1e-4 | n/a |
| G6 | Frozen repo gates: the fused SCAM and >=UNCERTAIN gates (`docs/TRANSACTIONAL_HARDNEG_FUSED_RESULTS.md`) | not run here: this candidate is student-only, and the fusion gates need the fused pipeline |

Report-only: GroundLM fraud visible recall (91 rows), campaign visible FPR, SmishX visible FPR, per-category
rates (Bank, UPI, telecom), and score bands.

**Kill:** any of G1 to G5 fails, or G6 cannot be assessed, means no candidate registration. The result is
reported as is. No second training run is chained without the orchestrator.

## 5. Candidates (ranked by expected effect and risk)

1. **Real-negative cross-fit augmentation (chosen, top).** Add cluster-disjoint GroundLM notice negatives
   (label 0, real) to the shipped recipe, R = 5. Expected effect: the largest real-notice gain for bank,
   UPI and telecom. Risk: real bank scams may be suppressed (they look like notices); GroundLM-style templates
   can still share families with ScamShield ham, so the cross-fit is an estimate, not a guarantee. The
   sample is small (about 566 negatives per fold).
2. **Reweight the 613 real bank ham rows (no new data).** Cheap, but the sample is too small to counter 13,212
   real bank scams and likely leaves the shortcut in place. Not run.
3. **Separate transactional-notice second-stage gate.** Trained on the bank, UPI and telecom subset, it
   would downgrade the student's flag. Highest effect potential, highest risk: scams impersonate bank
   notices, and cal-only context fusion was already rejected (PR #6). Not run before candidate 1's result.

Rejected: more synthetic notices (PR #10 killed), and threshold changes alone (proven insufficient).

## 6. Compute and data

- Kaggle only, private kernel `maadhavvh/rakshak-sms-groupneg` (new job `sms_groupneg`), GPU T4 if it helps, no
  internet, datasets: `maadhavvh/rakshak-data` (existing). No new dataset upload is needed, since the GroundLM
  split is computed inside the job from `external/external_groundlm.parquet` in that dataset.
- No Modal, AMD or paid compute. No public artifacts. Laptop: code, unit tests, the eval decision on the returned
  probabilities.
- Output: per-row probabilities as parquet and metrics JSON, ONNX of `M_full` and parity, all in the job output.

## 7. Limits

- GroundLM is the repo's in-sample benchmark (`docs/GROUNDLM_FPR.md`); the cross-fit halves the held-out
  population, so CIs are wide (about 566 negatives per fold). Passing G4 is a point-estimate gate.
- Training on GroundLM negatives means GroundLM is no longer a pure external test for `M_full`. Only the
  cross-fit numbers are reported as held-out.
- Independent Indian held-out validation remains OPEN (`docs/GROUNDLM_FPR.md`).
