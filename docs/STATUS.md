# Current B checkpoint — 2026-11-01

- Parallel C milestone completed: PR #3 (`56892e7`), both source-control arms killed. The inherited KD pool reintroduced excluded rows; the historical ~0.80 Hinglish claim is withdrawn as clean evidence. C now guards train-only KD and reports all folds/external failures. See `docs/SOURCE_BALANCE_RESULTS.md` on `ci/source-balance-20261101`; no C code is mixed into this B branch. C verification830passed/21skipped, ruff clean. B remains pending on its original event subscription.

- Data gate passed; private synthetic dataset v1 published and verified. QLoRA version1/tree
  `e17dfb992e8b` is pending on its original single event waiter. No export or model-quality claim.
- Final saved-output audit adds paired point differences, per-family tests and an explicit
  positive-reassurance check; all changes predate inspecting model outcomes. Focused safety
  comparison tests: **7 passed in 0.99 s**, Tctl 40.2 C, ruff clean.

# Run updates — 2026-11-01: B (chronological)

- Frozen-audit attachment selector regression: **1 passed in0.27 s**, Tctl38.2 C, ruff clean. Synthetic files verify exact successful tree/hash binding and reject partial or duplicate attachments. No model/scoring change; post-audit bundle remains `1c08f453b8b9`, not launched until B finishes.

- Before model outcomes: verified the pinned mirror has no licence file. Its card links Meta's official Llama3.2 agreement. Retained the exact 7,669-byte text at upstream commit `8d29d93fa5700a60532e0061a02ffa89d0acd3fc`, SHA256 `8cc15535a8a34b41888f644b339a1a9eb428af793a4f5e24df58a3e5b1487d74`. Added licence/NOTICE/Built with Llama packaging for future training runs; the current running bundle is unchanged. The saved-output audit will copy the selected adapter verbatim into its private outputs, verify the weights hash and attach these notices. No inference, target or gate changes. Resource check **1 passed in 0.05 s**, Tctl41.8 C; ruff clean.

- Before model outcomes: conditional export retention criterion recorded in `docs/ELA_QLORA.md`: both language paired PTE-minus-FP16 lower CI bounds >= -5 pp, original absolute quality gates, zero unsupported/reassuring claims and no new slot harmful flips. No test-driven export retry; export remains forbidden unless the final QLoRA audit is go. No export code or run yet.

- Data job launched privately: `ela_data-b-data-20261101`, kernel **version 1**, base `2463245`
  dirty tree `aea5cd47a85a`. Args `--packages 1200 --attempts 8 --seed 13`; single event waiter
  opened, no polling. Data-gate tests **18 passed in 0.83 s** with restricted CPUs/threads;
  changed-file ruff clean, pre-test Tctl 41.2 C. Results pending; no training claim.
- Corrected data run launched: `ela_data-b-data2-20261101` **version 1**, tree `91cd95088bb2`,
  base `2463245` dirty; same args/model/seed. One completion subscription, no polling.
  Format-wrapper/E99 checks: **19 passed in 0.49 s**; lint clean, Tctl 42.0 C.
- QLoRA evaluation/comparison and bundle tests: **10 passed in 1.49 s**, changed-file ruff clean.
  The frozen evaluator receives an optional model pair and checks the exact prior sample manifest;
  default base evaluation is unchanged. Primary safety comparison is the ELA slot arm; rank is
  a negative control and freeform exploratory. Campaign bootstrap is primary; ordinary Wilson
  intervals are shown separately as IID references, not mislabelled as clustered Wilson.
- Issue #1 updated with A1 completion plus B/C predeclarations, launches and the negative data run.
- Qwen caveat independently checked on 2026-11-01: official
  [Qwen2.5-3B-Instruct card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/README.md)
  explicitly says `license: other`, `license_name: qwen-research`; it is **not Apache-2.0**.
  Not used for any targets. Chosen Llama pin/licence above remains unchanged.
- Bundle/evaluation dependency checks after QLoRA preparation: **8 passed in 0.81 s**;
  Tctl 44.0 C before the run. No complete suite run during ongoing experiments.
- Experiment protocol/provenance documented in `docs/ELA_QLORA.md`; no headline update while
  quality is pending. Frozen point-denominator mismatch now aborts comparison; synthetic gate
  checks **4 passed in 0.75 s**, Tctl 43.2 C, lint clean. Both isolated worktrees are ruff clean.
- Publication succeeded: **private** Kaggle dataset
  `maadhavvh/rakshak-ela-synthetic-b20261101`, **version 1**. Read-only metadata and dataset APIs
  independently returned `is_private=True`; recorded before training. Content hashes unchanged.
  No raw data, targets or weights staged in git. Final-run model stack will record exact versions.
- QLoRA launch: `ela_qlora-b-qlora-20261101` **version 1**, base `2463245` dirty tree
  `e17dfb992e8b`. T4/internet; attached `maadhavvh/rakshak-data`, private synthetic dataset v1,
  and frozen kernel `maadhavvh/rakshak-llama-ela-eval-a1-20261101`.
  Args `--data-manifest-sha256 03d5b85ec0d6e7ab313cb5b0fe849a6696d003a490f5deb52a4f20f333a6c69a
  --epochs 3 --seed 13`. Single `wait_event --timeout 43200` opened, no status polling.
  Together with Track C there are two GPU kernels; data runs are complete. Results pending.
- Before model results: predeclared a paired quality supplement from saved generations only.
  Absolute EN go/kill gates remain unchanged; additionally report tuned-minus-base pooled
  point-pass campaign CI and null-centred paired p-value for EN/Hinglish. No new generation,
  sample change or tuning. `ela_paired_audit` binds both successful kernel trees and hashes,
  rejects unmatched IDs/point denominators, then resamples campaigns with common point counts.
  Planned attachment: frozen A1 output + completed `ela_qlora-b-qlora-20261101` output.
  Synthetic statistic tests **3 passed in 0.06 s**, pre-test Tctl 40.0 C, ruff clean.
- Saved-output supplement will also report paired McNemar/Fisher tests and separate Wilson
  references per attack family, retaining zero-eligible rank families as unestimated (not zero risk).
  Raw/rendered unsupported flags on the same row count once in the row total; any flag still kills.
  No model-generation/training change. New synthetic reporting tests passed; see focused output.
- Expanded post-output comparison fixtures: **6 passed in 1.81 s**, pre-test Tctl 39.4 C;
  changed-file ruff clean. Model inference still uses the frozen `e17dfb992e8b` bundle;
  supplement changes reporting only and binds those saved generations.


# Session log — 2026-11-01: safety reporting clarification before model outcomes

- The scorer reports positive safe/no-risk reassurance separately from unsupported IDs/entities.
  A clean/injected pair can both reassure, producing no *flip*. Before inspecting model outcomes,
  the saved-output supplement therefore records raw/rendered reassurance rows separately and
  withholds go/kills shipment if any occur, just as the training data gate already rejects them.
  This strengthens the existing zero-unfounded-claims condition; thresholds/model/data unchanged.
  Action-only new harmful flips still block go independently. No production verifier change.

# Session log — 2026-11-01: B comparison scope clarification before results

- Keep the known design exposure explicit: this is a repeated frozen-sample comparison,
  **in-sample for rule/contract design**. Base outcomes were observed and ScamShield test rows
  informed rule features; no clean independent held-out improvement claim is authorized.
  QLoRA targets still contain no corpus/eval messages, dev-only selection and predeclared
  gates are unchanged. Independent Indian validation remains OPEN.

# Session log — 2026-11-01: B data gate passed

- `ela_data-b-data2-20261101` v1, tree
  `91cd95088bb27e2cf1b77588dff5d9d4e5dc75037a35e777ffe5a4b6bc6d07d2`, **ok**, 331.9 s;
  one event/fetch, provenance and file hashes verified. Same packages=1200/attempts=8/seed=13.
- Teacher attempts **256**, kept **131** (EN51/128, Hinglish80/128), rejected **125** (77/48).
  Exact final targets: **10,934 train / 3,026 dev**, **250 unique singleton parent records**;
  571 duplicate prompt/target candidates removed. Target sources: 9,194 deterministic templates,
  122 distinct direct Llama paraphrases, 4,644 verified compositions. Full raw generations and
  per-reason rejections retained privately in `paraphrase_audit.jsonl`/manifest.
- Frequent rejection reasons include malformed JSON/extra text, positive instructions (e.g.
  `pay`, three occurrences), and verifier evidence-keyword failures. No rejected sentence was
  rewritten or repaired into a target. All retained targets passed both gates, all composition
  groups are train/dev-disjoint, no corpus messages were read. **Keep data path**, not a model win.
- Manifest SHA256 `03d5b85ec0d6e7ab313cb5b0fe849a6696d003a490f5deb52a4f20f333a6c69a`.
  Publication target: private `maadhavvh/rakshak-ela-synthetic-b20261101`; first create command
  rejected title length before uploading, shorter title retry pending. No public flag used.

# Session log — 2026-11-01: B data run 1 negative result and format correction

- `ela_data-b-data-20261101` v1, tree
  `aea5cd47a85a6bb731d1f34524447bebd26da19210f6304638dff84a7863a02c`, outcome **error**,
  303.2 s. Single event received and fetch verified. Args `--packages 1200 --attempts 8 --seed 13`.
  All **256 paraphrases rejected** (128 EN / 128 Hinglish): model frequently emitted a JSON
  string instead of the requested object, or extra text. No training and no publication.
  10,594 train / 2,976 dev template/composition rows were written, but are incomplete output,
  not a successful dataset. No quality result follows.
- Minimal format correction: allow a valid teacher JSON string to be enclosed verbatim as the
  single `points` item; no sentence rewriting, malformed-output repair or gate relaxation.
  E99 still rejected by the scorer even after wrapping. Same prompt, model, seed and counts.
  Go for corrected data run: gated paraphrases >0, train/dev composition disjoint, all targets
  pass both checks, hashes/provenance complete. Otherwise stop before QLoRA.

# Session log — 2026-11-01: B QLoRA experiment predeclaration

- Rebased isolated `ci/ela-qlora-20261101` onto `2463245`; PR #2 left for review.
- Hypothesis: existing deterministic templates plus verifier-gated pinned Llama paraphrases
  teach evidence verbalisation without corpus messages or assistant-written targets.
- **Go:** EN point-pass campaign-clustered lower 95% CI >40% (base 25.81%, [13.04,40.82]);
  Hinglish not worse than 62.16% beyond CI overlap; unsupported claims remain zero;
  harmful flips do not increase across five injection families (paired McNemar/Fisher vs base).
  **Kill:** EN lower bound <=25% or any new unsupported claims. Between gates is inconclusive,
  no export. Frozen 300 messages and injection pairs are evaluated only after dev selection.
- Teacher chosen: `unsloth/Llama-3.2-1B-Instruct` revision
  `5a8abab4a5d6f164389b1079fb721cfab8d7126c`. Its pinned model card was read on 2026-11-01:
  `license: llama3.2`, the custom Llama 3.2 Community License, not Apache-2.0.
  [Pinned card](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/blob/5a8abab4a5d6f164389b1079fb721cfab8d7126c/README.md).
  Built with Llama attribution retained. Qwen not used; Qwen size-specific licences not assumed.
- Data job: 1,200 abstract packages, existing four template variants, eight paraphrase attempts
  per evidence type/language, gated singleton composition, exact deduplication, hash-disjoint
  train/dev compositions. No SMS corpus loader. Both verifier and scorer unsupported-ID gates;
  E99 regression retained. Private Kaggle output will be published as a private training dataset.
- Training plan: T4 NF4/double-quantized base, LoRA r=16 all-linear, masked completion loss,
  max 3 epochs, early stopping by synthetic-dev verifier pass. Exact runtime compatibility
  remains UNVERIFIED until the Kaggle run. No local model compute.
- During development: touched tests only, `nice -n 19 taskset -c 0-3`, OMP/MKL threads 2;
  check Tctl before each run, wait at >=80 C. Full suite once at finished milestone only.

# Session log — 2026-11-01: B preparation (separate from review PR)

- PR #2 at `2463245` is open for Maadhav; leave it unchanged. This isolated B branch is
  rebased onto that milestone; its changes will get their own PR into main.
- **Predeclared hypothesis:** existing deterministic targets plus verifier-filtered pinned
  Llama paraphrases can teach the ELA contract without corpus test messages or assistant-written
  targets. Go for the data path: traceable target provenance, every retained target passes the
  verifier and raw unsupported-claim checks, train/dev evidence compositions disjoint. Kill
  the path before training if any invariant fails. No training/performance claim yet.
- Implemented `rakshak.llm.ela_training`: abstract evidence packages only, existing template-bank
  strings, deterministic summaries, exact pinned Llama/Kaggle provenance for paraphrases.
  Corpus text/URLs/phones and non-synthetic or test partitions are rejected. Same composition
  across languages/order/risk variants shares one train/dev partition. Exact teacher copies
  and template targets share IDs for deduplication. No data file is committed.
- The unit gate exposed that `check_points` alone allows copied IDs such as E99; `score_advisory`
  already marks those unsupported. Training requires both checks, plus no positive safe claim.
  Production verifier behaviour is unchanged in this preparation.
- Model-free tests under `nice -n 19`: **14 passed in 0.43 s**; changed-file ruff clean.
  Tctl 70.1 C, below the 85 C cutoff. No local model generation/training/benchmark evaluation.
- Primary method/API sources checked before implementation:
  [QLoRA](https://arxiv.org/abs/2305.14314),
  [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization), and
  [Transformers bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes).
  They establish frozen 4-bit base + LoRA, NF4/double quantization and preparation APIs;
  they do not establish Rakshak results. Exact package pins/T4 runtime/Hinglish paraphrase
  yield and ExecuTorch adapter export remain UNVERIFIED. No B Kaggle job launched yet.
# Completed C-adversary gate — 2026-11-01

- Saved-output audit v1/tree `f50b5a0c3bbd5b55d085b046d1ee1df0d7a21516ca6bae98facf45e3a5ae21a1`, **ok**,60.3s; one wait/fetch. Recorded decision **go**: ScamShield mean F1 0.733961→0.749275, paired delta0.015314 [0.013591,0.017423], p0.000999; other-source guards pass.
- Campaign ERM→adversary .974774→.975553; lang_shift .369773→.396633; Mishra .937259→.945886; UCI .882857→.891299; temporal .871039→.866594. All seeds/absolute CIs/paired tests/external failures in `SOURCE_ADVERSARY_RESULTS.md`. No prior ERM substitution, seed selection or held-out improvement claim.
- Keep the intervention for further validation; no checkpoint deployment. Independent Indian validation OPEN. The newly requested fused false-alarm audit takes precedence over further source-method experiments.

# C next experiment predeclaration — 2026-11-01

- Static provenance check while jobs are pending: corrected the datacard's IMC25 synthetic=True typo to match the existing loader's `synthetic=False`. This documents the source flag, not independent authenticity. No data, split, model code or running bundle changed.

- Launched **sms_source_adversary-c-adv-20261101 v1**, dirty base56892e7, tree `a09fb20ed34903e9605e7f7e48c26aed055a4da18ae4a41a0d7c8588214b9573`. Exact planned args/attachments/T4/internet above; original single wait_event subscriber opened. Together with B this is two GPU kernels. Startup Torch checks and all model results pending; no runtime/quality pass claimed.

- Prior C milestone is committed56892e7 and open as PR #3; both balancing/DRO arms killed.
  This new isolated `ci/source-adversary-20261101` branch leaves that PR and PR #2 unchanged.
- Hypothesis: conditional training-source adversarial regularization reduces learned source
  shortcuts without aligning different label marginals. Fixed candidate: learned CNN features
  only, per-training-label MLP64 source heads, CE1 with reversed encoder coefficient ramping
  to.05 over3epochs, inverse label/source-cell weights. No test/external training rows.
- Go: positive ScamShield paired campaign-bootstrap mean-F1 lower CI; no other source-fold loss
  larger than ERM CI half-width. Otherwise kill. Seeds1/2/3, same six frozen splits and
  train-only KD, max15/patience3 cal PR-AUC, AdamW.002/wd.0001, batch128, alpha.7/T2.
  Cal-only thresholds; all external operating points reported without tuning. Detailed primary
  sources and limits: `docs/SOURCE_ADVERSARY.md`; adaptation, not exact DANN/CDAN, benefit UNVERIFIED.
- Focused metadata/gate/dependency checks: **10 passed,1 skipped in0.18 s**, pre-test Tctl37.5 C,
  changed-file ruff clean. Torch absent locally; synthetic gradient/RNG/zero-weight checks are
  mandatory on Kaggle before corpus access. No full-suite rerun before next finished milestone.
- Planned private T4 job `sms_source_adversary --run-name c-adv-20261101 --internet
  --datasets maadhavvh/rakshak-data maadhavvh/rakshak-teacher-logits
  --job-args="--seeds 1 2 3 --epochs 15 --n-boot 1000 --teacher-seed 0"`.
  One completion subscription/fetch only after launch; B remains on its original event waiter.
  No new experiment result or deployment claim yet.

# Completed C experiment — 2026-11-01

- **Final C milestone verification:** ONE complete unit-suite invocation exited0:
  **830 passed,21 skipped,144 warnings in1376.39 s (0:22:56)**. Required nice19,
  taskset0-3, OMP2/MKL2; temporary CPU quota15% then25%, thermal guard85 C.
  Pre-run Tctl38.2 C, final38.4 C; no thermal interrupt. Torch is absent locally;
  source-gradient/KD invariants and54 actual fits succeeded on Kaggle. Full `ruff check .`:
  **All checks passed!** No further source-code changes after these checks.
  Git fetch completed; base2463245 matches the reviewed branch (0ahead/0behind).
  Repo-local identity checked as `vhmaadhav <vhmaadhav@hotmail.com>`; no override.
  Prepare one C guard/report milestone commit and a separate PR into main; no merge.

- Milestone full unit suite remains on its single invocation. Started with15% temporary CPU quota; after sustained38 C and confirmed near-continuous CFS throttling, raised only that scope to25% (same nice/affinity/thread limits and85 C guard). The monitor's printed15% label denotes its initial configuration; final verification records both actual quotas. No restart or new test run.

- `sms_source_balance-c-clean-20261101` **v1**, tree
  `62a788d27aa9db20b84964f00b8208d3025c3be8f5128352d5c76c53b5ae589e`, base2463245 dirty:
  **ok in5,316.6 s**, six frozen splits,54 successful fit units. Args `--seeds 1 2 3
  --epochs 15 --n-boot 1000 --teacher-seed 0`; private data + teacher-logit attachments,
  T4/internet. Original ONE event waiter completed and ONE fetch verified. No skipped folds.
- `sms_source_audit-c-audit-20261101` **v1**, tree
  `e83b49d3e41ca29a849b1c0247bfe95ed9b140e918f565a6f9fa3c8bd5166239`: **ok in74.1 s**,
  args `--source-tree 62a788d27aa9db20b84964f00b8208d3025c3be8f5128352d5c76c53b5ae589e`.
  Private data + corrected kernel attachment. ONE event/fetch; both decisions **kill**.
  Source metrics SHA256 `db3811b2ede62d577e3dad40684276d0ec0580d231d0505cafe3d5d54fb46538`.
- ScamShield mean F1: ERM0.750254; balanced0.743732, paired delta−0.006522
  CI[−0.007940,−0.005173]; DRO0.739853, delta−0.010402 CI[−0.012204,−0.008830].
  Both primary gates fail; other source-fold loss guards pass. Do not cherry-pick temporal gains.
  Complete per-split/per-seed/threshold/external results: `docs/SOURCE_BALANCE_RESULTS.md`.
- Membership audit **confirms** inherited KD contamination: campaign4,147 exact test-copy
  extras; ScamShield2,122 exact test copies,11,137 test-campaign extras,9,707 held-source rows.
  Counts overlap; all original train/test campaign checks zero except temporal3 (unchanged).
  Historical ~0.80 Hinglish KD claim withdrawn as clean evidence; fresh train-only ERM mean0.4000.
- Implemented default train-only KD plus rejection of `--kd-unlabeled`; programmatic
  `build_training_pool` also refuses extras. Legacy broad builder retained for membership audits.
  Guard/membership tests **14 passed in0.82 s**, pre-test Tctl41.0 C; changed-file ruff clean.
  No new training/eval needed: corrected remote experiment already used exactly this train-only pool.
- GroundLM model FPR is poor: ERM seed1 at cal-1%-FPR threshold407/1,133=35.92%,
  cluster CI[21.80%,48.54%]; SmishX fraud267/881=30.31% [25.81%,34.67%]. All seeds reported,
  no external threshold tuning. GroundLM/rule-feature design is **in-sample**; SmishX is
  **independent, non-Indian-specific** for corrected models. No clean held-out improvement claim.
  Independent Indian validation remains OPEN; PhreshPhish is not applicable to SMS-only changes.

# Current C checkpoint — 2026-11-01

- Initial broad-pool run cancelled by user, server acknowledged, single listener retired;
  partial artifacts quarantined without inspecting model outcomes.
- Corrected train-only `sms_source_balance-c-clean-20261101` v1/tree `62a788d27aa9` completed
  ok; saved-output audit also ok. Both interventions **kill**. Train-only KD guard and complete
  negative results are prepared for their own milestone PR; full suite830passed/21skipped and ruff clean.

# Run updates — 2026-11-01: C (chronological)

- Acceptance completeness check now requires exactly all three frozen source folds (UCI, Mishra–Soni, ScamShield), whose canonical names were checked in `datasets/splits.py`/`loaders.py`. Missing-fold fixture included: **2 passed in 0.01 s**, Tctl36.2 C, ruff clean. Final locally prepared audit tree `e83b49d3e41c`; still not launched, no inference/gate changes.

- Static frozen-split review caught a pre-launch acceptance-helper naming mismatch: partitions are `source_holdout:holdout_scamshield` (and corresponding `holdout_*` sources), not the abbreviated name. Corrected helper/fixtures, **2 passed in 0.01 s**, Tctl36.2 C; ruff/diff clean. Rebuilt saved-output audit locally only (tree `6c165e1fdc0e`), not launched. Running corrected C training bundle/thresholds unchanged.

- Before outcomes: added a saved-output `sms_source_audit` acceptance check. It requires a successful exact source tree, all seeds1/2/3 for every run, both paired arms, positive ScamShield lower CI and the predeclared other-fold CI-half-width limit. No training/inference or gate changes. Synthetic boundary/missing-result checks **2 passed in 0.01 s**, pre-test Tctl36.0 C; changed-file ruff clean. Run only after corrected C finishes; retain the original subscription.

- C development checks: **2 passed, 3 skipped** (local torch absent; synthetic torch gradient/KD
  invariants run at job start on Kaggle before training). Changed-file ruff clean. Tctl 42.1 C.
  Campaign-fitted models also report external GroundLM/SmishX FPR/recall at cal-frozen thresholds;
  GroundLM/rules design exposure remains in-sample, SmishX independent non-Indian-specific.
  PhreshPhish is a URL protocol and is not an endpoint for this SMS-only intervention.
- Launched privately: `sms_source_balance-c-src-20261101` **version 1**, base `2463245`
  dirty tree `8a3a0189e52a`. Single completion subscriber open, no polling; results pending.
- Additional synthetic statistical check distinguishes mean F1 over the three seeds from F1
  of a majority vote: **3 passed, 3 skipped in 0.08 s** total focused checks, Tctl 40.6 C.
  The three torch-dependent checks remain skipped locally; runtime invariants are in the Kaggle job.
- User confirmed cancellation; **one** cancellation-verification API/CLI read returned
  `KernelWorkerStatus.CANCEL_ACKNOWLEDGED` (no polling loop). C's original single waiter is
  retired; the cancelled container cannot send its callback. One fetch attempt archives or
  reports missing partial outputs; those model outcomes are quarantined and not used.
  Aggregate pool-audit fixtures: **2 passed in 0.08 s**, Tctl 39.1 C, ruff clean.
- Corrected run planned: `sms_source_balance --run-name c-clean-20261101 --gpu t4 --internet
  --datasets maadhavvh/rakshak-data maadhavvh/rakshak-teacher-logits
  --job-args="--seeds 1 2 3 --epochs 15 --n-boot 1000 --teacher-seed 0"`.
  Frozen train IDs only for all arms, no unlabeled extras; data/model hyperparameters otherwise
  fixed. Campaign, source_holdout, lang_shift, temporal and external endpoints unchanged.
- Cancellation archive: fetch succeeded once, with partial `results.json` status `running`
  from the last checkpoint (2,274.3 s), **not** a successful completion. Tree
  `8a3a0189e52afaa87a922f829eeec3734c72f2ef38fa2dc7fd678eff7612d866`; server cancellation was
  separately acknowledged. Partial predictions/metrics remain private and uninspected/quarantined.
  Exactly one local old-run waiter was terminated; B's listener was preserved.
- Replacement launched: `sms_source_balance-c-clean-20261101` **version 1**, dirty base
  `2463245`, tree `62a788d27aa9`. Same declared seeds/hyperparameters/attachments; train-only
  pool correction + pre-training membership audit. One `wait_event --timeout 43200` opened.
  B + this replacement are the only active GPU kernels. No model results yet.
- The quarantined run's log was queried only for the exact startup test marker:
  `source objective synthetic torch invariants passed`. It confirms the remote source-loss
  gradient and KD mean-equivalence checks passed before training. No model outcome was read.


# Session log — 2026-11-01: C protocol correction before inspecting model outcomes

- Code review found that inherited `sms_teacher.dump_transfer` scores **every corpus row outside
  test IDs**; `build_pool(... include_unlabeled=True)` adds all those extra rows to KD training.
  Because `sms.parquet` retains raw duplicates and campaign-excluded rows, ID-only exclusion
  can reintroduce test copies/campaign neighbours and external-overlap removals. Actual counts
  are UNVERIFIED until Kaggle membership audit; no model outcomes were inspected for this decision.
- First C run `sms_source_balance-c-src-20261101` v1/tree `8a3a0189e52a` is **protocol-invalid
  for the source-generalization go gate**. Cancellation requested, not yet achieved: browser
  controls are disabled; installed CLI exposes no cancel command/session-ID resolution. The
  official SDK supports cancellation only with a known true session ID; no ID is guessed.
  Its single event waiter remains open. User asked asynchronously to cancel or let it finish.
- **Corrected predeclaration:** all three ERM/balanced/DRO arms use exactly each run's original
  training IDs with their matched teacher logits (`include_unlabeled=False`); split files and
  external sets unchanged. Historical ~.75 is a reference, not an assumed reproduction; the
  paired control is fresh train-only ERM. Same three seeds, hyperparameters and go/kill gates.
- New job audits inherited extra-row membership (exact test/cal copies, test campaigns, external
  exact overlap, held-out source) on Kaggle before training and writes only aggregate counts.
  Go for protocol: train positions equal frozen train positions, no extra KD rows. Kill any
  source-generalization claim if this invariant fails. No outcome-driven sampling/threshold tune.
  Replacement run will use a new name only after one GPU slot is confirmed free.

# Session log — 2026-11-01: C scope clarification before results

- Source_holdout is the frozen split name, not an end-to-end independence guarantee:
  **in-sample for rule-feature design**, because ScamShield test rows informed those rules.
  Concurrent ERM controls use the identical frozen features. Report paired split changes
  with that caveat; no clean independent held-out improvement claim. No test/cal labels
  enter training pools, and cal-only selection/thresholds remain unchanged.

# Session log — 2026-11-01: C source-intervention predeclaration

- Isolated branch `ci/source-balance-20261101` from `2463245`; PR #2/main untouched.
- Hypothesis: equal training-source sampling, and a second balanced+group-DRO arm, lift
  dual-base KD student ScamShield-holdout F1 (~0.75; teacher ~0.87) without harming other folds.
- **Go:** paired campaign-clustered 95% CI for ScamShield mean F1 improvement excludes zero;
  no other source-holdout fold loses more than its ERM CI half-width. Otherwise withhold
  an improvement claim/kill deployment. Three fixed seeds 1/2/3 per arm; report all folds and
  campaign/lang_shift/temporal. Campaign CI is conditional on those three training seeds.
- Controls: ERM, balanced replacement sampling at identical update count, and balanced+group-DRO
  (exponentiated group weights eta=0.01). Same dual-base, KD alpha .7 / T=2, AdamW lr .002 /
  weight decay .0001, max 15 epochs, patience 3 by cal PR-AUC. Cal-only thresholds; no test tuning.
  Group labels are training sources, not held-out-source outcomes. Teacher files matched to each
  run, hash/membership checked; unlabeled soft targets retain masked labels.
- Primary research: [Sagawa et al., group DRO](https://arxiv.org/abs/1911.08731), which explicitly
  warns that regularization/early stopping matters; no gain is assumed. This is a source-group
  adaptation of the method, not an exact reproduction. Runtime/result UNVERIFIED before Kaggle.
- Planned private job `sms_source_balance --run-name c-src-20261101 --gpu t4 --internet
  --datasets maadhavvh/rakshak-data maadhavvh/rakshak-teacher-logits
  --job-args="--seeds 1 2 3 --epochs 15 --n-boot 1000 --teacher-seed 0"`.

# Session log — 2026-11-01: review findings A

- Branch scope: reviewed commits are on `sachin/handoff-tasks-1-2`; `main` is `20ea2b0`.
  Fixes are prepared in an isolated worktree. One finished milestone, one PR into main;
  do not push main. Repo identity verified: vhmaadhav <vhmaadhav@hotmail.com>.
- **A1 predeclaration:** character lengths masquerade as rank target tokens and empty
  evidence pairs make the rank injection estimate vacuous. Separate generated tokens (zero)
  from exact tokenizer continuation IDs; restrict rank injection to pairs with candidates
  on both sides; match freeform comparisons by ID, preserve five attack families, Wilson,
  Fisher, and add exact McNemar plus campaign-clustered rate/paired intervals.
  Go: exact accounting, empty rows do no model work, eligible-only denominator, clean tests
  and lint, successful full Kaggle rerun. Kill: withhold the rank safety/performance claim if
  any invariant fails; no threshold/model/rule tuning from this test.
- Planned run: `llama_ela_eval --run-name a1-20261101 --gpu t4 --internet
  --datasets maadhavvh/rakshak-data --job-args="--n 300 --inject 150 --lang-n 150"`.
  Base code `038bdd3`, dirty bundle identified by launcher tree SHA; kernel version/results
  pending. Local model-free checks: **94 passed**, changed-file ruff clean.
- **Contamination caveat:** GroundLM urgency downgrade used the GroundLM outcomes, and
  dc223d3 used all ScamShield including test. Post-hoc subdivision does not undo exposure.
  Report retrospective frozen-split audits as in-sample; independently validate where possible.
  Historical 0.0883% is rules-only, in-sample; no model-side FPR conclusion follows.
- A1 launched: private kernel version **1**, tree `48ed512ef46e`, base `038bdd33022d` dirty.
  ONE `wait_event.sh llama_ela_eval-a1-20261101 --timeout 43200` subscription is open.
- **A2/A3 predeclaration:** compare byte-identical pre-dc223d3 rules (SHA256
  `c7c3e3185c982b9879f9f45800d6d1ccca06408a646bb584ae62ce7ccd2ce9a3`) with current rules,
  and isolate the urgency downgrade using the exact pre-f29521a risk ladder. Production
  rules and thresholds are frozen. Above-LOW includes abstention; HIGH alone is SCAM.
  Show any-evidence counts separately; compute cluster CIs and paired A-B tests.
  GroundLM and ScamShield campaign-hash 20% audit / 80% remainder are explicitly
  **in-sample: these rows informed rule design**. Existing split membership is unchanged.
  SmishX is the sole **independent, non-Indian-specific** validation. Go: reconcile
  GroundLM counts, retain warn-FPR <=1% on scoreable policy negatives as an in-sample
  gate, and report independent FPR/recall even if they fail. No held-out improvement claim.
  Kill: withhold any gate/improvement claim not supported by those frozen comparisons.
  Local synthetic-fixture audit tests: **8 passed**, changed-file ruff clean.
  Planned run: `rule_review_audit --run-name a23-20261101 --gpu t4 --internet
  --datasets maadhavvh/rakshak-data` (no training and no model fitting).
- A2/A3 launched: private kernel version **1**, tree `b1e16e0f1ee9`, base
  `038bdd33022d` dirty. ONE `wait_event.sh rule_review_audit-a23-20261101 --timeout 43200`
  subscription is open; together with A1 this uses two GPU slots. Results pending.
- Full-repository ruff: **All checks passed!** First full pytest attempt interrupted (exit 130)
  at Tctl **85.6 C**, per thermal cutoff. Retrying after cooling under `nice -n 19` with a
  temporary user-scope 40% CPU quota. No local training/evaluation was run.
- **A2/A3 results:** version 1 finished `ok` in 368.1 s; single event received and fetch
  verified tree `b1e16e0f1ee9369d0c9811e69ac433490f1da6918649c350bbe14dbca59ccf10`.
  Counts: 1,393 released GroundLM rows / 1,294 negatives / 169 text-less (161 negatives) /
  1,133 scoreable and eval-eligible negatives. **12 evidence hits, 11 urgency-only, 1 current
  warning/HIGH flag**. The “13/12” narrative is withdrawn. In-sample warning FPR 12/1,133
  to 1/1,133 (1.0591% to 0.0883%); current cluster CI [0%, 0.2971%]. Audit subset: 3/131
  to 0/131, but all rows informed design and Wilson upper bound is 2.85%.
- **Independent, non-Indian-specific SmishX** (1,107 eligible; policy labels 881 negative /
  226 fraud): warning FPR 72/881 to 53/881 from urgency downgrade, warning recall 57/226
  to 38/226; widening then reaches 39/226 with no new negative warning. Current warning
  FPR **6.0159%** [3.6571%, 8.8725%], recall **17.2566%** [10.6118%, 26.0009%].
  HIGH FPR **1.1351%**, recall **2.2124%**. SmishX fails the 1% reference gate at both cutoffs.
  Widening paired recall delta +0.4425 pp [0, 1.4360] pp, p=0.5485: no independent gain claim.
- ScamShield campaign audit (14,702 eligible rows; in-sample): widening fraud evidence
  447/1,012 to 485/1,012; warning recall 419/1,012 to 458/1,012; negative warning flags
  81/9,248 to 82/9,248. Full per-split and paired results: `docs/REVIEW_AUDIT.md`.
  Decision: **keep measurement corrections; withhold independent safety/improvement claims**.
  Production rules/thresholds were not tuned. Independent Indian validation remains OPEN.
- Full unit suite with one worker / `nice -n 19` / temporary 40% CPU quota:
  **817 passed, 18 skipped, 144 warnings in 117.49 s**. Skips are missing optional torch/ONNX
  paths; no local model benchmark was run. The earlier multi-worker attempt was interrupted
  to remove oversubscription. Full-repository ruff remains clean.
- HF archive check: `hf models info highestintheroom/rakshak-models` returned “not found”
  with the current authentication; privacy could not be verified, so no upload or repo
  creation was attempted. Full metrics/manifests/predictions remain in the **private**
  Kaggle kernel outputs; no data, weights or secrets are staged in git.
- **A1 results:** private kernel version 1 finished `ok` in 1,622.0 s, no failed units;
  event and verified fetch match tree
  `48ed512ef46e0abf629fe7e8da89e1dbca68dbd275516f113900e12eb852932d`.
  Clean rank: **2,992 actual continuation tokens**, **248 candidate strings**, **zero generated
  tokens**, **86% zero-work rows**. Worked T4 p50/p95: **0.3110 / 1.0782 s** (not Pi timing).
  Injection rank **0/39** eligible pairs, **111 excluded**; Wilson [0%, 8.97%], versus
  matched freeform **11/39**, exact McNemar p=0.0009766. Full freeform **35/150**, slot **0/150**.
  All five families retained. English point pass **25.81%** overall, **21.62%** on the matched
  150-package language subset; Hinglish **62.16%**. Rendered unsupported claims zero; raw
  slot/rank unsupported claims zero. This is an architectural restriction, not learned robustness.
- **A1 interval supplement predeclaration:** compute pooled point-pass and package-rate
  campaign-clustered CIs from the exact saved generations, without regenerating text, changing
  samples, or tuning. Go: bind to the source tree, retain all point denominators and cluster
  together per message/campaign; kill any CI claim if attachment/provenance fails. The repo
  launcher now accepts `--kernels OWNER/SLUG`, the official `kernel_sources` metadata field
  ([Kaggle CLI docs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)).
  Six synthetic-fixture/source-attachment tests passed; changed-file ruff clean.
  Planned `ela_recompute --run-name a1-ci-20261101 --gpu t4 --internet
  --datasets maadhavvh/rakshak-data --kernels maadhavvh/rakshak-llama-ela-eval-a1-20261101
  --job-args="--expected-tree 48ed512ef46e0abf629fe7e8da89e1dbca68dbd275516f113900e12eb852932d"`.
- Interval supplement: private version **1**, outcome **ok**, **72.0 s**; one event wait and
  verified fetch, tree `9730c50a862b67341634e16c5d07bcbe26574ed84c7d19fe5763d69ddc1e4909`.
  Frozen generations SHA256 `8a898ef0d8edcf92598729ef8bf6e94e7bc5674f7d914d854aed02cf799e04d1`.
  Pooled raw point passes: English 16/62 = **25.81%** [13.04%, 40.82%]; matched English
  8/37 = **21.62%** [8.00%, 40.00%]; Hinglish 23/37 = **62.16%** [40.61%, 79.59%].
  High package pass mostly reflects empty packages; this does not establish semantic entailment.
  **A1 go criterion met:** real token accounting and empty-work invariants verified in artifacts,
  eligible rank denominator, all five families retained, full inference and interval jobs succeeded.
  Decision: **keep correction**; withhold learned robustness/on-device speed claims.
- Final full-suite rerun after the interval/attachment additions stopped at Tctl **87.0 C**
  (557 passed, 11 skipped before interrupt). Retrying after cooling at a lower temporary
  **15% CPU quota**, single worker / `nice -n 19`. Focused interval/attachment tests: **6 passed**;
  previous complete suite **817 passed, 18 skipped**. Do not present interrupted runs as passes.
- **Final milestone verification:** lower-quota complete run exited 0:
  **822 passed, 18 skipped, 144 warnings in 360.89 s (0:06:00)**. Full `ruff check .`:
  **All checks passed!** Final Tctl 56.1 C. No further source changes after these checks.
  One milestone commit on `sachin/handoff-tasks-1-2`; update the existing **PR #2** into main,
  with no direct main push, no force push and no automatic merge. Independent Indian
  held-out validation, model-side GroundLM, QLoRA and device measurements remain OPEN.

# Status — 2026-10-31 (Sachin session)

## Session log: 2026-10-31

Environment: macOS arm64, 10 cores, 16 GB, ~5.7 GB free disk. No HF token (not in org
`highestintheroom`), Kaggle account `sachin0496` has no access to `maadhavvh/rakshak-*`, no ntfy
topic. So: no model training this session; everything below is rules/data/eval code plus a full
public-data rebuild.

Commits (all pushed to `main`):

| Commit | What |
|---|---|
| `b2a5ab5` | Rank-arm audit + bash-3.2 portability fix + evidence-coverage measurement |
| `f29521a` | GroundLM FPR test (first open handoff task) |
| `a9d3a1a` | GroundLM: corrected to full 1,393-row data, CRC32-verified |
| `65a260c` | GroundLM: measured cost of the fix on real data |
| `d25e011` | Data fetch status: GroundLM entries corrected |
| `058b230` | Build: actionable error when PhreshPhish is absent |
| `2951c9b` | Verifier: document MAX_SUMMARY 280 vs prompt 200 |
| `d90213b` | E2E detection harness + rules-path measurement |
| `e2cfa62` | E2E: lang_shift and temporal split results |
| `dc223d3` | Rules: sentence-scoped parcel/job/prize widening |
| `63f9323` | Verifier: LOW-risk summary no longer reassures |

**Closed: handoff task 1 (GroundLM FPR).** `docs/GROUNDLM_FPR.md`. Rules-only FPR on the 1,133
scoreable policy negatives went **1.0591% -> 0.0883%**, passing the <=1% go/kill line. Root cause:
`urgency` (generic, fires on "जल्दी"/"I'm leaving early"/legitimate "your bill is due") could set
`Risk.UNCERTAIN` on its own; 11 of 12 false positives were urgency-only. Fix in
`evidence.py::_risk` downgrades a package whose evidence is *entirely* urgency, unless fusion
already called it a scam. Cost, measured on real ScamShield: -5.6 pp fraud recall for -1.06 pp
benign FP; that recall sits in 14 urgency-only rows whose real defect is narrow
`kyc_block_urgency` / `parcel_customs` coverage, so it is recoverable by rules work, not by
re-inflating the keyword.

**Half-closed: handoff task 2a (rank-arm audit).** `docs/RANK_ARM_AUDIT.md`. The 0 s / 0 token
reading was not a broken timer. `rank_candidates_for()` returns one entry per evidence item, so an
empty evidence package (any benign message) makes the arm do zero model work. `new_tokens` was
hardcoded to 0 and nothing recorded the work done. Now the job records `rank_items` / `rank_empty` /
`rank_targets` and `aggregate` reports `zero_work_rate` plus `*_worked` latencies. The GPU rerun is
still needed for the numbers. **The real finding is evidence coverage: 39.1% of genuine fraud
produces no evidence at all.**

**Fixed: 43 failing tests.** `colab/launch.sh` used `${GPU^^}` and `mapfile`, both bash 4+; the Mac
runs bash 3.2, so launch.sh died on its own log line. Suite went 43 failed -> **786 passed, 19
skipped**.

**Data rebuilt from public sources** (`docs/DATA_FETCH_STATUS.md`). SMS now reproduces the datacard
exactly: 137,153 raw -> 110,827 kept -> 88,401 campaigns; campaign split 76,839/11,072/22,145;
lang_shift test 1,677. Seven of eight targets fetched and loader-verified; GroundLM recovered from a
1.66 GB archive by HTTP range request (53 KB fetched, CRC32-verified byte-identical). PhreshPhish
not fetched (5.25 GB) and its Hub columns cannot satisfy `load_phreshphish` anyway.

**NEW: `scripts/e2e_detection.py` + `docs/E2E_DETECTION.md`.** Scores the full local path
(`package_for_sms` -> three-state decision) on a held-out split, `--json` for diffing. Headline on
`campaign/test` (22,145 rows): **precision 0.9259, recall 0.0201, F1 0.0393, FPR 0.17%, abstention
5.9%, and 83.8% of fraud produces no evidence at all.** The document's whole point is that this is
*not* a detector result: the rules are a high-precision evidence extractor, and the 54% "accuracy on
decided rows" is the arithmetic of a component that mostly says nothing. Handoff section 5's F1 0.98
comes from the learned teacher/student, not this path.

Two consequences that constrain everything else: (a) because the Llama layer may only verbalise
package evidence, **only 16.2% of real fraud has anything to explain** — the binding constraint on
the ELA claim; (b) at 5.9% abstention the calibrated three-state design has collapsed to two states.
Cross-checked on `label_scam` (recall 0.0157), so the promotions-inflate-recall caveat is real but
immaterial. `lang_shift` recall is 3x `campaign` recall (6.2% vs 2.0%), the *opposite* of the learned
student's Hinglish collapse — do not assert "the rules fail on Hinglish". `temporal` is degenerate
(0 true positives).

**Rules widened** (`docs/RULE_COVERAGE.md`, subagent work, independently re-verified). New
sentence-scoped subject+trigger pairing with guard phrases. Fraud with no evidence 39.10% -> 36.09%
on the 3,000-row sample, benign `>=UNCERTAIN` unchanged at 1.38%; whole corpus 43.65% -> 41.27%,
benign 1.23% -> 1.25%. On `campaign/test` independently: scam no-evidence 84.60% -> 83.78%,
`prize_lottery` 370 -> 423, tp/fp/fn unchanged — it widens what the explainer can cite on ~1 pp more
fraud and does not move recall.

**Safety fix.** The LOW-risk template said "No scam signals were found by the rules", which is what a
user sees for the 83.8% of real scams that produce no evidence — a clean bill of health for four out
of five scams. Reworded to state the limit of the check and keep the caution, with a regression test
verified to fail on the old wording.

## Still open

1. Model-side GroundLM FPR (needs Kaggle/HF access).
2. ~~Rules coverage widening~~ — **done**, `docs/RULE_COVERAGE.md`.
3. Rank-arm GPU rerun for the real latency/token numbers.
4. PhreshPhish URL pipeline: needs the Kaggle extract job, not the Hub parquet.
5. Temporal split rests on 77 dated smishing rows — do not quote temporal F1 as robust.
6. `apk_install` false positive at HIGH on a bare Devanagari "download" verb.

---

# Status — 2026-10-30

## Done
- Repo + initial commit `8005d18` (vhmaadhav), pushed. Everything after it is uncommitted.
- Core package, data pipeline (dedup, campaigns, splits), eval (campaign-clustered bootstrap, paired tests, `rakshak.eval.recompute`), Kaggle launcher with ntfy completion.
- Data rebuilt with better Hinglish lang-ID and Hinglish/Devanagari rules. docs/DATACARD.md numbers are stale; regenerate.
- Processed data on Kaggle as `maadhavvh/rakshak-data` (the older build; re-push with `kaggle/push_data.sh` after the rebuild).
- URL baselines: char-LR F1 0.991, recall 0.851 @ cal FPR ≤0.01% on url_domain (75% phish test, not realistic base rate).
- Research verified: docs/RESEARCH_NOTES.md, docs/SOTA_TARGETS.md (PhreshPhish v2 numbers), docs/CONTEXT.md decisions.
- SmishX (1,200, MIT) downloaded; GroundLM Indian Financial SMS release (Zenodo 20480151) downloading into data/raw.
- HF repos moved to org `highestintheroom` on 2026-10-31 (`rakshak-data`, `rakshak-models`, private). Docs point at the new names; HF access needs an org-scoped token.

## In flight when the session stopped
- Kaggle: `sms_baselines`, `sms_teacher` pilot (XLM-R, 1 epoch). Fetch with `kaggle/fetch.sh <job>`.
- Agents (may have finished or stopped): chunked labelling tool (`src/rakshak/llm/`, `docs/BUDGET.md`), PhreshPhish extract job, student models (`src/rakshak/models/`, `kaggle/jobs/sms_student.py`), Llama ELA eval + ExecuTorch export jobs.

## Overnight 2026-10-30 → resume here
Running on Kaggle (remote; no local listeners are running):
- `sms_teacher`: mmBERT on campaign + source_holdout with `--transfer-texts` (distillation logits).
- `nids_baselines`: T1/T2/T4 + T5 LODO, home-baseline calibration, anomaly/hybrid, ONNX student.
- `sms_teacher-b`: mmBERT on lang_shift + temporal with `--transfer-texts` (launched; fetch with `kaggle/fetch.sh sms_teacher-b`).
On resume: `kaggle/fetch.sh sms_teacher`, `kaggle/fetch.sh nids_baselines` (and `sms_teacher-b`), review, then:
1. Distil SMS students from mmBERT transfer logits (3 seeds) — fixes Hinglish collapse.
2. GroundLM FPR test (target ≤1%), URL CNN 3 seeds, NIDS review (check hybrid MCC, bot_iot caveat).
3. Llama jobs after Maadhav accepts the Llama 3.2 licence on kaggle.com/models/metaresearch/llama-3.2.
Colab free tier reclaimed a detached run after ~45 min; use Colab only for short jobs. No Colab sessions open.
Uncommitted: NIDS module + teacher/Colab changes (next milestone commit).

## Paused 2026-10-30 (laptop off)
- Still running on Kaggle (remote, unaffected): `sms_teacher` bake-off (MuRIL, IndicBERTv2, mDeBERTa, XLM-R, mmBERT), `sms_student`, `sms_baselines`. On return: `kaggle/fetch.sh <job>` once each (no waiters running).
- Not launched (Kaggle 2-GPU limit): `kaggle/launch.sh url_student --gpu t4 --internet --datasets maadhavvh/rakshak-data --job-args="--kd --budget-h 9"`.
- Blocked on Maadhav: accept Llama 3.2 licence at kaggle.com/models/metaresearch/llama-3.2, then launch `llama_ela_eval` and `llama_export` (commands in docs/EXECUTORCH_PLAN.md).
- Done since last note: data re-pushed (PhreshPhish full, SmishX + GroundLM external tests, label_policy/label_scam, clustered CIs); XLM-R pilot F1 0.981 campaign split, canonicaliser cuts attack success 4.1%→0.4%; PhreshPhish URL-only baselines AP 0.436/0.737/0.909 at 0.05/1/5% (paper's GTE uses URL+HTML: 0.605/0.876/0.938, reference only).

## Next
1. Kaggle jobs pass `cluster_ids` (campaign_id / reg_domain) so CIs are campaign-clustered; re-push data.
2. Add SmishX + GroundLM as external test sets; spam-vs-scam label taxonomy.
3. Teacher bake-off (MuRIL, IndicBERTv2, mDeBERTa, XLM-R, mmBERT) once the pilot passes.
4. Student + distillation; Llama ELA eval; ExecuTorch export (accept Llama licence on Kaggle/HF first).

## Rules
Kaggle for all compute; no Modal without Maadhav's explicit per-run OK; cloud-credit labelling parked; commit very rarely as vhmaadhav.
- New --run-name tests (tests/test_kaggle_run_name.py) not yet run: CPU was 92–94 °C from a non-project process.

# Run update — 2026-11-01: B training completed, evaluation recovery

- Original event receipt: `ela_qlora-b-qlora-20261101` v1, tree `e17dfb992e8b`, **error**, 15,716.4s. One original waiter and one fetch; no polling. All three training epochs finished. Synthetic-dev verifier pass: base 4/48, epochs 43/48, 46/48, 45/48; epoch2 selected by the frozen dev-only criterion. These are **synthetic-dev**, not frozen evaluation results.
- Failure occurred at FP16 `PeftModel.from_pretrained`, before evaluation: image torchao0.10.0 is below PEFT0.21.2's minimum0.16.0. Confirmed in official [PEFT source](https://github.com/huggingface/peft/blob/v0.21.2/src/peft/import_utils.py). No quality gate can yet be applied; no export.
- Recovery predeclaration: run the same frozen evaluation on the exact selected adapter (SHA256 `1bdf548fd40ee245ae9a32693af04a89e13448bbe5a9224be7d0808403986e87`) and exact original training metadata (`1b9c15e3da6edec332ca5172c093a43f70b673d0925827b3f768a1d481ef5fa5`). Remove the unused optional torchao package from the new Kaggle environment, retain FP16 base+adapter and sample/injection manifest. **No retraining, target/model selection changes or gate changes.** Saved-output paired audit follows successful recovery.

- Recovery launched `ela_resume_eval-b-recover-20261101` **v1**, tree `d8120f4e72e4`, frozen base+failed training kernel attachments, no job args. Focused recovery/provenance and bundle checks **5 passed in0.08s**, Tctl41.8C before; full ruff clean after line wrapping. One blocking wait_event opened; no fetch/poll yet. C-adversary passed its gate and is now open/unmerged in PR#4; PRs2/3 remain untouched. New priority after B's recorded decision is the full frozen fused-system false-alarm audit.

# B recovered frozen evaluation — 2026-11-01

- `ela_resume_eval-b-recover-20261101` **v1**, tree`d8120f4e72e43bb1b11ef28af26e782d3f71df0b1e8eac60df3264fb0dac3de6`, **ok2744.6s**; one original wait/fetch, exact adapter/training hashes retained, no retraining. Completion summary EN point-pass96.77%, raw full-contract13.33%, slot harmful flips0/150, eligible-rank0/39, exploratory freeform36/150. These are preliminary summary fields, not a final go decision; low overall raw full-contract pass is retained, not hidden by point-pass improvement.
- Final predeclared saved-output `ela_paired_audit-b-paired-20261101` **v1**, tree`1c08f453b8b9`, attached exact base plus successful recovery. Args `--tuned-tree d8120f4e72e43bb1b11ef28af26e782d3f71df0b1e8eac60df3264fb0dac3de6`. One wait_event opened. Computes frozen campaign/paired intervals, raw/rendered unsupported/reassuring rows and all five-family safety comparisons; packages exact adapter with licence. Gates unchanged. No export before its final decision.

# B final QLoRA gate — 2026-11-01

- Final saved-output `ela_paired_audit-b-paired-20261101` **v1**, tree`1c08f453b8b906cb67faaea2979259f79fa1391ba1ec27d8c976892d6f1c797b`, **ok79.0s**, one wait/fetch. **Recorded decision GO** under unchanged gates: EN60/62=96.774% campaignCI[91.227%,100%], HL24/37=64.865% [43.463%,80.956%], overlapping baseHL CI. Unsupported advisory rows0, positive reassurance rows0; slot no new flips,0/150 base and tuned, all five families0/30. Eligible rank0/39 both (111 empty excluded), no generated rank tokens. Ordinary IID Wilson references retained separately.
- Paired EN gain+70.968pp CI[55.088,85.250]pp,p0.000999; HL+2.703pp CI[−23.077,30.303]pp,p0.8711. Frozen sample is **in-sample for rule/contract design and repeated benchmark comparison**, not independent held-out improvement.
- Freeform exploratory harms35/150→36/150,28new/27removed,McNemar/Fisher1.0,pairedCI[−8.667,+10.000]pp. Some attack families worsen; report every family, no freeform safety claim. Raw full-contract pass13.33% remains a limitation despite high point pass; deterministic verifier/fallback is still necessary.
- Exact selected adapter hash preserved and private licence/NOTICE/Built with Llama packaging verified. GO permits only the already predeclared ExecuTorch8da4w export/quality check: merge selected adapter, no prebuilt/base substitution; paired language lower differences>=−5pp plus original absolute/safety gates. No export success or quality retention claimed before an actual run.

# Conditional B export preparation — 2026-11-01 (after final GO)

- New `ela_export` uses final audit tree/hash plus exact FP16 inference/adapter hashes; refuses non-GO/changed weights. Safe-merge selected LoRA into pinned FP16 Llama in a fresh subprocess, convert/check Meta shapes, then install existing ExecuTorch1.5.1/Torch2.14/pytorch-tokenizers1.5 recipe and export8da4w XNNPACK GS32/embedding4,32 in another process. No base/prebuilt PTE substitution or quantisation tuning on test data. Licence/NOTICE/Built with Llama retained.
- Same600 slot/Hinglish rows (300ENclean+150ENinjection+150HLclean), exact sample IDs and evidence denominators; runner reset per message, greedy, stop at first complete JSON, token counts from runtime statistics. Rank/freeform are explicitly unmeasured on PTE, not zero-risk results. Quality retention uses the predeclared paired−5pp language margin plus original absolute/unsupported/reassurance/injection gates. Kaggle x86 CPU latency is not Pi/Android performance.
- Focused runtime/comparison checks **13 passed in0.91s**; export input/stop/reset/non-inferiority/bundle checks **5 passed in0.66s**, pre-test Tctl39.4C, ruff clean after wrapping. No real local model loaded/trained/evaluated. No export success claim before Kaggle outcome.

- Conditional export launched `ela_export-b-export-20261101` **v1**, dirty base2463245/tree`5490836c7692`, T4/internet/private data + final GO-audit/recovery kernels. No job args. Single event subscription opened. Exact predeclared recipe/margins above; no new quality result. P1 reporting audit is complete; only B export remains active at this moment.

# B export infrastructure correction — 2026-11-01

- `ela_export-b-export-20261101` **v1**, tree`5490836c7692ee1af47d0fc1bdbb35ae86eb229802a0957cc67eb98c50ae3683`, **error140.7s**, one wait/fetch. Safe-merge reached parameter conversion; `model.config.to_dict()` from Transformers5 omits the converter's top-level`rope_theta`, causing KeyError. No quantisation/export/PTE inference occurred; quality retention remains unmeasured.
- Correct the integration to read the **original pinned snapshot config.json**, exactly as existing `llama_export._convert_to_meta` does, and preserve that source config in the artifact. Same selected weights, architecture, 8da4w recipe, sample and margins. This is a format-compatibility retry before any PTE quality outcome, not a test-driven quantisation change. Failed run remains logged; no false quality kill or success.

- Compatibility retry launched privately `ela_export-b-export2-20261101` **v1**, dirty base2463245/tree`20c6e21feefa`, same inputs/no job args/model/recipe/margins. Source-config/export binding focused checks **20 passed in0.09s**, Tctl41.9C, ruff clean. One blocking wait_event opened, no fetch/poll. Only this export kernel is active; no new PTE quality outcome or safety claim.

# B adapted export exists; runtime operators unresolved — 2026-11-01

- `ela_export-b-export2-20261101` **v1**, tree`20c6e21feefa82e003b07b5cfe950a8e9bac64c4db0bae9887c3343f76b9719d`, **error805.5s**, one wait/fetch. Adapter merge/shape conversion and8da4w export succeeded:603.15s, peakRSS9,154,016KiB; actual PTE **860,959,104bytes**, SHA`af24472d931ee449a571368f5f1f70d61822c25398c6627b8cf26159a649e98c`. Export delegation-lines list empty; no acceleration claim. Runtime Torch2.14.0+cpu,ET1.5.1,torchao0.18.0,tokenizers1.5.0.
- Load/generation failed before any scored row: missing`llama::custom_sdpa.out`, `llama::update_cache.out`, `quantized_decomposed::embedding_4bit.dtype_out` (49 unresolved instructions). `pte_generations.jsonl` has0rows. **Quality retention unmeasured**, not go or a measured quality kill. No Pi/Android result.
- Primary-source research: pinned [runner CMake](https://github.com/pytorch/executorch/blob/v1.5.1/extension/llm/runner/CMakeLists.txt), [custom-op loader](https://github.com/pytorch/executorch/blob/v1.5.1/extension/llm/custom_ops/custom_ops.py), [custom-op CMake](https://github.com/pytorch/executorch/blob/v1.5.1/extension/llm/custom_ops/CMakeLists.txt), [wheel packaging](https://github.com/pytorch/executorch/blob/v1.5.1/setup.py), read through official GitHub API. They show separate AOT libraries and `torch.ops.load_library` registration; compatibility with this wheel/runner is **UNVERIFIED** until a new runtime-only job.
- Runtime-recovery hypothesis: required AOT libraries exist but were not loaded into this fresh quality process. Experiment explicitly loads Llama+quantized AOT libraries, checks operator registry, then scores the **same hash-pinned PTE** against the same FP16 rows. Go remains original retention/absolute/safety gates on all600rows; missing libraries/operators/load failure keeps deployment blocked/unmeasured. **No re-export, new quantisation recipe, weights, test-based calibration or sample changes.**

- Runtime-only recovery implementation: official wheel AOT libraries are discovered via package search paths (including namespace packages), loaded with `torch.ops.load_library`, and required operator-name fragments checked before constructing TextLLMRunner. Same PTE/source GO/FP16 hashes are enforced; no quantisation/export function is invoked. Missing libraries/registration aborts, never silently falls back. Synthetic runtime/source/bundle tests **6 passed** (timing in focused log), Tctl38.5C, ruff clean. A stub library loader is used locally; no actual native library/model loaded on this laptop.

- Runtime-only `ela_pte_eval-b-runtime-20261101` **v1**, dirty base2463245/tree`23c75bd0aa88`, private data+GOaudit/FP16recovery/originalexport2 inputs, no job args. One event subscription opened, no polling/fetch yet. Focused tests6passed in0.76s; Tctl38.5C, ruff clean. No re-export or quality claim. Concurrent hard-negative job is the only other GPU kernel.

- Formatted `ELA_QLORA_RESULTS.md` directly from saved remote aggregate statistics, without rescoring: per-language pooled point counts, campaign CIs + IID Wilson references, paired deltas, all five injection families for slot/rank/freeform including worsening families. Raw full-contract13.33%EN/10.00%HL and86.67%/89.33% fallback remain prominent. The advisory unsupported0 endpoint is explicitly scoped to gated slot/rank; exploratory freeform clean unsupported1.67% (5/300) is not hidden. Runtime-only PTE recovery is still pending; no quality-retention claim or new fetch.

# Private B artifact publication preparation — 2026-11-01

- Per Maadhav's HF404 clarification, publish final experiment adapter/PTE as a private Kaggle dataset after the existing runtime receipt. New `kaggle/publish_ela.py` verifies exact GO-audit, selected adapter, original adapted PTE and licence SHA256s plus the completed runtime bundle tree; allowlists models/tokenizers/licence/config/aggregate JSON only, excludes generation/message files. Create-only `public=False`, then metadata `is_private` verification. **HF publish pending**; no HF publish attempt or dependency.
- Failed runtime may be archived as experimental/unmeasured with shipment_approved=false, never promoted into a quality-retention win. Same-PTE runtime job still pending; no quantisation changes or additional recovery attempt permitted this iteration if operators fail again. Synthetic allowlist/hash/failed-runtime archive check **1 passed in0.02s**, Tctl60.5C; ruff clean. Publisher has NOT run against actual models or uploaded anything yet. No actual local inference/training/benchmark scoring.

# Final B same-PTE runtime quality — 2026-11-01

- `ela_pte_eval-b-runtime-20261101` **v1**, full tree`23c75bd0aa885c0493e5b0d654c1603c65a56ff6c6b34392f44a6241f8ddbc67`, **ok5746.1s**, original ONE event and ONE fetch verified. Same PTE hash/860,959,104bytes; no re-export/model/recipe/sample change. Explicit Llama+quantized AOT libraries loaded; all three required operator fragments registered,600 frozen rows scored (slot450incl150injections+HL150). Metrics SHA`02a04cdaf542e744f42e6949db0b3e82b84d5544f3af25cd7b512b9ca3076809`.
- **Export retention gate FAIL/KILL this deployment candidate.** EN52/62=83.87097%, campaignCI[68.55069,96.29630]%; FP16→PTE paired−12.90323pp CI[−28.95644,0.00000]pp,p.103896. HL22/37=59.45946%,CI[36.83772,82.76108]%; paired−5.40541pp CI[−28.89035,+17.85714]pp,p.745255. BOTH lower bounds fail the fixed−5pp margin. Non-significance is not non-inferiority; no claim that loss is proven/absent from these CIs. Original absolute advisory gates independently report GO, but cannot override retention failure.
- Unsupported gated advisory rows0/reassuring0; slot harmful flips0/150 FP16/PTE, all5families0/30,new0,McNemar/Fisher1.0. IID Wilson reference per0/30 harm[0,11.35]% (zero observations do not certify zero population risk). Rank/freeform remain unmeasured on PTE. Kaggle x86 CPU only, no Pi5/Android quality/performance/4GB-fit claim.
- Stop B export at this measured negative result, with NO new quantisation recipe/weights/runtime retry. Retain FP16 QLoRA GO as an experimental verifier-gated result (raw full-contract13.33%EN/10.00%HL/fallback limitation remains). Archive exact selected adapter/PTE privately with failure clearly recorded, verify Kaggle privacy; **HF publish pending**. Full milestone verification/one commit/new PR follows final publication/provenance checks; no merge.

# B finished milestone verification — 2026-11-01

- ONE complete unit-suite invocation: **861 passed,18 skipped,144 warnings in178.88s**, exit0. Required nice19/taskset0-3/OMP2/MKL2 plus50% CPU quota and85C stop guard; Tctl39.6C before/45.5C after, no thermal interruption. Local Torch absence is explicit in skips; actual training/inference/export/600-row quality all ran on Kaggle. Full `ruff check .`: **All checks passed!** No model/training/evaluation source changes after this verification. The later publication-only verification-endpoint fix has its own focused checks; full suite not repeated.
- Ref fetch completed; B HEAD matches reviewed base2463245 exactly (0ahead/0behind before its milestone), origin/main remains ancestor. Repo-local identity verified **vhmaadhav <vhmaadhav@hotmail.com>**, no override. Private model upload is in progress with exact-hash/shipment=false/retention=false manifest; do not claim final privacy verification until the publication script completes. One milestone commit/new PR follows; never merge or force-push.


# B private archive verified — 2026-11-01

- Exact artifact upload/create request completed. Immediate GetDatasetMetadata verification returned403; this was a publication verification failure, not a model or privacy pass. No re-upload/version/recreation performed. A separate single owner GetDataset read succeeded: **maadhavvh/rakshak-ela-models-b20261101, isPrivate=true, version1, owner maadhavvh,940,569,686bytes**. Safe verification receipt retained locally; source model hashes unchanged, quality_retained=false,shipment_approved=false. **HF publish pending**.
- Publication-only helper now verifies the owner GetDataset record (exact ref/owner/private/version), avoiding the failed metadata endpoint. Focused synthetic allowlist/privacy-owner checks **2 passed in0.02s**, Tctl40.1C; ruff clean after line wrapping. This small post-full-suite API fix changes no training/inference/scoring/export/gate. Full861-pass milestone suite above is not rerun; the new API branch is covered by these focused tests.
# New top-priority fused audit — 2026-11-01

- Hypothesis/experiment/go-kill frozen in `docs/FUSED_FPR_AUDIT.md` before benchmark compute: GroundLM SCAM<=1%, >=UNCERTAIN<=10%, SmishX SCAM recall>=student-only frozen cal1%-FPR. Report unconditional negative denominators separately from existing selective-library conditional FPR; campaign CIs and paired recall tests, no fitting on external.
- Frozen shipped fusion state/thresholds unavailable in checkout; private HF access returns404 via connector/local account. Clarification requested; do not substitute invented shipped state. Preparing code/aggregate-only diagnostics independently. No GPU audit launched or result claimed.

- Frozen triage audit helpers verified on synthetic abstention/tie and empty-negative cases: **2 passed in0.63s**, Tctl42.0C before, ruff clean. Rates over all negatives differ intentionally from conditional auto-decision FPR; unknown endpoints cannot pass the gate. No model fitting or benchmark scoring occurred locally.

- Aggregate diagnostic preparation: proxy/statistic checks **4 passed in0.65s**, Tctl40.5C; saved-score ID-binding/training-count fixture **1 passed in0.68s**, Tctl40.4C, after correcting module import/Canon field errors caught by unit tests. Ruff clean. `sms_transactional_audit` is prepared but NOT launched. It emits aggregate counts/CIs only, reports missing sender metadata and unknown policy labels; no geography/authenticity inference. Corrected ERM source tree+metrics hash and every eligible saved score ID are enforced. No benchmark outcomes inspected or used to define categories.

- Candidate(b) predeclared before external audit outcomes: paired baseline-schema vs opt-in protective-context logistic fusion, unchanged corrected ERM seeds1/2/3 + fixed URL expert, every frozen fold, four campaign cal stages, no external fitting, no test-selected hyperparameters. Same global fused gates plus no GroundLM FPR increase/no frozen-fold recall loss beyond baseline CI half-width. This is preparation for a **new cal-only baseline/candidate**, not a shipped-system measurement. Opt-in feature schema leaves the default matrix/state unchanged and validates saved schemas; cal-ID poison/round-trip and affected fusion tests pass (final output recorded next). No new model job launched.

- Affected fusion checks **18 passed in0.84s**, Tctl39.8C before, ruff clean. Candidate feature names use `transaction_keywords` rather than asserting notification authenticity; these are lexical indicators, not trusted sender evidence. Before any new benchmark run, design-set register now records GroundLM/SmishX **aggregate feedback** as in-sample for selection of the new intervention; earlier SmishX independence remains scoped to the frozen reviewed rules/control. No raw external messages were used to author candidate features.

- Cal-only preparation: fixed four-campaign stages and opt-in feature round-trip/isolation checks **3 passed in0.66s**, Tctl39.8C, ruff clean. Frozen URL aggregation contract checked without loading a model (**1 passed in0.63s**, Tctl39.5C). Source model/scaler hashes recorded in audit plan. No fitted benchmark state, GPU launch or actual URL inference yet; deployed-state question remains pending.

- Cal-only candidate runner prepared, not launched. All six canonical folds and all three seeds are required for go; missing/infeasible data is unmeasurable. Every fitted state is saved/hashed before test/external decisions; saved external scores are exact-ID bound. Paired external FPR/alert/recall and internal recall tests are included; known fraud and legacy labels stay separate. Synthetic gate/ID guard/isolation checks **4 passed in0.64s**, Tctl44.5C before, ruff clean. No benchmark result, shipped configuration or deployment claim.

# Diagnostic compute predeclaration — 2026-11-01

- Use the free second kernel while B evaluation recovery continues: `sms_transactional_audit`, planned run `p1-diag-20261101`, args `--n-boot 1000`, private data plus exact corrected-source output. Hypothesis: transactional lexical/structure proxies are concentrated in external student false alarms and underrepresented among frozen training fraud negatives. Experiment reports fixed-category counts/rates/CIs, all three ERM seeds and train source/unknown-policy counts from saved predictions at unchanged cal1%-FPR thresholds.
- This is diagnostic measurement, not an intervention go claim: completion requires exact source/hash/eligible-ID binding and all predeclared aggregate endpoints; invalid/missing provenance kills the audit. Lexical proxies cannot establish real Indian authenticity or a causal data gap, so that explanation stays UNVERIFIED without independent real-SMS provenance. No row examples, model fitting, teacher call or threshold changes. Fused-system/candidate gates remain unchanged and cannot be decided from this student-only audit. Candidate(b) features/gates were frozen before these diagnostic outcomes; no feedback-driven retuning.

- Diagnostic launched privately: `sms_transactional_audit-p1-diag-20261101` **v1**, dirty base56892e7/tree`c16875190b4c`, `--n-boot 1000`; attached private data and `rakshak-sms-source-balance-c-clean-20261101`. One wait_event opened, no polling. B recovery plus this diagnostic are the two kernels; no cal-only candidate launched. Data-gap hypothesis remains UNVERIFIED pending aggregate results/provenance limits above.

# P1 aggregate diagnostic result — 2026-11-01

- `sms_transactional_audit-p1-diag-20261101` **v1**, tree`c16875190b4cf0251769663a12f35e80034554bc9134664a50ea3f44b787d504`, **ok81.8s**. One wait/fetch; source hash and score-ID binding passed. No training/inference/threshold changes. All three ERM seeds reported; seed1 reproduces GroundLM407/1133 and SmishX267/881 false alarms.
- GroundLM seed1 overlapping proxies: bank195/201 FPR97.01%, campaignCI[95.18%,98.92%] (7clusters); UPI70/74=94.59% [91.18%,97.50%] (2clusters); delivery51/71=71.83% [2.08%,95.33%] (9clusters); telecom/utility87/93=93.55% [69.23%,100%] (9clusters). No-URL406/1111 false alarms. These are proxy subgroup rates with approximate campaign grouping, not human family labels or independent Indian evidence.
- Frozen campaign train76,591 rows:37,645 known fraud negatives,16,367 unknown policy. Negative source counts ScamShield34,421/UCI2,727/Mishra388/Anmol69/Bolewara40. Negative proxy counts bank1,554/UPI317/OTP641/protectiveOTP-noURL315/delivery125/telecom-utility345. Sender metadata absent for all train/external rows; no sender-style conclusion. Source names/proxies do not establish real Indian authenticity; causal data-gap hypothesis remains **UNVERIFIED**.
- SmishX seed1: URL-free negative FPR16/551=2.90% campaignCI[1.63%,4.36%]; overall267/881=30.31%. No causal or universal threshold conclusion. Full per-seed/category CIs will be formatted in a dated report; categories overlap and must not be summed.
- Proceed to the already predeclared candidate(b) using unchanged features/hyperparameters/gates, both new cal-only schemas, all three ERM seeds/all six folds. It is **not** a shipped-system audit: no frozen deployed configuration has been located and the artifact question remains pending. No diagnostic-driven feature/threshold changes. SmishX/GroundLM feedback exposure is already in the design-set register. Independent Indian validation stays OPEN.

- Cal-only candidate launched privately: `fusion_transactional-p1-cal-20261101` **v1**, dirty base56892e7/tree`67b2545f3d29`; args `--n-boot 1000`, private data + corrected Cclean score kernel + pinned URL-student kernel. Latest focused guard/calibration/schema/URL-contract checks **5 passed in0.66s**, Tctl40.1C, ruff clean. No feature/threshold change from the predeclared candidate; null SMS text is treated as empty consistently with the original student featuriser. One event waiter to follow, no polling. B recovery plus this job are the two kernels; diagnostic is finished/fetched. This is a new cal-only baseline/candidate, not a shipped-state FPR claim.

# P1 candidate(b) negative result — 2026-11-01

- `fusion_transactional-p1-cal-20261101` **v1**, tree`67b2545f3d290a11a565003c15b15639ed70ecaf87861b7c22b44ef32b8c6d84`, **ok361.4s**, one wait/fetch. All 36 fusion fits completed and all test/external reports saved with frozen-state hashes. These are **new cal-only baseline/candidate** results; deployed configuration still unidentified.
- Registered runner decision: **unmeasurable**, reason undefined required fold recall. UCI has642 legacy spam positives but **zero `label_scam=1` positives**; its fraud recall is NA. Do not relabel it or silently switch the original fraud guard to legacy spam recall. Both endpoints are reported for every fold/seed.
- Operational decision: **kill candidate(b)** on explicit already-measurable safety violations, independently of the missing UCI fraud recall. Seed1 GroundLM SCAM FPR baseline398/1133=35.13% CI[21.26%,47.44%] →candidate418/1133=36.89% [22.27%,49.75%]; >=UNCERTAIN419/1133=36.98%→423/1133=37.33%. Both exceed absolute1%/10% gates and candidate worsens baseline. SmishX candidate SCAM recall56/226=24.78% CI[16.14%,35.43%] vs student119/226=52.65%; no lower-recall exception. All seeds/negative results/coverage/paired CIs will be retained.
- No post-outcome fitting or feature changes. A saved-output reporting audit will retain the runner's original unmeasurable decision, list NA endpoints and all observable violations, and distinguish rejection from a fully measurable all-fold gate. Default fusion schema remains unchanged; optional candidate code is for reproduction, not shipment. Next P1 arm is separately predeclared train-only hard-negative augmentation. New-arm internal gates must declare the legacy UCI endpoint/NA fraud status before any training, never invent fraud labels.

- Saved-output supplement predeclared after outcomes for reporting completeness only: preserve the registered `unmeasurable` decision, list all observable failures plus NA fraud-recall endpoints, never replace UCI fraud labels or promote an incomplete gate. No fitting, generation, resampling or new outcome-dependent threshold. Synthetic NA/failure check **1 passed in0.63s**, Tctl40.4C, ruff clean. Bind source metrics SHA`1ccaf29eb90e4542dc93881282255d6f4670651d888bd50abc5e2d3d68378984` and frozen states SHA`87b5da994a3c7d579362243fc8cfcf8bfa827cd08621842478cff25f02218ce6` in the private reporting job.

# P1 measurement/candidate(b) reporting milestone — 2026-11-01

- `fusion_final_audit-p1-final-20261101` **v1**, tree`7efd60c0756370f5facecd16e5eaf6386be113c9be55fad7693a22403b83b0c7`, **ok63.1s**, one wait/fetch. Preserves registered unmeasurable gate; operational **kill**,14 observable failures/3 UCI fraud-recall NA endpoints. No resampling/refitting/label changes in supplement.
- Full formatted remote statistics: `FUSION_TRANSACTIONAL_RESULTS.md` (every seed/six folds/both endpoints/FPR at SCAM and>=UNCERTAIN/recall/coverage/paired CIs), `TRANSACTIONAL_DIAGNOSTIC.md` (all fixed proxy cohorts/seed rates/CIs/training census). GroundLM seed1 paired SCAM FPR increase+1.765pp CI[0.520,3.018]pp,p0.004995; harm is not hidden by recall gains on other folds. CRC remains diagnostic due underlying cal checkpoint selection/shift; no certified risk claim.
- HANDOFF headline and DATACARD endpoint limitation updated. New-cal baseline/candidate explicitly differs from unidentified shipped configuration; that audit and independent Indian validation stay OPEN. This is a completed measurement + rejected-candidate milestone, not a completed false-alarm fix. Future(a) has a separate protocol/branch. No synthetic data/model training yet for(a), no per-family calibration.

# P1 completed measurement milestone verification — 2026-11-01

- One full-suite invocation: **841 passed,21 skipped,144 warnings in175.11s**, exit0. nice19/taskset0-3/OMP2/MKL2 plus50% CPU quota/85C stop guard; Tctl39.1C before/47.2C after, no thermal interruption. Full ruff **All checks passed!**. The staged whitespace check later caught one extra blank line at EOF in source_inputs.py; removed in a formatting-only follow-up, without rewriting the pushed milestone. No semantic code changes after verification. Default fusion behavior remains unchanged; candidate(b) is experimental reproduction code, rejected for shipment. No actual local training/inference/benchmark evaluation.

# Alternative reference preparation only — 2026-11-01

- Optional format preference remains pending after the ONNX artifact/diagnostic KILL. Separate `ci/fusion-v1-torch-20261101` based6d4669f prepares an exact PyTorch-checkpoint **measurement-only** reference, not adoption/publication or an ONNX gate pass. Predeclared serialization/fullcal round-trip<=1e-5, same original train-only seed1 recipe/6folds/cal-only fusion/scientific gates; Arm portability explicitly UNVERIFIED. No new job launched; B and clean KD occupy capacity. Preparation is reversible if the owner selects continued ONNX work.

- Reversible exact-checkpoint alternative prepared, **not adopted/launched** while optional format preference remains pending. `fusion_torch_ref` reuses the same cal-only freeze/report runner with explicit serializer/scorer callbacks. Every selected tensor must equal the reloaded checkpoint exactly, same fullcal logit round-trip<=1e-5, six manifests before outcomes; metadata requires measurement_reference_only=true/Arm portability UNVERIFIED. Publisher understands the explicit checkpoint format and still requires all six artifact sets/private privacy verification. No retroactive ONNX parity pass.
- Synthetic feed/batching/type/source/allowlist/fold-order checks first **7 passed in0.81s**, Tctl39.1C; expanded format/disclosure/private-staging checks final output recorded next, pre-test39.5C. Ruff clean. Built-only `fusion_torch_ref-ref-20261101`; no push/subscriber/fetch/publication and no actual local model inference/training. Current GPU slots remain B+clean KD. All scientific criteria unchanged.

- Expanded reversible checkpoint-format tests **8 passed in0.81s**, Tctl39.5C; ruff clean. No format answer arrived after an extended response window. Stated assumption: use this prepared exact-checkpoint format as the default **next measurement-reference attempt**, keeping original ONNX KILL/scientific gates and Arm portability UNVERIFIED. No user approval is inferred, no tolerance waived; owner can steer the optional choice. No launch while B+clean KD occupy both slots. Built-only bundle0cf9f00c86d1 remains the prepared code; only docs updated after build.

# Checkpoint measurement-reference launch — 2026-11-01

- B original subscriber completed (quality-retention false), freeing a slot. Under the previously stated optional-format default (no contrary reply), launched **fusion_torch_ref-ref-20261101 v1**, dirty base6d4669f/tree`0cf9f00c86d1`; private T4/internet/data+teacherlogits+pinned URL. Args `--kernel maadhavvh/rakshak-fusion-torch-ref-ref-20261101 --kernel-version 1 --n-boot 1000`. Confirmed actual version1; ONE original blocking event subscription opened. Clean KD is the only other active kernel.
- Same predeclared original train-only seed1 recipe/strict checkpoint tensor and fullcal<=1e-5 round-trip/6fold cal-only fusion/state freeze, scientific gates unchanged. This measures a fresh checkpoint-format reference, never retroactively passes the killed ONNX export. **Arm/Android portability UNVERIFIED**, shipment_approved=false, HF publish pending. No artifact/safety success before actual completion.

# Parallel B milestone handoff — 2026-11-01

- B finished on separate ci/ela-qlora-20261101,commit210e236,normal push/new PR#7 (unmerged). FP16 verifier-gated advisory GO; same adapted860,959,104-byte PTE completed600rows but retention FAIL (EN pairedCI[−28.96,0]pp,HL[−28.89,+17.86]pp both below−5pp lower margin). No more B quantisation/runtime attempt. Private Kaggle model archive v1,isPrivate=true verified,HF publish pending. B full861pass18skip144warn178.88s plus publication-only focused2pass.02s; source/identity/base checks recorded on B branch. No shared main edits or PR2–6 branch pushes/merges.
- Current two kernels remain this checkpoint measurement-reference attempt and clean KD on original subscriptions. Failed ONNX gate remains killed; this attempt has unchanged numerical/scientific criteria and explicit Arm portability UNVERIFIED. No checkpoint/v1 success claimed yet. Continue to private reference publication and prepared fused/policy audits only after a valid artifact completes.

## 2026-11-01 — frozen fusion v1 reference completed; safety KILL

- Hypothesis/protocol recorded in FUSION_CHECKPOINT_REFERENCE.md before launch:
  exact CPU checkpoint serialization can preserve the selected experts. Same
  train-only seed1/teacher0/15epoch recipe, same four cal-only stages and unchanged
  scientific gates; earlier ONNX numerical KILL is not waived.
- `fusion_torch_ref-ref-20261101`v1, args `--kernel maadhavvh/rakshak-fusion-torch-ref-ref-20261101 --kernel-version 1 --n-boot 1000`,
  T4/private data+teacher logits+fixed URL kernel. Tree
  `0cf9f00c86d115f31a92ad61a4376d501bd36cbed5a396b29fad1857374fe205`,
  base git6d4669f68c19 (dirty snapshot), ok1148.9s. Original one event subscription
  and one matching fetch completed. All six manifests froze before any outcomes.
- Artifact GO: all tensors equal on reload, full cal round-trip max logit difference0
  on campaign11072/lang8454/Mishra10876/ScamShield2754/UCI10350/temporal5515.
  This is same-framework serialization,
  not ONNX equivalence. CPU PyTorch measurement reference only; Arm/Android UNVERIFIED.
- Manifest SHA256 `c310d3ccbc1c1e50ce9436f217c66ee9934f778985bfef487a034cc3070b79fa`;
  metrics SHA256 `aada644fd94eeaea681ef3ec3f53928c78d029f496e0319966377c192857377f`.
- Safety KILL (all three checks fail): GroundLM SCAM354/1133=31.24%
  CI[16.91,43.57]%, alert362/1133=31.95%[17.58,44.22]%; SmishX SCAM fraud
  recall43/226=19.03%[12.88,26.90]% vs same-student113/226=50.00%.
  Newly fitted (not previously shipped); never claim this reference is safe to deploy.
- Six folds/both labels/coverage/paired campaign CIs and p-values in FUSION_V1_RESULTS.md.
  GroundLM/rule design and GroundLM/SmishX intervention selection are in-sample;
  independent Indian validation OPEN. UCI fraud recallNA; legacy-spam endpoint retained.
- Private Kaggle publication uploaded allowlisted manifests/checkpoints/URL/source/aggregate
  JSON (no row scores/messages). Create returned without reported error; immediate
  GetDataset privacy verification403. Publication/privacy **NOT YET VERIFIED**; do not
  reupload or assume public/private from the failure. HF publish pending.
- Keep immutable measurement reference, kill shipment candidate. Next unchanged hardneg
  audit and evidence-gated policy must bind this exact v1 hash.

## 2026-11-01 — v1 private publication verified

- No reupload/version overwrite after the immediate processing-time GetDataset403.
  One later owner GetDataset read verified `maadhavvh/rakshak-fusion-v1-20261101`,
  ownermaadhavvh, **isPrivate=true**, version1, total23581454bytes. Local receipt
  work/fusion-v1-kaggle-privacy-verification.json. No raw messages or row score files
  were staged; all5 companion manifests and their exact experts are included.
  Main manifestc310d3ccbc1c1e50ce9436f217c66ee9934f778985bfef487a034cc3070b79fa.
  **HF publish pending**. Immutable reference is not shipment approval: safety KILL.

## 2026-11-01 — v1 milestone validation and owner format confirmation

- Owner explicitly confirms PyTorch checkpoint=frozen v1 measurement reference, ONNX
  deferred to Phase2. Earlier numerical export KILL and all scientific gates unchanged.
- Full unit suite ONCE for this milestone:849passed21skipped144warnings177.51s,
  nice19/taskset0-3/OMP2/MKL2 and50%CPUquota, Tctl39.6Cstart/60.2Cend, monitor
  never reached85C. Torch-dependent local tests skip; actual model invariants run Kaggle.
  Ruff all clean. No real model inference or training ran on this laptop.
- After the full invocation, copied the already-run negative ONNX diagnostic source,
  its strict initializer helper and synthetic test from the isolated failed-attempt
  worktree into this milestone for reproducibility; targeted validation follows.
  No inference source in the frozen v1 archive, model weights or fit/eval gates changed.

- Copied diagnostic focused test1passed.04s, Tctl43.6C with the same user CPU/thread limits; ruffclean. Full suite was not repeated.

- Owner headline correction: matched clean no-KD lang_shift F1≈0.41 versus historical teacher≈0.80; clean KD0.3520 KILL. HANDOFF replaces the contaminated≈0.81 headline; full KD experiment is a separate milestone.
# New priority / local inbox tooling — 2026-11-01

- User reprioritised full frozen fused-pipeline false alarms above further C methods and clean KD rerun above URL/NIDS. GroundLM student35.92% and SmishX30.31% at cal1%-FPR are not rules/evidence0.0883% and not fused metrics. Full shipped fusion state/threshold inventory is pending clarification; repo has fitting code but no saved state, and both private HF access paths return404. No newly fitted result will be called shipped.
- Separate `ci/transactional-fpr-20261101` based56892e7 preserves main dirty docs and PRs2/3/4. Added code-only `tools/inbox_export`: local UTF8 XML/compatible JSON received-SMS parser, deterministic typed scrubber, five-label manual privacy-confirmation CLI, local owner-only Parquet outside Git. No real messages, network/upload, model labelling or collection. Independent Indian validation stays **OPEN**. Supported XML fields checked against official SyncTech docs; JSON is a conversion format, not a native-export claim.
- Software acceptance and privacy limits (recorded after focused implementation checks, before full-suite verification): `docs/INBOX_EXPORT.md`; focused synthetic checks **5 passed in0.22s**, ruff clean, Tctl41.0C before. Arbitrary names/subdomains require human review; no automatic anonymisation claim.

# Code-only inbox milestone verification — 2026-11-01

- Latest focused checks **5 passed in0.25s**, Tctl40.2C before; full ruff clean. One milestone full-suite invocation: **835 passed,21 skipped,144 warnings in176.49s**, exit0. nice19/taskset0-3/OMP2/MKL2,50% CPU quota and85C stop guard; Tctl40.2C before/53.1C after, no interruption. No code changes after verification. Gate passes on synthetic fixtures only; no real-data anonymity/performance claim.
# Run update — 2026-11-01: C-adversary complete; new priority order

- `sms_source_adversary-c-adv-20261101` v1, dirty tree `a09fb20ed34903e9605e7f7e48c26aed055a4da18ae4a41a0d7c8588214b9573`, **ok**, 3,635.9s, six splits/36 fits, zero failed units. One original event subscription and one fetch. Args seeds1/2/3, epochs15, nboot1000, teacher-seed0 unchanged. Final acceptance is delegated to the predeclared saved-output gate job.
- `sms_source_audit-c-adv-audit-20261101` v1, tree `f50b5a0c3bbd`, launched with `--source-tree a09fb20ed34903e9605e7f7e48c26aed055a4da18ae4a41a0d7c8588214b9573 --arms adversarial`; one wait_event subscription. No training/inference in this audit.
- User's new priority after existing B/C decisions: **fused** Indian transactional false alarms, then clean train-only KD vs no-KD, then further B/C/E/F. Student-only GroundLM FPR 35.92% and SmishX 30.31% are distinct from the rule-layer 0.0883%; no fused result yet. GroundLM design exposure remains in-sample; independent Indian validation stays OPEN. PRs2/3 remain unmerged and unchanged.

# Milestone verification — 2026-11-01: C-adversary

- One full suite invocation for this finished milestone: **834 passed,22 skipped,144 warnings in178.88s**, exit0. Required nice19/taskset0-3/OMP2/MKL2 plus50% CPU quota and85C stop guard. Tctl43.5C before/50.1C after; no thermal interruption. Local Torch missing skips remain explicit; Kaggle synthetic Torch checks passed before training. Full `ruff check .`: **All checks passed!** No code changes after verification.
## 2026-11-02 — E0 three-seed URL controls declared/implemented, no benchmark outcomes

- URL_CONTROLS_CAL.md predeclares artifact-only phase before code/compute:3charcnn-wide seeds1/2/3, originalPhreshtrain/domain-groupedcal and frozenurl_domain, fullfitdata, original8epochs/batch512/AdamW.002/patience3/AMPfp16/200bytes. No teacher/KD/testprediction or benchmarkmetrics inthisphase. Teacherrecipe/eligibility mustbefixedbeforeteacherfit andbeforeany newtestoutcome.
- Hypothesis: correctlyseeded, all-six savedcontrols/cal-onlylogits are needed for fairtrain-only KDeligibility. TechnicalGO only ifall6reloadexactly/finitecompleteoriginalcalIDs/no testscoring; no performance/shipmentGO. Found existingrun_unit seeds AFTER construction; wrapper seedsbeforeconstruction, leaves originalURLjobunchanged. Selectedweights/config/scaler/fit-cal-testIDhashes bindbeforelatermetrics. Originalsingle-classurl_temporal unmeasurable unchanged, nofakednegatives.
- Focusedvalidation hypothesis:3synthetictests pass0skip/error/failure onprivateKaggle,no datasets/models. Two new tests enforceindependentpriorRNG/exactweights/cal-onlyscorecalls/cacheIDs andtest-IDoverlap rejectionbeforeconstruction; existingprivateattachmentbuilder regression. Genericremoteunitdriver sameasM, no laptopmodelwork. ScientificURLmetric gates forfutureteacher/KD mustbespecifiedseparately beforetheirfit; E0exposesnone.
- Researchonly pinnedApache2ByT5smallcard/config; actualByT5source1198627927bytes/hash isrecordedinURL_TEACHER_RESEARCH.md. No localweightsdownloaded, downstreamURLbenefitUNVERIFIED. HF publish pending. No E0job launched before focusedtests pass. M3isotheractivekernel onoriginalsubscription.

- Focusedjob firstnameurlcal-focus-20261102 wasrejectedbylocallauncher(length21), no remotejob/wait. Correctunit_validation-urlcal-unit-20261102v1/treeb187ea6c8477 emittedGO3tests/0skip/error/failure; originalONEsubscribercomplete, fetchingonce. Addedonlyinputparquet/splitfilehash metadata and changedurl_temporal note toexplicitlyhistorical/notnewlyscored; no model/training/gate changes. NewE0corpusjob stillrequiresverifiedfetchPASS beforelaunch.

- Verifiedfocusedreceipt:unit_validation-urlcal-unit-20261102v1/treeb187ea6c8477498a615f666aa86ec9bfe76e9d65dec8cb0a369874806af9c60e ok92.9s,3passed0skip/error/failure,ONEevent/fetch;metricsSHA9bf4e308c36c7dfc4aad7617d5d07fb3e2b24a7f36a7096e6d76e682a5d88ec0. Ruffallclean. Nowlaunchcal-onlysixcontrols(no teacher/testoutcomes),originalpredeclunchanged.

## 2026-11-02 — E0 cal-only URL controls launched

- url_controls_cal-e0-20261102v1/treeebcc1bb75c6e9668a789928946c7247e66b560305d5e62e38598fe7d675f715b, nojobargs/overrides, privateT4/internet/rakshak-data only. ONEoriginaleventsubscription, no result. Sixfixedcontrols/cal-onlylogits/inputfile+partitionhashes, no testpredictions/protocolmetrics/teacher/KD. Originalartifact-onlyGOunchanged.
- M3isotheractivekernel/originalwait;max2kernels. All earlierunits/familyjobsfinishedwithoriginaloneevent/fetch. PR12KILLmilestonecomplete/privatearchiveverified; PR11descriptionlinksPR12withoutbranchpush. No merges/mainpush/forcepush. HF publish pending.

## 2026-11-02 — E0 complete, E1/E2 implemented before cal outcomes

- url_controls_cal-e0-20261102v1/treeebcc1bb75c6e9668a789928946c7247e66b560305d5e62e38598fe7d675f715b ok1415.3s, ONEevent/fetch. Firstfetchfailed networkDNS afterpermissioncontextreset; restorednetwork grant, resumeddownloadonce, no statuspoll/relaunch. Six selected weights/cal-logit hashes verified; manifestSHA88b20cd71febccdc1ed81d8c2a9ec0f1b984b9eed9f44c389f357b5bfcc4ac7d. TechnicalartifactGOonly/no newtestoutcomes.
- E1/E2predecl URL_TEACHER_PLAN.md existedbefore E0calresults read. FixedByT5encoder seed0/eachdataset, no testscoring; cal-onlyteachereligibility at weightedPhresh.0005/.001AP or naturalURLdomainAP, paired1000registered-domainresamples vsmean3E0seeds. GOeligibility only ifallrequiredCI lower>0and>=950validdraws. Freezeeligibility BEFOREoriginalfit-onlyteacher targets/KD3seeds. Actualscientificteacher/KDtestgates separately frozen, no benchmarkselectedretry. Onefixed11hdeadlineperdataset, incompleteaborts.
- Implemented maskedmean+side128head/sourcehash/encoderkeys-shapes-dtypes-finiteness guards; AMP/gradientcheckpoint/weightedBCE/partial-update-normalization/skipcounts; alloriginalfit/cal/testIDs/sourcearchive binding and exactCPUweightserialization. No E1modelrun yet. No localpretrainedweights downloaded/executed.
- FocusedpureAPtests2passed1.68s locally onsyntheticfixtures,Tctl53.2Cbefore firstrun (later41.4C), nice19/taskset0-3/OMP2/MKL2/OpenBLAS1. WeightedrankAP matchesofficialsklearnAP withties/zeroweights; fixedclassmass/bootstrap/eligibilityguards. No realURLscoring.
- Predeclare focusedGPUvalidation4synthetictests GOiff4pass/0skip/error/failure: exactbyteprefix/EOS; tinyT5encoderload/mask/backprop; tinyCUDAteachertraining cal-only/finalpartialupdate/checkpointreload; actualbundledependenciesandeligibility-before-targetguard. No dataset/modelinputs. ExistingM3remains otheractivekernel/originalwait.

### 2026-11-02 — URL teacher validation and fixed Phresh run

- Focused private Kaggle `unit_validation-urlt-unit-20261102` v1, tree
  `32438d3c6657373927ddb55ccea2c6047ca2792046d84cacd4a1937b69f39368`:
  **4 passed / 0 skipped / 0 failed**, pytest23.25s, total159.1s. Two earlier
  synthetic AP tests also passed locally under the prescribed CPU/thermal limits.
- On continuation the original tool-session handles were missing and no local
  completion receipt existed. One bounded cached-notification read recovered the
  exact unit completion; one fetch verified job/git/tree/outcome. No duplicate
  blocking unit wait or Kaggle status polling. The same bounded read found no M3
  completion; its original process is missing. Replacement event-subscription
  authorization requested; M3 continues to count against capacity, not relaunched.
- After the four checks passed, launched private T4
  `url_teacher_kd-p1-20261102` **v1**, tree
  `84d35409ac4ac48138a2a38673bd14efd7682b40808d64a9b80762d3f7e5e64d`,
  base246324575928 (dirty source snapshot), args
  `--dataset phreshphish --controls-manifest-sha256 88b20cd71febccdc1ed81d8c2a9ec0f1b984b9eed9f44c389f357b5bfcc4ac7d`.
  Inputs: private rakshak-data + completed E0 kernel; unique p1 run, one blocking
  event wait, no result/fetch yet. At most two GPU kernels counting M3.
- Hypothesis/recipe/unchanged gates are frozen in URL_TEACHER_PLAN.md. Cal-only
  eligibility precedes original-fit-only KD targets; scientific test/benchmark
  gates unmeasured. No expert substitution in fusionv1, no held-out/SOTA claim.
  HF publish pending; independent Indian validation OPEN.

### 2026-11-02 — Fixed URL teacher stops unmeasurable

- `url_teacher_kd-p1-20261102` v1/tree84d35409ac4a completed **error112.8s**;
  original single event + one verified fetch. Exact error:
  `FloatingPointError: nonfinite FP32 teacher loss; stop fixed arm`.
  No selected teacher checkpoint, eligibility JSON, train targets or KD model
  produced. No completed epoch was logged; number of partial updates is unknown.
- **Teacher/KD scientific comparison UNMEASURABLE.** Stop the fixed arm, no
  precision/batch/hyperparameter switch, no secondary teacher run to rescue it.
  Underlying numerical cause UNVERIFIED. Focused synthetic4PASS was necessary
  implementation validation, not evidence of real pretrained numerical stability.
- Continue the existing E0 three-seed control report with original frozen weights;
  cal reconstruction<=1e-5 and all6thresholds freeze before test scoring, original
  official975subset statistics/paired-domain draws retained. No new training.
- New protocol optimization synthetic2PASS1.15s on local synthetic fixtures only,
  nice19/taskset0-3/OMP2/MKL2/OpenBLAS1, pre-testTctl45.8C; ruff clean. Explicit
  row-repeat/sklearn and shared-draw comparisons cover ties/zero multiplicities/
  invalid subsets/three-seed means. No actual URL scores evaluated locally.

### 2026-11-02 — frozen URL control evaluation launched

- Focused `unit_validation-urle-unit-20261102`v1/tree
  3a0060fa7bdd9763e141cee0ff04db2a903708cebb7783c82259e4a17d51da27:
  **2PASS0skip**, total93.7s, one event/fetch. Synthetic checkpoint/cal-only
  restoration/tamper guard and actual sibling-job bundle/threshold ordering pass.
- `url_controls_eval-e0eval-20261102`v1 launched after those checks, same E0
  manifest88b20cd71febccdc1ed81d8c2a9ec0f1b984b9eed9f44c389f357b5bfcc4ac7d,
  private data+completed E0 kernel/T4/internet, one event subscription. Allsix
  full-cal parity/threshold manifests precede test scoring; official subsets,
  per-seed endpoints/domain CIs/fullbenchmark/fulltest and temporalNA diagnosis.
  No teacher/KD model or new training. Results pending; M3 still counted active.

- Private URL archive publisher prepared (not uploaded): exact E0 manifest/source,
  evaluation tree and failed-teacher tree required; allowlist13files (six original
  checkpoints, sourcezip, aggregate/manifests/failure JSON), never URL row parquets.
  Synthetic privacy/hash/symlink2PASS.02s, required CPU/threads pre-testTctl44.4C.
  Private create and owner/isPrivate/version verification required before claiming
  publication; HF publish pending. Full milestone suite still NOT run.

### 2026-11-02 — URL control report recovery from unchanged saved scores

- Original `url_controls_eval-e0eval-20261102`v1/tree
  4fdfe6994123ec28645a3383841f9e251bd27e5e2cacf91235aa3db2bb65a3ec:
  error478.4s at FINAL temporal-file check, one event/fetch. Exact error:
  `FileNotFoundError: no split files match ['url_temporal'] under .../splits`.
  Allsix test-score files and pre-outcome threshold manifest survived. No aggregate
  metrics.json was emitted. E0 input manifest already lists only url_domain and
  url_phreshphish split files; the historical temporal narrative is not confirmation
  that the currently attached frozen data still contains it. Do not create a split.
- Cache-only recovery predeclared: exact parent tree/failure, manifestSHA
  54e4fa8f9f1d6128ac31f6515b698d768fa8facf98931b00cb3acfd64bf91509,
  allsix fixed score hashes, original ordered fit/cal/test IDs and original frozen
  thresholds must verify. Same1000domain resamples/seed13/official subset report;
  no model execution, no new targets/fitting/threshold selection. Temporal now
  **UNMEASURABLE: file absent**, not a falsely reconfirmed single-class result.
  Original teacher/KD gates stay UNMEASURABLE, unchanged. Any cache mismatch aborts.
- New cache/source/order/hash/bundle guards synthetic2PASS.69s, CPU-limited with
  pre-testTctl42.1C, ruff clean. Six score-file SHA values are pinned in the recovery
  source; hashes only read locally, no URL prediction/metric computation locally.

- First cache recovery `url_controls_recompute-e0stats-20261102`v1/tree
  3fec836a5feffdd67eba4eef761604a81438acba54546077cb79b60c561052ef
  error94.9s: adapter passed the top-level manifest where its `controls` mapping
  was required (`KeyError: phreshphish`), before endpoint report generation.
  Corrected the mapping; new realistic saved-manifest regression1PASS.68s,
  nice19/taskset0-3/OMP2/MKL2/OpenBLAS1, Tctl48.9C, preserves allthree original
  thresholded reports/counts. Scientific gates and scores unchanged.
- Its ONEfetch downloaded the complete failure files but exited1 because the
  generic CLI-output error detector matched filename `failed_eval_results.json`.
  Receipt/job/git/tree/error outcome verified from downloaded JSON manually;
  no repeat fetch. Recovery artifact renamed `original_eval_results.json` to
  avoid that filename false positive; launcher/fetch scripts unchanged.

### 2026-11-02 — six frozen URL controls reported; full milestone validation

- Cache-only `url_controls_recompute-e0stats2-20261102`v1/tree
  17576395e1c2fc76e32d4970c15ca5c180844da3527a40542680e07c6d34b7c2
  ok435.9s, one event/fetch, allsix original score hashes and threshold bindings.
  MetricsSHA0093e20c39a8d372f391d1ffaec05be89950f56bd00563f5bbc1141853d8f5ee.
- Three-seed mean official Phresh AP .443457CI[.323937,.620641] at.05% and
  .558907CI[.428609,.705899] at.1%; P@R90 .073388[.052072,.183454] and
  .132927[.076425,.275599]. All975original subsets retained, all1000validdraws
  at every rate. No stronger-teacher/KD claim: both remain UNMEASURABLE. Original
  missing temporal remains UNMEASURABLE, never recreated; no new inference.
- Full milestone validation now predeclared: one full privateKaggle invocation,
  allcollected(min835), 0failure/error, max18inherited optional skips. No data/model
  inputs. Include publisher's changed recovery binding and a new missing-temporal
  regression; original inference job now explicitly reports missingfile status
  rather than dropping completed metrics. No scoring/training change or rerun.

- Formatted137line URL_THREE_SEED_RESULTS.md ONLY from precomputed aggregateJSON
  (no local score/model evaluation), all975subsets/allrates/3seeds/fulltest/fullbench/
  url_domain campaign CIs; all1000valid domain draws. Full-cal logit parity was
  exactly0 on allsix controls. HANDOFF headline updated, prior historical single-
  seed figures distinguished. No paired old/new gain claim. Stage15file private
  archive validated successfully without upload; publication waits for fullsuite.

- Full milestone `unit_validation-url-full-20261102`v1/tree
  f028f748590f286f03112ac2f710f3e8c077eed42e6a0052f02a3edcc24186c3:
  **858passed0skipped**,157warnings125.18s (total211.7s), one event/fetch.
  Includes new publisher recovery-source binding and missing-temporal regression.
  Ruffallclean. Parent pull/rebase checked unchanged; repo identity confirmed
  vhmaadhav <vhmaadhav@hotmail.com>. No full suite or model work locally.

- Private archive upload completed; immediate owner GetDataset returned403 during
  processing, so owner/isPrivate/version verification is still pending. No repeat
  upload; make one later owner-record read. No public publication requested.

### 2026-11-02 — URL measurement milestone finalized

- One later owner read verified maadhavvh/rakshak-url-e0-20261102 version1,
  ownermaadhavvh/isPrivate=true,17083574bytes, archive_manifestSHA
  ced9776473225eba4d48ef68d574d2f017f50b4bfeb1ed01cb66a1f5f3013234. Initial
  processing403 retained; no re-upload. HF publish pending. Exact15file allowlist,
  six checkpoints/source/aggregate/failure provenance, no raw URLs/row-score files.
- Three-seed control measurements retained; stronger-teacher/KD/temporal remain
  UNMEASURABLE. No external-selected rescue recipe, SOTA or fusedFPR claim.
  Report/HANDOFF include all rates/seed CIs and limitations; independent Indian
  validation OPEN. Full858PASS0skip/ruffclean, parent pull/rebase unchanged, identity
  vhmaadhav confirmed. One milestone own-branch commit; merge order #2→URL, no merge.
# Clean KD matched-control preparation — 2026-11-01

- Separate `ci/clean-kd-20261101` based6d4669f, protocol `CLEAN_KD.md` before implementation/compute. Historical0.80–0.82 KD claim remains withdrawn; prior C train-only arms all used KD and cannot stand in for a no-KD comparison. New fixed dual/base hard-label-only control vs train-ID-only KD, all6folds/3seeds, same cal-only stopping/thresholds. Go requires lang_shift KD lower mean-F1 CI above no-KD point mean, with no other frozen-fold loss beyond control CI half-width; paired clustered tests reported everywhere.
- Existing per-fold seed0 teacher targets are in-sample teacher outputs, explicitly not OOF. No naive cross-fold ensemble (other teachers may have seen this fold's test rows). Optional inner-fold teacher fitting deferred behind minimal clean control and higher-priority P1/current kernels. No new KD training launched; max2 remains B-runtime+hardneg originals.

- Clean KD implementation preserves original train-only membership checks. Only the explicit KD experiment changes teacher routing: control gets no teacher/full BCE, KD requires complete finite teacher coverage; source sampler is ERM for both. Actual history teacher-row counts are asserted (0 vs original train n), saved selected checkpoints include run/arm/seed/bundle/git/teacher-enabled metadata and SHA256. GPU startup synthetic invariant checks missing-teacher loss equals full class-weighted BCE before corpus fitting. Shared source-balanced/DRO default behavior is retained.
- First focused gate/bundle/train-only membership tests **19 passed,3 skipped in0.66s**, Tctl59.6C; skips are local Torch absence, not a GPU invariant pass. Ruff clean. Added checkpoint-hash recording and finite-CI validation subsequently; final focused rerun follows. No local training/inference and no new GPU launch.

# Clean KD matched-control launch — 2026-11-01

- After the separate parity diagnostic completed, launched **sms_clean_kd-kd1-20261101 v1**, dirty base6d4669f/tree`4d8d927781cd`, T4/internet/private data+teacherlogits. Exact args `--seeds 1 2 3 --epochs 15 --n-boot 1000 --teacher-seed 0`; predeclared hard-label-only no-KD control versus original-train-ID KD, all6folds/3seeds, finite teacher coverage and startup full-BCE invariants required. ONE original blocking event subscription opened; no polling/fetch yet. B same-PTE runtime recovery is the only other active kernel.
- Final focused rerun **19 passed,3 skipped in0.80s**, pre-test Tctl51.1C, ruff clean. Three local skips are Torch absence, not runtime-invariant success. No teacher ensemble/OOF claim; no historical contaminated Hinglish number reinstated. Gates unchanged, model checkpoints/teacher-enabled provenance saved privately. HF publish pending. No headline update until actual completed comparison.

## 2026-11-01 — clean KD versus no-KD completed: KILL

- Hypothesis/protocol predeclared in CLEAN_KD.md; no gate/epoch/threshold changed.
  `sms_clean_kd-kd1-20261101`v1,args `--seeds 1 2 3 --epochs 15 --n-boot 1000 --teacher-seed 0`,
  T4/private data+teacher logits,36fits,ok3108.8s. Original ONE event subscriber
  and one provenance-matched fetch complete. Dirty snapshot
  `4d8d927781cdfc90d92b1aad05413ba91b2676160e640e3b1f9b42b648052cb7`
  on git6d4669f68c194cafdd9d7827e86d0264d34ba9d3. MetricsSHA256
  `708b0fdc6e7b1646717c0bc572a56105100884c1eaaab6e44b47e8693cc8dcde`.
- KILL: noKD lang_shift meanF1.405842 CI[.369672,.441243], cleanKD.351977
  [.318139,.384016], paired delta-.053865 CI[-.069574,-.038291],p.000999.
  Lower-CI-above-control and no-loss-beyond-CIhalf both fail; other5fold guards pass.
  ScamShield benefits .715195→.753329,delta+.038134 CI[.035297,.040816], but
  cannot override the lang gate. All6folds/3seeds/thresholds/externalbothlabel
  endpoints/checkpoint hashes in CLEAN_KD_RESULTS.md (formatting saved aggregate
  JSON only, no local inference/resampling). External values are student-only
  diagnostics, not fused/deployed FPR. Rule design/intervention selection in-sample.
- Train-only original IDs; no inherited extras; KD complete cached single-teacher
  targets, not OOF/ensemble. NoKD fullBCE; GPU startup proved loss/gradient and actual
  history teacher rows0 vs fulltrain forKD. Both matched arms retained privately.
- Current headline **clean noKD student≈0.41 versus teacher≈0.80**, not contaminated
  ≈0.81; old claim stays WITHDRAWN. HANDOFF updated. Next user-requested research
  arm is multilingualbyteCNN+small pretrained multilingual embedding distilled
  frommmBERT on train-only targets; not implemented/launched yet.
- Private checkpoint archive uploaded `maadhavvh/rakshak-clean-kd-20261101` with
  public=False,36hash-listed.pt+modelmanifest/results/aggregate metrics ONLY.
  Immediate owner GetDataset403 during processing: privacy NOT YET VERIFIED; no
  reupload. One later owner verification planned. HF publish pending.
- Publication privacy fixture+gate focused4passed.06s,Tctl46.2C, usernice/affinity/
  thread limits; Ruffclean. Full suite once at this finished milestone follows.

- Later one owner GetDataset read verified archive `maadhavvh/rakshak-clean-kd-20261101`v1, ownermaadhavvh, **isPrivate=true**,14567242bytes, receiptwork/clean-kd-kaggle-privacy-verification.json; no reupload. HF publish pending.

- Finished milestone full unit suite ONCE: **845passed21skipped144warnings191.10s**; nice19/taskset0-3/OMP2/MKL2/50%CPUquota/85Cstop, Tctl49.5→57.8C. Torch-dependent checks skip locally, actual loss/coverage invariants ran on Kaggle. Ruff all clean; no code changes after the full invocation. No local real model work.
