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
