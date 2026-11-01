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

- One full-suite invocation: **841 passed,21 skipped,144 warnings in175.11s**, exit0. nice19/taskset0-3/OMP2/MKL2 plus50% CPU quota/85C stop guard; Tctl39.1C before/47.2C after, no thermal interruption. Full ruff **All checks passed!**, diff whitespace clean. No code changes after verification. Default fusion behavior remains unchanged; candidate(b) is experimental reproduction code, rejected for shipment. No actual local training/inference/benchmark evaluation.
