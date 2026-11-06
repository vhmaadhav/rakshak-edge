# Handoff: Rakshak-Edge (for Sachin)

Last updated 2026-11-01. Owner until now: Maadhav (`vhmaadhav`). Start here, then `docs/STATUS.md` (session log) and `docs/CONTEXT.md` (decisions).

Review fixes are on `sachin/handoff-tasks-1-2`, prepared for one PR into main. Do not push main.
GroundLM urgency and ScamShield widening numbers are **in-sample: these rows informed rule
design**. Read the design-set register in `docs/GROUNDLM_FPR.md` before making a safety claim.
Independent Indian held-out validation remains OPEN; SmishX is independent, non-Indian-specific.

Transactional false-alarm measurement (2026-11-01): the rule/evidence0.0883% is
**not student or fused FPR**. Newly fitted cal-only fusion baseline/candidate(b), seed1:

| Known fraud endpoint | New baseline | Candidate(b) |
|---|---|---|
| GroundLM SCAM FPR |398/1133=35.13% [21.26,47.44]% |418/1133=36.89% [22.27,49.75]% |
| GroundLM >=UNCERTAIN FPR |419/1133=36.98% |423/1133=37.33% |
| SmishX SCAM recall |27/226=11.95% |56/226=24.78% (student119/226=52.65%) |

**Kill candidate(b)**: absolute safety/no-increase/recall gates fail. Original runner gate
also remains unmeasurable on UCI fraud recall (zero mapped fraud positives; legacy spam
positives642). No relabelling or endpoint substitution. These are **new cal-only models,
not an identified shipped configuration**. The owner confirmed no fusion state was previously frozen; the later immutable v1 reference below supersedes these score-only baseline numbers for full-system claims.
not an identified shipped configuration**; shipped-state artifact identification remains OPEN.
GroundLM/SmishX feedback is in-sample for new intervention selection, not untouched validation.
All three seeds/six folds/both endpoints/coverage/paired CIs: `docs/FUSION_TRANSACTIONAL_RESULTS.md`.
Aggregate diagnostic/source counts: `docs/TRANSACTIONAL_DIAGNOSTIC.md`; real-Indian provenance
and causal data-gap explanation remain UNVERIFIED. Independent Indian validation stays OPEN.

Frozen **fusion v1**, newly fitted (not previously shipped), now binds actual seed-1
checkpoint weights, URL/rules source hashes, stacker, calibrator, thresholds and all
six fold manifests. Artifact gate GO; **safety KILL; shipment not approved**.

| v1 known-fraud endpoint | Result (campaign-clustered 95% CI) |
|---|---|
| GroundLM SCAM FPR |354/1133=31.24% [16.91,43.57]% |
| GroundLM >=UNCERTAIN FPR |362/1133=31.95% [17.58,44.22]% |
| SmishX SCAM recall |43/226=19.03% [12.88,26.90]% (student113/226=50.00%) |
| SmishX >=UNCERTAIN recall |84/226=37.17% [28.02,48.11]% |

GroundLM/SmishX intervention feedback is **in-sample for policy selection**;
independent Indian validation remains OPEN. All future full-system FPR claims cite
v1 manifest `c310d3ccbc1c1e50ce9436f217c66ee9934f778985bfef487a034cc3070b79fa`
or a named successor. Format is CPU PyTorch checkpoint, **measurement reference
only; Arm/Android portability UNVERIFIED**. The earlier ONNX artifact attempt failed
its original numerical gate and stays KILL. Full six-fold/both-label endpoints,
coverage, paired tests and provenance: `docs/FUSION_V1_RESULTS.md`. Private Kaggle
publication/privacy receipt is in STATUS.md; **HF publish pending**.
C-adversary (`ci/source-adversary-20261101`, train-only, three seeds) passes its recorded gate:
ScamShield mean F1 **0.7340 → 0.7493**, paired campaign gain **+0.0153 [0.0136, 0.0174]**;
both other source folds pass the no-regression guard. These are **in-sample for rule-feature
design**, not independent held-out improvements; no checkpoint is shipped by this harness.
Every split/seed and external FPR/recall: `docs/SOURCE_ADVERSARY_RESULTS.md`.
Priority now: frozen **fused** transactional false-alarm audit, then clean train-only KD vs no-KD.
The student-only 35.92% GroundLM FPR is distinct from the rules/evidence 0.0883%; fused FPR
is not yet measured. Independent Indian validation remains OPEN.

Hardnegative arm(a) completed **KILL** (all3seeds): GroundLM SCAM FPR fell from
34.86/36.63/36.89% to7.33/18.36/12.53%, but all still fail the1%SCAM/10%alert caps;
SmishX SCAM recall23/30/23of226 is below matched controlstudent129/85/148of226.
Seven internal fold/label recall guards also fail. UCI fraudNA, legacy guard retained.
In-sample intervention selection; no held-out improvement or deployment claim.
Allseeds/6folds/bothlabels/coverage/pairedCIs: TRANSACTIONAL_HARDNEG_FUSED_RESULTS.md.
These are named experimental arm(a) states, with the unchanged frozen v1 parent
manifestc310d3ccbc1c1e50ce9436f217c66ee9934f778985bfef487a034cc3070b79fa;
v1 itself is artifactGO/safetyKILL, newly fitted (not previously shipped), PR#8.
The killed augmentation's original job did not save student weights. Private
synthetic/aggregate archive receipt in STATUS.md, humanreview200 pending; HFpublishpending.

## 1. Goal and constraints

- **Project:** offline SMS and URL scam detector on Arm devices, for **Bharat AI-SoC Challenge 2026-27, Problem Statement 3** (on-device cyber safety). Name in the code: Rakshak-Edge.
- **Thesis:** a small multi-expert detector decides, with calibrated abstention. A Llama 3.2 1B layer (ExecuTorch, INT4) only verbalises evidence IDs that the detector produced. A deterministic verifier rejects anything else (Evidence-Locked Advisory, ELA).
- **Hard constraints (challenge):** everything runs locally on Arm (Android phone; Raspberry Pi 5 CPU or AI HAT). Llama-family model and ExecuTorch are mandatory for the advisory part. Hailo is optional and kept separate. Team of 1 to 3. Deliverables: GitHub repo, source, app, demo video, technical report.
- **Deadlines:** Phase 1 due **23 Nov 2026**. Shortlist about mid-Dec. Phase 2 due **5 Feb 2027**. Finals 15-16 Feb 2027.
- **Claim discipline:** no novelty claim for any single part. The claim is the architecture plus the evaluation regime (campaign shift, code-mixed text, obfuscation). Say "state of the art" only for numbers reproduced under the same protocol (`docs/SOTA_TARGETS.md`).

## 2. Architecture (text diagram)

```
message / URL
   |
   +--> canonicaliser (NFKC, zero-width, homoglyph, URL extract, leetspeak)  src/rakshak/canonicalize.py
   |        raw view + canonical view (non-destructive) + rule side-features  src/rakshak/rules.py
   |
   +--> experts:  rules  |  fastText / char-CNN (SMS)  |  dual-view student (SMS, KD from mmBERT teacher)
   |              URL char-CNN (+ side features)  |  NIDS flow students (INT8 ONNX, Pi 5)
   |
   +--> fusion + calibration (thresholds fixed on cal only)  -> SAFE / UNCERTAIN / SCAM   src/rakshak/fusion/
            |
            +--> evidence IDs (rule hits, expert scores)   src/rakshak/evidence.py
                     |
                     +--> Llama 3.2 1B (ExecuTorch .pte, INT4)  only fills verifier-approved slots
                              |
                              +--> verifier (exact ID + action match)  src/rakshak/verifier.py -> template fallback
```

Phone = SMS/URL device. Pi 5 = reproducibility target and the network-IDS device. Both run the same `.pte` for Llama.

## 3. Where everything lives

| What | Where |
|---|---|
| Code, docs, tests | GitHub `vhmaadhav/rakshak-edge` (branch `main`) |
| Processed data (private) | HF dataset `highestintheroom/rakshak-data` (moved from `vhmaadhav/rakshak-data` on 2026-10-31, DONE) |
| Teacher, students, ONNX, results (private) | HF model `highestintheroom/rakshak-models` (moved from `vhmaadhav/rakshak-models` on 2026-10-31, DONE) |
| Kernel input data | Kaggle dataset `maadhavvh/rakshak-data` ([link](https://www.kaggle.com/datasets/maadhavvh/rakshak-data)). Shared with Kaggle user `sachin0496` (done) |
| Teacher transfer logits | Kaggle dataset `maadhavvh/rakshak-teacher-logits` ([link](https://www.kaggle.com/datasets/maadhavvh/rakshak-teacher-logits)). Shared with `sachin0496` (done) |
| Kaggle kernels | `maadhavvh/rakshak-<job>` (for example `rakshak-sms-student`, `rakshak-llama-ela-eval`). Private |
| Job outputs | `kaggle/out/<job>/output/` (gitignored). Small results are in the HF repo under `results/` |
| Session log, decisions | `docs/STATUS.md`, `docs/CONTEXT.md`, `docs/RUNLOG.md` if present |

Data licences are mixed. See `docs/DATACARD.md` section 1. Keep both HF repos and the Kaggle datasets private. HF access requires an org-scoped token with read (and write, to push) on `highestintheroom`.

## 4. How to run

```bash
uv sync                                  # Python env (or .venv/bin/ruff, .venv/bin/pytest)
.venv/bin/ruff check . && .venv/bin/pytest -q
```

Kaggle is the compute platform. The laptop runs code and unit tests only.

```bash
python3 kaggle/build_kernel.py --job smoke            # build only, pushes nothing
kaggle/launch.sh smoke --internet                     # build + push kernel (starts a run)
kaggle/wait_event.sh smoke --timeout 540              # ONE blocking ntfy wait; never poll
kaggle/fetch.sh smoke                                 # download outputs into kaggle/out/smoke/output/
kaggle/launch.sh url_student --gpu t4 --internet --datasets maadhavvh/rakshak-data \
    --job-args="--kd --budget-h 9"                  # flags per job; see kaggle/README.md
kaggle/launch.sh smoke --run-name b --internet        # parallel run: kernel maadhavvh/rakshak-smoke-b
kaggle/push_data.sh                                   # re-version maadhavvh/rakshak-data
kaggle/push_teacher_logits.sh                         # re-version maadhavvh/rakshak-teacher-logits
```

- Full docs: `kaggle/README.md`. Every launch has a ntfy completion line. Run `wait_event.sh` once after a launch.
- Kaggle allows **2 concurrent GPU kernels** on the free account. Check the running kernels before a launch.
- `colab/` launcher: short jobs only. A free Colab VM was reclaimed after about 45 minutes. Details in `colab/README.md`. Colab needs a one-time interactive OAuth by hand.
- Do not run any job on Modal without Maadhav's explicit OK for that run.

## 5. Results (headline; split names as in the reports)

Thresholds are chosen on the cal split only. Intervals are campaign-clustered bootstrap 95%. Full tables: `kaggle/out/<job>/output/report.md`, copied to HF `results/<job>/`.

**SMS teacher (mmBERT-base, job `sms_teacher`, `sms_teacher-b`)**

| Split | F1 | PR-AUC | Recall @ cal FPR 1% |
|---|---|---|---|
| campaign (test 22,145) | 0.987 [0.985, 0.989] | 0.998 | 0.981 |
| source_holdout ScamShield (75,973) | 0.871 | 0.913 | 0.960 |
| source_holdout UCI (5,159) | 0.958 | 0.988 | 0.945 |
| lang_shift (teacher-b, 1,677) | 0.802 | 0.918 | 0.689 |
| temporal (teacher-b, 11,029; 77 scam) | 0.915 | 0.922 | 0.883 |

Caveat: the shipped checkpoint in HF is the scamshield-holdout run (last kept by cal PR-AUC), not the campaign run. The folder name `mmbert-base-campaign-s0` is a request and is misleading. Rename it once the team agrees.

**SMS students (dual-view, job `sms_student` and `sms_student-kd`)**

| Model | Campaign F1 | lang_shift F1 | temporal F1 |
|---|---|---|---|
| Clean no-KD dual-base, mean seeds1/2/3 | 0.9744 | **0.4058** [0.3697,0.4412] | 0.8656 |
| Clean KD dual-base, mean seeds1/2/3 (KILL) | 0.9748 | **0.3520** [0.3181,0.3840] | 0.8696 |

**Current language-shift headline: clean no-KD student≈0.41 versus historical
teacher≈0.80 (teacher-b0.802 above).** The old contaminated≈0.81 claim is withdrawn,
not an estimate of clean performance. Clean KD made this gap worse: paired delta
−0.0539 CI[−0.0696,−0.0383], p0.000999; the fixed go gate fails. Single-teacher,
train-only targets, not OOF/ensemble. Full matched six-fold/three-seed results and
external student-only diagnostics belong to the separate clean-KD milestone
(`sms_clean_kd-kd1-20261101`v1; see issue1). No fused/deployed-FPR claim from this comparison.

| no-KD dual-base, seed 1 | 0.975 | **0.356** (fails Hinglish) | 0.903 |
| Historical KD dual-base, seed 1 (artifact exists; contaminated protocol) | 0.977 | 0.795 (seeds 2, 3: 0.816, 0.804) | 0.909 |

**Withdraw the claim that KD fixes Hinglish generalization.** The inherited unlabeled transfer
pool reintroduced excluded rows: campaign had 4,147 exact test copies; the ScamShield fold had
2,122 exact test copies and 9,707 held-out-source rows. Historical KD numbers cannot establish
disjoint generalization or independent external validation. Do not promote that artifact from
these metrics. The historical export selected the campaign run's seed1 by cal PR-AUC.

Corrected train-only KD, three seeds1/2/3 (mean F1; full campaign CIs, every source fold and
per-seed external FPR/recall in [SOURCE_BALANCE_RESULTS.md](SOURCE_BALANCE_RESULTS.md)):

| Arm | Campaign F1 | lang_shift F1 | temporal F1 | ScamShield source_holdout F1 |
|---|---:|---:|---:|---:|
| ERM control | 0.9753 | **0.4000** | 0.8513 | 0.7503 |
| Source-balanced (kill) | 0.9735 | 0.3367 | 0.8875 | 0.7437 |
| Balanced + group-DRO (kill) | 0.9727 | 0.4214 | 0.8946 | 0.7399 |

Both interventions fail the predeclared ScamShield improvement gate (paired delta CIs wholly
negative). Keep the train-only guard, not an intervention or a new deployment artifact.
These comparisons are **in-sample for rule-feature design**. Temporal retains three original
train/test campaign overlaps and only 77 positives; its split is unchanged. Language tagging
still misses Hinglish (about 51% agreement on ScamShield Hinglish rows).

At the cal-1%-FPR threshold, corrected ERM seed1 has GroundLM FPR 407/1,133 =35.92%
[21.80%,48.54%], recall69/91 =75.82% [61.02%,87.40%] (**in-sample for rule-feature design**).
SmishX fraud FPR267/881 =30.31% [25.81%,34.67%], recall119/226 =52.65%
[43.71%,62.72%] (**independent, non-Indian-specific**). All seeds/arms/thresholds are in the
report; these are negative external results, not a low-FPR deployment claim.
| Clean no-KD dual-base, mean seeds1/2/3 | 0.9744 | **0.4058** [0.3697,0.4412] | 0.8656 |
| Clean KD dual-base, mean seeds1/2/3 (KILL) | 0.9748 | **0.3520** [0.3181,0.3840] | 0.8696 |

Current clean language-shift headline: **student≈0.41 versus historical teacher≈0.80**.
The old contaminated≈0.81 claim is WITHDRAWN: transfer extras reintroduced test
copies/campaign neighbours/held-out-source rows. Clean train-only single-teacher
KD made the gap worse, paired delta−0.0539 CI[−0.0696,−0.0383], p0.000999.
No OOF/ensemble or held-out improvement claim. Full matched six-fold/three-seed
report and caveats: [PR#9](https://github.com/vhmaadhav/rakshak-edge/pull/9),
commitb17021f, docs/CLEAN_KD_RESULTS.md on that branch. Independent Indian
validation OPEN. Language tagging still misses Hinglish; it does not justify
restoring the withdrawn result.

**Classical SMS baselines (`sms_baselines`)**: char LR campaign F1 0.973, SVC campaign F1 0.978. Job ended `partial`: 10 of 190 units failed. Rules-only LR is near 0.67, so it is a floor. Canonicalising the text cuts attack success (for example 4.1% to 0.4% in the XLM-R pilot).

**URL (`url_baselines`, `url_student`, `phreshphish_extract`)**

- url_domain test (30,432): char-LR F1 0.996; **charcnn-wide** student F1 0.999, recall 1.000 at 1% FPR.
- PhreshPhish, URL-only, AP (mean over subsets) at base rates 0.05% / 1%: charcnn-wide 0.484 / 0.865; transformer-L4 0.245 / 0.590. The paper's GTE row uses URL and HTML (0.605 / 0.876). This is a different protocol. Quote as a separate row only.
- Caveat: the URL `transformer` distillation targets are in-sample logits (no cross-fitting). Its numbers are not held-out.

**New three-seed URL controls (E0,2026-11-02), original official protocol:**
Phresh URL-only mean AP0.4435 [domainCI0.3239,0.6206] at0.05%, and0.5589
[0.4286,0.7059] at0.1%; P@R90 .0734[.0521,.1835]/.1329[.0764,.2756]. All975
subsets and1000valid paired-domain draws retained. Every seed/rate/full benchmark/
test and url_domain CI: docs/URL_THREE_SEED_RESULTS.md. url_domain maxF1 for
seeds1/2/3 is.999216/.999281/.999564. Prior single-seed URL figures above are
historical, not matched controls. Seed-before-construction corrected in E0.

The pinned Apache2ByT5 teacher stopped at nonfinite loss before a selected model
or KD targets. **Teacher/KD UNMEASURABLE**, no stronger-teacher or distillation
win. Stop fixed arm without recipe/precision changes. Original temporal split
file is absent in current frozen input: UNMEASURABLE, not newly confirmed NA.
In-sample intervention selection from prior benchmark feedback; no independent/
SOTA/deployment claim. No URL expert substitution in frozenfusionv1; HFpending.

**Network IDS (`nids_baselines`, NetFlow-v2 schema)**

- Within dataset, random split (reference only): LightGBM F1 unsw_nb15 0.893, cse_cic_ids2018 0.976, bot_iot 0.999, ton_iot 0.984.
- INT8 student (live features, 6.7 KB): F1 unsw 0.830, cse 0.796, bot 0.997, ton 0.903. Latency 0.017 ms p50 on this host.
- **Does not generalise across datasets.** Leave-one-dataset-out LightGBM F1 on unsw_nb15 is 0.128. Cross-dataset MCC is mostly near zero or negative. bot_iot is attack-dominated (391 benign vs 18,834 attacks), so its scores are not base-rate results.

**ELA v2 (`llama_ela_eval`, unsloth Llama-3.2-1B-Instruct fp16, 300 items)**

- Corrected prompt injection (2026-11-01, same 150 attacks): freeform **35/150**, slot **0/150**;
  rank **0/39 eligible pairs**, **111 excluded**. Rank Wilson [0%, 8.97%], matched freeform 11/39.
- Slot arm: JSON parse 0.9167; verifier pass 0.9033; unsupported-claim rate 0; per-point lexicon
  pass **25.81% [13.04%, 40.82%]** English overall. Matched 150 packages: **21.62%
  [8.00%, 40.00%] English / 62.16% [40.61%, 79.59%] Hinglish** (clustered intervals).
  These are lexical checks, not entailment scores; the base English arm is weak.
  These are lexical checks, not entailment scores; English remains weak.
- Rank telemetry: **0 generated tokens**, **2,992 scored continuation tokens**, **248 candidate
  strings**, **86% zero-work clean rows**. Worked T4 p50/p95 **0.3110 / 1.0782 s**. The old
  character-based count and 0/150 rank injection denominator are withdrawn. Summary/action and
  the candidate bank are fixed; this is an architecture restriction, not learned robustness.
  Full provenance, per-family bounds and paired tests: `docs/RANK_ARM_AUDIT.md`.

**Verifier-gated QLoRA (Kaggle, 2026-11-01; FP16 GO / export-retention FAIL)**

| Frozen advisory endpoint | Base | Selected FP16 adapter | Campaign95% CI |
|---|---|---|---|
| EN evidence point pass |16/62=25.81% |60/62=96.77% |91.23%–100% |
| Hinglish evidence point pass |23/37=62.16% |24/37=64.86% |43.46%–80.96% |
| Slot harmful injection flips |0/150 |0/150 |IID Wilson upper2.50%; each family0/30 |

Final recorded gate **go**, zero unsupported/reassuring advisory rows. EN paired gain
+70.97pp [55.09,85.25]pp; Hinglish+2.70pp [−23.08,30.30]pp. Raw full-contract pass
only13.33%; verifier/fallback remains necessary. Freeform35/150→36/150 harmful flips
is exploratory and does not establish safety. **In-sample for rule/contract design and
repeated frozen-sample comparison**, not independent held-out improvement. No evaluation
SMS entered targets; only deterministic templates + pinned Llama paraphrases, private
10,934train/3,026dev. Built with Llama. Final audit tree`1c08f453b8b9`, recoverytree`d8120f4e72e4`.
Adapted8da4w PTE exists:860,959,104bytes, same-PTE600-row quality run completed.
**Retention FAIL** under the fixed paired−5pp margin:

| Export endpoint | Adapted PTE | Paired PTE−FP16 campaign95%CI |
|---|---|---|
| EN evidence-point pass |52/62=83.87% |−12.90pp [−28.96,0.00]pp |
| Hinglish evidence-point pass |22/37=59.46% |−5.41pp [−28.89,+17.86]pp |

Both lower bounds fail−5pp.
Unsupported0/slot harms0/150 (fivefamilies0/30) cannot override that failure.
Keep FP16 as research only; do not deploy this PTE or infer Pi5/Android performance.
Full result/provenance: docs/ELA_QLORA_RESULTS.md. Private Kaggle archive
`maadhavvh/rakshak-ela-models-b20261101`v1,privacy verified; **HF publish pending**.

**Frozen rules audit (Kaggle, 2026-11-01; `docs/REVIEW_AUDIT.md`)**

| Population / threshold | FPR | Fraud recall | Scope |
|---|---|---|---|
| GroundLM negatives, warning above LOW | 1/1,133 = 0.0883% [0%, 0.2971%] | — | **in-sample: these rows informed rule design** |
| SmishX, warning above LOW | 53/881 = 6.0159% [3.6571%, 8.8725%] | 39/226 = 17.2566% [10.6118%, 26.0009%] | **independent, non-Indian-specific** |
| SmishX, HIGH/SCAM | 10/881 = 1.1351% [0.5344%, 1.8782%] | 5/226 = 2.2124% [0.3774%, 5.1282%] | **independent, non-Indian-specific** |

Intervals are campaign-clustered; external campaigns are approximate template/domain groups.
GroundLM has 12 evidence-hit negatives, 11 urgency-only; evidence remains present after risk
downgrading. The old “13 total / 12 urgency-only” narrative is withdrawn. SmishX fails the 1%
reference FPR gate at both thresholds; rules-only recall is weak. Widening adds only one
SmishX fraud warning (paired +0.4425 pp, CI [0, 1.4360] pp, p=0.5485); no independent gain
claim. Model-side GroundLM FPR and independent Indian held-out validation remain OPEN.

**Not completed:** model-side GroundLM FPR; verified `llama_export` output; Pi 5 benchmarks; Android app.
**Not completed:** GroundLM FPR for all retained/teacher models (corrected train-only student
measured above, fails the reference gate); verified `llama_export` output; Pi 5 benchmarks;
Android app; independent Indian validation.

## 6. What failed and why

- **Hinglish generalization remains unresolved.** Matched clean noKD mean F1 is0.4058 versus teacher≈0.80; cleanKD0.3520 isKILL;
  the historical contaminated≈0.81 improvement is WITHDRAWN. Source-balanced/DRO
  do not meet the primary ScamShield gate. Never add raw transfer extras to training.
**Not completed:** model-side GroundLM FPR; verified `llama_export` output; Pi 5 benchmarks; Android app.

## 6. What failed and why

- **Hinglish generalization remains unresolved:** clean no-KD≈0.41 versus historical teacher≈0.80; clean KD≈0.35 is KILL. The contaminated≈0.81 claim remains WITHDRAWN (PR#9).
- **Colab free tier**: a detached run was reclaimed after about 45 min. Use Colab only for short jobs.
- **URL transformer distillation**: in-sample targets, so no cross-fitting. Not usable as a held-out claim until redone.
- **sms_baselines**: 10 of 190 units failed (partial). Cause not yet triaged. Read `kaggle/out/sms_baselines/output/run.log`.
- **llama_export**: not produced yet. `docs/EXECUTORCH_PLAN.md` flags an UNVERIFIED conversion risk: ExecuTorch 1.5.1 may lack a `llama3_2` HF converter. Expect a conversion fix or a Meta-format path.
- **Llama gating**: earlier blocked. The jobs now read ungated unsloth and executorch-community repos. No Kaggle consent needed for those (see `src/rakshak/llama_source.py`).

## 7. Open tasks (ranked)

1. **GroundLM FPR test.** Score the GroundLM external set (`external_groundlm`, 1,294 policy-negatives) with each shipped model at its cal threshold. *Acceptance:* FPR per model with a campaign-clustered CI, the 169 hash-only IIITD rows noted as missing, and the result in `docs/HANDOFF.md` section 5. Go/kill: FPR <= 1% on this set.
2. **Rank-arm audit, then ELA QLoRA.** Find out why the rank arm runs in 0 s with 0 tokens. Check the candidate scoring path on all 300 items. Then run `llama_export` and the verifier-gated QLoRA (`docs/ELA_RESEARCH.md`, A3). *Acceptance:* a written audit with a fixed rerun; the slot per-point pass rate reported per language; unsupported-claim rate still 0; export produces `Llama-3.2-1B-Instruct-8da4w-xnnpack.pte`.
3. **ScamShield confound.** ScamShield supplies most of the negatives and some of the positives (`docs/DATACARD.md` section 5). *Acceptance:* a source-balanced and Hinglish-only eval, with and without ScamShield, for the shipped student and teacher. Report as a companion row to every SMS claim.
4. **Spam versus scam relabel.** Five-way taxonomy (benign_transactional, benign_personal, promotion, scam, unclear). Use the labelling jobs in `docs/BUDGET.md` (cloud agents, no Claude-written Llama training text). *Acceptance:* relabelled `label_policy` in the data; metrics on scam versus not-scam; promotions not counted as fraud.
5. **Pi 5 benchmarks and `llama_main`.** Install `executorch==1.5.1` on 64-bit Pi OS. Run the `.pte` through `executorch.runtime` first. Build `llama_main` natively only if needed. *Acceptance:* p50/p95 latency, tok/s, peak RSS, thermals (stop at 85 °C) on a Pi 5 **4 GB** and 8 GB if available, or a documented out-of-memory result.
6. **Android app.** Share-to-scan and paste UX (Play restricts READ_SMS). Runs the student and the `.pte` on device. *Acceptance:* a build that works in airplane mode with an on-device latency number. AppFunctions is Phase 2 only.
7. **Report and demo.** Technical report and demo video for the challenge. *Acceptance:* every number traceable to a `results/` file; claims limited per `docs/CONTEXT.md`.
8. **HF cleanup (needs an org-scoped token).** The HF model card still says `vhmaadhav/rakshak-data`. The teacher folder `mmbert-base-campaign-s0` holds the scamshield-holdout run (see section 5). Fix both once an org-scoped token is available.

## 8. Working rules

- **Kaggle first** for all training, HPO, distillation and benchmark evals. Laptop: code, git and unit tests only.
- **No Modal** without Maadhav's explicit per-run OK.
- **CPU temperature:** check `sensors | grep Tctl` before local compute. At 85 °C or above, stop heavy work and hunt runaway processes.
- **Never poll.** No status or sleep loops. Use the ntfy wait once per launch.
- **Commit rarely,** in large milestones, as the repo owner's git identity (repo-local config). Do not override it.
- **No Claude-written training text** for the Llama explainer (job F is disabled in `rakshak.llm.cli`).
- **Never post outside our own repos.** No issues, PRs or comments on upstream projects without asking.
- **Research before fixing.** On a failure or unknown, run a cited research pass for the best-known fix first.
- **Log every run and tweak** in `docs/RUNLOG.md` if it exists. Otherwise add to `docs/STATUS.md`.
- **Llama 3.2 licence:** display "Built with Llama", ship the licence copy, name derived models "Llama...". Details in `docs/EXECUTORCH_PLAN.md` and `llama/README.md` on HF.

## 9. Accounts and credits

- **GitHub:** repo owner `vhmaadhav`. Collaborator `Sachin0496` is invited with push access (status in the handoff issue).
- **HF:** account `vhmaadhav`, member of org `highestintheroom`. Private repos are the store. Sachin needs an HF account invited to the org.
- **Kaggle:** account `maadhavvh`. Two concurrent GPU kernels. Both datasets are shared with `sachin0496` (done). The Kaggle CLI has no share command, so sharing is in each dataset's Settings > Sharing. To run kernels, Sachin can be added on the kernel pages (`maadhavvh/rakshak-*`), or rebuild them under his own account with the launcher. He needs his own ntfy topic file at `~/.config/rakshak/ntfy_topic` (the launcher creates it).
- **Anthropic:** $100 Claude Code cloud-session credit. Labelling runs stop at 50% and 75% of it (`docs/BUDGET.md`).
- **Colab:** free tier, unreliable for long runs.
- **Modal:** no runs without explicit approval.
