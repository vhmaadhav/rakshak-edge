# Handoff: Rakshak-Edge (for Sachin)

Last updated 2026-11-06 (evening). Owner until now: Maadhav (`vhmaadhav`). Start here, then `docs/STATUS.md` (session log) and `docs/CONTEXT.md` (decisions).

## State on 2026-11-06 (read this first)

**Repo.** All of today's work (rbench, the edge service, Llama on the Pi, guards G1/G2, Fixes A–D, quiet detector-only
mode, Hinglish routing, the GroundLM edge FPR eval, the detector fix plan and group-negative split) is committed on `main` as one
batch on top of `7c2491b`, with `verifier.py`, `ela_eval.py` and the dashboard's benchmark tab left unformatted (decision e).
Local eval outputs (`/results/`) are gitignored because they contain dataset SMS text. No branches or PRs.
Full suite before the commit: **1362 passed, 28 skipped** (torch absent on the laptop); `ruff check .` clean.

**Phase 1 is due 2026-11-23 (17 days).** The on-device path now runs end to end on the Pi: SMS in, student decision,
rules, guarded Llama explainer, JSON out. The deliverable still needs an Android app, a demo video and
the report.

### Repo cleanup and formatting (2026-11-06)

- `ruff check` passes on every changed or new area. `ruff format --check` passes on all new files (`deploy/pi/service/*`,
  `deploy/pi/bench/*`, the new tests, `slot.py`, `edge_pipeline.py`, `test_edge_service.py`).
- `src/rakshak/ela_eval.py` and `src/rakshak/verifier.py` were **not** reformatted. Both were already unformatted at HEAD
  (26 and 30 lines of `ruff format` diff). The diff adds only logic: the `risk` parameter (G1/G2) and the lexicon and
  hedge code. `verifier.py` now also has about 110 unformatted lines of new code (the attribution word tuples, the
  guard branches). `deploy/pi/pi_dashboard.py` was formatted at HEAD; its new benchmark tab (uncommitted, about 470 lines)
  is not, and `ruff format --check` fails on it. Also a formatting decision (open decision e).

### Raspberry Pi 5 (details: `docs/PI_SETUP.md`, `deploy/pi/README`)

| Item | Value |
|---|---|
| Access | `ssh -i ~/.ssh/pi5 -o IdentitiesOnly=yes hermoine@pi5.local` (key login; sudo needs Maadhav's password, none used) |
| Board | Pi 5 Model B Rev 1.1, 4 GB, 4× Cortex-A76, Raspberry Pi OS Lite (Debian 13), boots from a USB 2.0 stick (no SD) |
| CPU | `asimddp` (dotprod) present, **no i8mm**: ExecuTorch uses the SDOT int8 path |
| Power | **No 27 W PSU.** `usb_max_current_enable=1` in `config.txt`. The PMIC is the energy source (not a calibrated meter) |
| Cooling | **No heatsink or fan.** Idle 44–55 °C; batches peak 70–73 °C. Throttle flags `0x0` at every start and end today |
| Thermal kit | `rakshak_thermald` (frequency-cap governor, **systemd, enabled and active**), `thermal_run.py` (job guard: pause 75, resume 65, kill 80, CSV log), `pi_dashboard.py` (**systemd `rakshak-dashboard.service`, enabled and active**). Dashboard: `ssh -i ~/.ssh/pi5 -N -L 8765:127.0.0.1:8765 hermoine@pi5.local`, then http://localhost:8765 |
| Models | `~/rakshak-bench/models/`: `sms_student` (ONNX, threshold 0.35127517580986023) and `llama32_1b_spinquant` (`.pte` 1,135,951,488 bytes). Never touched by sync |

Heavy Pi jobs go through `thermal_run.py`. The eval client will not start a batch at 65 °C or above, and the service refuses explainer calls at 75 °C or above (503). Report numbers with the
cooling and power state: final numbers want an Active Cooler and a 27 W (5 A) PSU.

### rbench (`deploy/pi/bench/`)

Benchmark harness for the Pi: `rbench list|run`, registry with SHA and byte pins (`registry.toml`), dataset and
model status labels (`ready`, `incomplete`, `present (not verified)`), a thermal guard that pauses and kills on
temperature, and `sync_bench.sh` (`--code-only`, never deletes, never writes under `models/llama32_1b_spinquant/`).
Tests: `tests/test_rbench_*.py`, `tests/test_pi_*.py`.

### Llama 3.2 1B on ExecuTorch on the Pi (details: `docs/LLAMA_PI_RESULTS.md`)

- **Versions:** executorch 1.5.1, torch 2.14.0+cpu, pytorch-tokenizers 1.6.0, torchao 0.18.0, tokenizers 0.23.3, Python
  3.13.5, kernel 6.18.50+rpt-rpi-2712. Prebuilt wheels only, no source build.
- **Model:** `executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET` at revision `911221e0`, `.pte` sha256
  `8715cdba…cda3b`. Llama 3.2 Community License. The Meta repo is gated, so the pinned community checkpoint is used.
- **Runtime fixes (runtime loads, not re-export):** preload `libcustom_ops_aot_lib.so` (custom SDPA and fast Hadamard
  kernels) and `libexecutorch_kernels_quantized.so` with `ctypes.RTLD_GLOBAL`; call `runner.reset()` before each prompt
  (KV cache overflow after about 6 prompts). Code: `deploy/pi/bench/rbench/runners/pte_llm.py`. No fused-INT4
  `xnn_status_invalid_parameter` seen at 1.5.1 (not tested with `per_op_mode`).
- **Speed and memory (valid 10-prompt runs, `ela_*`):** load 7.5–11.4 s (method load 2.6–3.9 s), TTFT 3.9–7.3 s, prefill
  45–84 tok/s, decode 9.8–11.6 tok/s, peak RSS 1.83–1.84 GB. Thermal: start 52–60 °C, max 68.8–72.7 °C, no pauses.
- **Prompt path, points vs slot (30 evidence rows, threads 2):** the points contract (one JSON from the model) gave
  raw advisory pass 1/30 and p50 9.93 s. The slot arm (code writes the JSON, the model writes one sentence per evidence
  item, stop at sentence end) gave raw advisory pass 15/30, point pass 29/46 (63%), p50 2.65–2.77 s. Final output
  passes 30/30 by construction (fallback to template).
- **slot_kw** (keyword in the assistant prefix): worse (raw advisory 12/30, point 24/46). Kept as opt-in only.
- **Slot figure correction (resolves the 0.91 vs 7.7% inconsistency):** `kaggle/out/llama_ela_eval/output/metrics.json`
  reports slot `verifier_pass_rate` 0.9133, but 260 of the 300 clean rows have zero evidence and a zero-point advisory
  passes trivially. On the 40 evidence-bearing rows the FP16 slot pass is **14/40 (35%)** and the point pass is
  **16/58 (27.6%)**. `docs/ELA_RESEARCH.md` 7.7% is an older contract, superseded. Quote 35%, not 0.91.
- **Edge prompt variant `describe_kw`** (Fix A, below) is the explainer prompt the service uses.

### Guards and fixes on the explainer (`src/rakshak/verifier.py`, `deploy/pi/service/edge_pipeline.py`)

- **G1 (no explainer for LOW):** LOW-risk advisories get the template only; no LLM call. Skipped 9 of 30 rows in the
  unguarded Part 1 run and 1 of 30 in the service run; changes no final output.
- **G2 (attribution guard):** at LOW any scam-intent attribution is rejected, negation included. At MEDIUM and UNCERTAIN
  attribution is allowed only inside a hedge (`may/could/might be a scam`, `likely a scam`, and the Hinglish forms).
  HIGH is not checked, so a HIGH point can say "is a scam" (one case today, row 22, "claims to be a police or government
  agency", which is correct). Tests: `tests/test_verifier_guards.py`.
- **Fix A (describe, do not judge):** `describe_kw` prompt. Trial on 11 points: 11/11 raw pass, against 6/11 for the old
  prompt. Known defect: it sometimes echoes "restates the fact" (2 of 11 in the trial).
- **Fix B (detector flags, no rule evidence):** a message with student score ≥ threshold and no rule evidence is
  UNCERTAIN with `VERIFY_INDEPENDENTLY` and a fixed summary, no explainer call (`path: detector_only`). Before this, two
  benign rows were shown "no known scam pattern". `evidence._risk` is unchanged (Kaggle evals depend on it).
- **Fix C (cleanup, deterministic, `generate_slot_advisory(cleanup=True)`):** removes prompt echoes at the end of a
  sentence and blanks exact or near-exact duplicate points (difflib ≥ 0.9); blanked points fall back to the template.
  Tests: `tests/test_edge_cleanup.py`. Its repeat rule no longer fires within one type, because of Fix D below.
- **Fix D (one point per evidence type, `edge_pipeline.collapse_by_type`):** the explainer runs once per type on the first
  matched item, and the advisory is built against that collapsed package (the verifier's one-point-per-item contract
  holds for it). The response keeps the full `evidence` list, and each point has `count` and `evidence_ids`. Pi re-run
  (40 rows, 3 threads): points 45 to 33, LLM calls 45 to 33, total p50/p95 2 096/6 350 to 1 706/3 012 ms (evidence 30).
  Row 21 went partial to template (its first call was empty). Tests: `tests/test_edge_collapse.py`. See
  `docs/EDGE_PIPELINE.md` section 3c.

### Edge service (how to run, what it measured)

```bash
~/rakshak-bench/deploy/pi/service/start_service.sh --threads 3     # service in unshare -rn, plus the socket relay
~/rakshak-bench/deploy/pi/service/stop_service.sh                  # relay, then service
ssh -i ~/.ssh/pi5 -N -L 8770:127.0.0.1:8770 hermoine@pi5.local     # then http://localhost:8770
# on the Pi: eval_service.py select | run | report; soak.py run --threads N --requests 100
```

- **Offline by construction:** the service runs in its own network namespace with `lo` only, no route, no DNS. The client
  talks over the Unix socket `run/edge.sock`. Verified in `docs/EDGE_PIPELINE.md` section 2.
- **API:** `POST /analyze` returns decision (risk, action, score, threshold, scam state), evidence, advisory (path, mode,
  points with `llm_text` and `cleanup`), verifier (G2 rejections), `llm_calls`, `timings_ms`, device (temperature,
  throttle, PMIC energy), RSS. 503 `cooling` at SoC ≥ 75 °C. `GET /health` and the page (no external assets).
- **Threads:** default is **3**. The soak (100 requests, 0 aborts, 0 throttle flags, one run per count) gave p50/p95 total
  2 400/9 265 ms (2 threads), 1 939/7 152 ms (3), 1 820/7 524 ms (4). 4 threads peaks at 73.0 °C; 3 threads gives the
  lowest p95 with more margin. The 3-vs-4 choice is a judgement, not a measured optimum.

**40-row eval, same rows as before (30 evidence + 10 benign), today's Fix C run at 3 threads.** "Before" is the Fix A/B run
at 2 threads (`results/edge_eval/before_cleanup_t2/` on the Pi). The thread count and cleanup changed together, but the
model's sentences are identical in both runs for all 40 rows (greedy decoding; checked), so the output differences are
the cleanup alone.

| Measure | Before (Fix A/B, 2 threads) | After (Fix C, 3 threads) |
|---|---|---|
| Raw point pass (model sentences that pass the verifier) | 38 of 45 (84%) | **30 of 45 (67%)** |
| Same, excluding the 8 blanked duplicates | 30 of 37 (81%) | 30 of 37 (81%) |
| Advisory mode llm / partial / template, evidence 30 | 22 / 2 / 6 | **19 / 5 / 6** |
| Advisory raw pass, called advisories, evidence | 22 of 29 | 19 of 29 |
| G2-rejected sentences (distinct) | 1 | 1 (row 25, "is likely an investment scam") |
| G1 skipped (LOW, no call), evidence / all 40 | 1 / 9 | 1 / 9 |
| Fix B path (detector only), all 40 | 2 | 2 (rows 35, 38, both benign label 0) |
| Decisions, evidence (HIGH / MEDIUM / UNCERTAIN / LOW) | 1 / 20 / 8 / 1 | 1 / 20 / 8 / 1 |
| Cleanup: points echo-stripped / deduplicated (blanked) | n/a | **2 / 8** (of 45 points) |
| Final output passes the contract | 40 of 40 | 40 of 40 |
| Total p50 / p95, evidence 30 (ms) | 2 256 / 7 520 | **2 096 / 6 350** |
| Total p50 / p95, all 40 (ms) | 2 062 / 6 305 | **1 814 / 5 420** |
| LLM per advisory, evidence (ms, n = 29) | 2 275 / 7 618 | 2 133 / 6 417 |
| LLM per call (ms, n = 45) | 2 062 / 2 568 | 1 791 / 2 543 |
| Energy per request, evidence (J, p50 / p95 / mean) | 8.9 / 30.8 / 12.8 | 8.0 / 27.9 / 11.9 |
| Energy per request, all 40 (J, p50 / p95 / mean) | 7.8 / 27.4 / 9.6 | 7.3 / 25.1 / 8.9 |
| RSS after batch (MB, peak) | 1 676 (1 839) | 1 676 (1 839) |
| SoC at batch start, max during batch | 55.4 °C, 70.3 °C | 54.3 °C, 71.9 °C |
| 503 refusals, throttle flags at start and end | 0, `0x0` | 0, `0x0` |

Per-stage latency after Fix C, evidence 30 (p50 / p95, ms): canonicalise 0.90 / 1.60; detect (student ONNX) 3.53 / 6.70;
evidence and decision 0.98 / 2.54; verify and assemble 2.60 / 4.55. Energy is the PMIC rails (`vcgencmd pmic_read_adc`),
not a calibrated meter. The 2 Fix-B rows make no LLM call, so they are excluded from the LLM columns (the report script
counts them as "called"; read the LLM rows here, not the script's "all").

**Cleanup changes, all 10 (hand-checked, 4 cases).** The two echo strips (rows 0 and 14) and the eight duplicates (rows 4
×5, 11 ×2, 28 ×1) were all examined against the SMS text in `results/edge_eval/rows.json`:
- Row 0 (prize SMS, label 1, MEDIUM): "This message claims a prize or lottery." Correct. The echo "and has been matched"
  is removed.
- Row 14 (electricity disconnection SMS, label 1, MEDIUM): "This message threatens electricity disconnection." Correct.
  "and restates the fact" is removed.
- Row 4 (Prize Patrol SMS, label 1, MEDIUM): the model sentence is kept ("...is asking the reader to do something"),
  correct but generic. The five repeats became the template "This message claims a prize or lottery.", so the advisory
  still shows the same sentence six times. **Not fixed: the dedupe moves the repetition into the template.**
- Row 28 (bit.ly SMS, label 1, MEDIUM): the model sentence is kept ("contains a shortened link and is likely to be a link to
  a website or service"), correct. The exact repeat is the template "This message contains a shortened link." Repetitive.
Verdict: 4 of 4 factually correct, no overclaim; two of the four are still repetitive. Each of the 8 blanked duplicates had
passed the verifier before, which is why the raw pass fell from 38 to 30 of 45 while the pass on the remaining sentences
stayed at 30 of 37.

**Other hand-check (carried from today's earlier runs).** Section 3 of `docs/EDGE_PIPELINE.md` has 8 outputs: 6 correct
and 2 mild overclaims (rows 3 and 25). Section 2 has 6 (no unsupported claim; one HIGH "is a scam" stated unhedged by policy). A small
sample, not a wrong-claim rate.

**Residual defects.** (1) repeated identical template points (above; resolved by Fix D, see ranked task 1); (2) the model echoes prompt words ("restates the
fact", "matched" in the trial); (3) 4 of 45 calls end unterminated and fall back to the template; (4) generic clauses
("is asking the reader to do something") pass the verifier and add little; (5) a HIGH point can state scam intent as fact;
(6) the student flags benign bank, UPI and telecom notices at high confidence (see GroundLM below).

### GroundLM full-pipeline false-positive rate (laptop, no LLM; `docs/GROUNDLM_EDGE_FPR.md`)

In-sample: GroundLM rows informed rule design (`docs/GROUNDLM_FPR.md`). Independent Indian validation is still OPEN.
Decision path: shipped student threshold 0.35127517580986023, rules, state, `build_package`, the Fix B override, and
`choose_action`. The explainer is not run; it does not change risk or action. Scores come from the laptop ONNX run.

1,133 scoreable policy negatives (Wilson 95%):

| Alert | FP / 1,133 | Rate | 95% CI |
|---|---:|---:|---|
| (a) student score ≥ threshold | 471 | 41.57% | [38.73, 44.46] |
| (b) any rule evidence | 12 | 1.06% | [0.61, 1.84] |
| (c) risk ≥ MEDIUM (warning) | 1 | 0.09% | [0.02, 0.50] |
| (d) risk ≥ UNCERTAIN (visible notice with Fix B) | 472 | 41.66% | [38.82, 44.55] |
| (e) HIGH only | 1 | 0.09% | [0.02, 0.50] |

SmishX (independent, non-Indian; recall on 226 smishing, ham FPR on 606 legitimate):

| Alert | Recall | Ham FPR |
|---|---:|---:|
| (a) student | 69.47% | 9.08% |
| (b) rule evidence | 25.66% | 4.95% |
| (c) risk ≥ MEDIUM | 17.26% | 3.14% |
| (d) risk ≥ UNCERTAIN | 76.55% | 9.57% |
| (e) HIGH only | 2.21% | 1.49% |

470 of the 472 (d) flags have no rule evidence (student-only). They fall in bank (161 of 161) and UPI (146 of 146)
legitimate messages and in the telecom and balance-notice templates. Most are confident (403 of 471 at score ≥ 0.8), so this is
a calibration and training problem. A threshold cannot fix it: the smallest threshold with GroundLM FPR ≤ 1% is 0.99793,
which drops campaign recall to 71.9% and SmishX scam recall to 5.8%. The go criterion (≤ 11 FPs on 1,133) is met only by
alert (c), the MEDIUM-and-above warning. The visible-notice alert (d) is about 40 times over it.

### Open decisions for Maadhav

(a) **DECIDED (2026-11-06): quiet for demo; verify mode kept behind `detector_only_mode`; detector fix on bank/UPI/telecom
notices remains the research task.** Detector-only flags (score at or above threshold, no rule evidence) are LOW / IGNORE with
the neutral summary "No specific scam pattern was found in this message." (`advisory.path` = `detector_only_quiet`). No Llama call.
Setting: `--detector-only-mode {quiet,verify}` or `RAKSHAK_DETECTOR_ONLY_MODE`; default quiet. Measured: GroundLM visible-warning FPR
**0.18%** (2 of 1,133, [0.05, 0.64]) against 41.66% in verify; SmishX visible recall 23.45% (quiet) against 76.55% (verify); campaign
test recall 16.2% against 98.1%. The recall cost is accepted for the demo and is stated in the demo. Details: `docs/EDGE_PIPELINE.md`
("Detector-only mode"), `docs/GROUNDLM_EDGE_FPR.md`.
(b) **Cooler and 27 W PSU** for the final benchmark numbers. Without them, latency and throttle numbers are for this setup only.
(c) **Android phone:** which device, and is one available? The Android app (ExecuTorch LlamaDemo with the `executorch-android`
1.5.1 AAR) needs one for the demo.
(d) **Commit timing.** DECIDED (2026-11-06): committed as one batch.
(e) **verifier.py formatting.** DECIDED (2026-11-06): left as is; no formatting-only commit.

### Ranked next tasks

1. DONE by Fix D (checked 2026-11-06 on the Pi's latest `results/edge_eval/responses.jsonl`, the Fix D run at 3 threads):
   row 4 is now one `prize_lottery` point (count 6), and row 28 is one `shortened_url` point (count 2). None of the 40 rows has two points
   with text similarity ≥ 0.8. No re-run needed.
2. Final numbers with the cooler and the 27 W PSU: 100-request soak at 3 and 4 threads, a live 503 gate test (forced
   temperature), and a repeat of the 40-row eval. Report the cooling and power state with every number.
3. DONE (Hinglish, 2026-11-06, `docs/EDGE_PIPELINE.md` section 5). Per-request routing by `detect_lang`; `--hinglish-mode`
   defaults to `template` (verifier Hinglish templates, no LLM, 40 of 40 pass). The Hinglish Llama prompt does not win: raw point pass
   8 of 37 (22%), 29 of 37 calls unterminated (the model quotes the matched text and stops on the quote mark), and the sentences
   that pass are often wrong Hinglish (10-sentence model check, not native-speaker). Routing: 2 of the 40 English eval rows route
   hinglish (`906572c8cc42edc04053`, `62046b0c0c23c1bb48fb`); the other 38 are unchanged. Hinglish eval set: 30 scam + 10 benign
   (`results/edge_eval_hinglish/rows.json`). Detector recall 30 of 30 scam, 0 of 10 benign flags; one benign rule false alert (row 34).
   Open: a Hinglish-tuned explainer prompt or stop rule (not tuned on the eval rows); native-speaker review.
4. Android app from the ExecuTorch LlamaDemo (AAR 1.5.1), calling the Pi service or the on-device model. Due 2026-11-23.
5. Demo video and the report (Phase 1 deliverables), with the telemetry from the service's `/health` and `/analyze`.
6. Decide the HIGH "is a scam" policy (stated fact vs hedged) and the "likely" hedge limit (G2 only matches "likely" directly
   before the term; row 25 is the one G2 rejection).
7. DONE: committed as one batch (2026-11-06).

### Untested

- A live 503 cooling refusal (only mocked in `tests/test_edge_service.py`).
- Fix A beyond 11 trial points; the `describe_kw` echo of "restates the fact".
- Fix B beyond 2 real hits in the 40 rows (5 in the scan); the UNCERTAIN choice is not user-tested.
- Hinglish explainer output through the service.
- More than 40 requests per eval (soak 100 requests on each thread count, one run each).
- Thermal behaviour past 73 °C, and any run with a cooler.
- The page in a real phone browser (desktop at 375 px only). The relay under concurrent browser connections.
- Restart after a crash (the scripts assume a clean stop).
- (Done) Full test suite before the commit: 1362 passed, 28 skipped.

## Earlier notes (2026-11-01 to 11-03)

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

Evidence-required alert policy also **KILL** (2026-11-02), no v2/wiring:
GroundLM SCAM0/1133 (degenerate empirical clusterCI[0,0], not a population guarantee),
alert4/1133=.353%; SmishX alertrecall175/226=77.43% vsrules39/226/student113/226,
but10internal fold/label recall guards fail. GroundLM SCAMrecall0/91 must remain
visible; SmishX fraudalertFPR306/881=34.73%. NativeURLcalONLY18087originalrows,
threshold.6308487409530461; exactIDs/hash/threshold frozen beforeSMSloading.
Named policy manifestc51d8f55102637a67da2786d90b9a7abfc053546a0e8ff18cd771890dd35635a
binds unchangedv1experts. Original escalation remains UNMEASURABLE, native1adapter
failure retained. In-sample policyselection, independentIndianvalidationOPEN.
Full endpoints/pairedCIs/coverage/risk–coverage: EVIDENCE_REQUIRED_ALERT_RESULTS.md.
Next lastP1arm(c)per-familycalibration is separately predeclared; no thresholdfishing.

Last P1 arm(c), template-cal-c1 (2026-11-02): **KILL**,9external gate failures.
All90family fits were sparse under fixed cal minimums and used exactglobalfallback;
no intervention/map was fitted. GroundSCAMFPR38.83/29.83/30.71%,alert40.86/32.39/31.42%
across clean noKDseeds1/2/3; SmishSCAMrecall20/29/106of226 belowstudent118/116/122.
Allinternal retention guards pass by identicaloutputs; UCIknownfraudNA/legacyretained.
These are a named newcohort, not a replacement for immutablev1; **in-sample
intervention selection**, no independentIndian/heldout gain. No more P1threshold
arms or wiring from this result. Fullfolds/seeds/CIs/pairedtests/calcounts:
`docs/TEMPLATE_CALIBRATION_RESULTS.md`. IndependentIndianinboxvalidation OPEN.
Current clean studentlangF1≈.41 versus teacher≈.80, contaminated≈.81 WITHDRAWN;
next multilingualrepresentation experiment is separate and has no measured result yet.

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

**New compact multilingual research candidate (M3,2026-11-02), mean seeds1/2/3:**

| Matched M3 arm | Campaign F1 | lang_shift F1 [campaign CI] | Temporal F1 |
|---|---|---|---|
| Fresh byte noKD control | 0.9747 | 0.4633 [0.4268,0.4951] | 0.8877 |
| Multilingual KD hybrid, researchGO | 0.9764 | **0.8916 [0.8630,0.9171]** | 0.8644 |
| Multilingual noKD diagnostic only | 0.9762 | 0.8726 [0.8468,0.8953] | 0.8459 |

Paired language gain+0.4283 CI[0.3909,0.4622], p0.000999; allsix fold-loss guards
and18classifier-size checks pass. **In-sample intervention selection**, legacy
spam-inclusive F1, conditional on3fixed seeds, single teacher0/trainIDs only.
NoKDhybrid is not promotable, and this comparison does not isolate KD's added
benefit. Classifier tensors4,600,490bytes; tokenizer34,393,406bytes excluded from
8MiB gate. Arm/export/runtime remain UNVERIFIED; no deployment approval.

**Transactional safety is unresolved:** candidate GroundLM fraud FPR39.19/34.86/
38.31% and SmishX25.09/23.50/29.06% at cal1%-FPR, seeds1/2/3. These are student-only,
not fusedv1 claims. GroundLM/SmishX feedback makes this intervention selection
in-sample; independent Indian validation OPEN. Allseed/fold/external endpoints,
paired CIs, original failures/source hashes: docs/MULTILINGUAL_RESULTS.md.

**Prior clean KD experiment below is a different cohort, not M3's matched control:**

| Model | Campaign F1 | lang_shift F1 | temporal F1 |
|---|---|---|---|
| Clean no-KD dual-base, mean seeds1/2/3 | 0.9744 | **0.4058** [0.3697,0.4412] | 0.8656 |
| Clean KD dual-base, mean seeds1/2/3 (KILL) | 0.9748 | **0.3520** [0.3181,0.3840] | 0.8696 |

**Prior clean language-shift reference: no-KD student≈0.41 versus historical
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

- **Hinglish evidence has a new research candidate, not a shipment.** M3hybrid0.8916 versus its fresh byte0.4633 passes its fixed research gate; the previous clean no-KD0.4058/teacher≈0.80 and clean KD0.3520KILL remain separate reference runs;
  the historical ~0.80 improvement used a contaminated unlabeled pool. Source-balanced/DRO
  do not meet the primary ScamShield gate. Never add raw transfer extras to training.
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
5. **Pi 5 benchmarks and `llama_main`.** The Pi is set up (see "State on 2026-11-06"). Install `executorch==1.5.1` in a venv on the Pi. Run the `.pte` through `executorch.runtime` first, under `deploy/pi/thermal_run.py`. Build `llama_main` natively only if needed. *Acceptance:* p50/p95 latency, tok/s, peak RSS, max temperature and throttle flags (guard pauses at 75 °C, stops at 80 °C; no cooler fitted) on the Pi 5 **4 GB**, or a documented out-of-memory result.
6. **Android app.** Share-to-scan and paste UX (Play restricts READ_SMS). Runs the student and the `.pte` on device. *Acceptance:* a build that works in airplane mode with an on-device latency number. AppFunctions is Phase 2 only.
7. **Report and demo.** Technical report and demo video for the challenge. *Acceptance:* every number traceable to a `results/` file; claims limited per `docs/CONTEXT.md`.
8. **HF cleanup (needs an org-scoped token).** The HF model card still says `vhmaadhav/rakshak-data`. The teacher folder `mmbert-base-campaign-s0` holds the scamshield-holdout run (see section 5). Fix both once an org-scoped token is available.

## 8. Working rules

- **Kaggle first** for all training, HPO, distillation and benchmark evals. Laptop: code, git and unit tests only.
- **No Modal** without Maadhav's explicit per-run OK.
- **CPU temperature:** check `sensors | grep Tctl` before local compute. At 85 °C or above, stop heavy work and hunt runaway processes.
- **Never poll.** No status or sleep loops. Use the ntfy wait once per launch.
- **Pi 5 has no cooler.** Run heavy Pi work only through `deploy/pi/thermal_run.py`. No stress tests, overclocking, voltage changes or disabling thermal protection. Check `vcgencmd get_throttled` before and after every benchmark.
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

## Provenance note (2026-11-02, ported from Codex NIDS worktree)

Historical IDS CIs are **IID-stratified**, not host-clustered: allfour source
copies lacked host columns, so T4 was unavailable. K5000 target baseline attack
shares .2374/.1886/.982/.7798 do not establish mostly-benign-home operation.
Provenance recovery and any future source-only covariance candidate remain OPEN;
see docs/NIDS_NEXT_RESEARCH.md. No new model or split is introduced here.
