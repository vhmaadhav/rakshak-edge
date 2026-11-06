# Llama 3.2 1B SpinQuant INT4 on the Raspberry Pi 5 (ExecuTorch 1.5.1, rbench)

Date: 2026-11-06. Device: Pi 5, 4 GB, 4x Cortex-A76 (dotprod, no i8mm), Debian 13, no cooler,
Python 3.13.5 venv at `~/rakshak-bench/.venv`. Model: `executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET`
at revision `911221e0f49d3354cbd4697a9c09feee25cadec7`, `.pte` sha256 `8715cdba...cda3b` (verified).

## Verdict

- **It works.** The prebuilt `.pte` loads and generates intelligible text on the Pi, offline, through
  the rbench harness, using the repo's ELA prompt construction and verifier.
- **The quality on this task is poor.** On 10 evidence-bearing SMS prompts, the verifier passes 0 of 10.
  Outputs are mostly truncated or malformed JSON (empty `points`, `points` as a string), so the
  template fallback is 100%. Speed is usable (about 5-7 s per prompt at 2 threads, about 3 s prefill
  latency). The on-device explainer does not yet meet the ELA contract.
- The first 10-prompt runs (`llama_*`) are **invalid for quality**: their random prompts had zero rule
  evidence (see "Sampling fix" below). They stay in `results/` and are marked superseded in this doc.

## Versions

| Component | Version | Notes |
| --- | --- | --- |
| executorch | 1.5.1 | pip, aarch64 cp313 wheel |
| torch | 2.14.0+cpu | `--index-url https://download.pytorch.org/whl/cpu` (the version `install_executorch.py` names at v1.5.1) |
| pytorch-tokenizers | 1.6.0 | resolved by executorch 1.5.1 (requires >=1.5.0) |
| torchao | 0.18.0 | required by executorch 1.5.1 (`>=0.18,<0.19`) |
| tokenizers / tiktoken | 0.23.3 / 0.14.0 | resolved dependencies |
| numpy / pandas / pyarrow | 2.5.4 / 3.1.1 / 26.0.1 | rbench and ELA code |
| Kernel / Python | 6.18.50+rpt-rpi-2712 / 3.13.5 | |

Install (no compilation; both are prebuilt wheels):

```bash
~/rakshak-bench/.venv/bin/pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu
~/rakshak-bench/.venv/bin/pip install "executorch==1.5.1"
```

No ExecuTorch or `llama_main` source build was done on the Pi or the laptop.

## What had to be done to make the prebuilt .pte run (runtime, not re-export)

1. **Missing Llama custom kernels.** The first load failed at method load with
   `Missing operator: llama::custom_sdpa.out` and `llama::fast_hadamard_transform.out`
   (`Failed to load method 'forward'`). The wheel ships both kernels in
   `executorch/extension/llm/custom_ops/libcustom_ops_aot_lib.so`, but importing the Python package
   does not register them. Loading that shared library with `ctypes.RTLD_GLOBAL` before the runner
   does (checked: 251 registered ops, including both kernels).
2. **Missing quantized embedding kernel.** Next error: `quantized_decomposed::embedding_byte.dtype_out`.
   Fixed the same way with `executorch/lib/libexecutorch_kernels_quantized.so`.
3. **KV cache overflow.** The runner keeps its position across `generate()` calls. After about 6
   prompts it failed with `pos_ 1819 + num_prompt_tokens 284 >= max_context_len 2048`. Fixed by
   calling `runner.reset()` before each prompt.

These are runtime loads of binaries shipped in the wheel. They are not source builds and not
re-exports. The code is in `deploy/pi/bench/rbench/runners/pte_llm.py` (`_preload_kernels`).

## Commands

Smoke test (16 tokens, one prompt, before the rbench wiring; scratch script on the Pi, not synced):

```bash
cd ~/rakshak-bench && .venv/bin/python smoke/smoke16.py \
  models/llama32_1b_spinquant/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte \
  models/llama32_1b_spinquant/tokenizer.model 16 "<chat-template prompt>"
```

Doctor and registry check:

```bash
cd ~/rakshak-bench && .venv/bin/python deploy/pi/bench/bin/rbench list models
# llama32_1b_spinquant   pte_llm   ready
.venv/bin/python deploy/pi/bench/bin/rbench list models --no-hash   # present (not verified)
```

Benchmark (the 10-prompt matrix; `--threads` 2 then 4, `--max-new-tokens` 64 (the rbench default) and 24):

```bash
cd ~/rakshak-bench
.venv/bin/python deploy/pi/bench/bin/rbench run --model llama32_1b_spinquant \
  --dataset sms_ela_campaign_test --limit 10 --threads 2 --max-new-tokens 64 --name ela_t2_m64 --json
# likewise --threads 2 --max-new-tokens 24, --threads 4 --max-new-tokens 64 and 24
```

Offline run (no network; `unshare -rn`, which needs no sudo; unprivileged user namespaces work here):

```bash
unshare -rn .venv/bin/python deploy/pi/bench/bin/rbench run --model llama32_1b_spinquant \
  --dataset sms_ela_campaign_test --limit 10 --threads 2 --max-new-tokens 64 --name ela_offline_t2_m64 --json
```

Offline proof, inside the same namespace: the only interface is `lo`. `curl -sI https://huggingface.co`
returns rc 6 (no DNS) and `curl -sI https://1.1.1.1` returns rc 7 (no route). Outside the namespace
the Pi has `eth0` and `wlan0`, and SSH stays up because the namespace is only around the benchmark.

## Metrics (10 prompts, seed 0, evidence-bearing sample, valid runs)

Columns: `load_s` = whole `load()` (preload, runner init, warmup generate that loads the method);
`model_load_ms` = the method load alone (from the runtime's stats); TTFT, prefill and decode from the
runtime Stats per prompt, averaged over the 10 prompts. Peak RSS is the process max RSS. Temperatures in
°C. All runs: `throttled=0x0` before and after, 0 pauses, 0 kills.

| Run | Threads | max_new | load_s | model_load_ms | TTFT ms | prefill tok/s | decode tok/s | new tok (mean) | peak RSS MB | nonempty | JSON parse | verifier pass | template fallback | temp start / max / end | power W avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ela_t2_m64 | 2 | 64 | 7.45 | 2558 | 5539 | 59.2 | 11.6 | 41.9 | 1842 | 1.0 | 0.8 | 0.0 | 1.0 | 51.8 / 70.0 / 66.1 | 4.14 |
| ela_t2_m24 | 2 | 24 | 8.29 | 2904 | 5832 | 56.1 | 11.6 | 23.0 | 1843 | 1.0 | 0.0 | 0.0 | 1.0 | 57.3 / 68.8 / 65.0 | 3.80 |
| ela_t4_m64 | 4 | 64 | 9.43 | 3226 | 3938 | 83.7 | 10.1 | 41.9 | 1827 | 1.0 | 0.8 | 0.0 | 1.0 | 59.5 / 72.7 / 64.5 | 3.88 |
| ela_t4_m24 | 4 | 24 | 11.40 | 3915 | 4097 | 79.6 | 9.8 | 23.0 | 1842 | 1.0 | 0.0 | 0.0 | 1.0 | 60.0 / 68.8 / 63.9 | 3.59 |
| ela_offline_t2_m64 | 2 | 64 | 10.75 | 3813 | 7255 | 45.0 | 10.5 | 41.9 | 1843 | 1.0 | 0.8 | 0.0 | 1.0 | 59.5 / 64.5 / 61.1 | 3.08 |

Per-prompt wall time at 2 threads / 64 tokens: p50 about 9.3 s, p95 about 10.1 s (run `ela_t2_m64`).
Throughput: 0.11 prompts/s. The offline run is slower (about 11 s per prompt, mean clock 1591 MHz against
2154 MHz for `ela_t2_m64`). The thermal governor (`rakshak_thermald`) capped `scaling_max_freq`
(`cap_mhz_min` 1500 to 2200 in the thermal blocks), which is the likely cause; it was not isolated.
Treat speed differences under about 20% as noise.

Earlier runs on the same Pi, invalid for quality (the sample had zero rule evidence, see below). Speed
numbers are for the short system-prompt-dominated prompts and are not comparable to the table above:

| Run (superseded) | Threads | max_new | JSON parse | verifier pass | template fallback | decode tok/s | TTFT ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama_t2_m64 | 2 | 64 | 1.0 | 1.0 | 0.0 | 11.7 | 4639 |
| llama_t2_m24 | 2 | 24 | 1.0 | 1.0 | 0.0 | 11.8 | 4824 |
| llama_t4_m64 | 4 | 64 | 1.0 | 1.0 | 0.0 | 10.3 | 3245 |
| llama_t4_m24 | 4 | 24 | 1.0 | 1.0 | 0.0 | 10.3 | 3354 |
| llama_offline_t2_m64 | 2 | 64 | 1.0 | 1.0 | 0.0 | 12.0 | 5304 |

Result directories on the Pi: `~/rakshak-bench/results/` (run IDs in `index.jsonl`; the valid runs are
`ela_*`, the superseded ones `llama_*`). The laptop has no copy yet (`rbench-remote pull` would copy them).

Single-run 16-token smoke test (one prompt, before the rbench wiring): load 0.09 s for the runner
init, 3.43 s for the method load, TTFT 416 ms for 32 prompt tokens (about 77 tok/s prefill), decode
about 13 tok/s, peak RSS 1604 MB. Output: "A smoke detector is a safety device that detects the
presence of smoke in the air".

## Sample outputs (three prompts, `ela_t2_m64`, verbatim after stripping `<|eot_id|>`)

1. Row `39f0ecce180e54679369` (mode `partial`, verifier fail, 35 tokens):
   `{"lang": "en", "summary": "The message claims a prize or lottery, but does not specify what the prize is.", "points": ["", ""]}`
2. Row `477eaf15ac735a8f4b27` (mode `template`, verifier fail, 36 tokens):
   `{"lang": "en", "summary": "The message claims a prize or lottery, but does not specify what prize or lottery it is.", "points": "MEDIUM"}}`
3. Row `d02552160d8cabf80d3f` (mode `template`, verifier fail, 56 tokens):
   `{"lang": "en", "summary": "Be cautious of messages asking for payment or fees to be sent to a parcel or customs", "points": "Matched text: 'duty registration at malad east pay' is a possible indicator of a fraudulent message."}`

Failure modes seen across the 10 prompts: empty points (`point is empty`), `points` not a list, a
missing lexicon keyword for the evidence type (`no en lexicon keyword for prize_lottery`), and
outputs cut at the budget. At 24 tokens every output is truncated before the JSON closes (JSON parse 0.0).
At 64 tokens the mean length is 41.9 tokens and two outputs are cut.

## Sampling fix (why the first numbers were invalid)

The first 10-prompt runs drew random rows from the campaign SMS test. A 200-row probe showed that the
rules fire on only 17 of 200 rows (8.5%), and all 10 sampled rows had zero evidence. Those rows test
only the "No facts were provided" path, so the verifier passed trivially. The worker now draws
seeded random rows that have at least one evidence item (`_ela_rows` in `worker.py`). It scanned 152
rows to find 10 (hit rate 0.066). The denominator notes record this. The 8.5% and 6.6% hit rates are
themselves a finding: the rules cover a small fraction of the campaign set, so most SMS never reach
the explainer.

## Orchestrator notes (answered)

- **Fused INT4 `xnn_status_invalid_parameter` (ExecuTorch 1.2.0 blog, `per_op_mode` workaround):**
  not observed on 1.5.1 aarch64 in any run here (the 16-token smoke test and all the benchmark runs
  above, both thread counts and both budgets). Not tested with `per_op_mode`; not needed so far.
- **`generate_from_pos` and raw logits:** neither is exposed by the installed Python bindings. The
  `TextLLMRunner` stub (`executorch/extension/llm/runner/_llm_runner.pyi`) has `generate`, `prefill`,
  `reset`, `stop`, `get_vocab_size`, and `Stats`. `prefill()` rebuilds the KV cache without generating,
  which may be a route to prefix reuse, but this was not built or tested. A C++ runner would be
  needed for `generate_from_pos` and logits.
- **Load separate from per-prompt timing:** done. One process loads once (`load()` runs a 1-token warmup
  that loads the method), then all prompts run in that process. `load_s` and `model_load_ms` are reported
  separately from TTFT and per-prompt latency.

## Thermal and power

Every run started below 65 °C (51.8 to 60.0 °C), with `throttled=0x0` before and after. Peak temperature
was 72.7 °C in the 4-thread run (the 75 °C pause never triggered), with 0 pauses and 0 kills. The
`rakshak_thermald` governor capped the clock to 1500 to 2200 MHz during the series. Average power was
3.1 to 4.1 W during generation (the guard's estimate, not a PMIC measurement). Energy per token was not
computed.

## Failures and deviations

- `rbench-remote doctor` could not run from the Pi (it is a laptop wrapper); the Pi-side launcher
  `deploy/pi/bench/bin/rbench` works. A mistaken laptop-wrapper call on the Pi failed harmlessly.
- `/usr/bin/time` is not installed on the Pi; peak RSS comes from `getrusage` in the runner.
- The `ela_generations` dataset is still disabled (it holds outputs, not prompts) and still appears in the
  model's compatible-dataset list. It is not used.
- The rbench result for `ela_*` has `json_parse_rate` 0.0 at 24 tokens: the budget is too short for the
  explainer, not a parse bug.

## Status labels (step 6)

- `llama32_1b_spinquant` now shows `ready` (both files hashed and matching, and sizes pinned:
  `.pte` 1,135,951,488 bytes, `tokenizer.model` 2,183,982 bytes). With `--no-hash` it shows
  `present (not verified)`. A pinned file with the wrong size shows `incomplete`. A file with no pin
  shows `present (sha not pinned)`.

## QLoRA prebuilt model

Not downloaded and not run. It is disabled in `registry.toml` with an empty sha256. It would need the
same two runtime preloads. Whether it is worth trying depends on the verifier problem above, not on
load or speed: the SpinQuant model's JSON failures are the blocker, and a different checkpoint may or
may not fix them. Not evaluated.

## Files changed (laptop repo, uncommitted)

- `deploy/pi/bench/rbench/runners/pte_llm.py`: rewritten against the real API (preloads, warmup, reset,
  threadpool, stats).
- `deploy/pi/bench/rbench/worker.py`: ELA prompt construction and verifier scoring, evidence-sampled
  rows (`_ela_rows`), runtime facts in the `llm` block.
- `deploy/pi/bench/rbench/datasets.py`: ELA rows keep the SMS text.
- `deploy/pi/bench/rbench/registry.py`, `deploy/pi/bench/rbench/fetch.py`: `incomplete` and
  `present (not verified)` labels, `bytes` pins, clearer fetch error.
- `deploy/pi/bench/registry.toml`: Llama `bytes` pins; new dataset `sms_ela_campaign_test` (task
  `ela_prompts`).
- `tests/test_rbench_registry.py` (status labels, Llama pins), `tests/test_rbench_pte_runner.py` (new,
  mocked ExecuTorch API).
- `docs/LLAMA_PI_RESULTS.md` (this file).

Tests: `tests/test_rbench_*.py` and `tests/test_pi_*.py` pass (119); `ruff check` and `ruff format --check`
pass on `deploy/pi/bench` and those tests. Sync: `deploy/pi/bench/sync_bench.sh` copied the code; it never
touched `models/llama32_1b_spinquant/`.

## Untested

- The ExecuTorch C++ runner (`llama_main`), not built, by design.
- Verifier pass on a proper template or prompt; the 0.0 verifier pass is the measured output of the prebuilt
  model, not of the prompt design.
- Thread counts beyond 2 and 4; `per_op_mode`; `prefill()`-based prefix reuse; energy per token.
- Sample size: 10 prompts per run. Rates are noisy at that size (0.8 means 2 of 10 failed).

## Points vs slot (2026-11-06, Pi 5, 30 rows, threads 2, seed 0)

Question: can the on-device Llama pass the verifier if code builds the JSON and the model writes only
short per-evidence sentences? Two arms on the same 30 evidence-bearing rows (`_ela_rows`, seed 0):

- **points** (`--ela-mode points`, the section above): one call per advisory, the model writes the whole
  JSON from `build_points_chat_prompt`. `--max-new-tokens 64`.
- **slot** (`--ela-mode slot`, `deploy/pi/bench/rbench/slot.py`): code writes `lang`, the template
  `summary` and the point list. For each evidence item the model gets a compact prompt (short system
  text, one `FACT:` line, assistant prefix `This message`) and writes one sentence. Generation stops at
  a sentence end, a newline, a quote or a brace (`stop()` on the runner), with a 20-token budget. A
  sentence is kept only if it ends in a sentence terminator. Otherwise, or if the verifier rejects it,
  that point is the template sentence. Runs: `cmp_points_t2` (run dir `20261106T140943Z-ae6068`) and
  `cmp_slot_t2` (`20261106T140733Z-f9c237`), both under the thermal guard, 0 pauses, 0 kills.

Columns: latency is per advisory (whole advisory, all calls). "Raw" means the verifier applied to the
model's own output before any template fallback. "Final" means the advisory the user sees.

| Metric | points | slot |
| --- | --- | --- |
| Latency p50 / p95 (s, per advisory) | 9.93 / 10.75 | 2.65 / 8.72 |
| Latency mean (s) | 9.83 | 3.65 |
| TTFT, mean per call (ms) | 6396 | 1533 |
| Calls per advisory (mean) | 1.00 | 1.53 |
| Prompt tokens per advisory (mean) | 318 | 145 |
| Prompt tokens per call (mean) | 318 | 94.5 |
| New tokens per advisory (mean) | 39.6 | 18.5 |
| Model JSON valid (raw output parses) | 0.60 | n/a (model writes no JSON; assembled output 1.00) |
| Raw verifier pass, advisory level | 0.033 (1/30) | 0.50 (15/30) |
| Raw verifier pass, point level | 0.022 | 0.630 |
| Template fallback, advisory level (mode != llm) | 0.967 (29/30) | 0.50 (15/30) |
| Template fallback, point level | 0.978 | 0.370 |
| Final output passes verifier | n/a (fallback is in render) | 1.00 (30/30) |
| Mode counts (llm / partial / template) | 1 / 7 / 22 | 15 / 15 / 0 |
| Stop predicate fired (per call) | n/a | 0.913 |
| Slot status (ok / unterminated / empty) | n/a | 42 / 4 / 0 |
| Thermal: start / max / end (C) | 59.5 / 70.5 / 65.0 | 48.5 / 70.5 / 66.1 |
| Thermal: pauses, throttled before/after | 0, 0x0 / 0x0 | 0, 0x0 / 0x0 |
| Mean CPU frequency (MHz), power avg (W) | 1780 / 3.48 | 2188 / 4.11 |

Notes on the table:

- The slot p95 (8.7 s) is one slow advisory on a long run, not a steady state: the mean is 3.65 s and the
  p50 is 2.65 s. Thermal peaks were the same (70.5 C) in both runs. The frequency caps were different
  (1700 MHz for points, 1900 MHz for slot), so the two latency columns are not on equal clocks.
- The points "JSON valid" 0.60 is the parse rate of the raw output (`extract_json_object`). The same
  outputs fail the points contract on schema, so most advisories fall back.
- The slot raw verifier pass (0.50) is much lower than its point pass (0.63) because one empty or
  rejected point fails the whole advisory on the raw check. The final output is always valid by
  construction (`final_verifier_pass_rate` 1.00).

### Why the earlier 0.91 and 7.7% figures do not compare

- `kaggle/out/llama_ela_eval/output/metrics.json` (FP16, unsloth, 300 items, Oct 31) reports slot
  `verifier_pass_rate` 0.9133. That figure is inflated. 260 of the 300 clean rows have **zero evidence**
  (`points_n` 0), and the verifier passes a zero-point advisory trivially. On the 40 evidence-bearing rows
  the pass rate is 14/40 (35%) and the per-point pass is 16/58 (27.6%). The same file shows
  `per_point_pass_rate` 0.2759.
- The HANDOFF figure 0.9033 is not in this tree (the copy here has 0.9133). The English per-point lexical
  pass in HANDOFF (25.81%) matches the evidence-bearing numbers, not the 0.91 advisory figure.
- `docs/ELA_RESEARCH.md` 7.7% is an earlier v1 contract (one JSON with evidence ids and few-shot
  examples, copy bias). It is a different contract and is superseded.
- Conclusion: the evidence-bearing advisory pass for the FP16 points contract is about 35%, not 91%.
  The Pi numbers above use evidence-bearing rows only, so they are the right comparison.

### Sample outputs

Points, `cmp_points_t2` (first three rows, verbatim after `<|eot_id|>` removal):

1. `39f0ecce180e54679369` (partial, verifier fail): `{"lang": "en", "summary": "The message claims a prize or lottery, but does not specify what the prize is.", "points": ["", ""]}`
2. `477eaf15ac735a8f4b27` (template, verifier fail): `{"lang": "en", "summary": "...", "points": "MEDIUM"}}`
3. `d02552160d8cabf80d3f` (template, verifier fail): `{"lang": "en", "summary": "...", "points": "Matched text: 'duty registration at malad east pay' is a possible indicator of a fraudulent message."}`

Slot, `cmp_slot_t2`, five rows picked by stepping through the 30 (output points, then the model's sentence
status and source):

1. `39f0ecce180e54679369` (SMS: "Congrats! You have won the 1st Prize Maruti ALTO ... Contact PAYTM Helpline"; facts: claims a prize x2). Points: `"This message claims a prize or lottery."` (template, first sentence unterminated: "...without providing any information about the") and `"This message is likely a scam because it claims a prize."` (model, verifier pass).
2. `5d2deddf5eab4d4872ee` (SMS: a Canara Bank notice saying the bank does not ask for PIN; fact: time pressure "immediately"; risk LOW). Model sentence `"is trying to rush you into making a decision."` rejected (no urgency lexicon keyword). Output: `"This message uses time pressure (turant, abhi, 24 hours)."` (template).
3. `71a5e4a84a7833e550f3` (SMS: "Hey it's [name] from twitter Could I ask an urgent favour please?"; fact: time pressure "urgent"). Model `"is trying to scare you into taking action quickly."` rejected (same lexicon reason). Output: template sentence.
4. `fd95b9eca7b8972b358c` (SMS: "Your parcel line up for delivery Track the progress here http://bit.ly/ 376QzLp"; fact: shortened link). Model: `"This message may be a scam because it contains a shortened link."` (verifier pass).
5. `ad35048242561a897177` (SMS: tournament offer with "https://bit.ly/RUp11"; fact: shortened link). Model: `"This message may be a scam because it contains a shortened link."` (verifier pass).

### Hand-check verdicts (slot outputs against their SMS and facts)

1. Row 1: the model sentence is correct (the SMS offers a prize and asks for a registration contact). It is the only correct model-written point in that row. The first point is the template.
2. Row 2: the template sentence is correct as a rule description (the text does contain "immediately"), but the SMS is a bank notice that warns against sharing PIN. The rule is a false positive on this message, and the model's "rush you" sentence was a fair paraphrase, so the lexicon rejection cost accuracy here, not safety.
3. Row 3: the model sentence "trying to scare you" overstates a benign-looking request ("urgent favour"). The template, which is what was shown, is a neutral description. Verdict: the template is the safer output; the rule fires on a non-scam.
4. Row 4: correct. The link is shortened and the sentence says so, with a hedge ("may be a scam").
5. Row 5: correct, same as row 4. The hedge is appropriate.

Verdict: of the five hand-checked outputs, the model-written sentences were correct in rows 1, 4 and 5 and
fair but rejected in rows 2 and 3. No hand-checked output made a wrong claim. The verifier is keyword-based,
so it passes wrong claims as long as a keyword is present; these five were not wrong. The sample is too
small to show the wrong-claim rate. The stronger concern is recall: the lexicon rejects accurate paraphrases.

### Verdict

- Slot mode does what it was designed for. The final advisory passes the verifier 30/30 by construction.
  The raw model output passes the verifier in 15/30 advisories, against 1/30 for the points contract on the
  same rows. Advisory-level template fallback is 50% (slot) against 97% (points). Slot is about 3.7 times
  faster at p50 with about half the prompt tokens per advisory.
- It does not make the on-device model good at this. Only 63% of the model's own point sentences pass the
  verifier. The rest are template text. The model's sentences that pass are correct in these five cases.
- The main limit is the lexicon. Accurate paraphrases ("trying to rush you") are rejected, so the lexicon,
  not the model, sets much of the pass rate.

### Untested

- Prefix reuse (`prefill()` once, then per item): not built.
- Hinglish slot: English only.
- The 30 rows are evidence-bearing from one seed. The 5 hand-checked outputs are not a sample for the
  wrong-claim rate. Speed numbers are one run each, not repeated.
- The slot p95 is one outlier and was not isolated.
- Points-mode results use the new worker code; the points path is unchanged in logic, but the
  `point_pass_rate` and `prompt_tokens_per_advisory_mean` fields are new.

### Files changed (laptop repo, uncommitted)

- `deploy/pi/bench/rbench/slot.py` (new): prompt, clean, assembly, per-item generation.
- `deploy/pi/bench/rbench/worker.py`: `--ela-mode` flag; `run_llm` branches on it; new llm-block fields.
- `deploy/pi/bench/rbench/guard.py`: `RunSpec.ela_mode`, passed to the worker.
- `deploy/pi/bench/rbench/cli.py`: `rbench run --ela-mode`.
- `deploy/pi/bench/rbench/runners/pte_llm.py`: `generate(..., stop_when=)` calls `runner.stop()`.
- `tests/test_rbench_slot.py` (new, 13 tests), `tests/test_rbench_pte_runner.py` (2 stop tests).
- `docs/LLAMA_PI_RESULTS.md` (this section).

Tests: `tests/test_rbench_*.py` and `tests/test_pi_*.py`: 134 passed. Ruff check and format check pass on
`deploy/pi` and those tests. The Pi has the same code (`sync_bench.sh --code-only`); the model directory
was not touched. No process of mine is left on the Pi (`pgrep rbench` returns nothing).

## slot vs slot_kw (2026-11-06, Pi 5, same 30 evidence-bearing rows, threads 2, seed 0)

Question: does putting the evidence keyword into the assistant prefix raise the model's own
verifier pass, without loosening the verifier? Both modes ran back to back on the same rows
(`_ela_rows`, seed 0), each under the thermal guard, 0 pauses, 0 kills, `throttled=0x0` before and after.

- **slot** (`--ela-mode slot`): prefix `This message`, as before. Run `cmp_slot_t2b`
  (`~/rakshak-bench/results/20261106T142746Z-f0523e`). Rerun of the earlier `cmp_slot_t2` (`20261106T140733Z-f9c237`),
  same code path; the numbers differ slightly (p50 2.77 s here vs 2.65 s before).
- **slot_kw** (`--ela-mode slot_kw`, `deploy/pi/bench/rbench/slot.py`): the prefix is
  `This message <clause> because it`, where `<clause>` is the type's `TYPE_PHRASE` with its parenthetical
  examples dropped. The model completes the clause. The system text is shorter than slot's so the prompt
  stays near 95 tokens. Stop rules (sentence end, newline, quote, brace), the 20-token budget, the verifier and
  the template fallback are unchanged. Run `cmp_slotkw_t2` (`~/rakshak-bench/results/20261106T142959Z-0f02f9`).

`verifier.py`, the lexicon, `prompts.py` and `evidence.py` are not changed.

### Comparison

Latency is per advisory (all calls). "Raw" means the verifier on the model's own sentences, before
fallback. "Final" means the advisory the user sees.

| Metric | slot | slot_kw |
| --- | --- | --- |
| Latency p50 / p95 (s) | 2.77 / 8.64 | 3.07 / 9.43 |
| Latency mean (s) | 3.75 | 3.84 |
| TTFT, mean per call (ms) | 1560 | 1597 |
| Prompt tokens per call (mean) | 94.5 | 85.1 |
| Prompt tokens per advisory (mean) | 144.9 | 130.4 |
| New tokens per call (mean) | 12.0 | 12.0 |
| New tokens per advisory (mean) | 18.5 | 18.4 |
| Calls per advisory (mean) | 1.53 | 1.53 |
| Stop predicate fired (per call) | 0.913 | 0.739 |
| Call status ok / unterminated | 42 / 4 | 26 / 20 |
| Raw verifier pass, advisory level | 0.50 (15/30) | 0.40 (12/30) |
| Raw verifier pass, point level | 0.630 (29/46) | 0.522 (24/46) |
| Template fallback, advisory level (mode != llm) | 0.50 (15/30) | 0.60 (18/30) |
| Template fallback, point level | 0.370 | 0.478 |
| Final output passes verifier | 1.00 (30/30) | 1.00 (30/30) |
| Mode counts (llm / partial / template) | 15 / 15 / 0 | 12 / 18 / 0 |
| Thermal start / max / end (C) | 46.9 / 70.5 / 65.5 | 60.0 / 69.4 / 68.3 |
| CPU frequency cap (MHz) | 1800 to 2400 (mean 2098) | 1800 for the whole run |
| Power avg (W, guard estimate) | 4.13 | 3.54 |

Frequency caps differ: slot ran between 1800 and 2400 MHz, slot_kw was held at 1800 MHz for the whole run
(it started at 60 C, slot at 47 C). Latency is therefore not on equal clocks. slot_kw's p50 and p95 are
about 9 to 11% higher, which is in the range the cap could explain; it was not isolated.

### Per-type raw point pass (verifier on the model's sentence, 46 points)

| Evidence type | slot | slot_kw |
| --- | --- | --- |
| prize_lottery | 15/17 | 11/17 |
| parcel_customs | 2/2 | 2/2 |
| account_suspension | 3/4 | 3/4 |
| urgency | 0/13 | 2/13 |
| investment_returns | 1/2 | 2/2 |
| shortened_url | 5/5 | 4/5 |
| suspicious_url | 1/1 | 0/1 |
| electricity_disconnect | 1/1 | 0/1 |
| digital_arrest | 1/1 | 0/1 |
| credential_request, upi_collect_or_pay, kyc_block_urgency, apk_install, remote_access_app, job_task_scam, brand_domain_mismatch | not in sample (0 rows) | not in sample |

### Prefixes used (slot_kw), one per evidence type

Each prefix is `This message <clause> because it`. Every one was checked with `has_lexicon_keyword` in
`tests/test_rbench_slot_kw.py`, and each prefix plus a neutral completion passes `check_points`.
The keyword is the lexicon word that the verifier matches.

| Type | Prefix | Keyword the verifier accepts |
| --- | --- | --- |
| credential_request | This message asks for an OTP, PIN or password because it | otp |
| upi_collect_or_pay | This message uses a UPI collect or payment request because it | upi |
| kyc_block_urgency | This message uses KYC urgency or a block threat because it | kyc |
| account_suspension | This message claims an account or card is suspended because it | account |
| prize_lottery | This message claims a prize or lottery because it | prize |
| apk_install | This message asks for an app or APK installation because it | app |
| remote_access_app | This message asks for a remote-access app because it | remote |
| electricity_disconnect | This message threatens electricity disconnection because it | electricity |
| parcel_customs | This message asks for a parcel or customs fee because it | parcel |
| job_task_scam | This message offers a task or job with easy earnings because it | job |
| investment_returns | This message promises guaranteed investment returns because it | investment |
| digital_arrest | This message claims to be a police or government agency because it | police |
| brand_domain_mismatch | This message links to a domain that does not match the brand named because it | domain |
| shortened_url | This message contains a shortened link because it | shortened |
| suspicious_url | This message contains a suspicious link because it | suspicious |
| urgency | This message uses time pressure because it | time pressure |

Note: the prefix states the rule's own fact, but "because it" is a causal connective, and it invites
the model to give a reason the rule did not establish. That is what happened (see the hand-check).

### Why slot_kw did not help

- **More run-ons.** 20 of 46 completions ended unterminated (slot: 4). 8 had a quote character, which the
  stop rule treats as a cut. The other 12 ran to the 20-token budget with no sentence end. The typical run-on was "...which is a common tactic used by
  scammers to try to", which has no sentence end inside 20 tokens.
- **Keyword present, point still failing.** On prize_lottery the keyword was in the prefix, but 6 of 17 points
  still failed, all six as unterminated run-ons or empty output. The keyword is no longer the bottleneck; sentence termination is.
- **Urgency barely moved** (0/13 to 2/13). The two passes were the two LOW-risk messages below.
- **The sentences that pass are not all true.** See the hand-check.

Difference in raw point pass (0.63 vs 0.52 on 46 points) is about 5 points, which is within noise at this
sample size. The verdict below does not rest on it: the completions show the failure, not the rate.

### Hand-check: 8 slot_kw sentences (against the SMS and its fact)

The SMS text is from the campaign test split and was reconstructed locally with the same sampler (row ids match
the run). Verdict key: correct / overclaim (a detail the SMS does not support) / wrong (contradicts the SMS).

| # | Row | Risk | SMS (short) | slot_kw sentence (after the prefix) | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | 39f0ec | MEDIUM | "Congrats! You have won the 1st Prize Maruti ALTO ... For Registration please Contact PAYTM Helpline" | "...because it is trying to trick you into sending money or personal information to claim it." | overclaim (mild): the prize and the contact request are in the SMS; "trying to trick you" and "sending money" are inferred |
| 2 | d02552 | MEDIUM | "gudi padwa special offer bhk ready flats ... registration at malad east pay homeloan" (a property ad) | "This message asks for a parcel or customs fee because it is likely a scam." | **wrong**: the SMS is a flat offer, not a parcel or customs fee. The rule false positive is repeated and then labelled a scam |
| 3 | 4f6af7 | LOW | "URGENT - Dear Commonwealth Bank Customer, Please review and confirm your details to avoid suspension" | "...uses time pressure because it implies that the recipient must respond immediately, which is not a typical way to communicate." | overclaim: "URGENT" and "avoid suspension" are in the SMS; "not a typical way to communicate" is not, and is a claim about how banks write |
| 4 | f2d12a | LOW | "GOOD NEWS SAMSUNG ... ANNUAL REWARD 2022 TO PROCESS URGENT SEND YOUR NAME,ADDRESS,PHONE NO,AGE" | same sentence as #3 | overclaim: same unsupported "not a typical way to communicate". The message does ask for personal details, so LOW risk for it is itself a miss |
| 5 | ee2cdc | MEDIUM | "We have locked your account due to unusual activity. Kindly verify your identity at <URL>" | "claims an account or card is suspended because it is locked." | correct (restates the SMS) |
| 6 | 0ac611 | MEDIUM | "Your card is on hold. Send 1 and unblock." | "claims an account or card is suspended because it is on hold." | correct (restates the SMS) |
| 7 | e006f6 | MEDIUM | "...last week our clients made profit rs lac on our tips to subscribe our stock tips ..." | "promises guaranteed investment returns because it is likely a scam." | overclaim: the SMS advertises tips and past profits; "guaranteed" is not in it (it comes from the rule's phrase) |
| 8 | deb054 | MEDIUM | "intraday and delivery trading profit nse mcx ... automatic buysell signal ... free demo call" | "promises guaranteed investment returns because it is likely a scam." | **wrong** on the detail: no guarantee in the SMS; it advertises a trading-signal demo. The rule's "guaranteed" is repeated as fact |

Tally: correct 2, overclaim 4, wrong 2.

**Benign and LOW-risk messages (the focus).** Nine of the 30 rows are LOW risk. Most are benign:
a Twitter "urgent favour", an out-of-office reply, a telco data offer, an F1 quiz, a telco "offer of the week" (Hinglish),
and a Canara Bank notice that warns against sharing a PIN. For these, all 7 slot_kw completions were
unterminated or rejected, so the advisory shows the template. But the raw completions are the warning:
- The wording "...which is a common tactic used by scammers to try to" was written for the Twitter favour,
  the F1 quiz, the telco offer and the Hinglish offer (the out-of-office reply got a close variant). Each is a
  false claim about a benign message. Only the 20-token budget stopped it from reaching the user. The verifier
  would have passed the same text if it had finished: checked on the Twitter favour row, the prefix plus that
  completion passes `check_points`, because the keyword is in the prefix and the text has no banned verb or word.
- The two LOW-risk messages that did pass (#3, #4) carry the unsupported "not a typical way to communicate".
- The benign rows' template text ("This message uses time pressure (turant, abhi, 24 hours).") is a rule
  false positive on messages like the bank notice. That false positive is the same in slot mode, and is
  the rule's problem, not the prefix's.

So: on benign messages the forced prefix did not produce a shown false claim in this sample, but only
because the budget cut the completion. The keyword prefix makes the model rationalise the rule. That is the
risk the prefix introduces, and it is larger than the pass-rate gain it was meant to give.

### Verdict

- **slot_kw does not beat slot here.** Raw point pass 0.52 vs 0.63, raw advisory 0.40 vs 0.50, and
  template fallback rises from 50% to 60% of advisories. Both modes give a final output that passes the
  verifier 30/30, because the fallback is unchanged.
- It shortens the prompt (85 vs 95 tokens per call) and has the same tokens per advisory and TTFT, but it
  makes the completions run longer. The 20-token budget now cuts 20 of 46 sentences.
- **Keep `slot` as the default.** `slot_kw` stays in the tree as an opt-in mode (`--ela-mode slot_kw`).
- The hand-check says the same thing from the other side: the prefix produces overclaims ("guaranteed",
  "not a typical way to communicate", "is a common tactic used by scammers") and two wrong sentences.
  The verifier does not catch any of them, because it checks keywords, not truth.

### Thermal and process notes

- Pi start temperatures: slot 46.9 C, slot_kw 60.0 C (checked before each run; the start rule was below 65 C).
  Both runs passed `throttled=0x0` before and after. Peak 70.5 C and 69.4 C. No pauses or kills (`status: ok`).
- Frequency cap differs between the runs (see the comparison). Treat the latency gap as indicative only.
- Nothing of mine is left on the Pi. `pgrep` shows only the system services (`pi_dashboard.py`, `rakshak_thermald.py`).

### Untested

- Connective variants (", which" as in the first example, or no connective) and a larger budget (28 tokens) to
  reduce run-ons. These might improve slot_kw; they were not run.
- Equal-clock latency: slot and slot_kw were not run at the same frequency cap.
- Types with no sample rows (7 of 16 types), Hinglish, and sample sizes above 30 rows.
- Pi-side unit tests were not run. The slot_kw logic is covered by the laptop tests with a mocked runner only.
- Statistical comparison: 46 points is too few to separate 0.63 from 0.52.

### Files changed (laptop repo, uncommitted)

- `deploy/pi/bench/rbench/slot.py`: `keyword_clause`, `kw_prefix`, `slot_kw_prompt`, `SLOT_KW_SYSTEM`;
  `clean_sentence(..., prefix)`; `generate_slot_advisory(..., keyword_prefix=)` with `prefixes`, `point_checks`, `mode`.
- `deploy/pi/bench/rbench/worker.py`: `SLOT_MODES`, `slot_kw` branch, `point_checks` and `prefixes` in the per-advisory records.
- `deploy/pi/bench/rbench/cli.py`: `--ela-mode` accepts `slot_kw`.
- `tests/test_rbench_slot_kw.py` (new): one parametrised test per evidence type (prefix keyword via `has_lexicon_keyword`,
  and prefix plus neutral completion via `check_points`), plus assembly, fallback, stop rules, prompt and default-mode checks.

Tests: `tests/test_rbench_*.py` and `tests/test_pi_*.py` pass (176). `ruff check` passes on `deploy/pi` and those tests.
`ruff format --check` fails only on `deploy/pi/pi_dashboard.py`, which this change does not touch.
The Pi has the same code (`sync_bench.sh --code-only`, 143 files); the model directory was not touched.

## Guards (G1, G2), slot arm, 2026-11-06

Two deterministic guards, opt-in (`rbench run --guards`; `verifier.check_points(..., risk=)`). G1: a LOW-risk
advisory gets no explainer call and is the template. G2: at LOW any attribution of scam intent is rejected; at
MEDIUM only a hedged form ("may|could|might be a scam|fraud|phishing|fake", and since 2026-11-06 "likely a
scam|fraud|phishing") is allowed; HIGH is not checked. Full rule, term list and the hedge regex:
`docs/EDGE_PIPELINE.md` section 1.

### Fixes A and B on the guards, 2026-11-06 (re-measured through the service, same 40 rows)

Fix A changes the explainer prompt: it describes what the message says and does not judge it (variant
`describe_kw`, `rbench.slot.PROMPT_VARIANTS`; the default slot prompt is unchanged). In a 10-row trial (11 points,
held out from the 40 eval rows) it gave 11 of 11 raw points passing, against 6 of 11 for the old prompt. The
chosen prompt echoes "restates the fact" in 2 of 11 points and is 104 tokens (target about 95).

Result on the 40 service rows: raw point pass 10 of 45 became 38 of 45 (22% to 84%); G2-rejected sentences 30 became
1; advisory mode llm / partial / template for the 30 evidence rows 7 / 2 / 21 became 22 / 2 / 6. Final contract
40 of 40 before and after. The G1 skips are 1 (evidence) and 9 (all 40).

Fix B is at the pipeline, not in `evidence._risk`. A student-flagged message with no rule evidence is now UNCERTAIN
with a fixed "detector flagged, no specific pattern" summary (no explainer call, `path: detector_only`). Before the
fix, 2 of the 40 rows (benign label, detector 0.59 and 0.44) were shown "no known scam pattern". Full before/after
tables, hand-check of 8 outputs and the residual defects: `docs/EDGE_PIPELINE.md` section 3.

Same 30 evidence rows, `--ela-mode slot --limit 30 --threads 2 --seed 0 --guards`, run `20261106T144210Z-ab584c`:

| Metric | With guards | Same sentences, G2 off |
| --- | --- | --- |
| G1 skipped (LOW, no call) | 9 of 30 | |
| Raw advisory pass, 21 called | 7 (33%) | 15 (71%) |
| Raw point pass, 37 points | 10 (27%) | 29 (78%) |
| Template fallback, advisory (30) | 23 (77%) | 15 (50%) |
| Final output passes verifier | 30 of 30 | 30 of 30 |
| Latency per advisory, 30 (p50 / p95, s) | 2.40 / 8.36 | |

G2 rejected 23 points, 10 distinct sentences, for example "This message is likely a scam because it claims a
prize." and "This message is trying to rush you into making a decision." (the full list is in the EDGE_PIPELINE
document). G1 changed no final output: the 9 LOW rows fell back to the template in the unguarded run too.
