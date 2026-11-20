# Explainer speed and termination: pre-registration

Written before any candidate was measured on the real `.pte`. The baseline, candidates, rows, gates
and rules below are fixed here; the results are in `docs/EXPLAINER_SPEED_RESULTS.md`. A failed gate
is reported as failed.

Scope: the English explainer of the edge SMS pipeline (Llama 3.2 1B Instruct, SpinQuant INT4 `.pte`,
ExecuTorch 1.5.1). It writes one short sentence per evidence item. The verifier and the template
fallback are not changed.

## 1. Baseline (the shipped pipeline)

The baseline is the call `EdgeAnalyzer.analyse` makes for an English advisory today:

- Package: `collapse_by_type(package)` (one evidence item per type, first occurrence).
- Call: `rbench.slot.generate_slot_advisory(llm, package, lang="en", max_new_tokens=SLOT_MAX_NEW_TOKENS
  (20), guards=True, variant="describe_kw" (EDGE_PROMPT_VARIANT), cleanup=True)`.
- Prompt: the `describe_kw` system text (DESCRIBE_SYSTEM) and user turn
  `FACT: <fact>\n\nWrite the sentence for this fact.`, assistant prefix `This message <type clause> and`.
- Stop: `slot_stop` (first sentence end, or a newline, quote or brace), the `<|eot_id|>` stop of the
  direct runner (EOS ids from the `.pte`), and the 20-token cap.
- One call per collapsed evidence type. G1 skips the LLM for LOW risk.
- Runner: `PteDirectRunner` with prefix reuse (the system turn is kept in the KV cache).
- Sampling: greedy (argmax).

The harness reproduces this call exactly (`run_explainer(candidate="baseline")`, tested as equal to
the direct `generate_slot_advisory` call). It does not run the detector, so the advisory is not
rebuilt end to end; the explainer call is the only thing it varies.

## 2. Rows

- The 30 evidence rows of the edge eval: `rbench.worker._ela_rows(root, sms_ela_campaign_test, 30,
  seed 0)`. This is the same seeded sampler as `eval_service.select`, so the rows are the Pi's
  evidence group.
- The 10 benign LOW rows are not run. The G1 guard skips the LLM for them in every arm, so they cannot
  change any gate. This is a stated deviation from "40 rows".
- Gate rows and calibration rows are the same 30 rows. The only statistic taken from them for
  tuning is the tight cap (section 3, rule T).

## 3. Candidates

All candidates use the same runner, cleanup, verifier and template fallback as the baseline.

| Arm | What changes | Cap (tokens) |
| --- | --- | --- |
| `baseline` | none (shipped) | 20 |
| `c1_short` (C1, prompt only) | the length rule "Keep it under 15 words." is added to the system turn (inside the reused prefix); the rest is the baseline | 20 |
| `c1_tight` (C1, cap) | C1 with the tight cap | tight (T) |
| `c2_batch` (C2) | one call per advisory: a numbered list of facts in the user turn, one numbered sentence per fact, assistant prefix `1. This message`, the same short-sentence rule; parsed line by line; a point with no line or no sentence end falls back to the template | tight (T) per point |
| `c3_trim` (C3) | C1 with the tight cap and the trailing line `Write the sentence for this fact.` removed from the user turn | tight (T) |

Rule T (tight cap): cap = ceil of the linear-interpolated p95 of `new_tokens` over the baseline calls
whose sentence status is `ok` (terminated). It is computed once, from the baseline arm's first repeat,
before any other arm runs. Its value is recorded in `metrics.json` (`tight_cap`).

Hinglish, the template-only Hinglish path, and the detector-only path are not changed by any candidate.

Prompt lookup decoding (candidate D) is not a runnable arm. It needs a `.pte` whose forward returns the
logits for every input position. The probe (`explainer_speed.py run --probe`) records the output shape
for a 1-token and a 4-token input. If the output has only the last position, D is reported as not
possible with this `.pte`, and the re-export flag needed is named in the results document.

## 4. Metrics and definitions

Counted per advisory, on the same rows, greedy decoding, repeat 0 for counts (every repeat is also run
and must give identical outputs, which is checked).

- Unterminated point: a point whose sentence status (before cleanup) is not `ok`. For per-item arms the
  status is that of the call (`unterminated`, `empty`). For `c2_batch` it is the status of the point's
  line (`unterminated`, `missing`).
- Raw verifier pass: the number of called points that pass `check_points` on the model's own sentences
  (before the template fallback). The baseline count is the reference.
- Final contract: the number of advisories whose final output passes `check_points` (the template
  fallback is applied first). It must equal the number of advisories.
- Tokens processed per advisory, steady state: sum over calls of `(prompt_tokens - prefix_tokens)` plus
  the single-token decode forwards. This assumes the system-turn prefix is in the KV cache, which is what
  the service sees after the first advisory. Counted by the runner, not timed.
- Forward calls per advisory, steady state: every `forward` call, minus the prefix forward when it ran
  in that call.
- x86 wall time: the sum of the call wall times per advisory (`llm_wall_ms`), p50 and p95, over all
  repeats. Arms run in blocks (all rows of one arm, then the next) so that the prefix cache stays warm
  within a block; the arm order rotates across repeats; `taskset` and `nice` are noted in the results.

## 5. Gates (all must hold against the baseline on the same rows)

- **E1** unterminated points = 0 (over the rows, repeat 0).
- **E2** raw verifier pass count >= the baseline's raw pass count.
- **E3** final output contract passes for 100% of advisories (template fallback allowed).
- **E4** tokens processed per advisory (steady state) <= 0.70 x baseline, OR forward calls per advisory
  (steady state) <= 0.70 x baseline.
- **E5** x86 p50 of `llm_wall_ms` per advisory: the ratio candidate/baseline has a point estimate below 1
  AND the upper end of its 95% bootstrap CI (2,000 resamples of rows, independent for the two arms) is
  below 1. Wall time is not a gate on the Pi projection.

A candidate passes if E1 to E5 all hold. The shipped default changes only if a candidate passes all
five. A candidate that passes only some gates is reported with the failed gates named; it is not made the
default.

The Pi projection is a derived estimate (`PI_BASELINE_TOTAL_P50_MS` = 989 ms, the shipped direct-runner
total p50 on the Pi, times the x86 ratio). It is labelled as derived in every table. It is not a
measurement, and a Pi re-measurement is required before any claim about the Pi.

## 6. Diagnosis of the 12.9% unterminated outputs

Measured, not assumed, on the baseline arm: the count of calls by `stop_reason` (`eos`, `stop_when`,
`max_new_tokens`). An unterminated call that stops on `max_new_tokens` is a cap cut; one that stops on
`eos` is the model ending a turn without a sentence end. The diagnosis is stated from these counts. The
earlier notes (docs/LLAMA_PI_RESULTS.md, "slot vs slot_kw") attribute the cases to run-ons at the cap.

## 7. Limits stated in advance

- x86 is not the Pi. Only the hardware-independent counts (tokens, forward calls, verifier counts) are
  primary. The VM is a shared 4-vCPU machine: its wall times are noisy and are reported with the CI.
- Greedy decoding on x86 and on the Pi may give slightly different sentences (the direct runner's
  prefix-reuse note reports the same effect). The gate counts are for x86 only.
- A single evidence set of 30 rows gives small counts; a gate that passes on 30 rows is not a claim
  about the whole campaign.
- The tight cap uses the gate rows (rule T). This is the one tuning statistic, stated here.

## 8. Pi re-measurement (after a pass)

The Pi command is `deploy/pi/bench/explainer_speed.py run` with the Pi's `~/rakshak-bench` as `--root`,
the Pi's `.pte` and `tokenizer.model` paths, `--threads 3`, `--reps 3`. The results document will carry
the exact command.
