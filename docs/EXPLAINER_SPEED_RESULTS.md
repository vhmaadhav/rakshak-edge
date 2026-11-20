# Explainer speed and termination: results (x86 VM, real .pte)

Plan and gate rules: `docs/EXPLAINER_SPEED_PLAN.md` (written before measuring). Raw numbers:
`docs/results/explainer_speed/metrics.json`. Per-advisory records (540 KB, including every sentence)
are on the VM at `/data/workspace/rakshak/explainer/run1/advisories.jsonl`.

Run: 2026-11-20, InstaBox VM (x86_64, 4 vCPU shared with other jobs, 3 threads), SpinQuant INT4 `.pte`
(sha256 `8715cdba...cda3b`, verified), ExecuTorch 1.5.1, greedy decoding, 30 evidence rows (seed 0, the
Pi's evidence group), 3 repeats in interleaved blocks, `--probe`.

**Verdict: no candidate passes all five gates. The shipped default is unchanged.** The best arm,
`c3_trim`, passes E1, E2 and E3 and fails E4 and E5.

## 1. Results

Counts are over the 21 LLM-called advisories (9 of 30 are skipped by G1 in every arm). Token and forward
counts are steady state per advisory, over all 30 advisories. Wall time is the sum of call times per
advisory, over the 63 called advisory-repeats (21 x 3), with a 95% bootstrap CI on the ratio to baseline.

| Arm | Unterminated called points (of 25) | Raw verifier pass (of 25 called points) | Final contract | Tokens processed / advisory (ratio) | Forward calls / advisory (ratio) | x86 LLM p50 ms (ratio, 95% CI) | Pi projection, derived (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (shipped) | 3 | 16 | 30 of 30 | 41.03 | 5.90 | 615.4 (p95 1361) | 989 (shipped) |
| c1_short (length rule, cap 20) | 4 | 20 | 30 of 30 | 40.03 (0.976) | 4.90 (0.831) | 573.7 (0.932, 0.809 to 1.079) | 922 |
| c1_tight (length rule, cap 10) | 4 | 20 | 30 of 30 | 40.03 (0.976) | 4.90 (0.831) | 546.1 (0.887, 0.790 to 1.052) | 878 |
| c2_batch (one call, cap 10 per point) | 5 | 18 | 30 of 30 | 40.57 (0.989) | 6.43 (1.090) | 635.1 (1.032, 0.889 to 1.184) | 1021 |
| c3_trim (cap 10, no user instruction) | **0** | 19 | 30 of 30 | **34.73 (0.847)** | 5.43 (0.921) | 525.7 (0.854, 0.741 to 1.055) | 845 |

The tight cap is 10 tokens: the p95 of the 22 terminated baseline calls (rep 0) is 10.0, and the longest
terminated call is 10. The Pi projection is a derived estimate only (989 ms times the x86 ratio); it is not
a measurement.

## 2. Gate verdicts (pre-registered rules)

| Arm | E1 unterminated = 0 | E2 raw pass >= baseline (16) | E3 contract 100% | E4 tokens or calls <= 0.70 | E5 p50 ratio < 1 and CI high < 1 | Passed all |
| --- | --- | --- | --- | --- | --- | --- |
| c1_short | fail (4) | pass (20) | pass | fail (0.976 / 0.831) | fail (0.932, CI high 1.079) | no |
| c1_tight | fail (4) | pass (20) | pass | fail (0.976 / 0.831) | fail (0.887, CI high 1.052) | no |
| c2_batch | fail (5) | pass (18) | pass | fail (0.989 / 1.090) | fail (1.032, CI high 1.184) | no |
| c3_trim | pass (0) | pass (19) | pass | fail (0.847 / 0.921) | fail (0.854, CI high 1.055) | no |

## 3. Diagnosis of the unterminated outputs (measured)

The 3 baseline unterminated calls all stopped on `stop_when` after a quote character, with no sentence end
before it. Examples (rep 0): `" matched text is "` (4 tokens), `" matched the text "`, `" matches the text "`. None
reached the 20-token cap, and no call stopped on EOS. So for this shipped describe_kw path the cause is the
stop rule itself (`slot_stop` fires on a quote, newline or brace), not the token budget, and not the
`<|eot_id|>` set (EOS ids in the `.pte` are 128009 and 128001, as the runner expects).

The earlier attribution in `docs/LLAMA_PI_RESULTS.md` (run-ons that hit the 20-token cap) describes the
slot_kw arm. It is not what happens on this path. The Pi's 4-of-31 figure in `docs/EDGE_PIPELINE.md` was
not reproduced here: this x86 baseline gives 3 of 25 called points.

The batched arm's unterminated points are a different failure: 2 partial batch calls hit the per-point cap
(`max_new_tokens`), and others stopped on the terminator count with a missing line.

A quote-tolerant stop (stop on the first terminator and treat quotes as text) is the obvious candidate for
this diagnosis. It was not pre-registered and was not run, so it has no result here.

## 4. Prompt lookup decoding (candidate D): not possible with this `.pte`

The probe loads the `.pte` and runs `forward` with a 1-token and a 4-token input. Both return shape
`[1, 128256]`, i.e. only the last-position logits. `per_position_logits` is false. Prompt lookup verification
needs logits at every drafted position, so it cannot run on this export. The fix is a re-export with the
exporter's option that returns full logits; the exact flag name was not checked against the exporter source,
which is not in the installed wheel, so it is marked unverified.

## 5. Other findings

- The verifier's lexicon match is case-sensitive: `OTP` and `PIN` do not satisfy the `credential_request`
  keyword, but lowercase `password` does. The explainer's uppercase identifiers fail the check. Found while
  writing the batch test; not a speed result, but it is a caveat for any prompt change.
- The raw pass here (16 of 25 for the baseline) is not the Pi's 27 of 33. Different denominators (the Pi
  run used the detector's decisions and the service's G1), and x86 greedy numerics differ. Compare
  candidates within this table only.
- Steady-state tokens per advisory are dominated by the prefill of about 36 newly processed tokens per call
  (the system turn is reused). Decode is about 5 tokens per advisory. Termination and cap changes move decode
  only, so E4 is hard to reach with them. `c3_trim` removes prefill tokens (the trailing instruction), which
  is why it reaches the largest token reduction.

## 6. Deviations from the plan (stated)

1. **E5 on called advisories only.** The first run crashed (ZeroDivisionError in the bootstrap) because
   9 of 30 advisories are G1-skipped with zero LLM time in every arm. The crash wrote no output. The fix
   restricts E5 and its bootstrap to LLM-called advisories and guards the zero denominator. The all-advisory
   p50 is kept in `metrics.json` (`advisory_wall_ms_p50_all_including_skipped`). This was changed after a
   crash, not after seeing any candidate's result.
2. **The detector is not run.** The rows are the Pi's evidence group, and the G1 decision uses the
   label-free proxy scam state (`collapse_by_type` plus `skip_llm_for_risk`). The 10 benign rows are not run;
   G1 skips them in every arm.
3. **Tight cap from the gate rows.** Rule T takes one statistic (p95) from the baseline on the same 30 rows,
   as the plan allows.
4. **Interleaving.** Arms run in blocks, not per row, so the system-turn prefix stays warm within a block. The
   arm order rotates across repeats. The VM is shared: wall times are noisy, and the CI is the only basis
   for E5.

## 7. Code

- `deploy/pi/bench/rbench/runners/pte_direct.py`: forward-call, decode-forward, prefix-forward and stop-reason
  accounting per call.
- `deploy/pi/bench/rbench/slot.py`: prompt variants `describe_kw_short` and `describe_kw_trim` (optional
  `suffix`); the shared `_finish_points` tail. The baseline call is unchanged.
- `deploy/pi/bench/rbench/explainer_candidates.py`: the candidate table, `run_explainer`, the batched path,
  `AccountingGenerator` (Generator protocol) and `summarise_accounting`.
- `deploy/pi/service/edge_pipeline.py` and `rakshak_edge_service.py`: `explainer_candidate` and
  `explainer_cap` flags. Default `baseline`; Hinglish keeps the shipped path. Setting `RAKSHAK_EXPLAINER_CANDIDATE`
  changes the English explainer, so do not set it in production.
- `deploy/pi/bench/explainer_speed.py`: the harness (`run --probe`) and the gate computation.
- `tests/test_explainer_speed.py`: 14 tests with a fake generator. The explainer, edge, rbench, llama and
  llm suites pass on the VM (489 passed, 6 skipped for missing artifacts).

## 8. Pi re-measurement

Requires the Pi's `~/rakshak-bench` with `data/processed` (the registry paths) and the `.pte` and tokenizer
in place:

```
python deploy/pi/bench/explainer_speed.py run \
  --root ~/rakshak-bench \
  --model ~/rakshak-bench/models/llama32_1b_spinquant/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte \
  --tokenizer ~/rakshak-bench/models/llama32_1b_spinquant/tokenizer.model \
  --out ~/rakshak-bench/results/explainer_speed --reps 3 --threads 3 --limit 30 --probe
```

Run it on a cool, clean-power device, and report the wall-time CI from `metrics.json` (E5). The counts
(tokens, forward calls, verifier pass, unterminated) should match the x86 run except where greedy numerics
differ.

## 9. Recommendation

Keep the shipped default. Next step, if the team wants the termination gap closed: test a quote-tolerant
stop as a pre-registered arm. For speed, E4 needs a larger prefill cut than prompt trimming gives; a
shorter fact text in the user turn is the lever, and it is a verifier-facing change that needs its own gate.
