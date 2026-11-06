# Edge pipeline on the Raspberry Pi 5 (resident service, guards G1 and G2)

Date: 2026-11-06. Device: Pi 5, 4 GB, no cooler, Python 3.13.5 venv at `~/rakshak-bench/.venv`,
executorch 1.5.1, onnxruntime 1.31.1. Models: SMS student `sms_student` (ONNX, threshold 0.35127517580986023
from the registry) and Llama 3.2 1B SpinQuant INT4 `.pte` (both `ready`, hashed at service start).
Threads 3 (default since the soak in section 4; was 2). The service is not a systemd unit; start and stop are user-level scripts.

## Status

- Part 1 (guards G1 and G2, opt-in): implemented, unit-tested, re-run on the Pi with the slot arm.
- Part 2 (resident service, offline, end-to-end): implemented, run on the Pi: 40 requests through the
  service (30 evidence-bearing rows from the rbench sampler, seed 0, plus 10 benign LOW rows), all 200 OK,
  no cooling refusals, max SoC 69.7 C, no throttle flags.
- Fixes A and B (2026-11-06, section 3): the explainer now describes instead of judging (raw point pass
  10 of 45 to 38 of 45), and a scam-flagged message with no rule evidence is no longer shown as "no known
  scam pattern". Re-measured on the Pi on the same 40 rows; section 3 has the before/after.
- Not in the box: a 503 cooling refusal on real hardware (only mocked); a real-time demo video; the
  Android app; any model change. Nothing is committed.

## 1. Guards (G1 and G2)

Implemented in `src/rakshak/verifier.py`. Both are opt-in: `risk=None` (the default) keeps the
existing contract, so the Kaggle jobs and the existing callers are unchanged.

**G1, no explainer text for LOW risk.** `skip_llm_for_risk(package)` is True when `package.risk` is
LOW. The advisory is then the template only (`template_only_advisory`): the summary and one template
point per evidence item. The explainer is not called, and the advisory records `llm_skipped_low_risk`.
Enabled by `guards=True` in `rbench.slot.generate_slot_advisory` (the worker's `--guards`, applies to
every ELA mode) and always in the service.

**G2, attribution guard.** `attribution_problems(text, risk)` runs on each LLM point and LLM summary:
- **HIGH:** not checked.
- **LOW:** any match is rejected. A negation is not an exemption ("not a scam" is rejected).
- **MEDIUM and UNCERTAIN:** a match is allowed only inside a hedge. The hedges are
  `(may|could|might) be (a|an)? (scam|fraud|phishing|fake)` and
  `(scam|fraud|dhoka|dhokha|thagi) ho (sakta|sakti|sakte) hai`. Any attribution term outside a hedge
  is rejected, so "may be a scam, and scammers want your PIN" fails.
- **Matching:** case-insensitive, on word boundaries, multi-word phrases allow any whitespace.
- **Terms:** the list is `_ATTRIBUTION_WORDS` and `_ATTRIBUTION_PHRASES` in `verifier.py`. It covers
  the scam/fraud/phishing family (scam, scammer, fraud, fraudster, phishing, con man, swindle, ...),
  deception and intent (trick, cheat, steal, lure, impersonate, pretend, hack, malicious, fake, bogus,
  exploit, tactic, victim), the phrases "trying to rush|scare|pressure|trick|...", "rush|scare|pressure
  ... you", "take advantage", "designed to steal|trick|...", "intends to steal|...", and Hinglish
  equivalents (dhoka, dhokha, dhokebaaz, thag, thagi, jaalsaaz, chor, chori, loot, nakli, farzi, ...).
- A rejected point becomes the template point (the existing fallback). Templates are deterministic and
  are not run through G2 by the final checks (the LOW template summary says "no known scam pattern",
  which is a negated mention by design).

Tests: `tests/test_verifier_guards.py` (the doc's benign sentences rejected at LOW and MEDIUM, hedged
forms accepted at MEDIUM, HIGH unchecked, `risk=None` unchanged, the template passes at every risk).
`tests/test_rbench_guards.py` (G1 makes zero generate calls; G2 lists rejections).

### Part 1 result: slot arm, same 30 evidence rows, guards on

`rbench run --model llama32_1b_spinquant --dataset sms_ela_campaign_test --ela-mode slot --limit 30
--threads 2 --seed 0 --guards`. Run `20261106T144210Z-ab584c`, 0 pauses, 0 kills, throttled `0x0`
before and after, SoC 47.4 to 70.5 C.

| Metric | With G2 (shipped) | Same sentences, G2 off |
| --- | --- | --- |
| G1: advisories skipped (LOW, no LLM call) | 9 of 30 | (no guard: 9 LLM calls) |
| LLM advisories (called) | 21 | 21 |
| Raw advisory pass (all points pass, before fallback) | 7 of 21 (33%) | 15 of 21 (71%) |
| Raw point pass (points written by the LLM) | 10 of 37 (27%) | 29 of 37 (78%) |
| Advisory template fallback (mode != llm, all 30) | 23 of 30 (77%) | 15 of 30 (50%) |
| G2-rejected sentences | 23 points, 10 distinct sentences | (not applied) |
| Final output passes the contract | 30 of 30 | 30 of 30 |
| Latency per advisory, all 30 (p50 / p95, s) | 2.40 / 8.36 | (pre-guard run: 2.77 / 8.64, other clock) |
| Latency per advisory, LLM-called 21 (p50 / p95, s) | 3.00 / 9.84 | |

"Same sentences, G2 off" is the earlier unguarded run `20261106T142746Z-f0523e`, re-scored with the
new verifier at `risk=None`. It reproduces the earlier 15/30 and 29/46 figures, so the comparison is
on the same text. Greedy decoding and the same prompts make the sentences identical for the 21 rows that
call the explainer.

**G2 rejected sentences (10 distinct, 23 points):**
1. "This message is likely a scam because it claims a prize."
2. "This message is likely a scam because it claims a prize or lottery."
3. "This message is likely a scam because it claims you have won a prize or lottery."
4. "This message is likely a scam because it claims you have won something."
5. "This message is likely a scam because it promises guaranteed investment returns."
6. "This message is a scam because it asks for payment to receive a parcel or customs fee."
7. "This message is a scam because it asks for payment without providing any information about the parcel or the fee."
8. "This message is a scam because it threatens to disconnect electricity, which is a serious and potentially life-threatening action."
9. "This message is trying to rush you into making a decision."
10. "This message is trying to trick you into sending money or revealing sensitive information."

**What G1 saved and what it did not change.** The 9 LOW rows each had an explainer call in the earlier
unguarded run. Their final output is identical with and without G1 (9 of 9), because every one of those
sentences failed the verifier anyway (no lexicon keyword, or an attribution). G1 removes 9 LLM calls
(about 9 x 2.5 s of compute and about 9 x 10 J) without changing what the user sees.

**Note on the benign examples from the doc.** The Twitter "urgent favour" and the Canara PIN notice
sentences are the test cases in `test_verifier_guards.py`. In this run they were not produced, because
G1 skipped those rows. They are rejected at LOW and MEDIUM by unit tests, and the explainer would have
produced attribution sentences for them (the earlier run).

## 2. Resident service

### Start, stop, and the offline proof

Files are in `deploy/pi/service/` and synced by `deploy/pi/bench/sync_bench.sh` (code-only included).

```bash
# on the Pi, from anywhere
~/rakshak-bench/deploy/pi/service/start_service.sh     # service in unshare -rn, plus the relay
~/rakshak-bench/deploy/pi/service/stop_service.sh      # relay, then service; SIGTERM, KILL after 20 s

# laptop, to open the page (the relay listens on the Pi's 127.0.0.1:8770)
ssh -i ~/.ssh/pi5 -N -L 8770:127.0.0.1:8770 hermoine@pi5.local
# then open http://localhost:8770 (phone width works)
```

`start_service.sh` runs:
`setsid unshare -rn sh -c 'ip link set lo up && exec .venv/bin/python deploy/pi/service/rakshak_edge_service.py ...'`
It waits (no poll loop) on a FIFO that the service writes once it listens, with a 300 s limit. Files:
`run/edge_service.pid`, `run/edge_service.log` (JSON lines), `run/edge.sock`, `run/edge_bridge.pid`,
`run/edge_bridge.log`.

**Why there is a relay.** The service's network namespace is separate. An unprivileged user cannot
join it from outside (`nsenter` fails on `setgroups`, and `setns` needs `CAP_SYS_CHROOT`). The
service therefore serves the same API on the Unix socket `run/edge.sock` (sockets are files, so they
cross namespaces). `socket_bridge.py`, which runs outside the namespace and makes no outbound
connection, relays host `127.0.0.1:8770` to that socket. The service's own TCP listener is
`127.0.0.1:8770` inside its namespace.

**Offline proof (on the Pi, 2026-11-06):**
- The service's own report (`/health`, `network`, read from `/proc/self/net` in its namespace):
  `interfaces: ["lo"]`, `routes: 0`, `egress_probe: "failed: Network is unreachable"`.
- A fresh `unshare -rn` with the same setup: `lo` only, `routes 0`, `curl https://huggingface.co` rc 6
  (no DNS), `curl https://1.1.1.1` rc 7 (no route).
- The service's namespace id differs from the SSH shell's (`net:[4026532517]` vs `net:[4026531833]`).
- The service is reachable only through the relay and the socket; host port 8770 is refused from
  outside (before the relay started, rc 7).

### API

- `POST /analyze` with `{"sms": "..."}` (1 to 2000 characters). Response (200):
  - `decision`: `risk` (LOW, UNCERTAIN, MEDIUM, HIGH), `action` (the system's choice, never the model's),
    `score` and `threshold` (student probability and its threshold), `scam_state`, `evidence_count`.
  - `evidence`: `id`, `type`, `weight`, `text` (the matched span).
  - `advisory`: `path` (`explainer`, `low_template` for G1, `detector_only_quiet` or `detector_only` for Fix B; added
    2026-11-06; see "Detector-only mode" below),
    `mode` (`llm` all points from the model, `partial` some from the template, `template`
    none or LOW), `llm_skipped_low_risk`, `summary` (always the template), `points` (each with `text`,
    `source` llm or template, `llm_text` what the model wrote, `check` ok and problems), `action`.
  - `verifier`: `final_contract_ok`, `g2_rejected` (sentence and problem per rejected point).
  - `llm_calls`: per call `status`, `prompt_tokens`, `new_tokens`, `ttft_ms`, `wall_ms`, `stop_hit`.
  - `timings_ms`: `canonicalise`, `detect`, `evidence`, `llm_calls` (list), `llm_total`, `verify`, `total`.
    `verify` is the slot total minus the generate-call time (assembly and checks).
  - `device`: `start` and `end` (`temp_c`, `throttled`, `power_w`), `energy` (`energy_j`, `mean_w`,
    `samples`, `duration_s`, `method`), `rss_mb`, `rss_peak_mb`.
  - Errors: 400 (bad JSON, empty or over-long `sms`), 404, 500 (analysis failed, logged), 503 `{"error": "cooling"}`
    when the SoC is at or above 75 C (the explainer does not run; logged as `refused_cooling`).
- `GET /health`: models (status, load times), threads, uptime, requests, `refused_cooling`, RSS and
  peak RSS, device, `network`.
- `GET /`: the page (no external assets; `textContent` only; checked at 375 px, no horizontal scroll).

Requests run one at a time (a lock). Each request is one JSON log line with the temperatures, the
risk, the timings and the energy.

### Decision and the shipped path it uses

`src/rakshak` already has the pieces; the service composes them without new detection logic:
- canonicalise: `rakshak.canonicalize.canonicalize`
- student score: `rbench.runners.onnx.SmsRunner` (featurise plus onnxruntime), threshold from the registry
- rules: `rakshak.rules.run_rules`
- decision state, mirroring `rakshak.nids.pi_monitor.assess_flow`: score >= threshold gives `scam`;
  otherwise rule evidence gives `uncertain`; otherwise `benign` (`edge_pipeline.scam_state_for`)
- risk and allowed actions: `rakshak.evidence.build_package`, action `choose_action`
- explainer: `rbench.slot.generate_slot_advisory(..., guards=True)` (slot mode, G1 and G2), Llama `.pte`
  through `rbench.runners.pte_llm` (preloads, reset, stop)

### Thermal

Before each request: temperature read. At or above 75 C: 503, no Llama. The eval client does not start a
batch at or above 65 C. The last batch peaked at 69.7 C, below the limit.

### Eval: 30 evidence rows + 10 benign LOW rows, through the service

Rows: `eval_service.py select` (on the Pi, outside the namespace) writes `results/edge_eval/rows.json`.
The 30 evidence rows are `rbench.worker._ela_rows` (seed 0), the same as Part 1. The 10 benign rows are
seeded label-0 rows whose decision at selection time is LOW (the first 10 label-0 rows all were). The
batch: `eval_service.py run` (40 POSTs, serial, over the socket) and `report`. Raw files:
`results/edge_eval/responses.jsonl`, `summary.json`, `run_meta.json`.

Decisions (service, student plus rules): evidence group, risk HIGH 1, UNCERTAIN 8, MEDIUM 20, LOW 1.
Benign group, LOW 10. The student's decisions differ from Part 1's label-free proxy, so the LOW count is
1 of 30 here against 9 of 30 in Part 1.

**Per-stage latency (ms, p50 / p95).**

| Stage | Evidence 30 | Benign LOW 10 |
| --- | --- | --- |
| Canonicalise | 0.78 / 1.56 | 0.65 / 0.82 |
| Detect (featurise + student ONNX) | 3.18 / 6.46 | 2.59 / 3.60 |
| Evidence and decision | 0.86 / 2.35 | 0.57 / 1.02 |
| LLM, per advisory (n = 29 called; 0 benign) | 2 727 / 8 973 | (not called) |
| LLM, per call (n = 45 calls) | 2 346 / 3 043 | |
| Verify and assemble | 2.05 / 3.89 | 0.27 / 0.27 |
| Total, all requests | 2 691 / 8 838 | 4.10 / 5.70 |
| Total, LLM-called advisories only (n = 29) | 2 733 / 8 997 | |

Over all 40 requests: total p50 2 392 ms, p95 7 243 ms. The p95 is driven by advisories with several
explainer calls (one per evidence item, up to 6). Mean TTFT per call 1.52 s (p50 1.49 s).

**Energy, power, memory, temperature.**

| Measure | Evidence 30 | Benign LOW 10 | All 40 |
| --- | --- | --- | --- |
| Energy per request, J (p50 / p95 / mean) | 10.08 / 36.6 / 14.6 | 0.067 / 0.080 / 0.068 | 9.39 / 30.4 / 11.0 |
| Energy per LLM-called advisory, J (p50 / p95 / mean, n = 29) | 10.20 / 37.2 / 15.1 | | |
| Mean power per request, W (p50 / p95) | 3.96 / 4.94 | 2.77 / 3.31 | 3.84 / 4.91 |

Method: `vcgencmd pmic_read_adc`, sum of V x A over the PMIC rails (`pi_common.pmic_power_w`),
sampled every 0.25 s during the request, trapezoid-integrated (29 requests). Requests shorter than one
sample interval (11 benign requests) use mean power times duration. This is the PMIC's reading, not a
calibrated or wall-socket meter, and it covers the whole board's rails the PMIC sees.

- RSS of the service: 1 676 MB after the batch (peak 1 839 MB), on a 4 GB Pi.
- SoC: start 49.9 C; max 69.7 C; no 503 refusals; throttled `0x0` at every start and end.
- Batch wall time 111 s for 40 requests.

**Advisory quality (service, 40 requests).**

| Measure | Evidence 30 | All 40 |
| --- | --- | --- |
| Advisory mode llm / partial / template | 7 / 2 / 21 | 7 / 2 / 31 |
| Points written by the model that pass the verifier (raw) | 10 of 45 (22%) | 10 of 45 |
| Advisories with all points passing (raw) | 7 of 29 (24%) | 7 of 29 |
| G2 rejected sentences | 30 | 30 |
| G1 skipped (LOW, no LLM) | 1 | 11 |
| Final output passes the contract | 30 of 30 | 40 of 40 |

The raw pass is low, and most of the drop is G2 (the same pattern as Part 1). The guards keep the
final output safe, but the on-device model contributes a sentence to only a minority of advisories.

### Hand-check: 6 outputs (3 scam, 3 benign)

Read against the SMS text and the rule evidence in each response.

1. Row `ee2cdc56a23fb9dbfa26` (evidence, label 1, MEDIUM, mode llm). SMS: "We have locked your
   account due to unusual activity. Kindly verify your identity at <URL>". Point: "This message may be a
   scam because it claims you have an account or card that has been suspended." **Correct and hedged.**
   Minor overclaim in wording: the SMS says "locked", not "suspended" (the rule's phrase). The action
   (CONTACT_BANK_OFFICIAL) is safe.
2. Row `906572c8cc42edc04053` (evidence, label 1, HIGH, llm). SMS (Hinglish): "National Cyber Cell se bol raha
   hoon. Investigation ke liye aapki identity verify karni hogi." Point: "This message is a scam because it
   claims to be a police or government agency." **Correct in substance, unhedged.** HIGH is not checked by
   G2 (per the rule), so "is a scam" passes. This is the one place the service states scam intent as fact;
   it is a policy choice, and it is stated here so it can be revisited. Action REPORT_1930_OR_CHAKSHU is right.
3. Row `fd95b9eca7b8972b358c` (evidence, label 1, MEDIUM, llm). SMS: "Your parcel line up for delivery Track
   the progress here http://bit.ly/ 376QzLp". Point: "This message may be a scam because it contains a
   shortened link." **Correct and hedged.** The SMS URL is truncated in the data; nothing is invented.
4. Row `05d06be1613843224f60` (benign, label 0, LOW, template). SMS is garbled (mixed Hindi and markup).
   Summary: "These checks found no known scam pattern, which does not make the message safe." **Correct:
   the rules found no pattern and the text says so.** No LLM call.
5. Row `7b4f4217d15746d59bc6` (benign, label 0, LOW, template). SMS: "I uploaded mine to Facebook".
   Same template summary. **Correct; no overclaim.**
6. Row `b8604793f6e6e1c61695` (benign, label 0, LOW, template). SMS: "Check ur Skype messages for the link".
   Student score 0.3225 (threshold 0.3513), no rule evidence. Same template summary. **Correct for the
   decision, but close to the threshold.** Had the score been 0.36, the decision would be `scam` with no
   rule evidence, and `evidence.py` gives LOW (see finding 1).

Verdict: no hand-checked output made an unsupported factual claim. One output (case 2) states scam intent
without a hedge at HIGH, by the rule. Six is a small sample, not a wrong-claim rate.

### Findings and open issues

1. **(Resolved in section 3, service path only.)** A student-flagged scam with no rule evidence is LOW risk. `evidence._risk` returns LOW when the state is
   `scam` and there is no evidence (the rule needs evidence to escalate). With G1 that message gets the
   template "found no known scam pattern", which contradicts the detector. Not hit in this batch (the
   nearest benign score was 0.3225), but it is reachable. This is the shipped risk mapping and was not
   changed here. A decision for the owner: map `scam` without evidence to UNCERTAIN, or keep LOW and word
   the LOW template differently.
2. **The guard removes most of the model's sentences.** The explainer's sentences were mostly attribution
   ("is likely a scam", "is trying to rush you"). G2 makes the advisory safe, but the on-device model writes
   one passing sentence in about one point in five.
3. **The student and the label-free proxy disagree on risk.** The evidence rows are 1 LOW under the student
   and 9 LOW under the proxy. Latency and energy numbers from Part 1 are for the proxy path.
4. **Energy is the PMIC sum**, not a calibrated meter. Per-request energy is dominated by the explainer calls.

## 3. Fixes A and B (2026-11-06, second run)

Two fixes after the first 40-row run (section 2). Same 40 rows (`results/edge_eval/rows.json`, unchanged),
same service path, re-run with `eval_service.py run` and `report`. The section 2 numbers are kept in
`results/edge_eval/before_fixes/` on the Pi (`summary.json`, `responses.jsonl`, `run_meta.json`).

### Fix A: describe, do not judge

**Problem.** The explainer wrote "This message is (likely) a scam because ..." and G2 rejected it. Raw point
pass was 10 of 45.

**Change (edge only).** `rbench.slot.PROMPT_VARIANTS` holds three describe variants. The default slot prompt
is unchanged, and `generate_slot_advisory(..., variant=None)` is the default. In all variants the system text
asks for a plain restatement of one fact and forbids judging the message or the sender, and the system text
contains no attribution word (tested). The variants:
- `describe`: the describe system text, prefix "This message".
- `describe_kw`: the describe system text, prefix "This message <type clause> and" (the keyword is in the
  prefix, as in slot_kw).
- `describe_plain`: a shorter restate-only system text, prefix "This message".

**Trial.** `deploy/pi/service/prompt_trial.py trial --rows 10`, run on the Pi through rbench.slot with the
guards on (G1, and G2 with the likely hedge below). The 10 rows are evidence rows with risk other than LOW,
taken from the seed 0 pool of 80 and skipping the 40 eval rows, so the choice did not use the reported rows.
11 points in total. Three new variants were tried, plus the default prompt as the control.

| Variant | Raw point pass | Raw advisory pass | G2-rejected points | Prompt tokens (mean) | Call wall p50 |
| --- | --- | --- | --- | --- | --- |
| slot (default, control) | 6 of 11 | 5 of 10 | 3 | 95.1 | 2 045 ms |
| describe | 10 of 11 | 9 of 10 | 0 | 99.1 | 1 631 ms |
| **describe_kw (chosen)** | **11 of 11** | **10 of 10** | 0 | 104.3 | 2 275 ms |
| describe_plain | 9 of 11 | 8 of 10 | 0 | 75.1 | 1 517 ms |

**Choice and hand-check of the trial sentences.** describe_kw has the highest raw pass and the most specific
sentences (for example "This message contains a shortened link and is likely to be a link to a website or
service."). Two defects read by hand: in 2 of 11 points the model echoes the prompt ("... and restates the
fact."), and the prompt is 104 tokens, about 10% over the 95-token target. `describe` is the next-best and is
closer to the token target; it is the fallback if the echo matters. The gap between describe_kw and describe
is one point out of 11, so the choice is a weak basis and is recorded as one.

**G2 change (verifier.py, `LIKELY_HEDGE_RE`).** At MEDIUM and UNCERTAIN, "likely a|an scam|fraud|phishing"
is now a hedge, like "may be a scam". LOW stays strict, HIGH is not checked. Any other attribution term in
the same sentence is still rejected ("likely a scam, and scammers want your PIN" fails). The lexicon is
unchanged. Tests: `tests/test_verifier_guards.py` (the old rejection of "is likely a scam" at MEDIUM was
removed; the hedged cases and the second-term case are added). Limit: the hedge needs "likely" directly before
the term. "likely an investment scam" is still rejected (row 25 below). In the trial and in the eval the
describe prompt almost never wrote "likely a scam", so this change had little effect on the numbers.

### Fix B: detector flags the message, no rule evidence

**Problem.** A message with student score at or above the threshold and no rule evidence has scam state
"scam", and `evidence._risk` maps that to LOW. The LOW template then said "These checks found no known scam
pattern", which contradicts the detector. `evidence._risk` is not changed (Kaggle evals depend on it).

**Rule, verify mode (`edge_decision.py`, used by `EdgeAnalyzer.analyse`).** When scam state is "scam" and there
is no rule evidence, and `detector_only_mode` is `verify` (the default is `quiet`, see "Detector-only mode"):
- The package risk becomes **UNCERTAIN** (not LOW, not MEDIUM). Allowed actions are `VERIFY_INDEPENDENTLY`,
  and the advisory action is `VERIFY_INDEPENDENTLY`. The reason for UNCERTAIN over MEDIUM: the verifier needs
  non-empty reasons for HIGH and MEDIUM, and without evidence MEDIUM would always fail the contract.
  UNCERTAIN is the state the NIDS path uses for "not sure".
- No explainer call. The advisory has `path: "detector_only"`, mode `template`, no points, and the fixed
  summary `DETECTOR_ONLY_SUMMARY`: "Our on-device detector flagged this message, but no specific scam pattern
  was matched. Do not click links or give out codes until you have checked with the sender through an official
  channel."
- Wording: the first draft said "Don't click links or share codes". The verifier's positive-instruction rule
  flags "share" even after "Don't", so the text was reworded. The rule was not loosened. The contract check
  passes on the final text (tested).

**Tests.** `tests/test_edge_service.py`: the path is taken at the threshold and above; below the threshold
without evidence the LOW template still applies; a message with evidence still uses the explainer; the
summary contains no "no known scam pattern" when the detector fired; `evidence._risk` still gives LOW for
scam with no evidence. Both modes are covered (see below).

### Detector-only mode (decided 2026-11-06 for the demo)

Open decision (a) in `docs/HANDOFF.md` is decided: **quiet for the demo**, with verify kept behind a setting.

- **Setting.** `detector_only_mode` in `EdgeAnalyzer` (`deploy/pi/service/edge_pipeline.py`), and on the service
  `--detector-only-mode {quiet,verify}` or env `RAKSHAK_DETECTOR_ONLY_MODE`. Default `quiet`. `GET /health` reports
  it. Logged per request as `path` and `detector_only`.
- **Decision code.** `deploy/pi/service/edge_decision.py` (`decide_edge`, no model import). The pipeline and
  `eval_groundlm.py` both call it, so the eval cannot drift from the service (`tests/test_eval_groundlm.py` checks
  agreement on the same inputs).
- **quiet (default).** Detector-only flag (score >= threshold, no rule evidence): risk **LOW**, action **IGNORE**
  (the same no-warning action as a benign message), no Llama call. `advisory.path` = `detector_only_quiet`,
  `advisory.summary.text` = **"No specific scam pattern was found in this message."** It does not say the message is
  safe, and it does not say "no known scam pattern". The verifier accepts it (tested). The response still carries
  `decision.detector_only: true` and `decision.score`, so the dashboard and the eval can count these.
- **verify.** The Fix B behaviour as above: risk UNCERTAIN, action VERIFY_INDEPENDENTLY, `advisory.path` =
  `detector_only`, summary `DETECTOR_ONLY_SUMMARY`. Kept for later; not the demo default.
- **Not changed in either mode.** Rule evidence and the explainer path; `evidence._risk`; the student threshold.
- **What quiet costs.** Student-only scams are no longer shown as warnings. On SmishX, the visible-notice recall
  falls from 76.6% to 23.5%, and on the campaign test split from 98.1% to 16.2% (`docs/GROUNDLM_EDGE_FPR.md`).
  The bank, UPI and telecom false alarms are no longer shown either. The detector fix (retrain or calibrate on
  those notices) is the research task, not this setting.
- **Tests.** `tests/test_edge_service.py` (both modes, the neutral summary, the verifier contract, the health field)
  and `tests/test_eval_groundlm.py` (decision agreement with the pipeline, `apply_mode` against a fresh decide).

**Real cases.** `prompt_trial.py scan-detector-only` found real test-split messages in this state: the first
5 hits were within the first 126 rows of the split, all with score 0.70 to 0.99 and no rule hit, all LOW
before the fix. The 40-row eval contains 2 of them (rows 35 and 38 below). No mocked score was needed for the
real path. Unit tests use a constructed text with no rule hit.

### Re-run with quiet as the default (Pi, 2026-11-06, 40 rows, threads 3)

Run once after `sync_bench.sh --code-only`. The SoC started at 54.9 C with throttle `0x0`. The previous
run's responses are kept on the Pi in `results/edge_eval/verify_before_quiet/`. Only rows 35 and 38 changed.

| Measure | verify (before) | quiet (default) |
| --- | --- | --- |
| Fix-B path (detector only, no evidence) | 2 (`detector_only`, both benign group) | **2** (`detector_only_quiet`, rows 35 and 38) |
| Row 35 and 38 decision | UNCERTAIN / VERIFY_INDEPENDENTLY | **LOW / IGNORE** |
| Row 35 and 38 summary | "Our on-device detector flagged this message ..." | "No specific scam pattern was found in this message." |
| Rows 0 to 34 and 36 to 39 (decision, advisory, evidence, LLM status) | reference | identical (checked field by field, timings excluded) |
| Contract ok, HTTP 200 | 40 of 40 | 40 of 40 |
| Total p50 / p95 (all 40, ms) | 1569.7 / 2934.9 | 1629.9 / 2935.6 |

The p50 difference is not a controlled comparison: one run each, and this run peaked at 73.6 C (the handoff
records earlier peaks of 70 to 73 C). Treat it as noise until repeated.

### Re-measured: service, same 40 rows (run 2026-11-06, 98 s batch)

| Measure | Before (section 2) | After (fixes A and B) |
| --- | --- | --- |
| Raw point pass (points written by the model that pass the verifier) | 10 of 45 (22%) | **38 of 45 (84%)** |
| Raw advisory pass, called advisories | 7 of 29 (24%) | **22 of 29 (76%)** evidence; 24 of 31 all |
| Advisory mode llm / partial / template, evidence 30 | 7 / 2 / 21 (all 40: 7 / 2 / 31) | **22 / 2 / 6** (all 40: 22 / 2 / 16) |
| G2-rejected sentences | 30 (10 distinct) | **1 (1 distinct)** |
| G1 skipped (LOW, no LLM call) | 1 evidence, 11 all | 1 evidence, 9 all |
| Fix-B path (detector only, no evidence) | not present (2 rows said "no known scam pattern") | **2** (both in the benign group) |
| Decisions, evidence 30 (HIGH / MEDIUM / UNCERTAIN / LOW) | 1 / 20 / 8 / 1 | 1 / 20 / 8 / 1 (unchanged) |
| Decisions, benign 10 (LOW / UNCERTAIN) | 10 / 0 | 8 / 2 (Fix B) |
| Final output passes the contract | 40 of 40 | 40 of 40 |
| LLM calls (n) and status | 45 | 45: 41 ok, 4 unterminated |

**Latency (ms, p50 / p95).**

| Stage | Evidence 30, before | Evidence 30, after | Benign 10, before | Benign 10, after |
| --- | --- | --- | --- | --- |
| Canonicalise | 0.78 / 1.56 | 0.74 / 1.53 | 0.65 / 0.82 | 0.64 / 0.80 |
| Detect (student ONNX) | 3.18 / 6.46 | 3.02 / 6.31 | 2.59 / 3.60 | 2.50 / 3.51 |
| Evidence and decision | 0.86 / 2.35 | 0.83 / 2.33 | 0.57 / 1.02 | 0.54 / 1.02 |
| LLM per advisory (called) | 2 727 / 8 973 (n = 29) | 2 275 / 7 618 (n = 29) | not called | not called (the 2 Fix-B rows take no LLM path) |
| LLM per call | 2 346 / 3 043 (n = 45) | 2 062 / 2 568 (n = 45) | | |
| Verify and assemble | 2.05 / 3.89 | 2.02 / 4.54 | 0.27 / 0.27 | 0.26 / 0.27 |
| Total, all requests | 2 691 / 8 838 | 2 256 / 7 520 | 4.10 / 5.70 | 3.95 / 5.61 |

Total over all 40 requests: p50 2 392 to **2 062** ms, p95 7 243 to **6 305** ms.

**Energy, memory, temperature.**

| Measure | Before | After |
| --- | --- | --- |
| Energy per request, evidence 30 (J, p50 / p95 / mean) | 10.08 / 36.6 / 14.6 | 8.92 / 30.8 / 12.8 |
| Energy per LLM-called advisory, evidence (J, p50 / p95 / mean) | 10.20 / 37.2 / 15.1 | 9.03 / 31.1 / 13.2 |
| Energy per request, all 40 (J, p50 / p95 / mean) | 9.39 / 30.4 / 11.0 | 7.76 / 27.4 / 9.61 |
| Energy per request, benign 10 (J, p50 / mean) | 0.067 / 0.068 | 0.068 / 0.067 |
| RSS after batch, MB (peak) | 1 676 (1 839) | 1 676 (1 839) |
| SoC: batch start, max | 49.9 C, 69.7 C | 55.4 C, 70.3 C |
| 503 refusals, throttle at start and end | 0, `0x0` | 0, `0x0` |

The energy method is unchanged (PMIC rails, not a calibrated meter, section 2). The batch numbers are
lower than before; the run did not isolate why, so no cause is claimed.

**Hand-check: 8 outputs (4 scam including 1 HIGH, 4 benign or Fix B).** Read against the SMS text in
`rows.json` and the response. Verdicts: correct, overclaim (says more than the message supports), wrong.

| Row | Case | Output (the point or the summary) | Verdict |
| --- | --- | --- | --- |
| 22 | HIGH, digital_arrest, template after the model's "asks ... to provide information" was rejected by the positive-instruction rule | "This message claims to be a police or government agency." | **Correct.** The SMS claims to be a government cell. No "scam" wording at HIGH. |
| 0 | MEDIUM, prize, LLM | "This message claims a prize or lottery and has been matched." | **Correct, with a wording leak**: "has been matched" echoes the prompt. |
| 3 | MEDIUM, partial (2 LLM points, 1 template) | template: "This message claims an account or card is suspended." | **Overclaim.** The SMS says the SIM document is pending and will be blocked; it does not say "suspended". The LLM points ("uses time pressure ...") are correct. |
| 20 | MEDIUM, parcel fee, template (the model's "asking the recipient to pay" was rejected) | "This message asks for a parcel or customs fee." | **Correct.** |
| 25 | MEDIUM, investment, template after G2 rejected "is likely an investment scam" | "This message promises guaranteed investment returns." | **Overclaim** (mild). The SMS reports clients' profits and invites a subscription; "guaranteed" is the rule's phrase, not the SMS's. |
| 35 | benign, label 0, detector 0.59, no rule hit (Fix B) | "Our on-device detector flagged this message, but no specific scam pattern was matched. Do not click links ..." | **Correct.** Says what the system found and does not claim a pattern. |
| 38 | benign label 0, detector 0.44, no rule hit (Fix B) | the same Fix-B summary | **Correct.** The SMS is a Netflix-style billing link on a lookalike domain; the rules did not match it, and the advice (do not click) is the safe reading. The label says benign: this may be label noise. |
| 32 | benign, LOW, score 0.32, no rule | "These checks found no known scam pattern, which does not make the message safe. ..." | **Correct** for the decision. |

Tally of the 8: 6 correct (one with a wording leak), 2 overclaim (rows 3 and 25), 0 wrong. A small sample.
Two more issues seen outside the 8 and not fixed here:
- Row 17 (LOW, urgency only, detector 0.01): the reply "I will get back to you after 8 am" is shown with "uses
  time pressure (turant, abhi, 24 hours)" and then "no known scam pattern". The urgency rule (weight 0.4) is weak.
  This is the existing rule, not Fix B.
- Row 4 (MEDIUM, six prize items): the same sentence is written six times. The verifier passes it, so the
  advisory is repetitive and says little.

**Residual defects, all visible in the trial and the eval.**
- 4 of 45 calls ended "unterminated" (no sentence end before the token budget), so those points fell back to the
  template (rows 3, 13, 21, 26).
- The model echoes prompt words: "matched" (row 0), "restates the fact" (trial).
- The explainer writes generic sentences ("is asking the reader to do something"). They pass the verifier but
  add little.
- G2 limits: the hedge needs "likely" directly before the term (row 25). The one G2 rejection in the eval is
  that case.

**Changed files for fixes A and B.** `deploy/pi/bench/rbench/slot.py` (`PROMPT_VARIANTS`, `variant_prompt`,
`variant=` in `generate_slot_advisory`), `src/rakshak/verifier.py` (`LIKELY_HEDGE_RE` in `attribution_problems`),
`deploy/pi/service/edge_pipeline.py` (`EDGE_PROMPT_VARIANT = "describe_kw"`, `DETECTOR_ONLY_SUMMARY`,
`detector_only_package`, the detector-only path, `advisory.path`), `deploy/pi/service/eval_service.py` (report:
`advisory_path`, `g2_distinct_sentences`), `deploy/pi/service/prompt_trial.py` (new: trial and Fix-B scan),
`tests/test_rbench_describe.py` (new), `tests/test_edge_service.py`, `tests/test_verifier_guards.py`.
Pi, not in the repo: `results/edge_eval/prompt_trial/` (records, summary), `results/edge_eval/before_fixes/`
(section 2 numbers), and the new `results/edge_eval/responses.jsonl`, `summary.json`, `run_meta.json`.

## 3b. Fix C: output cleanup (2026-11-06, deterministic, edge only)

Three defects in the last 40-row run (section 3, "Residual defects") are fixed in code after the
model writes its sentences. The prompt, the verifier rules and the G2 lexicon are unchanged. Opt-in:
`generate_slot_advisory(..., cleanup=True)`. The edge pipeline passes it; other callers keep the old
behaviour.

**Rules (applied in order, to each point the model wrote, before assembly and verification):**
1. **Echo strip** (`rbench.slot.strip_echo`). A prompt echo at the end of a sentence is removed: the
   clause `and restates the fact`, `and has been matched`, `is matched` (with an optional `and` or `which`
   before it). The remainder is kept only if it starts with "This message", has at least four words and
   does not end on a function word (`and`, `because`, `it`, `the`, ...). Otherwise the point is "" and falls
   back to the template (status `echo_invalid`). Echo phrases in the middle of a sentence are not touched.
   Generic clauses such as "is asking the reader to do something" are not echoes and are not stripped.
2. **Duplicate** (`rbench.slot.duplicate_points`). After the echo strip, a point whose normalised text
   (lower case, no punctuation) equals an earlier point's, or has a difflib ratio of at least 0.9, is
   blanked. The first occurrence is kept. The blanked point falls back to the template (status
   `duplicate`).
3. **Verifier.** Every point still goes through `check_points` and `render_advisory`. A blanked point is
   a template point, so the final output passes the contract as before.

**Consequences, stated so they are not a surprise.** A duplicate is not counted in `g2_rejected` (it is
blanked before the verifier sees it). A point that the model wrote as "" is still "unterminated" as before.
`llm_text` in the response is the model's original sentence; `text` is the final sentence;
`cleanup` (`echo_stripped`, `duplicate_of`) is on each point.

**Bad strings from the last eval (`results/edge_eval/responses.jsonl`, unchanged rows):**

| Row | Model sentence | After Fix C |
| --- | --- | --- |
| 0 | "This message claims a prize or lottery and has been matched." | "This message claims a prize or lottery." |
| 14 | "This message threatens electricity disconnection and restates the fact." | "This message threatens electricity disconnection." |
| 4 | "This message claims a prize or lottery and is asking the reader to do something." x 6 | first kept, five template points |

Tests: `tests/test_edge_cleanup.py` (the strings above, the no-echo and invalid-remainder cases, the
dedupe rule, the cleanup-off path, and end-to-end through `EdgeAnalyzer`). `tests/test_edge_service.py`
was changed for the repeat rule: the FakeLLM test that repeats one sentence now asserts the first point is
LLM text and the repeats are template (its G2 count is 1, not 2).

Not done: the model's repeated generic clause ("is asking the reader to do something") still passes the
verifier and is shown once per distinct evidence type. A rule to stop that would change what counts as
an acceptable point and is left for a decision.

## 3c. Fix D: one explainer call and one point per evidence type (2026-11-06)

Fix C left the repeats as template sentences ("the same line, six times" became "the template line, five
times"). Fix D removes the repeats at the source. `edge_pipeline.collapse_by_type` keeps the first evidence item
of each type; the explainer is called once per type, on that item's matched text. The advisory is built against
that collapsed package, so `check_points` needs exactly one point per type and passes. The response keeps the
full `evidence` list; each advisory point has `evidence_ids`, `count` (how many items of the type matched), and
`evidence_id` of the first. `advisory.evidence_items` is the full count. Risk, actions, the summary (the same
first-three-types template) and the G1/G2 guards are unchanged. The count is not written into the sentence: the
verifier checks the model's sentence, and a number added after the check would not be checked.

Tests: `tests/test_edge_collapse.py` (row 4 shape: six prize_lottery items; row 28 shape: two shortened_url items;
mixed types; the forced-LOW template path), with a mock runner. `tests/test_edge_cleanup.py` and
`tests/test_edge_service.py` were changed for the new call count.

**Re-run, same 40 rows (`results/edge_eval/before_collapse_t3/` is the Fix C run at 3 threads), Pi, 3 threads:**

| Measure (30 evidence rows, 29 called) | Before (Fix C) | After (Fix D) |
|---|---|---|
| Points per called advisory, mean / max / total | 1.55 / 6 / 45 | **1.14 / 3 / 33** |
| LLM calls per called advisory, mean / total | 1.55 / 45 | **1.14 / 33** |
| Raw point pass | 30 of 45 (67%) | 26 of 33 (79%) |
| Cleanup: echo stripped / duplicates blanked | 2 / 8 | 2 / 0 |
| Mode llm / partial / template | 19 / 5 / 6 | 22 / 1 / 7 |
| Final output passes the contract (per-type points) | 30 of 30 | 30 of 30 |
| Total p50 / p95 (ms), evidence 30 | 2 096 / 6 350 | **1 706 / 3 012** |
| LLM per advisory p50 / p95 (ms) | 2 133 / 6 417 | 1 718 / 3 013 |
| LLM per call p50 / p95 (ms) | 1 791 / 2 544 | 1 614 / 2 119 |
| Total p50 / p95 (ms), all 40 | 1 814 / 5 420 | 1 570 / 2 935 |
| Energy per request, evidence (J, p50 / p95 / mean) | 8.0 / 27.9 / 11.9 | 7.2 / 16.4 / 8.2 |
| SoC at start / max during batch; throttle | 52 / 71.9 C; `0x0` | 52.7 / 71.9 C; `0x0` |
| 503 refusals | 0 | 0 |

The raw point pass is lower in count but not in rate: before, 30 of 37 non-duplicate points passed (81%); after,
26 of 33 (79%). The model's sentences are unchanged for every row that had one call.

**Hand-check.** Row 4 (prize-giveaway SMS, label 1): one point, "This message claims a prize or lottery and is asking
the reader to do something.", count 6. Correct, and the generic clause is the known residual. Row 28 (SMS with two
bit.ly links and one other link, label 1): one point, "This message contains a shortened link and is likely to be a
link to a website or service.", count 2. Correct; it does not claim anything about the non-shortened link.

**Trade-off seen in the run.** Row 21 went from partial to template: its first item's call returned an empty
sentence, so the one point for that type fell back to the template. Before, the second item's call had a valid
sentence. One call per type gives no second try. Retrying the next item on an empty result would restore it at the
cost of one more call, and is not done here.

## 4. Soak and thread count (2026-11-06)

**Method.** `deploy/pi/service/soak.py run --threads N --requests 100`, run on the Pi outside the service's
namespace, over the Unix socket. Request k uses eval row k mod 40 (`rows.json`), back-to-back with no idle gap.
A 1 Hz sampler records SoC temperature, throttle flags, ARM clock and the cpufreq cap. Abort rules: SoC at or
above 79 C, throttled not `0x0`, or temperature unreadable for 3 samples (the sampler then stops the service).
A 503 is recorded and the same row retried after 3 s; it does not count toward the 100. Per request: total
and LLM latency, temperatures, power, energy, cap and clock before and after. Each run starts and stops the
service with `start_service.sh --threads N` and `stop_service.sh`; the Pi cooled below 55 C between runs
(`soak.py cool`). The cpufreq cap is set by the system governor `rakshak_thermald.py` (ceiling 2400 MHz,
floor 1500 MHz, steps of 100 MHz from 68 C); this soak did not change it.

**Results (one run each, 100 accepted requests, 0 aborts, 0 throttle flags).**

| Measure | 2 threads | 3 threads | 4 threads |
| --- | --- | --- | --- |
| Total p50 / p95 (ms) | 2 400 / 9 265 | 1 939 / 7 152 | 1 820 / 7 524 |
| LLM per advisory p50 / p95 (ms, n = 77 called) | 2 626 / 9 707 | 1 994 / 8 367 | 1 996 / 7 516 |
| LLM per call p50 / p95 (ms, n = 123) | 2 414 / 2 703 | 1 994 / 2 409 | 1 895 / 2 255 |
| Decode tok/s (aggregate) | 13.06 | 12.11 | 11.10 |
| Requests/min, last 50 | 20.1 | 23.2 | 25.7 |
| Requests/min, whole run | 19.5 | 23.9 | 25.2 |
| Max SoC temp (C) | 69.7 | 71.9 | 73.0 |
| Cap at start (MHz) | 1900 | 2400 | 1900 |
| Min cap (MHz) | 1800 | 1500 | 1500 |
| Time to first cap reduction (s) | 61.5 | 18.4 | 18.5 |
| Energy per advisory p50 / p95 (J) | 7.81 / 32.6 | 7.21 / 29.9 | 6.76 / 28.1 |
| Energy per LLM advisory p50 (J) | 8.65 | 8.19 | 7.80 |
| 503s | 0 | 0 | 0 |

Modes (same rows, all runs): llm 53, partial 13, template 34.

**Caveats that limit the comparison.** The governor state carried over between runs. The 3-thread run started
at the 2400 MHz ceiling; the 4- and 2-thread runs started at 1900 MHz, so their early requests ran slower
than a fresh start would. Both 3 and 4 threads reached the 1500 MHz floor, so the last-50 rate for those two
is taken at the same cap. The 2-thread run never reached the floor. One run per thread count; the p95 values
depend on a few long advisories (several explainer calls), so differences of 5 to 10 percent are within noise.

**503 gate.** Not exercised live. The SoC peaked at 73.0 C, below the service's 75 C gate, so no request was
refused. The gate is covered only by the unit tests in `tests/test_edge_service.py` (mocked temperature). The
recovery path after a live 503 is untested.

**Recommendation: default 3 threads.** The service default changed from 2 to 3 (`rakshak_edge_service.py`,
`--threads`). Reasons: 2 threads is the slowest on every latency measure and has the highest energy per
advisory. 4 threads is the fastest on p50 and throughput, but its peak SoC is the highest (73.0 C, 2 C under the
75 C gate and 6 C under the 79 C abort), its p95 is slightly worse than 3, and its energy advantage is about
7 percent. 3 threads gives most of the throughput with the lowest p95 and more thermal margin. This is one run
per count, so the choice between 3 and 4 is a judgement, not a measured optimum.

**Files.** `deploy/pi/service/soak.py` (new), `deploy/pi/service/rakshak_edge_service.py` (default threads 3).
Raw output on the Pi: `~/rakshak-bench/results/soak/t2`, `t3`, `t4` (`requests.jsonl`, `samples.jsonl`,
`summary.json`).

## 5. Hinglish (2026-11-06)

**What changed (edge only; English output is unchanged).** Each request is routed by `detect_lang(raw text)`
(`src/rakshak/datasets/lang.py`): `hinglish` goes to the Hinglish path, `en`, `hi_deva` and `other` to the
English path (hi_deva and other stay English for now). The request may set `"lang": "auto" | "en" | "hinglish"`
(default `auto`). The response carries `decision.lang_detected` (always the detector's value) and `advisory.lang`
(the language used). `--hinglish-mode {template,llm}` (env `RAKSHAK_HINGLISH_MODE`) selects the Hinglish explainer:
- `template`: no Llama call; the verifier's Hinglish templates (`advisory.path` = `template_hinglish`).
- `llm`: a Hinglish slot prompt (`deploy/pi/bench/rbench/slot.py`, `hinglish_prompt`; English instructions, the fact
  in `TYPE_PHRASE["hinglish"]`, assistant prefix "Yeh message", same stop rule and token budget). The sentence is
  checked by the verifier with `lang="hinglish"` (G1, G2, cleanup and collapse as for English), and a point that
  fails falls back to the Hinglish template.
- Detector-only flags on Hinglish text use Hinglish versions of the two fixed summaries (`DETECTOR_ONLY_*_HINGLISH`).

**Routing check on the existing English eval set (40 rows, `results/edge_eval/rows.json`).** 38 route `en`. Two
rows route `hinglish` under `auto`: `906572c8cc42edc04053` (evidence, "National Cyber Cell se bol raha hoon ...",
a real Hinglish SMS) and `62046b0c0c23c1bb48fb` (evidence, a mixed offer SMS). Two benign rows are `hi_deva` and
`other` and route `en`. Re-running the English 40 through the new service (`--lang auto`, template mode) gives the
same decision, evidence, advisory text, verifier result and LLM-call records as the Fix C run for the other 38
rows (`results/edge_eval_hinglish/english_regression/`). The two Hinglish rows differ, as intended.

**Eval set.** `deploy/pi/service/build_hinglish_rows.py` (seed 0) builds it from `data/processed/sms.parquet`, the
`lang_shift` test partition, and `detect_lang == "hinglish"`. For that partition detect_lang agrees with the corpus
`lang` column on all 398 rows. 30 scam rows with rule evidence are drawn round-robin over the evidence types (rarest
first; primary types: prize_lottery 7, digital_arrest 6, urgency 6, shortened_url 5, apk_install 2, and one each of
kyc_block_urgency, parcel_customs, account_suspension, suspicious_url). 10 benign Hinglish rows are drawn at random.
File on the Pi: `~/rakshak-bench/results/edge_eval_hinglish/rows.json` (laptop copy: `results/edge_eval_hinglish/rows.json`).
Only 89 of the 320 Hinglish test scams have rule evidence, so the eval is evidence-heavy by design.

**Arms** (same 40 rows, service on the Pi, 3 threads, `eval_service.py run --rows ... --out ... --lang ...`):
A = baseline, `--lang en` (what users get today); B = `auto` with `--hinglish-mode template`; C = `auto` with
`--hinglish-mode llm`. Raw files: `~/rakshak-bench/results/edge_eval_hinglish/{A_baseline_en,B_template,C_llm}/`.

Decisions are identical in all arms (the decision does not depend on language): scam rows HIGH 6, MEDIUM 18,
UNCERTAIN 6, LOW 0; benign rows LOW 9, MEDIUM 1. Detector flag (student score at or above threshold): 30 of 30
scam rows, 0 of 10 benign. User alert (MEDIUM or HIGH): 24 of 30 scam rows, and 1 of 10 benign. The benign alert is
row 34, "mera card block hai i will give you around by tomorrow": a rule false positive (`account_suspension`, "card
block"; score 0.12, threshold 0.35). It is not a model output.

| Measure | A: forced en | B: hinglish template | C: hinglish llm |
|---|---|---|---|
| Advisory language | en 40 | hinglish 40 | hinglish 40 |
| Path (explainer / low_template / template_hinglish) | 31 / 9 / 0 | 0 / 0 / 40 | 31 / 9 / 0 |
| Mode llm / partial / template (all 40) | 22 / 1 / 17 | 0 / 0 / 40 | 6 / 2 / 32 |
| Raw point pass (model sentences, verifier, lang of arm) | 28 of 37 (76%) | n/a (templates, 37 of 37 pass) | **8 of 37 (22%)** |
| G2 rejected sentences | 2 (below) | 0 | 0 |
| LLM calls unterminated / total | 4 / 37 | 0 / 0 | **29 / 37** |
| Final output passes the contract | 40 of 40 | 40 of 40 | 40 of 40 |
| Total latency p50 / p95, all 40 (ms) | 1 677 / 3 937 | 6 / 10 | 2 597 / 6 240 |
| LLM per advisory p50 / p95 (n = 31 called, ms) | 1 778 / 3 979 | n/a | 2 936 / 6 381 |
| LLM per call p50 / p95 (ms) | 1 682 / 2 088 | n/a | 2 814 / 3 736 |
| Energy per request J, p50 / p95 / mean (all 40) | 6.7 / 16.6 / 7.0 | 0.08 / 0.10 / 0.075 | 9.5 / 27.8 / 10.1 |
| SoC at batch start / max during batch | 48.3 / 70.3 | 56.0 / 58.2 | 52.1 / 70.8 |
| Throttle flags at start and end | 0x0 | 0x0 | 0x0 |

Energy is the PMIC rails (`vcgencmd pmic_read_adc`), not a calibrated meter. The Pi had no cooler or 27 W PSU, and
the eval client's 65 C start gate was respected (each arm started below 65 C; the llm arm peaked at 70.8 C, under the
75 C refusal limit). RSS 1 676 MB after each batch.

**G2 rejections (arm A only, English prompt on Hinglish text).** Rows 12 and 24: "This message claims to be a police
or government agency and is likely to be a scam." Problem: attribution at MEDIUM outside a hedge (`likely to be`
is not in the `likely a scam` hedge form). Arm C produced no G2 rejections; its failures were lexicon misses, not
attribution.

**Why C fails (diagnostic, not an arm).** The service does not store raw generations, so the Pi was stopped and a
short diagnostic (`~/rakshak-bench/run/diag_hinglish.py`, six unterminated rows, same prompt and model) printed them.
The model quotes the fact's matched text and stops on the quote mark, which the stop rule treats as a sentence
break. Examples: `' ka matlab hai "'` (row 0), `' ka hinglish hai: "'` (row 6), `' "install our app" ka matlab hai.<|eot_id|>'`
(row 3). It also rewrites the instruction as a translation task. The Hinglish prompt was not tuned on these rows,
because that would contaminate the eval, so no fix is shipped.

**Hand-check (model check, not a native-speaker check).** Ten llm-arm outputs read against the SMS text and its
rule evidence: the eight sentences that passed the verifier in arm C (rows 2, 4, 12, 22, 23, 24, 27, 29), and two raw
sentences from the diagnostic that did not pass (rows 3 and 5). Quoted as written:
1. row 2 (suspicious_url; SMS links whatsapp-verify-account-now.net): "Yeh message Shaq wala link deta hai." Fact correct, natural.
2. row 4 (suspicious_url; SMS links to a bare IP address): "Yeh message ka anubhav karna hai ki Shaq wala link deta hai." Fact correct; "ka anubhav karna hai" is wrong Hinglish.
3. row 12 (digital_arrest; "ACP Sharma bol raha hoon"): "Yeh message police agency ke sarkari agency hone ka dawa karti hai." Fact correct; repeats "agency", garbled.
4. row 22 (prize_lottery; "Inaam" offer with bit.ly): "Yeh message ka aadhar Inaam ya lottery ka dawa hai." Fact correct; "ka aadhar" is wrong.
5. row 23 (prize; cash and bike claim): "Yeh message ka aadhar lottery ke dawa ke bare mein hai." Fact correct; vague, "ka aadhar" wrong.
6. row 24 (digital_arrest, same SMS pattern as row 12): "Yeh message police agency ke sarkari agency hone ka dawa karti hai." Same as 3.
7. row 27 (digital_arrest; "Crime Branch se bol raha hoon"): "Yeh message ka sarkar ke police agency ke liye dawa hai." Meaning unclear; no overclaim.
8. row 29 (prize; cash and bike claim): "Yeh message ka aadhar lottery ke dawa ke bare mein hai." Same as 5.
9. row 3 (apk_install, raw, not kept): "\"install our app\" ka matlab hai." Meta quote; fact not restated.
10. row 5 (shortened_url, raw, not kept): "\"bit.ly\" ka short link hai." Fact correct; quoted.

Verdict: no sentence makes an unsupported factual claim or attributes scam intent (all describe what the SMS says or
asks for). But 2 of the 10 read as natural Hinglish (rows 2 and 23), and the rest contain a wrong or nonsense phrase.
This is a model check by the author, not a native-speaker check.

**Default and why.** `template`. The llm arm does not win: its raw point pass is 8 of 37 (22%), 29 of 37 calls end
unterminated, and the sentences that do pass are often wrong Hinglish. The template arm is fully verifier-checked,
sub-10 ms, and the same decisions. The service runs in `template` mode (`--hinglish-mode` default; the running
service was started in that mode and `/health` reports it).

**Untested.** A native-speaker review of the Hinglish output. Hinglish on hi_deva and other scripts (routed English).
The Hinglish prompt on a different model or with a stop rule that accepts a quote mark. A Hinglish detector
threshold: the student's threshold was not changed for Hinglish. The 65 C gate was never exercised at its limit.

## Untested

- A live 503 cooling refusal (only mocked in `tests/test_edge_service.py`).
- Fix A on more than 10 trial rows: the variant choice rests on 11 points, so it is not a stable ranking.
  The describe_kw echo of "restates the fact" and the 104-token prompt are open.
- Fix B at scale: 2 real hits in 40 rows and 5 in the scan. Quiet mode (the default since the demo decision)
  has not been checked with users either, and it hides these flags from the user by design.
- Hinglish output quality beyond the 10-sentence model check (section 5); no native-speaker review.
- More than 40 requests, a longer soak, and thermal behaviour past 70 C.
- The page in a real phone browser (checked in the desktop browser at 375 px emulation only).
- The relay under concurrent browser connections (the relay is single-request in practice; the service is serial).
- Restart after a crash (the scripts assume a clean stop; a stale pid file is handled by `kill -0`).

## Files

Repo (uncommitted): `src/rakshak/verifier.py` (G1, G2, `risk` opt-in in `check_points` and
`render_advisory`), `src/rakshak/ela_eval.py` (`score_advisory(..., risk=None)`),
`deploy/pi/bench/rbench/slot.py` (`guards`, `g2_rejected`, `point_problems`),
`deploy/pi/bench/rbench/worker.py` and `guard.py` and `cli.py` (`--guards`),
`deploy/pi/bench/sync_bench.sh` (service code), `deploy/pi/service/` (new: `rakshak_edge_service.py`,
`edge_pipeline.py`, `edge_device.py`, `socket_bridge.py`, `eval_service.py`, `index.html`,
`start_service.sh`, `stop_service.sh`), `tests/test_verifier_guards.py`, `tests/test_rbench_guards.py`,
`tests/test_edge_service.py` (new), `docs/EDGE_PIPELINE.md` (this), `docs/LLAMA_PI_RESULTS.md` (guards section).

Pi (not in the repo): `~/rakshak-bench/run/` (pid, log, socket, while running), `~/rakshak-bench/results/edge_eval/`
(`rows.json`, `responses.jsonl`, `summary.json`, `run_meta.json`), and the Part 1 run
`~/rakshak-bench/results/20261106T144210Z-ab584c/`. No file under `models/` was touched.
