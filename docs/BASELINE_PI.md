# Rakshak-Edge baseline v0: Raspberry Pi 5 measurements

Companion to `docs/BASELINE.md` (frozen v0). This file holds the Pi half: provenance, score parity with the laptop,
device performance, explainer quality on the Pi's INT4 `.pte`, and the NIDS device row. Measured 2026-11-07,
09:37 to 09:55 IST, on the Pi (`hermoine@pi5.local`). Raw files are on the Pi under
`~/rakshak-bench/results/baseline_v0/` (laptop side: none copied into the repo).

Conventions: CI is a percentile bootstrap 95% interval over the requests (5,000 resamples, seed 0), unless marked
otherwise. "p50 [CI]" means the point estimate with its interval. Power is the PMIC ADC sum (V x A over the Pi 5
rails), not a calibrated meter.

## Caveats (read first)

- **No cooler.** Passive board on a weak USB supply (`usb_max_current_enable=1` per BASELINE). The SoC reached 70.8 C
  in the soak. The governor (`rakshak_thermald`) capped the clock to 1500 MHz during it. No throttle flag (`0x0`) was
  seen at any sample. Numbers are for this setup only.
- **PMIC is not a calibrated meter.** Energy in J is an integral of PMIC rail power over each request. At 4 ms per
  request (Hinglish template path) the per-request energy is near the reading floor, so those J values are noise-level.
- **Clock.** At first contact `date` read 2026-11-07 09:37:57 IST and `timedatectl` said `System clock synchronized: yes`.
  The Pi had just booted (uptime 1 min). Timestamps in the raw files are from that clock.
- **Code state is WIP, not clean v0.** See Provenance. The SMS path is v0 plus remediation fields. The NIDS row is WIP.
- **Only one service start for the whole batch.** Llama load time and the RSS numbers are from that start.
- **Two Hinglish rows in the English batch.** The English 40 was sent with the service default (`lang` not sent,
  which means `auto`), so 2 of its rows routed to the Hinglish template path. Counted as they came out.

## Provenance

| Item | Value | Check |
|---|---|---|
| Laptop git HEAD | `ac45aeaa2f10fba7de7d9a2a8c8441f716136192` (`main`) | `git rev-parse HEAD` |
| Laptop working-tree diff | `git diff` sha256 `84f3f2efe9df9d6c7827d4d55f1283a58af67c6890e82d06a0e11fa2ed332e6f` (9 tracked files; untracked files not in it) | `git diff \| sha256sum` |
| Code on the Pi | `sync_bench.sh --code-only` from the laptop tree at the time above. 149 files under `src/rakshak`, `deploy/pi/bench`, `deploy/pi/service`, `pi_common.py`, `thermal_run.py`. Per-file sha256 list identical to the laptop | `sha256sum` of each file, compared |
| Pi extra file | `deploy/pi/service/netns_exec.sh` (sha256 `ddb58c9d...`) exists on the Pi only, left from an earlier sync. Not in the service path | listed, not used |
| SMS source files (v0 hashes) | `rules.py` 9bde4ae2..., `canonicalize.py` 892a89ad..., `evidence.py` df8300c6..., `verifier.py` f10890b5... | identical on the Pi, the laptop, and `git show ac45aea:` |
| SMS path diff vs v0 | `deploy/pi/service/edge_pipeline.py` adds `remediation` and `remediation_ok` fields. One v0 line (the `verifier` dict) is reformatted with the new key. No decision logic changed | `git diff ac45aea -- deploy/pi/service/edge_pipeline.py` |
| Artifact sha256 on the Pi | SMS student `5341e14c...2466` (v0); Llama `.pte` `8715cdba...cda3b` (v0, `Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte`); NIDS INT8 `d3dfdcb1...8b9e14` (v0); URL student `1bbd2c57...213fc7` and side scaler `c6b3fc96...5b6501e` (both equal the BASELINE values; loaded only by the parity script) | `sha256sum` on the Pi |
| Service start | `start_service.sh --threads 3` (run with `bash`; the first attempt under `sh` failed at a bash test and started nothing). Defaults: `detector_only_mode=quiet`, `hinglish_mode=template`, `temp_limit=75.0`. Threads reported `3` | `/health` at start and at end |
| Service at the end | pid 1487, status `ok`, `requests=300`, `refused_cooling=0`, detector_only `quiet`, hinglish `template`. Left running | `/health` |

Model load times at the service start: Llama `load_s` 15.77 (method `model_load_ms` 2585); SMS 0.875 s; NIDS 0.008 s.
An earlier start (pid 51388, stale, from before this session) reported Llama `load_s` 7.56 (`model_load_ms` 2558).

## Section A: score parity, Pi vs laptop

Sample: fixed seed 0 (`numpy.random.default_rng(0).permutation`), drawn with the repo's `rbench` loaders
(`load_rows`, registry `eval_mask` and `policy_negative_scoreable` filters).

- SMS, 300 rows: 200 from `sms_campaign_test` (campaign test partition, eval_ok and not synthetic, pool 22,145) and 100
  from `groundlm_fpr` (policy negatives with text, pool 1,133). Labels: 95 positive (all from the campaign rows), 205 negative.
- URL, 300 rows: from `url_domain_test` (pool 30,432), 214 positive, 86 negative.
- Laptop: `.venv` (Python 3.12, onnxruntime 1.31.0). Pi: `~/rakshak-bench/.venv` (Python 3.13.5, onnxruntime 1.31.1).
  Both use the repo's own featurisation (`rbench.runners.onnx`) and 1 intra-op thread.

| Model | n | max abs prob diff | mean abs prob diff | max abs logit diff | Decision threshold | Flips | Positives (both) |
|---|---|---|---|---|---|---|---|
| SMS student (v0) | 300 | **0.0** | **0.0** | **0.0** | 0.35127517580986023 (v0) | **0 of 300** | 132 |
| URL student (`url_student`) | 300 | **0.0** | **0.0** | **0.0** | 0.5 (see note) | **0 of 300** | 269 |

Result: zero flips, and the outputs are bit-identical on this sample, so the Kaggle quality numbers stand for the Pi
(same ONNX bytes, same featurisation, same arithmetic in this run).

Notes on the parity result:
- The **URL threshold**: `url_student` has no registered threshold (BASELINE.md, and `registry.toml` notes). The
  thresholds in `docs/URL_THREE_SEED_RESULTS.md` belong to other URL units. The 0.5 used here is a placeholder, so the
  URL flip count only tests device agreement, not decisions.
- The SMS sample is for parity, not quality. Within it, 34 of 100 GroundLM policy negatives score at or above the
  v0 threshold (the same on the laptop and the Pi), and 4 of 105 campaign negatives do. Those 34 are a device-side
  observation, not a quality claim.

## Section B: device performance (v0 settings, 3 threads)

Service at 3 threads, default modes. All runs: 0 aborts, 0 HTTP 503 (the 75 C gate was not reached, max SoC 70.8 C).

### Idle power (60 s, service loaded, no requests)

| Measure | Value |
|---|---|
| PMIC power, mean (sd) | **1.558 W** (0.221) |
| Range over 60 1 Hz samples | 1.328 to 2.253 W |
| SoC during idle | 42.2 to 45.5 C |
| Throttle flags | `0x0` |

File: `results/baseline_v0/idle_power.json`.

### B1, English 40 rows (`results/edge_eval/rows.json`) and B2, Hinglish 40 rows (`results/edge_eval_hinglish/rows.json`)

Same service, same run order (B1 then B2), sent with `eval_service.py run --out`. B1 is the default (`lang` not sent).
B2 sent `--lang auto`. Files: `results/baseline_v0/edge_en/`, `results/baseline_v0/edge_hinglish/`.

| Measure | B1 English 40 (all) | B1 evidence 30 | B2 Hinglish 40 (template) |
|---|---|---|---|
| Request total p50 [CI] (ms) | 1 393.5 [1 252, 1 573] | 1 570.3 [1 394, 1 712] | 4.5 [4.0, 5.4] |
| Request total p95 [CI] (ms) | 2 931.6 [1 901, 4 576] | 3 010.9 [1 930, 4 576] | 7.9 [6.0, 13.4] |
| Request total p99 [CI] (ms) | 3 993.2 [2 073, 4 576] | 4 142.5 [2 082, 4 576] | 12.0 [6.9, 13.4] |
| LLM calls (total) | 31 (27 ok, 4 unterminated) | 31 | 0 |
| Per LLM call p50 [CI] (ms) | 1 558.1 [1 390, 1 611] | same | n/a |
| Per LLM call p95 [CI] (ms) | 1 967.1 [1 819, 2 098] | same | n/a |
| TTFT p50 [CI] / p95 (ms) | 1 101 [1 092, 1 142] / 1 217.5 | same | n/a |
| Generated tokens: per request (mean) / per call (mean, p50) | 4.7 / 6.06, 7 | 6.27 / 6.06, 7 | 0 |
| Decode tok/s, aggregate over calls | 13.38 | same | n/a |
| Energy per request, PMIC raw, p50 [CI] (J) | 6.70 [5.4, 7.5] | 7.5 [6.7, 8.1] | 0.10 |
| Energy per request, idle-subtracted, p50 [CI] (J) | 4.30 [3.4, 5.1] | 5.1 [4.3, 5.3] | ~0.09 (noise floor) |
| Energy per request, raw p95 (J) | 14.6 | 15.8 | 0.1 |
| J per generated token, raw / idle-subtracted | 1.25 / 0.85 | 1.25 / 0.85 | n/a (no tokens) |
| Modes (llm / partial / template) | 21 / 1 / 18 | 21 / 1 / 8 | 0 / 0 / 40 |

Per-request energy at 4 ms is near the PMIC reading floor (about 0.1 J per request): the B2 J values are not a
measurement of template work.

### Soak, 200 requests at 3 threads (`soak.py run --threads 3 --requests 200`)

Back-to-back, 40 rows cycled, no idle gap. Files: `results/baseline_v0/soak_t3/` (`requests.jsonl`, `samples.jsonl`,
`summary.json`, `stdout.log`).

| Measure | Value |
|---|---|
| Accepted / attempts / 503 / aborts | 200 / 200 / 0 / 0 |
| Request total p50 [CI] / p95 [CI] / p99 [CI] (ms) | 1 817.7 [1 673, 1 914] / 3 824.9 [2 567, 4 443] / 6 119.3 [4 264, 6 563] |
| Per LLM call p50 [CI] / p95 [CI] (ms, n = 155) | 1 962.8 [1 886, 2 035] / 2 547.9 [2 444, 2 555] |
| Modes (llm / partial / template) | 105 / 5 / 90 |
| Generated tokens: per request (mean), total | 4.7, 940 |
| Decode tok/s, aggregate | 11.89 |
| Energy per request, raw p50 [CI] / p95 (J) | 6.4 [6.1, 6.8] / 15.0 |
| Energy per request, idle-subtracted p50 [CI] / p95 (J) | 3.5 [3.2, 3.7] / 8.7 |
| J per generated token, raw / idle-subtracted | 1.21 / 0.70 |
| Requests per minute, whole run | 36.7 |
| SoC at request start, min / max (C) | 49.4 / 70.8 |
| Clock cap (governor) start / min | 2 400 / 1 500 MHz; first cap reduction at 19.5 s |
| Throttle flags seen | `0x0` only |

### Llama load, RSS, SoC summary

| Measure | Value |
|---|---|
| Llama load, this start (wall, `load_s`) | 15.8 s (method load `model_load_ms` 2 585) |
| Llama load, earlier start | 7.6 s (method load 2 558) |
| RSS, steady (after start / after 300 requests) | 1 655.8 MB / 1 679.9 MB |
| RSS, peak (`VmHWM`) | 1 839.3 MB |
| SoC at request start, min / max (B1, B2) | 45.5 to 68.6 C (B1); 48.8 to 52.1 C (B2) |
| SoC max during batches | B1 69.2 C; B2 53.2 C; soak 70.8 C |
| Throttle flags, all runs | `0x0` |
| 503 refusals, all runs | 0 |

Wall time: B1 51.4 s for 40 requests; B2 3.1 s; soak 326.9 s.

## Section C: explainer quality on the Pi's INT4 `.pte` (from B1 and the soak)

Counted by `eval_service.py report` (points on called advisories) and checked by hand from `responses.jsonl`.

| Measure | This run (B1, English) | Fix D run in BASELINE / EDGE_PIPELINE | Difference |
|---|---|---|---|
| Evidence rows / rows called | 30 / 29 | 30 / 29 | none |
| Raw point pass (model sentence passes verifier), on called points | **27 of 33 (82%)** | 26 of 33 (79%) | +1 point |
| Evidence rows where every point passes raw | 23 of 30 | not reported | new |
| LLM share of final points (llm source, called points) | **25 of 33 (76%)** | not reported in this form | template 8 of 33 |
| Advisory mode, evidence 30 (llm / partial / template) | 21 / 1 / 8 | 22 / 1 / 7 | one more advisory on template |
| Unterminated calls | **4 of 31 (12.9%)** | 4 of 45 (8.9%, Fix C run; Fix D not reported) | not comparable (other denominator) |
| G2 rejections | **1**: "This message promises guaranteed investment returns and is likely an investment scam." (`attribution of scam intent at MEDIUM: scam`) | 1 sentence ("is likely an investment scam", Fix C) | same type, different row |
| Final contract pass, evidence 30 / all 40 | **30 of 30 / 40 of 40** | 30 / 30; 40 / 40 | none |
| Cleanup (echo stripped / duplicates blanked) | 2 / 0 | 2 / 0 | none |
| Soak (200 requests) modes llm / partial / template | 105 / 5 / 90 | n/a | n/a |
| Hinglish (B2) path / raw pass | template 40 of 40; raw template pass 37 of 37 | 37 of 37 template pass (v0 Hinglish table) | none |

Differences flagged: the raw pass rose by one point and one advisory moved to template. The model is run with the
same prompt on the same `.pte`, so the difference is either run-to-run variation in generation or the WIP service
code (remediation fields only in the SMS path). The run did not separate the two. The unterminated rate is higher than
the Fix C figure, but the denominators differ and BASELINE does not give the Fix D unterminated count.

The 4 unterminated calls are the status the service records ("no sentence end before the token budget"). Each has
`stop_hit=true` in the record, which the service does not explain in its field name. Not resolved here.

## Section D: NIDS device row (WIP code, not v0)

Synthetic pcaps from `deploy/pi/service/make_test_pcaps.py` (default profile, seed 0), built on the laptop with scapy
(files only, nothing sent) and copied to the Pi. 5 repeats of each through `POST /analyze_flows` over the service's
Unix socket. The network path is not part of v0 (BASELINE.md). Files: `results/baseline_v0/nids_device/`
(`requests.jsonl`, `summary.json`). Pi pcaps: `results/baseline_v0/pcaps/`.

| pcap | Flows | Flow source | Risk, alerts | Request total p50 (ms), range | Flows/s p50 | PMIC energy p50 (J) | 503 |
|---|---|---|---|---|---|---|---|
| benign | 14 | pcap_nfstream | UNCERTAIN, 0 alerts | 75.7 (75.3 to 307.1) | 185 | 0.24 | 0 of 5 |
| port_scan | 100 | pcap_nfstream | HIGH, 1 alert | 98.4 (94.9 to 98.9) | 1 017 | 0.31 | 0 of 5 |
| ssh_brute | 60 | pcap_nfstream | HIGH, 1 alert | 91.5 (86.7 to 109.2) | 656 | 0.28 | 0 of 5 |
| dns_long | 40 | pcap_nfstream | HIGH, 1 alert | 78.4 (78.2 to 78.8) | 510 | 0.24 | 0 of 5 |

Label this row "WIP code, not v0". The benign pcap gives UNCERTAIN with no alerts, not LOW: that is an observation
about the WIP network decision on synthetic traffic, not a claim about real traffic. n = 5 per row, so the ranges are
min to max rather than a CI.

## Pi rows for BASELINE.md

| Row | Pi value (v0 artifacts) | Denominator and config | Source on the Pi |
|---|---|---|---|
| SMS score parity, Pi vs laptop | max abs prob diff 0.0; mean 0.0; **0 flips** at 0.35127517580986023 | 300 SMS (200 campaign test, 100 GroundLM policy neg); 1 intra-op thread on both devices | `results/baseline_v0/parity/parity_pi.json`, `parity_inputs.json` |
| URL score parity, Pi vs laptop | max abs prob diff 0.0; mean 0.0; 0 flips at 0.5 (placeholder, no v0 threshold) | 300 URLs, url_domain test | same |
| Idle power | 1.558 W mean (sd 0.22) | 60 s, SoC 42 to 46 C, service loaded | `idle_power.json` |
| Request total p50 / p95 / p99, English 40 (all) | 1 394 [1 252, 1 573] / 2 932 [1 901, 4 576] / 3 993 [2 073, 4 576] ms | 40 requests, 3 threads, 0 503 | `edge_en/responses.jsonl`, `edge_en/summary.json` |
| Request total p50 / p95, English evidence 30 | 1 570 [1 394, 1 712] / 3 011 [1 930, 4 576] ms | 30 evidence rows | same |
| Request total p50 / p95, Hinglish 40 (template) | 4.5 [4.0, 5.4] / 7.9 [6.0, 13.4] ms | 40 rows, no LLM | `edge_hinglish/` |
| Request total p50 / p95 / p99, soak | 1 818 [1 673, 1 914] / 3 825 [2 567, 4 443] / 6 119 [4 264, 6 563] ms | 200 requests, 3 threads, 0 aborts | `soak_t3/summary.json` |
| Per LLM call p50 / p95 | 1 558 [1 390, 1 611] / 1 967 [1 819, 2 098] ms (B1, n = 31); 1 963 / 2 548 ms (soak, n = 155) | calls | same |
| TTFT p50 / p95 | 1 101 / 1 218 ms | 31 calls, B1 | `edge_en/responses.jsonl` (`llm_calls`) |
| Decode tok/s (aggregate) | 13.4 (B1); 11.9 (soak) | calls | same |
| Tokens per request (mean) | 4.7 (all 40 B1; soak); 6.3 (evidence) | 188 tokens in B1, 940 in soak | same |
| Energy per request, raw p50 / p95 (J) | 6.7 / 14.6 (B1 all); 6.4 / 15.0 (soak) | PMIC, not calibrated | same |
| Energy per request, idle-subtracted p50 (J) | 4.3 (B1 all); 3.5 (soak) | idle 1.558 W | same |
| J per generated token, raw / idle-subtracted | 1.25 / 0.85 (B1); 1.21 / 0.70 (soak) | | same |
| Llama load | 15.8 s wall at this start (method 2.6 s); 7.6 s earlier | 3 threads | `run/edge_service.log` (ready event) |
| RSS steady / peak | 1 656 to 1 680 MB / 1 839 MB | service | `/health` |
| SoC max / throttle / 503 | 70.8 C (soak), 69.2 C (B1); `0x0` throughout; 0 refusals | all runs | `soak_t3/samples.jsonl`, `run_meta.json` |
| Explainer raw point pass (evidence) | 27 of 33 (82%) | 33 called points, 30 evidence rows | `edge_en/summary.json` |
| Explainer LLM share of final points | 25 of 33 (76%) | 33 called points | same |
| Explainer advisory mode (evidence 30) | llm 21, partial 1, template 8 | 30 rows | same |
| Unterminated calls | 4 of 31 (12.9%) | 31 calls | same |
| G2 rejections | 1 | 40 rows | same |
| Final contract pass | 30 of 30 evidence; 40 of 40 | | same |
| Hinglish explainer (template) | 40 of 40 template; raw template pass 37 of 37 | 40 rows | `edge_hinglish/summary.json` |
| NIDS flows/s p50, synthetic (WIP) | benign 185; port_scan 1 017; ssh_brute 656; dns_long 510 | 5 repeats each; WIP code | `nids_device/summary.json` |
| NIDS request p50 (WIP) | 76 / 98 / 92 / 78 ms | same | same |

## Open items

- Not measured: an ambient-temperature or cooled run; a second soak at other thread counts (BASELINE's v0 already has
  2, 3 and 4 threads from the earlier session); a Llama-side prompt-injection run on the Pi (BASELINE still has it missing).
- The explainer numbers come from one B1 run plus the soak. They are not averaged over repeats.
- The GroundLM flag rate (34 of 100 policy negatives at the v0 threshold) is a device-parity observation from this
  sample. It needs a full evaluation before it is compared with any quality target.
