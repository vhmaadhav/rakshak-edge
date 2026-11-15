# URL phishing detector in the edge service: results (2026-11-12)

Plan: `docs/URL_SERVICE_PLAN.md` (pre-registered). Threshold job `maadhavvh/rakshak-url-threshold` v1 (CPU, ok, 5 units).
Outputs are in `kaggle/out/url_threshold/output/`. The system eval is `deploy/pi/service/eval_groundlm.py run --mode quiet`,
run on the laptop with and without `--url-scores … --url-threshold τ_url`.

## Threshold

**τ_url = 0.6308487409530461**: the smallest calibration score with benign FPR ≤ 1 % on the rebuilt 5 % grouped carve
(18 087 rows: 11 347 benign, 6 740 phish; cal FPR 0.996 %, cal recall 89.6 %). It was written before any other set was
scored. The ONNX, scaler and six data files were checked by sha. The scaler refit on the carve's fit rows matches the
shipped scaler.

## Gates

| # | Gate | Result | Pass |
|---|---|---|---|
| U1 | PhreshPhish `official.test` FPR at τ_url | 0.92 % [0.86, 0.99] Wilson (91 260 benign); recall 92.0 %, precision 98.8 % | ✔ |
| U2 | GroundLM quiet-mode visible warnings (1 133) | 2 → **5** (0.44 % [0.19, 1.03]). The 3 new ones are EMI/loan messages with links (3 of the 5 in that bucket). URL/click bucket: 0 of 19 | **✘** (limit 4) |
| U3 | SmishX smishing quiet-mode visible recall (226) | 53 → **181** (23.5 % → 80.1 %) | ✔ |
| U4 | Pi parity, service path, 300 rows | **bit-identical, 0 flips**: 300/300 laptop (x86_64, ORT 1.31.0) = Pi (aarch64, ORT 1.31.1); 0 flips at τ_url vs Kaggle and between machines | ✔ |
| U5 | Pi added latency p95 | **added p95 = 2.3 ms** (decision time, verify 10.5 ms vs quiet 8.2 ms, n=95 each, LLM excluded); 300 `/analyze_url` server total p95 5.8 ms. Thermal caveat below | ✔ (≤ 50 ms) |

**U2 fails, so the pre-registered fallback applies:** URL-model evidence on `/analyze` is shown in **verify mode
only**. The shipped quiet mode keeps today's behaviour (2 / 1 133 visible, SmishX visible recall 23.5 %).
`POST /analyze_url`, where the user asks about a link explicitly, returns the URL verdict in every mode.

## Reported, not gated

- SmishX legitimate, quiet-mode visible FPR with the URL path on: 22 → 52 / 606 (3.6 % → 8.6 %). Of the 57 legit
  SmishX messages with a URL, 43 cross τ_url.
- `url_domain` test (out of domain, our non-PhreshPhish URL table): **FPR 69.4 %** at τ_url, recall 94.4 %. The
  threshold transfers within PhreshPhish but not to our other benign URL sources. That is the main limitation of
  this model and the reason quiet mode stays off.
- Feature skew (2 000 PhreshPhish test URLs inside a template SMS): `canonicalize` returns the dataset string for
  99.3 %, the max score difference is 3.4e-4, and there are 0 flips at τ_url.
- The campaign-test rows in the system eval are unchanged, because the URL job did not score campaign test. URL
  effects on campaign test are not measured.
- PhreshPhish base-rate APs (unchanged from `docs/BASELINE.md`): 0.484 / 0.599 / 0.809 / 0.865 / 0.941 at
  0.05 / 0.1 / 0.5 / 1 / 5 %.

## What a different choice would take (not done)

Making URL evidence visible in quiet mode needs a new pre-registration. Two possible ways: a stricter τ_url (for
example 0.1 % cal FPR), or showing URL evidence only when the SMS student or a rule agrees. Either way it must be
judged on GroundLM and on an independent legit-with-links set, given the 69 % out-of-domain FPR.

## U4 (laptop, 2026-11-12)

Script `deploy/pi/service/url_parity.py` (tests in `tests/test_url_parity.py`). It scores the 300 rows of
`kaggle/out/url_threshold/output/parity_300.parquet` with the service path: `canonicalize(url).urls`, then
`best_url_hit` on a scorer with an empty cache (one ONNX call per message, as in `/analyze_url`). ONNX and scaler are
the `kaggle/out/url_student/output` files, sha-checked (1bbd2c57…, c6b3fc96…). Laptop: x86_64, Python 3.12.15,
numpy 2.5.4, onnxruntime 1.31.0. Laptop Tctl was 40–68 °C during the run (never near 85).

- Raw path (`score_urls([url])` on the dataset string): 300 / 300 float32 bit-equal to the Kaggle `score` column,
  max abs diff 0.0.
- Service path (`canonicalize` then score): max abs diff vs Kaggle 1.45e-5 (canonicalized string differs from the
  dataset string on some rows, the known feature skew). **0 flips at τ_url** (300 rows, all with a URL).

## U4 Pi half and U5 (Pi, 2026-11-12)

Code on the Pi: `deploy/pi/bench/sync_bench.sh --code-only` (175 files, models not touched). Model files on the Pi
(`~/rakshak-bench/models/url_student`) pass the sha pins (1bbd2c57…, c6b3fc96…).

**U4 Pi.** `deploy/pi/service/url_parity.py` on the Pi, same 300 rows (`run/parity_300.parquet`), service code path.
Result (`--compare`, laptop vs Pi): raw path 0 / 300 float32 bit mismatches, service path 0 / 300 mismatches, **0 flips at
τ_url**. Pi: aarch64, Python 3.13.5, ORT 1.31.1. Pi SoC 46.9–50.7 °C, throttle 0x0 for the batch.

**U5.** Service at 3 threads (`start_service.sh --threads 3`), stopped afterwards. Latency numbers come from the
server's own timings (response `timings_ms`, and the service log `total_ms`, `llm_total_ms`). Client wall time is
in the table row (a). First 5 requests of each batch are not recorded (warm-up).

| Measure | n | p50 ms | p95 ms | Note |
|---|---|---|---|---|
| (a) `/analyze_url` server total (canonicalise + URL + rules) | 295 | 4.79 | 5.75 | client wall p95 81 ms (through the relay) |
| (b) `/analyze` decision, **verify** mode, 100 URL SMS | 95 | 8.01 | 10.50 | total minus explainer; 146 explainer calls |
| (b) `/analyze` decision, **quiet** mode, same 100 SMS | 95 | 6.50 | 8.16 | URL scorer never called in quiet mode; explainer calls 82 in chunks 2–5 (chunk 1 not summed) |

**Added p95 (verify − quiet) = 2.35 ms ≤ 50 ms: U5 passes.** Mean difference 1.48 ms. The verify batch also used
the URL path on all 100 messages, so this is the cost of the URL scorer inside `/analyze` (the explainer is excluded
from the decision time).

Thermal caveat (not met as planned): the rule was to start a batch below 55 °C and abort at 65 °C. Each batch started
below 55 °C, and the throttle flag stayed 0x0 throughout. But the SMS batches were not checked between requests, and
their logged `temp_start_c` went above 65 °C: verify batch max 71.4 °C, quiet batch max 65.9 °C. The explainer makes
these batches heat the Pi. Both batches were therefore run hotter than the rule allows, and the verify batch was
hotter than the quiet one, so the quiet/verify comparison is confounded by temperature (the verify decision times are
higher, which if anything makes the URL cost look larger). The URL-only batch (a) stayed at 57.6 °C max.

## Pi state

Service: stopped at the end (relay and service pids gone, port 8770 closed). Last SoC reading 52.9 °C, throttle 0x0.
No files on the Pi were changed except `~/rakshak-bench` code (sync_bench) and `run/` scratch (parity_300.parquet,
parity_urls.json, u5_*.json, url_parity_pi.json). No sudo, no pip install, no config changes.

## Thermal note on U5 (2026-11-12)

The U5 batches each started below 55 °C with throttle `0x0`, but the temperature was not checked between requests,
and the logged start temperatures of the later SMS batches reached 71.4 °C (verify) and 65.9 °C (quiet). That is
above the project's 65 °C abort rule. There was no throttle event and the Pi ended at 52.9 °C. Because the verify batch
ran hotter than the quiet one, the 2.35 ms added-p95 figure is temperature-confounded. It is far under the 50 ms gate
either way. Later Pi runs go through `deploy/pi/thermal_run.py` with `--kill 65`.

## Corroborated URL evidence in quiet mode (amendment 2026-11-12 night): FAILS U2

`eval_groundlm.py run --mode quiet --url-scores … --url-threshold 0.6308 --url-quiet-policy corroborated` (laptop).
URL evidence visible in quiet mode only when the SMS student is over 0.35128 or a rule fired.

| Gate | Result | Pass |
|---|---|---|
| U2 GroundLM quiet visible (1 133) | 2 → **5** (0.44 % [0.19, 1.03]) | ✘ (limit 4) |
| U3 SmishX smishing quiet visible (226) | 53 → **143** (63.3 %) | ✔ |

SmishX ham quiet visible FPR 22 → 39 / 606 (3.6 % → 6.4 %). The 3 extra GroundLM warnings are the same EMI/loan messages
with links: the student is over its threshold on them, so "corroboration" does not filter them. Quiet mode keeps the
`off` policy (shipped default). The `--url-quiet-policy` option and `edge_decision.URL_QUIET_POLICY` stay in the code
with default `off`.
