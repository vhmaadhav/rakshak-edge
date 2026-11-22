# Fast inference on the Pi 5: verdict-first and the explanation table

Branch `perf/fast-inference`. Measured 2026-11-22 on the Pi 5 (4 GB) at 2.4 GHz with `rakshak-thermald` stopped.
Peak temperature 76.3 °C. All arms were valid (no thermal, throttle or clock trip).
Bench scripts: `deploy/pi/bench/perf/` (see its README). Results are not committed.

## What changed

- **Verdict-first response.** `/analyze` returns the verdict at once. The explanation comes from a table or from
  `GET /explanation/<id>` (`--response-mode verdict_first`). The default stays `sync`.
- **Explanation table.** Sentences are generated once offline by the same Llama runner, per (language, prompt variant,
  type, has-slot). The placeholder `[[SLOT]]` marks where the message's matched text goes. At run time the real slot is
  added as a deterministic value clause (`slot_clause`), so the model never writes the slot. The verifier (`check_points`,
  G2 included) runs at build time per sample message and again at run time on the filled sentence. Failing entries fall
  back to the template. Code: `deploy/pi/service/explanation_table.py`, `deploy/pi/bench/build_explanation_table.py`.
- **Device-read fix.** The full device snapshot costs three `vcgencmd` subprocesses, about 30 ms. The verdict path now
  re-reads it at most every `--device-ttl` seconds.
- **Dedup.** `duplicate_points` in `deploy/pi/bench/rbench/slot.py` checks `real_quick_ratio` and `quick_ratio` before
  the full `SequenceMatcher.ratio()`. These are upper bounds, so the result is identical and the cost is lower.
- **Multi-worker, batch and UDS.** `--workers N` starts N verdict processes sharing the port (SO_REUSEPORT), each with
  its own detector. The explanation store is a Manager dict, so any worker answers `GET /explanation/<id>`.
  `--batch-onnx` batches ONNX calls. `socket_bridge.py` relays 127.0.0.1:8770 to the service's Unix socket.
- **Niced LLM process.** The explainer runs in its own process at `--llm-nice` (default 10), optionally pinned with
  `--llm-cpus`. It is the only process that runs the LLM. Children exit with the supervisor (PR_SET_PDEATHSIG).

Files: `deploy/pi/service/{edge_workers,explanation_table,rakshak_edge_service,edge_pipeline,socket_bridge}.py`,
`src/rakshak/{canonicalize,url_features}.py`, `src/rakshak/models/{features,encoding}.py`.
Tests: `tests/test_{edge_workers,explanation_table,edge_verdict_first}.py`.

## Flags

| Flag | Default | Effect |
|---|---|---|
| `--response-mode {sync,verdict_first}` | `sync` (env `RAKSHAK_RESPONSE_MODE`) | Verdict returned before the explainer finishes. |
| `--explanation-table PATH` | none (env `RAKSHAK_EXPLANATION_TABLE`) | Built table. Absent: every point uses the template. |
| `--device-ttl SECONDS` | `1.0` | Max age of the cached device snapshot on the verdict path. |
| `--workers N` | `1` | verdict_first only: N verdict processes on one port. `1` serves in the supervisor. |
| `--batch-onnx` | off (env `RAKSHAK_BATCH_ONNX=1`) | Batched ONNX inference. |
| `--llm-nice N` / `--llm-cpus LIST` | `10` / unset | Priority and CPU pinning of the explainer process. |
| `--explainer-cache N` | env default | Explainer result cache size. The benches used `0`. |

`--native` is not part of this work.

## Parity

- Decision parity, `parity.py` + `compare.py` on the Pi, real student ONNX: **0 verdict flips on 5,080 messages**
  (both 40-row eval sets plus a seeded 5,000-message sample of `eval_ok` SMS).
- INT8 ONNX was **rejected**: it flips 1 verdict. The FP32 model stays.
- `--batch-onnx`: **0 flips on the Pi** against the per-message path.
- Table entries: the verifier also checks each filled sentence at run time, not only at build time.

## Results (Pi 5, 2.4 GHz)

| Metric | OLD (2b3c1bd) | NEW (verdict_first + table) | Change |
|---|---|---|---|
| e2e p50 / p95 | 794 ms / 1,871 ms | 5.6 ms on table hits (hit rate 90%) | ~140x |
| Throughput, sequential | 1.04 req/s | 250/s | ~240x |
| TTV HTTP p50 (p95, p99) | | 5.5 ms (9.1, 14.2) | 144x |
| Throughput, batch 1024 | | 350/s | |
| Throughput, 4 workers x 16 clients | | 691/s | |
| Throughput, 4 workers x 16 clients, batched | | 1,050/s | ~1,010x vs OLD |
| Miss: LLM, isolated | | 1.2 s | |
| Miss: 20-message burst p50 | | 11.8 s | |

**1.5 GHz column** (rakshak-thermald cap, see gaps):

| Metric | OLD | NEW |
|---|---|---|
| p50 | 1,114 ms | 7.5 ms |
| Throughput | 0.85 req/s | 166/s sequential, 701/s concurrent |

## Known gaps

- **14 of 68 table keys are rejected by the verifier.** Those points fall back to the template. Coverage is lower than
  the table size suggests.
- **Single LLM queue on misses.** Misses wait behind each other: 1.2 s isolated, 11.8 s p50 in a 20-message burst.
- **1.5 GHz cap.** `rakshak-thermald` throttles the Pi (HOT 68 °C, HARD 73 °C). Without a cooler, sustained load
  hits that cap. The 2.4 GHz numbers need the Active Cooler and the 27 W PSU to repeat.

## Operational limits

Request bounds. They are checked before the body is parsed, on `/analyze`, `/analyze_batch`, `/analyze_url` and
`/analyze_flows`. The service binds loopback only (`127.0.0.1`, `::1`, `localhost`); any other address is refused.

| Bound | Default | Flag | Answer when exceeded |
|---|---|---|---|
| Body size | 20 MiB (the flow cap, `nids_pipeline.MAX_BODY_BYTES`) | `--max-body-bytes N` | 413, decided from the header; the body is not read |
| Content-Length | required | none | 411 when missing; 411 for `0` on `/analyze_flows`; 400 when not a non-negative integer |
| Body read deadline | 10 s for the whole body | `--read-timeout S` | 408; a body that stops early is 400 |
| SMS or batch item length | 2,000 characters | none | 400 |
| URL length | 2,000 characters | none | 400 |
| Batch size | 1 to 1,024 messages | none | 400 |

The 20 MiB default is also enough for a full batch of 1,024 messages of 2,000 characters.

Explainer jobs (`--response-mode verdict_first`):

| Limit | Default | Flag | Effect |
|---|---|---|---|
| Queue depth | 64 jobs waiting for the LLM | `--explain-queue-depth N` | A job beyond this is refused. The explainer process has one job in flight on top of the queue. |
| Job and result TTL | 120 s | `--explain-ttl S` | A job still waiting after this is not run. A finished explanation older than this is dropped from the store. |
| Store size | 4,096 finished explanations | `--explanations-kept N` | Oldest finished explanations are dropped first. Pending jobs are not dropped. |

Fallback. The verdict and the template advisory are always in the `/analyze` or `/analyze_batch` answer. Only the explainer
sentence depends on the job:

- Accepted: `"explanation": {"status": "pending", "id": ..., "explanation_source": "llm"}`, read at `GET /explanation/<id>`.
- Queue full: `"explanation": {"status": "fallback", "id": null, "explanation_source": "template", "reason": "overload"}`.
- Job expired: `GET /explanation/<id>` returns `"status": "expired", "explanation_source": "template", "reason": "expired"`.

Counters, in `GET /health`:

- `http_requests_total` and `http_requests_by_endpoint`: one per HTTP request, whatever the answer.
- `batch_requests_total`: one per `/analyze_batch` request.
- `messages_total`: messages scored. One per `/analyze`, and N per batch of N. `/analyze_url` and `/analyze_flows` are not messages.
- `refused_cooling`: 503 answers for the SoC temperature limit. Still counted per verdict process.
- Everything above except `refused_cooling` is service-wide (shared between the verdict processes).
- `requests`: deprecated alias of `http_requests_total`, kept so existing readers keep working. Its old meaning
  (messages plus URL and flow requests, and batch items counted twice) is gone. Read `http_requests_total` or
  `messages_total` instead.
- `explain_jobs.overloaded` and `explain_jobs.expired`: service-wide. Every verdict process adds to one shared set of
  counters, so with `--workers N` the totals are exact whichever process answers.
- `explain_jobs.llm`: the explainer process's own counts (`completed`, `failed`, `expired`).
- `workers`: restarts per supervised process (`llm`, `http0`, ...) and whether it has failed for good.
- `response_mode_config`: the mode in use, its source (`flag`, `env` or `default`), and the default (`sync`).

Response mode. The default is `sync`, as before. `--response-mode` or `RAKSHAK_RESPONSE_MODE` selects `verdict_first`.
`GET /health` shows the mode and where it came from.

Worker restarts. A supervised worker that dies with a failure is restarted. Each death and restart is a JSON line in
the log (`worker_died`, `worker_restarted`) and is counted in `workers`. After 5 restarts of one worker the service
stops (`worker_failed`). When the explainer process restarts, pending jobs older than the TTL are marked expired. A job
the explainer was running when it died is lost, and its client sees `expired` once it is older than the TTL.

## Reproduce

1. Build the table on the Pi: `deploy/pi/bench/build_explanation_table.py --root ~/rakshak-bench --out PATH`.
2. Start with `--response-mode verdict_first --explanation-table PATH --workers 4 --device-ttl 1.0`.
3. Run the arms in `deploy/pi/bench/perf/` (see its README). Check parity with `parity.py` and `compare.py`.
