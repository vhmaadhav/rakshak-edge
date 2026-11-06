# rbench: benchmark Rakshak models on the Pi 5

`rbench` runs one registered model on one registered dataset on the Pi, under the thermal guard,
and writes a versioned JSON result with latency, quality, and system telemetry. Agents use the CLI
(`--json`, stable exit codes). Humans use the **Benchmarks** section of the Pi dashboard.

Code: `deploy/pi/bench/` (package `rbench`). Registry: `deploy/pi/bench/registry.toml`.
Tests: `tests/test_rbench_*.py`. Pi side: `~/rakshak-bench` (repo `src/`, `deploy/pi/`, plus `models/`,
`data/`, `results/` and `.venv/`).

## Setup

1. Sync the code and the registered files from the laptop (never deletes, skips files whose sha256
   already matches, never writes under `models/llama32_1b_spinquant/`):

   ```bash
   deploy/pi/bench/sync_bench.sh --dry-run   # list what would move
   deploy/pi/bench/sync_bench.sh             # copy (code, then files)
   ```

2. Create the Pi venv once (no torch): `python3 -m venv ~/rakshak-bench/.venv` and install
   `onnxruntime numpy pandas pyarrow`. Optional for the LLM runner: `executorch`.

3. Check the machine and the artifacts:

   ```bash
   deploy/pi/bench/rbench-remote doctor --json
   ```

## Commands

Run on the Pi, or through the laptop wrapper (`rbench-remote <args>` passes everything through):

| Command | What it does |
| --- | --- |
| `rbench list models [--json] [--no-hash]` | Registered models, status (ready, not downloaded, sha mismatch, disabled), compatible datasets |
| `rbench list datasets [--json] [--no-hash]` | Registered datasets and status |
| `rbench run --model M --dataset D [--limit N] [--threads T] [--warmup K] [--seed S] [--name NAME] [--max-new-tokens N] [--json]` | Run one benchmark under the thermal guard; writes `results/<run_id>/` and one line in `results/index.jsonl` |
| `rbench results [--last N] [--json]` | Recent runs from the index |
| `rbench show RUN_ID [--json]` | The full `result.json` of one run |
| `rbench compare RUN_A RUN_B [--json]` | Metric deltas; reports whether the two runs are comparable (same dataset, model, threads) |
| `rbench fetch MODEL` | Download a pinned prebuilt `.pte` into the bench root and verify its sha256. Refuses models with no `hf` entry |
| `rbench doctor [--json] [--no-hash]` | Python, imports, versions, CPUs, RAM, temperature, throttling, git sha, artifact states |
| `rbench manifest [--json]` | Files `sync_bench.sh` sends (used by the sync script) |

Laptop wrapper: `deploy/pi/bench/rbench-remote pull` copies the Pi's `results/` into
`bench_results/` (gitignored). It never deletes local results.

Example:

```bash
deploy/pi/bench/rbench-remote run --model sms_student --dataset sms_campaign_test --limit 300 --threads 1 --json
deploy/pi/bench/rbench-remote compare <run_a> <run_b> --json
```

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | ok |
| 2 | bad usage, incompatible model and dataset, a disabled entry, or an unknown id |
| 3 | missing artifact, sha mismatch, or a missing dependency (e.g. no onnxruntime, no executorch) |
| 4 | the thermal guard killed the run (80 °C) |
| 5 | runtime error, or the bench lock is held by another run |

With `--json`, errors print `{"error": "...", "exit_code": N}` on stdout.

## Thermal guard

Every `run` goes through the guard (`deploy/pi/thermal_run.py`): pause at 75 °C, resume at 65 °C,
kill at 80 °C. A lock file (`results/.rbench.lock`) allows one benchmark at a time. The guard's
per-second telemetry becomes the `thermal` block of the result.

## Result schema (`rbench/1`)

`results/<run_id>/result.json`, plus `items.jsonl` (per-item rows), `telemetry.csv` and `run.log`.
Top-level keys:

- `schema`, `run_id`, `ts`, `name`, `status` (`ok`, `error`, `thermal_killed`), `error`
- `model`: `id`, `kind`, `task`, `path`, `sha256` per file, `threshold`, `notes`
- `dataset`: `id`, `sha256`, `n_file`, `n_filtered`, `n_used`, `denominator_notes`
- `params`: `limit`, `threads`, `warmup`, `seed`, `max_new_tokens`
- `env`: Python, package versions, kernel, git sha, CPU summary
- `perf`: `load_s`, `latency_ms` (p50, p95, p99, mean), `infer_ms`, `throughput_items_s`,
  `peak_rss_mb`, `latency_scope`, `warmup_items`
- `llm`: `ttft_ms`, `prefill_tok_s`, `decode_tok_s`, `json_parse_rate`, `nonempty_rate`, and null
  fields for the verifier metrics (not measured yet). Null for non-LLM runs
- `quality`: `task`, `key_metric` (`{name, value}`), per-task block (threshold metrics, ROC-AUC,
  average precision, FPR with Wilson 95% CI). `key_metric` is null when the run measures no quality
- `thermal`: `temp_start_c`, `temp_max_c`, `temp_end_c`, `throttled_before`, `throttled_after`,
  pauses, power and clock summary
- `parity`: max and mean logit difference against the stored logits, when the model has them

`index.jsonl` rows are a compact subset (run id, status, model, dataset, n, p50, p95, key metric,
max temperature, throttling). `rbench results` and the dashboard read those rows.

Always report the denominator with a quality number. `groundlm_fpr` reports the scoreable count
(1,133 of 1,294 policy-negatives in the current file), and `n_used` says what was scored.

## Registry

`registry.toml` lists `[[models]]` and `[[datasets]]`. Paths are relative to the bench root.

Model fields: `id`, `kind` (`onnx_sms`, `onnx_url`, `onnx_nids`, `pte_llm`), `task` (implied by the
kind), `enabled`, `path`, `sha256`, `threshold` (score >= threshold is scam), `notes`, plus
`source` (the laptop file `sync_bench.sh` copies). Sidecars: `side_scaler`, `side_scaler_source`,
`side_scaler_sha256`, `parity_logits`, `parity_logits_source`, `parity_logits_sha256`, `extras`
(each with its own `path`, `sha256`, `source`), and `hf` (repo, revision, files) for downloads.

Dataset fields: `id`, `task` (`sms_binary`, `url_binary`, `groundlm_fpr`, `nids_latency`,
`ela_prompts`), `enabled`, `path`, `sha256`, `source`, `split_file`, `split_sha256`,
`split_source`, `split_part` or `split_column` and `split_value`, `row_filter` (`none`,
`eval_mask`, `policy_negative_scoreable`), `columns`, `synthetic`, `notes`.

Rules:

- A model runs only on datasets with a compatible task (`COMPATIBLE` in `rbench/registry.py`).
  Otherwise `run` exits 2 and names the datasets it accepts.
- A disabled entry must say why in `notes`.
- A blank `sha256` means the file is not pinned. `doctor` reports that state.
- `source` blank means the file is not synced from the laptop (for example the Llama files, which
  the Pi downloads itself).

To add a model: export the ONNX file to `kaggle/out/...` (or wherever the laptop copy lives), compute
its sha256, add a `[[models]]` block with `source` and the threshold from its training metrics,
then run `sync_bench.sh` and `rbench-remote list models`. To add a dataset, use the same steps with
`[[datasets]]`. Keep the feature code in `src/rakshak`, not in rbench.

## Dashboard

The Pi dashboard (`deploy/pi/pi_dashboard.py`, port 8765) has a **Benchmarks** section. It lists
recent runs from `index.jsonl`, shows a run's details, and starts a run from a form. Its API is
`/api/bench/{options,runs,run/<id>,status}` (GET) and `/api/bench/run` (POST). Only 127.0.0.1 and ::1
clients may use it, and it runs one benchmark at a time (409 while one is running).
