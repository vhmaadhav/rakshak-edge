# Fast-inference bench scripts (Pi 5)

Scripts used for the numbers in `docs/PERF_FAST_INFERENCE.md`. Scripts only; no result data.

| File | What it does |
|---|---|
| `lib.sh` | Thermal guard, device logging and the arm runner, sourced by the arm scripts. |
| `arm_old.sh` | Arm A: the old baseline service (`/analyze`, sync response mode) at 2.4 GHz. |
| `arm_new.sh WORKERS CLIENT_PER` | Arm B/C: the verdict-first service with table, workers and batch/UDS arms. |
| `armbc.py` | Arms B/C helper: `ttv PASSES` (time-to-verdict pass, writes `ttv_records.json`), `drain`, `http N HITS`, `mp N`. |
| `armbc2.py` | Verdict-first arm helper, called by `arm_new.sh`: `ttv`, `drain`, `seq`, `batch`, `clients`, `batchpar`, `uds`. |
| `llm_one.py` | One isolated explainer call on a table miss (LLM latency and tokens/s). |
| `parity.py` | Decision parity: one process per (tree, mode), one JSON line per message. |
| `compare.py` | Diffs two `parity.py` outputs: field mismatches, verdict flips, max score difference. |

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `PERF_DIR` | `/home/hermoine/rakshak-perf` | Pi-side repo checkout with `src/`, `deploy/pi/service`, `run/`. |
| `TOOLS_DIR` | `/home/hermoine/pf-tools` | Where the helpers write `ttv_records.json` and `hits.json`. |
| `OLD_HOME` / `NEW_HOME` | `/home/hermoine/bl-home` / `/home/hermoine/pf-home` | Home dirs for the old and new trees (each holds `rakshak-bench`). |
| `ONNX_PATH` | `/home/hermoine/rakshak-bench/models/sms_student/best_student.onnx` | Student ONNX used by `armbc.py`. |
| `SMS_PARQUET` | `/home/hermoine/rakshak-bench/data/processed/sms.parquet` | Source SMS parquet (`text`, `eval_ok`). |

## Running

1. Copy this directory to the Pi (for example `~/pf-tools/`, keeping the layout) and set the env vars if the paths differ.
2. Start the service and run the OLD arm: `bash arm_old.sh`.
3. Run the NEW arms: `bash arm_new.sh 4 16` (4 workers, 16 clients). `arm_new.sh 1 ...` also runs the LLM-miss arm.
4. Parity, on the Pi: `python3 parity.py --tree $PERF_DIR --mode new --onnx $ONNX_PATH --out new.jsonl`, the same with `--mode old`, then `python3 compare.py old.jsonl new.jsonl`.

Each arm samples the device before it starts and aborts (marks INVALID) if the temperature reaches 78 °C, the clock drops below 2350 MHz, or any throttle bit is set. Keep `rakshak-thermald` stopped for the 2.4 GHz runs.

The arm scripts also call `start_service.sh`, `stop_service.sh`, `eval_service.py` and `soak.py` from `deploy/pi/service`, and read `results/edge_eval/rows.json` from the bench tree.
