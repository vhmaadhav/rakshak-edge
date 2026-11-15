# Encoder on the Pi: W1d results (mmBERT-small seed 0, fp16 and nbits8)

Status: **W1d not measured to completion. The gate cannot be evaluated from these runs.** Both
variants hit the bench's own 64 C stop during the first sequence length. The numbers below are
partial and are labelled as such. Procedure: `docs/SOTA_PUSH_PLAN.md` ("W1d procedure"). Prepared and
run 2026-11-12. Nothing committed.

## Inputs (all hashes verified on laptop and Pi)

| File | Bytes | sha256 (first 8) | Source |
|---|---|---|---|
| `model_fp16_storage.onnx` | 478,526,907 | c556a0e2 | Kaggle `maadhavvh/rakshak-sms-encoder-export-small-s0`, parity passed |
| `model_nbits8_b32.onnx` | 441,925,896 | 86be24e2 | same kernel; decision agreement 1.0000, max abs dp 0.0285 (gate 0.02, reported as a variant) |
| `encoder_rows_cal500.npz` | 522,964 | ce9eb3f4 | built on laptop: 500 rows, seed 0, from `source_holdout:holdout_scamshield` cal (2,754 rows); 9 truncated at 128; token p50 33, p95 89 |
| `tokenizer.json` / `tokenizer_config.json` | 34,363,441 / 573 | f231fae3 / 02055b88 | same kernel (not in manifest; used only to build rows) |

Pi location: `~/rakshak-bench/models/encoder/small_s0/`. Runtime: onnxruntime 1.31.1, provider
`CPUExecutionProvider`, 3 threads, batch 1, sequential execution, graph optimisation `ORT_ENABLE_ALL`.

## Changes made to get the bench to run

- `deploy/pi/bench/build_encoder_rows.py` and `deploy/pi/bench/encoder_bench.py`: `main()` called
  `parse_args(list(argv or []))`, so running the script directly saw no arguments and exited with a usage
  error. Both now pass `None` through, so argparse reads `sys.argv`. No other logic changed.
- `tokenizers==0.23.3` installed into the laptop `.venv` (needed by the row builder only).
- Code synced to the Pi with `sync_bench.sh --code-only` (178 files, twice: before and after the fix).

## Pre-run checks

- Laptop: Tctl 39 to 57 C during the row build, well under 85 C.
- Pi before each start: temp 47.7 to 49.4 C, throttle `0x0`, `rakshak-edge` inactive, no
  `thermal_run`, `encoder_bench` or `rbench` process.
- `~/rakshak-bench/results/.rbench.lock` exists (0 bytes, dated 6 Nov). It is **not held**
  (`flock -n` succeeded). The guard uses a persistent lock file, so its presence is normal. I did not
  delete it. Flagged for awareness; it does not affect these runs.

## Runs (all under `thermal_run.py --pause 63 --resume 58 --kill 65`, bench `--max-temp 64`)

| Run | Start C | Max C (bench) | End C | Pauses | Status | Samples at stop |
|---|---|---|---|---|---|---|
| fp16, attempt 1 | 49.9 | 64.2 | 63.7 | 0 | stopped at 64 C, exit 3 | seq 32 only, p50 50.01, p95 50.62, p99 51.13 ms |
| fp16, attempt 2 (after cooldown to 46 C) | 47.7 | 64.2 | 64.2 | 0 | stopped at 64 C, exit 3 | seq 32 n=81: p50 50.39, p95 50.74, p99 51.50 ms |
| nbits8_b32 (after cooldown to 46 C) | 49.4 | 64.2 | 64.8 | 2 (3.1 s paused) | stopped at 64 C, exit 3 | seq 32 n=65: p50 92.71, p95 94.31, p99 864.9 ms |

Stop reason in each case: the SoC reached 64.2 C during the seq-32 pass, before seq 64 or 128 started.
The guard's own CSV shows the cause: under 3-thread inference the SoC rises about 2 C/s
(48.5, 50.7, 50.1, 55.6, 59.0, 60.0, 60.6, 62.8 C over 8 s). The 64 C stop is reached in about 8 to 10 s.
The procedure's rows-of-200 run, at 3 seq lengths, needs about 20 minutes, which this heating rate
does not allow without pauses or a cooler.

The nbits8 run had two guard pauses. Per procedure step 6, a run with a pause is discarded, so its p99 is not
a valid figure. Its seq-32 p50 and p95 are shown for completeness only.

## Memory

Peak RSS is the encoder's `VmHWM` from the bench JSON.

| Variant | Encoder peak (VmHWM) | Llama explainer peak (ADDED, not measured together) | Estimated sum | Budget | Fit (estimate) |
|---|---|---|---|---|---|
| fp16 | 902.4 MB (924,016 kB) | 1,839.3 MB (`docs/BASELINE_PI.md`) | 2,741.7 MB | 3,796 MB | fits by estimate |
| nbits8_b32 | 867.2 MB (888,016 kB) | 1,839.3 MB | 2,706.5 MB | 3,796 MB | fits by estimate |

The 3,796 MB budget is a derived rule from the procedure (4,049 MB total minus the 253 MB idle `used` read).
It is still to be confirmed as a gate. The estimate is a sum and was not measured with the Llama
explainer running. A real co-run is the follow-up.

## Latency gate (p95 at most 400 ms per SMS, batch 1, 3 threads)

- Only seq 32 was measured, for both variants. The procedure's gate uses `realistic.latency_ms.p95`
  from the 500 real rows. That pass was **skipped** in both runs (`realistic.skipped: true`).
  **The gate cannot be evaluated.**
- For seq 32, p95 was 50.7 ms (fp16) and 94.3 ms (nbits8, pause-affected). Both are far below 400 ms,
  but seq 32 is the shortest bucket and most SMS sit at seq 33 to 89 tokens.
- Seq 64 and 128 were not measured.

## Temperature and throttle flags

- Pi start and end temps are listed in the table above. Throttle flag `0x0` before and after every run.
  The Pi's `vcgencmd get_throttled` read `0x0` at all points, so no under-voltage flags were seen.
- Cooldown before the second and third starts: one bounded check (sleep 120 s), 46.3 C. The first check
  read 58.4 C after the first attempt and 57.9 C after the nbits8 run.
- End state: Pi at 51.8 C, throttle `0x0`, `rakshak-edge` inactive, no bench process running.

## ORT MatMulNBits

- The session ran on `CPUExecutionProvider` only. No fallback or unsupported-op warning was found in
  the bench logs.
- The bench does not profile kernels, so whether ORT used the MatMulNBits kernels (versus a
  dequantise-then-MatMul path) is **not verified** by these runs. A kernel profile is needed to confirm.

## Raw files

- Pi: `~/rakshak-bench/run/encoder-small_s0-fp16.attempt1-stopped-at-64C.json`,
  `~/rakshak-bench/run/encoder-small_s0-fp16.attempt2-stopped-at-64C.json`,
  `~/rakshak-bench/run/encoder-small_s0-nbits8_b32.json`, plus the guard CSV and log under `~/rakshak-bench/run/`.
- Laptop copies: `bench_results/encoder/` (gitignored).

## Verdict and what would unblock W1d

W1d is **not passed**. Neither the memory fit nor the p95 gate has a valid measurement for the full run.
The fit is an estimate and the p95 gate was never reached.

The Pi has no cooler and runs on USB power (`docs/STATUS.md`, 2026-11-11). Under the procedure's
thermal limits, a 20-minute run cannot complete in one cycle. Options for Maadhav to decide:

1. Fit the cooler and 27 W supply (already pending in `docs/STATUS.md`), then rerun.
2. Allow a cooldown between sequence lengths in the bench (a procedure change; needs approval).
3. Measure at seq 32 only, with the realistic pass reduced, and label the result partial.

Nothing has been committed.

## Procedure amendment (2026-11-12, before re-running)

Reason: the continuous-load procedure in `docs/SOTA_PUSH_PLAN.md` ("W1d procedure") reaches the bench's
64 C stop in about 8 s on the cooler-less Pi, so it cannot measure the real workload. Real SMS arrive one
at a time. This amendment changes how the load is applied. The gates, the 400 ms p95 gate and the
3,796 MB fit rule are unchanged, and so are the thermal guard settings and the PI RULES (no sudo, no pip
install, no config changes).

1. **Realistic set under a duty cycle.** The 500-row set (`encoder_rows_cal500.npz`, real token lengths,
   one pass) runs in bursts of 10 messages, batch 1, 3 threads. The 500 rows are the same as in the
   first attempt. Warmup is 5 untimed messages before the first burst.
2. **Cooldown before each burst.** Before every burst the SoC must be below 55 C. Each wait is bounded at
   120 s. A wait that ends at 120 s without reaching 55 C is a non-cooling wait. After 3 consecutive
   non-cooling waits the run is aborted and recorded as stopped early.
3. **Burst abort at 63 C.** A reading of 63 C or more during a burst aborts that burst. The burst is
   discarded and recorded (index, temperature, messages run). The run continues after the next cool
   wait. The bench's own `--max-temp 64` stays as the hard stop.
4. **Stall rule (added).** A single message taking more than 1000 ms also discards its burst and is
   recorded. This catches a `thermal_run` SIGSTOP at 63 C, which would otherwise inflate one latency.
5. **Latency only inside bursts.** Tokenisation and cooldown waits are not timed.
6. **Reported.** p50, p95 and p99 per variant on the realistic set; token-length distribution; latency by
   length bucket (1-32, 33-64, 65-96, 97-128); peak RSS (VmHWM); the added Llama estimate against 3,796 MB;
   max temperature; throttle flags before and after; discarded bursts; the duty cycle achieved (measured
   messages per minute, cooldown waits included); the gate verdict (p95 at most 400 ms, and fits).
   The duty cycle is an upper bound on the rate in practice, since there are no idle gaps between bursts
   other than the cooldown waits.
7. **Synthetic sweep (secondary, fp16_storage only).** Seq 32, 64 and 128: one burst of 20 messages per
   length, with the same cooldown rule and the same 63 C abort.
8. **Order.** `fp16_storage` (primary, parity passed) first. Then `nbits8_b32` (reported). The two runs are
   sequential, and the Pi must be below 55 C before each one.

Implementation: `encoder_bench.py` gets `--burst N`, `--cool-below C` (default 55), `--max-cool-wait S`
(default 120), `--burst-abort-c C` (default 63) and `--only realistic|sweep|all` (default all). Without
`--burst` the original continuous behaviour is unchanged.

Pass rule, unchanged: `realistic.latency_ms.p95` at most 400 ms, and the added estimate at most 3,796 MB.

## Results under the amended procedure (2026-11-12)

Runs: `deploy/pi/bench/encoder_bench.py --rows encoder_rows_cal500.npz --threads 3 --burst 10 --cool-below 55
--max-cool-wait 120 --burst-abort-c 63 --only realistic --warmup 5 --max-temp 64`, under
`thermal_run.py --pause 63 --resume 58 --kill 65`. Batch 1, ORT 1.31.1, CPUExecutionProvider. Raw JSON on the Pi:
`~/rakshak-bench/run/encoder-small_s0-fp16-burst.json`, `~/rakshak-bench/run/encoder-small_s0-nbits8_b32-burst.json`,
`~/rakshak-bench/run/encoder-small_s0-fp16-sweep.json`.

### Realistic set (500 rows, one pass, 50 bursts of 10)

| Variant | n | p50 ms | p95 ms | p99 ms | Duty (msgs/min) | Wall s | Bursts discarded | Cool waits (non-cooling) | Max SoC C | Start / end C | Throttle |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16_storage (primary) | 500 | 53.97 | **115.23** | 165.69 | 143.3 | 209.3 | 0 of 50 | 50 (0) | 62.0 | 47.7 / 59.3 | 0x0 before and after |
| nbits8_b32 (reported) | 500 | 97.77 | **162.42** | 210.91 | 74.7 | 401.8 | 0 of 50 | 50 (0) | 62.6 | 54.3 / 60.9 | 0x0 before and after |

Latency by token-length bucket (ms; p50 / p95):

| Bucket | n (fp16) | fp16 | n (nbits8) | nbits8 |
|---|---|---|---|---|
| 1-32 | 237 | 41.5 / 56.3 | 237 | 82.8 / 105.5 |
| 33-64 | 202 | 63.0 / 86.4 | 202 | 108.2 / 132.7 |
| 65-96 | 42 | 101.2 / 121.2 | 42 | 147.2 / 169.6 |
| 97-128 | 19 | 165.4 / 166.8 | 19 | 208.0 / 214.6 |

Token lengths (500 rows, after truncation at 128): min 3, p50 33, p95 89, mean 38.8. Untruncated max 220;
9 rows truncated at 128.

Memory (encoder only, VmHWM): fp16 924,608 kB (902.9 MB); nbits8 888,640 kB (867.8 MB).

Added Llama estimate (not measured together; companion 1,839.3 MB from `docs/BASELINE_PI.md`):

| Variant | Encoder peak MB | + Llama MB | Sum MB | Budget MB | Fits (estimate) |
|---|---|---|---|---|---|
| fp16_storage | 902.9 | 1,839.3 | 2,742.2 | 3,796 | yes |
| nbits8_b32 | 867.8 | 1,839.3 | 2,707.1 | 3,796 | yes |

### Gate verdict

- Latency gate (realistic p95 at most 400 ms): **PASS** for both variants (fp16 115.2 ms, nbits8 162.4 ms).
- Memory fit (estimated sum at most 3,796 MB): **PASS by estimate** for both (2,742 and 2,707 MB). Not measured
  with the Llama explainer running; the co-run remains the follow-up.
- Burst behaviour: no burst was discarded in either run; every cooldown wait reached below 55 C. No guard pauses
  (`thermal_run` pauses 0 in fp16; nbits8 also no pause). The Pi ended at 56.2-56.8 C, throttle 0x0,
  `rakshak-edge` inactive, no bench process running.

### Synthetic sweep (secondary, fp16_storage, one burst of 20 per length): INCOMPLETE

- Seq 32: one burst of 20, no discard. p50 50.11, p95 60.71, p99 64.62 ms.
- Seq 64: the run reached **66.4 C** and stopped at the bench's 64 C hard stop before any seq-64 message was
  timed. The seq-64 burst was not discarded at 63 C, so the 63 C abort did not catch the rise.
- Seq 128: not run.
- Cause (code, to fix before any re-run): the 5 warmup messages run before each burst loop and are not
  temperature-gated, so the 63 C abort cannot see the heat they add. The hard stop then fires on a reading at
  66.4 C. The fix is to gate warmup too, and rerun the sweep with that change.

### Projection for mmBERT-base (~307 M params): estimate only, not measured

- Parameters: base fp32 file 1,230,787,007 B (`kaggle/out/sms_encoder_export-base-teacher/output/results.json`),
  so about 307.7 M at 4 bytes each. Vocab 256k and hidden 768 are assumed for the embedding, giving 196.6 M
  embedding and about 111.1 M non-embedding. Small: 140.6 M total (98.3 M embedding, 42.3 M non-embedding,
  from the initializers of `data/raw/encoder_small_s0/model_fp16_storage.onnx`).
- Latency scaling by non-embedding FLOP ratio (111.1 / 42.3 = 2.62, embedding lookups excluded): fp16 p50
  about 142 ms, p95 about 302 ms. This is an optimistic estimate: it ignores memory-bandwidth and attention
  growth, and the base's larger weights make the real figure likely higher. It stays below 400 ms on this
  basis, but the margin is smaller.
- RSS: fp16_storage keeps the embedding in fp32 and the layers in fp16 (as in the small file). Base file
  estimate 962 MB, plus the small model's measured runtime overhead of 447 MB (VmHWM minus file size), gives
  about 1.41 GB. Adding Llama's 1,839 MB gives about 3.25 GB, under 3,796 MB by estimate. The fp32 file
  (1,231 MB) gives about 1.62 GB, and about 3.46 GB with Llama, also under budget by estimate.
- Base `fp16_storage` parity: **PASS** (kernel `maadhavvh/rakshak-sms-encoder-export-base-teacher`, relaunched after the protobuf fix): decision agreement 1.0000, max |dp| 0.0016, max |dmargin| 0.105, Spearman 0.999999, file 1 009 021 459 B (fp32 1 230 787 007 B). nbits8 b32/b128: decision 1.0000 but max |dp| 0.089 / 0.071 (fail the 0.02 gate); nbits4 fail; dynamic INT8 0.829 (constant-scam failure, as for small). The base model on the Pi is NOT measured yet; the projection above stands as an estimate.

### Not verified

- ORT's MatMulNBits kernel use (not profiled).
- The sweep for seq 64 and 128 (see above).
- Llama co-run RSS.
- Base latency and RSS (projection only).

## mmBERT-base fp16_storage (measured)

Status: **INCOMPLETE. Stopped early at the 64 C hard stop after 110 of 500 rows. The gate is not evaluable.**
Run 2026-11-12, under the amended procedure (realistic set, bursts of 10, cool below 55 C, burst abort at 63 C,
3 threads, batch 1, warm-up 5). Nothing committed.

Model: kernel `maadhavvh/rakshak-sms-encoder-export-base-teacher`, `model_fp16_storage.onnx` (1,009,021,459 B,
sha256 `ab5b9563`, matches `export_manifest.json`). Downloaded only the four requested files
(`kaggle kernels output --file-pattern`), into `data/raw/encoder_base_teacher/` (gitignored). Rows: the same 500
seeded cal rows as small_s0 (`ce9eb3f4`, identical bytes, since the small and base tokenizers are the same
`tokenizer.json`, `f231fae3`). Pi copy: `~/rakshak-bench/models/encoder/base_teacher/`, both hashes verified on the Pi.

Before and after:

| Point | SoC C | Throttle | Other |
|---|---|---|---|
| Start (before copy and run) | 47.7 | 0x0 | `rakshak-edge` inactive, no bench process, 3,775 MB available |
| End (after stop) | 55.1 | 0x0 | `rakshak-edge` inactive, no bench process |

Code change in the same session: warm-up is now temperature-gated (`run_warmup` in `encoder_bench.py`): cool below
55 C first, end early at the 63 C burst abort, stop the run on a cooldown abort. Tests added
(`tests/test_encoder_bench.py`, 45 pass). Ruff clean. Synced with `sync_bench.sh --code-only`.

### Latency (measured, kept messages only)

| n kept | p50 ms | p95 ms | p99 ms | Mean ms |
|---|---|---|---|---|
| 110 of 500 | 114.9 | 224.4 | 267.4 | 122.4 |

By token-length bucket (kept messages only):

| Bucket | n | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|
| 1-32 | 59 | 84.2 | 115.5 | 133.1 |
| 33-64 | 43 | 145.5 | 196.9 | 216.2 |
| 65-96 | 7 | 234.5 | 264.4 | 267.8 |
| 97-128 | 1 | 327.6 | n/a | n/a |

Bias: discarded bursts are not counted, and the kept set is skewed to short messages (59 of 110 are 32 tokens or
fewer, while the full 500-row set has median 33 tokens, p95 89). The partial p95 is not comparable to the full set.

### Bursts and duty

- 45 bursts of 10. 11 kept, 34 discarded: 24 at 63.1 C, 9 at 63.7 C, 1 at the 64 C hard stop (the last burst, index 44).
- Warm-up: 5 messages, untimed, no abort.
- Cooldown: every wait ended below 55 C (non-cooling waits 0). Each wait was 1 to 10 s. The SoC went from 57 to 63 C
  between bursts, so bursts kept starting hot.
- Duty achieved: 19.5 messages per minute over 338.8 s (110 kept messages, cooldown waits included). The 500-row
  set would need about 26 minutes at this rate, which is above the 10-minute headroom the procedure requires.
- `thermal_run`: exit 3, state done, pauses 0, paused 0 s. Its own max reading was 62.8 C. The bench's JSON max is
  64.2 C. The bench samples per message and the guard samples once per second, so the 64.2 C reading was a
  short spike the guard did not record.

### Memory (estimate, not measured together)

| Variant | Encoder peak (VmHWM) MB | + Llama MB (`docs/BASELINE_PI.md`) | Sum MB | Budget MB | Fits (estimate) |
|---|---|---|---|---|---|
| fp16_storage (base) | 1,782.9 | 1,839.3 | 3,622.2 | 3,796 | yes, margin 174 MB |

Pi free RAM before loading: 3,775 MB available (above the 2,000 MB abort floor).

### Gate verdict

- Latency gate (realistic p95 at most 400 ms): **NOT EVALUABLE**. The run stopped at 110 of 500 rows. The partial
  p95 of 224 ms is shown for reference only.
- Memory fit (estimated sum at most 3,796 MB): **PASS by estimate** (3,622 MB). Not measured with the Llama
  explainer running.

### Comparison with the projection (`docs/SOTA_PUSH_PLAN.md`, base projection in the section above)

| Quantity | Projected | Measured | Note |
|---|---|---|---|
| p50 ms | 142 | 114.9 (partial) | Partial, skewed short |
| p95 ms | 302 | 224.4 (partial) | Partial, skewed short |
| Encoder RSS | about 1,410 MB | 1,782.9 MB (VmHWM) | Measured is about 370 MB higher |
| Sum with Llama | about 3,250 MB | 3,622.2 MB (estimate) | Still under 3,796 MB, with less margin |
| Duty / completion | none assumed | 110 of 500 rows, stopped | The 63 C burst abort discards most bursts |

### Flags and not verified

- Throttle flags: 0x0 at start, during and at the end.
- The run did not complete, so no synthetic sweep was run (the procedure allows it only after a finished realistic
  run with the Pi below 55 C and 10 minutes of headroom).
- Discarded bursts: 34 of 45 (listed above).
- ORT MatMulNBits: not relevant to fp16_storage; kernel use not profiled.
- Raw JSON on the Pi: `~/rakshak-bench/run/encoder-base_teacher-fp16-burst.json`. Laptop copy:
  `bench_results/encoder/encoder-base_teacher-fp16-burst.json` (gitignored).

### What would unblock this

The 63 C abort, at 5 to 10 s of cooldown per burst and about 1 C of heating per burst, cannot finish 500 rows
at this duty on the cooler-less Pi. Options for Maadhav to decide: fit the cooler and 27 W supply (pending in
`docs/STATUS.md`), then rerun; or change the procedure (longer cooldown, smaller bursts, or a lower abort
threshold). Any change needs approval first.
