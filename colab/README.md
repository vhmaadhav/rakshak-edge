# Colab job launcher

The Colab launcher runs one rakshak job on a Google Colab VM. It is the Colab counterpart of
`kaggle/launch.sh`: the same job files (`kaggle/jobs/<job>.py`), the same bundle of `src/rakshak/`,
the same results.json, and the same ntfy completion line, sent as `rakshak-colab-<job>`.

The design has no long-lived connection and no polling. Each step is a short `colab` call:

1. `colab/launch.sh JOB ...` (start): new session, optional data upload, upload of the runner, then a
   short `colab exec` that starts the runner **detached** on the VM and returns. The VM keeps running.
2. `colab/wait.sh JOB` (completion): one blocking ntfy subscription (`kaggle/wait_event.py --platform colab`).
   When the completion line arrives, it runs `colab/finish.sh JOB`.
3. `colab/finish.sh JOB` (finish): packs `/content/out` on the VM, downloads it, extracts it, and stops the
   VM. `--abort` stops the VM without downloading.

The CLI is `google-colab-cli` (`colab`, version 0.7.5 when this was written), installed with
`uv tool install google-colab-cli` into `~/.local/bin`. Official docs:
`gh api repos/googlecolab/google-colab-cli/readme --jq .content | base64 -d`.

## Authentication (Maadhav does this once)

The launcher never authenticates for you. Run this by hand first, because it is interactive:

```bash
colab usage        # read-only: shows the compute-unit balance; triggers the login on first use
```

Default `--auth oauth2` (what the launcher uses):

1. The CLI prints an authorization URL. It uses the remote copy-paste flow with redirect
   `https://sdk.cloud.google.com/applicationdefaultauthcode.html`.
2. Open the URL, sign in with the Google account that has Colab, copy the code Google shows.
3. Paste the code at the CLI prompt.
4. The refresh token is kept in the OS keyring (service `google-colab-cli`, user `oauth_token`). If the
   keyring is locked or unavailable, the CLI asks again on every command.

Optional config: `-c PATH` (default `~/.colab-cli-oauth-config.json`). When that file is missing, the
CLI uses its bundled OAuth client config. Local state: `~/.config/colab-cli/sessions.json` (sessions),
`~/.config/colab-cli/settings.json`.

Alternative `--auth adc` (Application Default Credentials):

```bash
gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
export RAKSHAK_COLAB_AUTH=adc      # makes launch.sh pass --auth adc to every colab call
```

`userinfo.email` is required by the session backend (without it assign, unassign and sessions return
HTTP 401). The `--auth` flag is global, so it must come before the subcommand.

`colab usage` is the check that auth works. (`colab whoami` exists but is hidden from `--help`.)

## Cost rule

- Default GPU is **T4**. It runs without a flag.
- **Any other GPU** (`L4`, `A100`, `H100`), and **`--high-mem`**, spend compute units. The launcher refuses
  them unless `--yes-paid` is passed. Maadhav gives that approval per run; never add `--yes-paid` on your
  own. The refusal happens before the VM is requested.
- The VM is stopped on any failure of the start phase (EXIT/INT/TERM trap). On success it is **kept running**
  until `colab/finish.sh` (or `colab/wait.sh`, which calls it) stops it. `finish.sh` always stops the VM,
  even when the download fails.
- `colab usage` is printed before the start and after each finish, so the spend is visible in the logs.

## Usage

```bash
# Dry run: prints every colab command and the runner build; executes no colab command.
colab/launch.sh sms_teacher --gpu T4 --data --budget-h 8 --job-args="--models microsoft/mdeberta-v3-base --no-fp16" --dry-run

# Start (returns in about a minute; the job runs on the VM afterwards):
colab/launch.sh sms_teacher --gpu T4 --data --budget-h 8 --job-args="--models microsoft/mdeberta-v3-base --no-fp16"

# Completion, once: one blocking subscription, then finish.sh. Run it in the background; it returns on the event.
colab/wait.sh sms_teacher                 # default timeout: the wall-clock limit + 30 min
colab/wait.sh gpu_smoke --timeout 900     # explicit timeout in seconds

# Only if the run must be abandoned (or the VM was reclaimed): stop it without downloading.
colab/finish.sh sms_teacher --abort

# Fetch and stop by hand, after a completion you saw elsewhere (for example in the ntfy app):
colab/finish.sh sms_teacher

# Only some files, instead of the job's default set (quote globs; '!' removes matches):
colab/launch.sh url_student --data-files urls.parquet 'splits/url_*.json' --dry-run
```

Flags of `launch.sh`: `JOB` (lowercase, letters digits `_` `-`), `--gpu T4|L4|A100|H100` (default T4), `--cpu`
(no accelerator: `colab new` gets no `--gpu`; cannot be combined with `--gpu`), `--high-mem`, `--job-args="FLAGS"`
(passed to the job as `ctx.argv`), `--budget-h H` (the job's own budget in hours, the same value as in
`--job-args`; the runner's hard limit is H plus 30 min), `--data` (uploads the job's default data, see below),
`--data-files PATTERN...` (only the files matching the patterns; implies `--data`; it takes every word up to the
next `--option`), `--yes-paid`, `--dry-run`.

Flags of `finish.sh`: `JOB`, `--abort`, `--dry-run`. Flags of `wait.sh`: `JOB`, `--timeout SECONDS`, `--dry-run`.

Environment: `RAKSHAK_COLAB_AUTH` (`adc` adds `--auth adc` to every colab call), `RAKSHAK_COLAB_CHUNK_MB` (upload
part size in MiB, default 32, allowed 1 to 50).

Default data per job (patterns are relative to `data/processed`):

| Job family | Patterns | Raw size | Archive, parts |
|---|---|---|---|
| `sms_*` | `sms.parquet`, `splits/*.json` except `url_*`, `external/**` | 46.5 MiB | 34.3 MiB, 2 parts |
| `url_*` | `urls.parquet`, `urls_phreshphish.parquet`, `splits/url_*.json` | 150.1 MiB | 111.8 MiB, 4 parts |
| `llama_*` | `sms.parquet`, `splits/campaign.json` | 33.2 MiB | 27.5 MiB, 1 part |
| any other job | everything in `data/processed` | | |

The sizes are from a dry run on the laptop with the current `data/processed`. Each family's default set
contains the marker file its jobs look for (`sms.parquet`, `urls.parquet` or `urls_phreshphish.parquet`).

Each launch uses a new session name, `rakshak-<job>-<yymmddHHMMSS>`, so a failed `colab new` cannot stop
another run's session. The session is recorded in `colab/out/JOB/session.json` (job, session name, accelerator,
launch epoch, budget, wall-clock limit, `state`: `running`, then `finished` or `aborted`).

## What a run does

1. `kaggle/build_kernel.py --platform colab --job JOB --wall-limit-h H` writes `colab/out/JOB/kernel/run.py`, with
   the bundle, the git label, the tree hash, the notify topic and the hard limit embedded. H is `--budget-h` plus
   0.5, or 11.5 when `--budget-h` is not given.
2. `--data` or `--data-files`: `colab/data_bundle.py build` packs the selected files into
   `colab/out/JOB/data/rakshak-data.tar.gz`, cuts it into `rakshak-data.part-NNN` files of at most
   `RAKSHAK_COLAB_CHUNK_MB` MiB, and writes `manifest.json` (sha256 and size of the archive, of each part and of each
   file). `colab upload` sends the manifest and each part; `colab/remote_unpack_data.py` checks each part, joins them,
   checks the archive, extracts it with `filter='data'` and checks every file, into `/content/rakshak-data`.
   Nothing is extracted if a check fails.
3. `colab new -s SESSION --gpu GPU [--high-mem]` (no `--gpu` with `--cpu`), then `colab upload` of `run.py` to
   `/content/rakshak_run.py`.
4. `colab exec -f colab/remote_start_job.py` (short, timeout 120 s). It starts `/content/rakshak_run.py` with
   `subprocess.Popen` in its own session (`start_new_session`), stdout and stderr appended to
   `/content/out/runner.stdout.log`, PID in `/content/out/runner.pid`. It waits one second: a runner that exits in
   that second is a failed start, and the VM is stopped. Then it returns.
5. The runner (on the VM, a fresh Python process) installs the pinned pip set, unpacks the bundle, runs the job, writes
   `/content/out/results.json` and POSTs the completion line `rakshak-colab-<job> ok|partial|fail git=... tree=...`.
   Its status is `error` when the job raises. Its hard limit is a watchdog thread: past `--wall-limit-h` the run is
   recorded as `error` with `WallClockLimit`, the completion is sent, and the process exits, even while the job is
   stuck in a C call. Whichever of the job and the watchdog ends the run first writes the result, once.
6. `colab/wait.sh` blocks on the ntfy topic (it replays cached events since `launch_epoch`), then `colab/finish.sh`:
   `colab exec remote_pack_out.py` packs `/content/out`; `colab download` saves it to
   `colab/out/JOB/output/out.tar.gz`; `colab stop` runs; the tarball is extracted to `colab/out/JOB/output/`, so the
   results are in `colab/out/JOB/output/out/results.json`. Every failure before the stop still reaches the stop.

Everything is logged to `colab/out/JOB/launch.log` (start) and `colab/out/JOB/finish.log` (finish). `colab/out/` is
gitignored.

## If a run is lost

Colab free sessions can be reclaimed, and a reclaimed VM never sends a completion line. `colab/wait.sh` then exits 124
after its timeout and stops nothing. Check once:

```bash
colab sessions                          # is the session still listed?
colab/finish.sh JOB --abort             # if it is gone or the run is dead: marks the session aborted, stops it
```

A dropped ntfy stream (exit 1 from `wait.sh`) costs nothing: nothing is stopped, and rerunning `colab/wait.sh JOB`
replays the cached completion line.

## Files

- `launch.sh`: start phase (args, paid-GPU refusal, budget, data, runner upload, detached start, session record, traps).
- `wait.sh`: one blocking completion subscription, then `finish.sh`.
- `finish.sh`: pack, download, always stop, extract; `--abort` stops only.
- `data_bundle.py`: laptop-side selection, archive, chunking and manifest for `--data` and `--data-files`.
- `remote_unpack_data.py`, `remote_pack_out.py`: small standard-library scripts run on the VM by `colab exec -f`.
- `remote_start_job.py`: the detached starter run on the VM by `colab exec -f`. It returns after the runner has started.
- `out/`: generated runner, data archive, session record, downloaded output, logs (gitignored).
- Shared with Kaggle: `kaggle/build_kernel.py --platform colab` (runner template with the watchdog),
  `kaggle/wait_event.py --platform colab`, `src/rakshak/eval/data.py` (`COLAB_DATA` candidate).

## Limits

- Live check (2026-10-30, `gpu_smoke` on a free T4, detached design): start phase returned in under a minute with the
  runner detached (pid 571); the completion arrived as `rakshak-colab-gpu-smoke ok`; `finish.sh` fetched the output
  and stopped the VM. results.json: status `ok`, GPU `Tesla T4, 15360 MiB`, torch 2.11.0+cu130, total 42.9 s.
  `colab sessions` afterwards: none.
- Earlier live check: `smoke --cpu --data-files splits/campaign.json` ran end to end on a CPU VM with the blocking
  design (results.json status `ok`, 3/3 samples, VM stopped). The detached design is covered by
  `tests/test_colab_launcher.py` (fake CLI, watchdog run for real) and by the live `gpu_smoke` check above.
- The runner's exit code is no longer seen by the launcher: the start cell returns before the job ends. The result is
  the completion line (`ok`, `partial`, `fail`) and `results.json`. `wait.sh` exits 1 on `fail`.
- The VM's Python must have `tarfile` extraction with `filter='data'` (Python 3.12 and later) for `--data`.
- Upload size, measured with one free CPU session (`colab new -s rakshak-probe`, no GPU, then stopped): 10, 25 and
  50 MiB files uploaded; 100 MiB got `400 Client Error: Bad Request`. Parts are 32 MiB by default, so each one is below
  a size that worked.
- Remote paths: a relative remote path (`probe-rel-1m.bin`) uploaded without error but did not show up in `/content`,
  so the launcher always uploads to absolute `/content/...` paths.
- The data unpack is still a blocking `colab exec` (timeout 900 s). It runs before the runner starts, so a dropped
  connection there ends the start phase and the VM is stopped.
- Colab can disconnect or reclaim a VM at any time; see "If a run is lost".
