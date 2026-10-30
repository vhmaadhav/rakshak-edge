# Colab job launcher

`colab/launch.sh` runs one rakshak job on a Google Colab VM. It is the Colab counterpart of
`kaggle/launch.sh`: the same job files (`kaggle/jobs/<job>.py`), the same bundle of `src/rakshak/`,
the same results.json, and the same ntfy completion line, sent as `rakshak-colab-<job>`.

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
- Every launch stops its VM. `colab stop` runs in an EXIT/INT/TERM trap once `colab new` is requested,
  and again explicitly after the download. A failed run still downloads what `/content/out` holds.
- `colab usage` is printed before and after each launch so the spend is visible in the log.

## Usage

```bash
# Dry run: prints every colab command and the runner build; executes no colab command.
colab/launch.sh sms_teacher --gpu T4 --data --job-args="--models microsoft/mdeberta-v3-base --no-fp16" --dry-run

# Real run (blocking; run it in the background from the caller, never poll it):
colab/launch.sh sms_teacher --gpu T4 --data --job-args="--models microsoft/mdeberta-v3-base --no-fp16"

# Only some files, instead of the job's default set (quote globs; '!' removes matches):
colab/launch.sh url_student --data-files urls.parquet 'splits/url_*.json' --dry-run

# Completion, once: the same replay-and-wait script as Kaggle, with the Colab prefix.
kaggle/wait_event.sh sms_teacher --platform colab
```

Flags: `JOB` (lowercase, letters digits `_` `-`), `--gpu T4|L4|A100|H100` (default T4), `--cpu` (no
accelerator: `colab new` gets no `--gpu`; cannot be combined with `--gpu`), `--high-mem`,
`--job-args="FLAGS"` (passed to the job as `ctx.argv`), `--data` (uploads the job's default data, see
below), `--data-files PATTERN...` (uploads only the files matching the patterns; implies `--data`; it
takes every word up to the next `--option`), `--yes-paid`, `--dry-run`. Environment: `RAKSHAK_COLAB_AUTH`,
`RAKSHAK_COLAB_TIMEOUT` (runner timeout in seconds, default 43200; colab exec's own default is 30 s),
`RAKSHAK_COLAB_CHUNK_MB` (upload part size in MiB, default 32, allowed 1 to 50).

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
another run's session. The session name appears in `launch.log`.

## What a run does

1. `kaggle/build_kernel.py --platform colab --job JOB` writes `colab/out/JOB/kernel/run.py`, with the
   bundle, the git label, the tree hash and the notify topic embedded. No kernel metadata is written.
2. `--data` or `--data-files`: `colab/data_bundle.py build` packs the selected files into
   `colab/out/JOB/data/rakshak-data.tar.gz`, cuts that archive into `rakshak-data.part-NNN` files of at most
   `RAKSHAK_COLAB_CHUNK_MB` MiB, and writes `manifest.json` (the sha256 and size of the archive, of each part and
   of each file). Then `colab upload` sends the manifest to `/content/rakshak-data.manifest.json` and each part to
   `/content/rakshak-data.part-NNN`, and `colab/remote_unpack_data.py` checks each part, joins them, checks the
   archive, extracts it with `filter='data'` and checks every file's size, into `/content/rakshak-data`. Nothing is
   extracted if a check fails. Old parts in `colab/out/JOB/data` are removed before each build.
3. `colab new -s SESSION --gpu GPU [--high-mem]` (no `--gpu` with `--cpu`). Then `colab upload` sends
   `run.py` to `/content/rakshak_run.py`, and ONE blocking `colab exec -f colab/remote_run_job.py` starts
   it in a fresh Python process (`sys.executable`, not the VM kernel). Its stdout and stderr are printed
   to the cell output and copied to `/content/out/runner.stdout.log`. The runner's exit code is the exit
   code of that exec: 0 when `results.json` status is `ok` or `partial`, 1 for `fail` or `error`, and
   `launch.sh` then reports the failure after it has packed and stopped the VM.
   On the VM the runner works in `/content/out` (`RAKSHAK_WORK`). Pinned pip installs from
   `kaggle/requirements-kaggle.txt` run first, inside that fresh process. Data is found through
   `resolve_data_root`, which checks `/content/rakshak-data` after `$RAKSHAK_DATA` and
   `/kaggle/input/rakshak-data`.
4. `colab/remote_pack_out.py` packs `/content/out`; `colab download` saves it to
   `colab/out/JOB/output/out.tar.gz` (extract with `tar -xzf`; the archive holds `out/results.json` and
   the rest).
5. `colab stop -s SESSION`, then `colab usage`.

Everything is logged to `colab/out/JOB/launch.log`. `colab/out/` is gitignored.

## Files

- `launch.sh`: the launcher (args, paid-GPU refusal, traps, dry-run).
- `data_bundle.py`: laptop-side selection, archive, chunking and manifest for `--data` and `--data-files`.
- `remote_unpack_data.py`, `remote_pack_out.py`: small standard-library scripts run on the VM by `colab exec -f`.
- `remote_run_job.py`: the wrapper `colab exec -f` runs. It starts `/content/rakshak_run.py` in a child
  interpreter, streams its output and passes its exit code on. The runner is never exec'd in the kernel:
  the kernel already has numpy imported, and the runner's pip install upgraded it under that live process
  (`ImportError: cannot import name '_slice' from 'numpy._core.umath'`, run 2026-10-30).
- `out/`: generated runner, data archive, downloaded output, launch log (gitignored).
- Shared with Kaggle: `kaggle/build_kernel.py --platform colab`, `kaggle/wait_event.py --platform colab`,
  `src/rakshak/eval/data.py` (`COLAB_DATA` candidate).

## Limits

- Live checks (2026-10-30): `smoke --cpu --data-files splits/campaign.json` ran end to end on a CPU VM
  (results.json status `ok`, 3/3 samples, VM stopped). A T4 `gpu_smoke` run on the same day did NOT
  complete: `colab exec` reported `RuntimeError: Connection was lost.` in the CLI's own first cell, before
  `remote_run_job.py` ran, and the launch was then ended by SIGTERM (exit 143). No output was downloaded and
  no GPU name was recorded. GPU and torch detection on Colab is therefore still unproven. The exit-code path
  for a failed runner is covered by `tests/test_colab_launcher.py` only, not live.
- The wrapper depends on `colab exec` propagating a non-zero exit (a `SystemExit` from the wrapper) to the
  launcher as a failure. The launcher treats any non-zero `colab exec` as failed and still packs, downloads and
  stops, so a failure is never reported as `done` by the launcher's exit code.
- The environment report includes `gpu` (nvidia-smi name and memory, or null). It is read by the runner
  before the job starts. torch is still checked only by jobs that import it (`gpu_smoke`, `sms_teacher`).
- The VM's Python must have `tarfile` extraction with `filter='data'` (Python 3.12 and later) for `--data`.
- Upload size, measured with one free CPU session (`colab new -s rakshak-probe`, no GPU, then stopped): 10, 25 and
  50 MiB files uploaded; 100 MiB got `400 Client Error: Bad Request`. The real cap lies between 50 and 100 MiB,
  and the probe did not narrow it further. Parts are 32 MiB by default, so each one is below a size that worked.
  The 400 on the 200 MB archive fits that size limit (the same path style worked at 50 MiB).
- Remote paths: a relative remote path (`probe-rel-1m.bin`) uploaded without error but did not show up in `/content`,
  so the launcher always uploads to absolute `/content/...` paths. The probe did not look for it elsewhere.
- Colab can disconnect a VM mid-run. The exec then ends with an error, the launcher downloads whatever
  `/content/out` holds and stops the VM.
