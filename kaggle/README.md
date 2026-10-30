# Kaggle job launcher

All training, hyperparameter search and evaluation for rakshak-edge run on private Kaggle kernels, never on this
machine. Each job is one private script kernel, `maadhavvh/rakshak-<job>`, built from the working tree. The flow:
build, launch, wait once for the notification, fetch.

## Commands

```bash
python3 kaggle/build_kernel.py --job smoke            # build only: writes kaggle/out/smoke/kernel/, pushes nothing
kaggle/launch.sh smoke --internet                     # build + kaggle kernels push (starts a Kaggle run)
kaggle/wait_event.sh smoke --timeout 540              # ONE blocking ntfy subscription; exit 0 on completion
kaggle/fetch.sh smoke                                 # downloads outputs once into kaggle/out/smoke/output/
kaggle/push_data.sh                                   # creates/versions private dataset maadhavvh/rakshak-data
```

`launch.sh` takes the build flags below after the job name. `wait_event.sh` never polls, sleeps or retries. Run it
once after a launch. It exits 124 if `--timeout` passes first, and nothing is written in that case.

## Build flags (`build_kernel.py`, also passed through by `launch.sh`)

| Flag | Effect |
|---|---|
| `--gpu [t4\|p100]` | GPU on. Sets `enable_gpu: true` and `machine_shape` in `kernel-metadata.json`. Default `t4` when given with no value. Omit for CPU. |
| `--internet` | Internet on. Needed for the pinned pip install and the completion notification. Off by default. |
| `--datasets OWNER/SLUG ...` | Attach Kaggle datasets as `dataset_sources`. Data appears under `/kaggle/input/<slug>/`. |
| `--models OWNER/MODEL/...` | Attach Kaggle model sources. |
| `--job-args="FLAGS"` | Flags for the job, passed to `run(ctx)` as `ctx.argv`, e.g. `--job-args="--hpo 8 --adv-aug 0.3"`. Use the `=` form when the first flag starts with `-`. Kernels get no argv, so this is how jobs take their options. |
| `--require-clean` | Refuse to build while `src/rakshak/`, `kaggle/jobs/_common.py` or the job file has uncommitted or untracked changes. |

GPU mechanism: `kaggle kernels push` reads `machine_shape` from the metadata (or `--accelerator`). Kaggle CLI 2.2.4
(`kagglesdk`) lists exactly two GPU shapes, `NvidiaTeslaT4` and `NvidiaTeslaP100`. There is no "T4 x2" value, so
`--gpu t4` is the closest option. Check the GPU count reported in `results.json` (`environment`, plus
`metrics.json` `env.gpu`) after the first GPU run.

## What a build bundles

`build_kernel.py` writes `kaggle/out/<job>/kernel/run.py` and `kernel-metadata.json`. `run.py` embeds:

- the working-tree files of `src/rakshak/`, `kaggle/jobs/_common.py` and `kaggle/jobs/<job>.py` (gzip tar, base64).
  The repository has no commits yet, so the bundle is the working tree, not `git archive HEAD`. The build records
  `git_label` (first 12 characters of HEAD, or `no-commit`), `dirty` (true when those paths have uncommitted or
  untracked changes), and `tree_sha256` (over the bundled files). Those fields identify the exact code that ran.
- the pinned pip set from `kaggle/requirements-kaggle.txt` (versions from `uv.lock`). It is installed only when
  the kernel has internet. Without internet, the preinstalled packages are used and the run log says so.

If `src/rakshak/__init__.py` or the job file is missing, the build exits with an error and writes nothing.

## Completion notification (ntfy)

- The topic is a random `rakshak-<32 hex>` string in `~/.config/rakshak/ntfy_topic` (mode 600, directory mode 700).
  `build_kernel.py` creates it on the first `--internet` build. `RAKSHAK_NTFY_TOPIC` overrides it. It is never
  printed or committed. `kaggle/out/` is gitignored, and the generated `run.py` holds the URL because the kernel is
  private.
- The kernel posts one line at the end, including on exception:
  `rakshak-<job> ok|fail git=<sha12> tree=<hash12>[+dirty] <summary>`. `summary` is set by the job in
  `ctx.results['summary']`.
- `wait_event.py` (run by `wait_event.sh`) opens one stream with `since=<launch epoch>`, so a fast job is not
  missed. It accepts only a message for this job and this build's git label, and writes
  `kaggle/out/<job>/completion.json`. Only the topic subscription touches the network, and the URL is never printed.
- Kernels built without `--internet` send no notification, so `wait_event.sh` refuses them. Rebuild with
  `--internet` to wait on them.

## Outputs

`fetch.sh` downloads everything the kernel left in `/kaggle/working`: `results.json`, `run.log`, and whatever the
job writes (for smoke, `metrics.json`). When `completion.json` exists, `fetch.sh` checks that the downloaded
`results.json` has the same job, git label, tree hash and outcome as the notification. The output folder is emptied
first, so a failed download cannot leave an older run in place.

## Writing a job

Add `kaggle/jobs/<name>.py` with `run(ctx) -> bool`:

- `ctx.log(msg)` writes to `run.log`. `ctx.results` is the dict saved as `results.json`. Set `ctx.results['summary']`
  for the notification line.
- `ctx.work` is `/kaggle/working`. `ctx.src` is the unpacked `src/` (already on `sys.path`, so `import rakshak` works).
- `_common.py` gives `write_metrics`, `seed_everything`, `timed` and `env_report` (Python, CPU, `nvidia-smi` GPU, torch
  and CUDA when installed, and the project's package versions).
- Return `False` for a clean failure. An exception sets `status: error` and still notifies.

The laptop runs none of this. `build_kernel.py` only bundles and writes files. `smoke` (CPU) and `gpu_smoke`
(`--gpu t4`) are the jobs that have been run through the launcher.

## Datasets

`kaggle/push_data.sh` packages `data/processed/` as the private dataset `maadhavvh/rakshak-data` (create on the
first run, version after that). It stages a copy under `kaggle/out/rakshak-data/`, so the repo `data/` folder is
never modified. It creates `data/processed/README.txt` as a placeholder if the folder is missing, and it refuses to
upload when no file other than that placeholder exists. The dataset license is set to `other` and should be
set properly before any dataset is made public. Data stays private.

## Files

- `build_kernel.py`: bundles and writes the kernel (no compute, no push).
- `launch.sh`, `fetch.sh`, `wait_event.sh` + `wait_event.py`: push, download, and the one-shot notification wait.
- `push_data.sh`: the dataset upload.
- `requirements-kaggle.txt`: pinned runtime set (numpy has one pin per Python range).
- `jobs/_common.py`, `jobs/smoke.py`: shared helpers and the first job.
- `jobs/gpu_smoke.py`: torch, CUDA device and one matmul checked against the CPU (`kaggle/launch.sh gpu_smoke --gpu t4 --internet`).
- `out/`: generated kernels, launch epochs, downloads and logs. Gitignored.
