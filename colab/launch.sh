#!/usr/bin/env bash
# Usage: colab/launch.sh JOB [--gpu T4|L4|A100|H100 | --cpu] [--high-mem] [--job-args="FLAGS"]
#                            [--data | --data-files PATTERN...] [--yes-paid] [--dry-run]
#
# Runs one rakshak job on a Google Colab VM: the Colab counterpart of kaggle/launch.sh. Steps:
#   1. build the runner: kaggle/build_kernel.py --platform colab -> colab/out/JOB/kernel/run.py
#   2. (--data, or --data-files) select files under data/processed, pack them into an archive,
#      split it into parts of at most RAKSHAK_COLAB_CHUNK_MB MiB, and write a manifest with sha256
#      sums (colab/data_bundle.py). Then colab upload the manifest and each part, and
#      colab/remote_unpack_data.py joins, verifies and unpacks them into /content/rakshak-data.
#   3. colab new (default GPU T4, or no accelerator with --cpu). The runner is uploaded to
#      /content/rakshak_run.py, then ONE blocking colab exec of colab/remote_run_job.py, which runs the
#      runner in a fresh Python process (the caller backgrounds the launch; nothing here polls).
#   4. colab exec packs /content/out; colab download saves it to colab/out/JOB/output/out.tar.gz
#   5. colab stop. This is always run: an EXIT/INT/TERM trap stops the VM once colab new was requested.
# Any GPU other than T4, and --high-mem, spend Colab compute units. They are refused unless --yes-paid
# is passed for that run, and Maadhav gives that approval per run. --dry-run prints every colab command
# and runs none of them (the local bundle is still built, under colab/out). Everything is logged to
# colab/out/JOB/launch.log. Nothing here polls or sleeps.
# Auth and config: colab/README.md. RAKSHAK_COLAB_AUTH=adc adds --auth adc to every colab call (default:
# oauth2). RAKSHAK_COLAB_TIMEOUT sets the runner's colab exec timeout in seconds (default 43200).
# RAKSHAK_COLAB_CHUNK_MB sets the upload part size in MiB (default 32, at most 50: a 50 MiB upload
# succeeded in the probe and 100 MiB was rejected with HTTP 400).
set -euo pipefail

usage() {
  echo 'usage: colab/launch.sh JOB [--gpu T4|L4|A100|H100 | --cpu] [--high-mem] [--job-args="FLAGS"] [--data | --data-files PATTERN...] [--yes-paid] [--dry-run]' >&2
}

# default_data_patterns JOB: the data a job needs when --data is given without --data-files.
default_data_patterns() {
  case "$1" in
    sms_*) printf '%s\n' 'sms.parquet' 'splits/*.json' '!splits/url_*.json' 'external/**' ;;
    url_*) printf '%s\n' 'urls.parquet' 'urls_phreshphish.parquet' 'splits/url_*.json' ;;
    llama_*) printf '%s\n' 'sms.parquet' 'splits/campaign.json' ;;
    *) printf '%s\n' '**' ;;
  esac
}
LOG=""
die() {
  echo "launch: $*" >&2
  if [ -n "$LOG" ]; then echo "launch: $*" >>"$LOG"; fi
  exit 2
}

if [ "$#" -lt 1 ]; then
  usage
  exit 2
fi
case "$1" in
  -h|--help) usage; exit 0 ;;
esac
JOB="$1"
shift
if ! [[ "$JOB" =~ ^[a-z][a-z0-9_-]*$ ]]; then
  die "the job name must start with a letter and use only lowercase letters, digits, '_' and '-'"
fi

GPU=T4
GPU_GIVEN=0
CPU=0
HIGH_MEM=0
JOB_ARGS=""
DATA=0
DATA_PATTERNS=()
YES_PAID=0
DRY_RUN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu) [ "$#" -ge 2 ] || die "--gpu needs a value"; GPU="$2"; GPU_GIVEN=1; shift 2 ;;
    --gpu=*) GPU="${1#--gpu=}"; GPU_GIVEN=1; shift ;;
    --cpu) CPU=1; shift ;;
    --high-mem) HIGH_MEM=1; shift ;;
    --job-args) [ "$#" -ge 2 ] || die "--job-args needs a value"; JOB_ARGS="$2"; shift 2 ;;
    --job-args=*) JOB_ARGS="${1#--job-args=}"; shift ;;
    --data) DATA=1; shift ;;
    --data-files)
      DATA=1
      shift
      # Every following word up to the next --option is a PATTERN (quote globs in the shell).
      while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do DATA_PATTERNS+=("$1"); shift; done
      [ "${#DATA_PATTERNS[@]}" -gt 0 ] || die "--data-files needs at least one PATTERN"
      ;;
    --data-files=*) DATA=1; DATA_PATTERNS+=("${1#--data-files=}"); shift ;;
    --yes-paid) YES_PAID=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; die "unknown option: $1" ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
SLUG="${JOB//_/-}"
OUT="$HERE/out/$JOB"
LOG="$OUT/launch.log"
RUNNER="$OUT/kernel/run.py"
DATA_OUT="$OUT/data"
CHUNK_MB="${RAKSHAK_COLAB_CHUNK_MB:-32}"
EXEC_TIMEOUT="${RAKSHAK_COLAB_TIMEOUT:-43200}"
# A new name per launch: a failed colab new can never stop a session that another run is using.
SESSION="rakshak-$SLUG-$(date -u +%y%m%d%H%M%S)"
COLAB=(colab)
if [ -n "${RAKSHAK_COLAB_AUTH:-}" ]; then
  COLAB+=(--auth "$RAKSHAK_COLAB_AUTH")
fi
mkdir -p "$OUT"

log() {
  printf '%s\n' "$*" | tee -a "$LOG"
}
# run COMMAND...: logs the command, then runs it (output goes to the log too) unless --dry-run.
run() {
  local shown
  printf -v shown '%q ' "$@"
  if [ "$DRY_RUN" = 1 ]; then
    log "[dry-run] ${shown% }"
    return 0
  fi
  log "+ ${shown% }"
  "$@" 2>&1 | tee -a "$LOG"
}
# run_local COMMAND...: a local helper that runs even in --dry-run (it only writes under colab/out).
run_local() {
  local shown
  printf -v shown '%q ' "$@"
  log "+ ${shown% }"
  "$@" 2>&1 | tee -a "$LOG"
}

STOP_PENDING=0
stop_vm() {
  [ "$STOP_PENDING" = 1 ] || return 0
  STOP_PENDING=0
  if ! run "${COLAB[@]}" stop -s "$SESSION"; then
    log "launch: colab stop -s $SESSION did not succeed. If the VM is still listed, stop it by hand."
  fi
}
on_exit() {
  local rc=$?
  trap - EXIT
  stop_vm || true
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

report_usage() {
  log "colab usage ($1):"
  run "${COLAB[@]}" usage || log "launch: colab usage failed ($1); continuing"
}

log "== $(date -u +%FT%TZ) launch $JOB gpu=$GPU cpu=$CPU high_mem=$HIGH_MEM data=$DATA dry_run=$DRY_RUN"
log "session: $SESSION (stopped on exit)"
GPU="${GPU^^}"
case "$GPU" in
  T4|L4|A100|H100) ;;
  *) die "unknown GPU '$GPU': choose T4 (default), L4, A100 or H100" ;;
esac
if [ "$CPU" = 1 ]; then
  [ "$GPU_GIVEN" = 0 ] || die "--cpu and --gpu cannot be combined"
  GPU=CPU
fi
PAID=""
if [ "$GPU" != T4 ] && [ "$GPU" != CPU ]; then
  PAID="--gpu $GPU"
fi
if [ "$HIGH_MEM" = 1 ]; then
  PAID="${PAID:+$PAID, }--high-mem"
fi
if [ -n "$PAID" ] && [ "$YES_PAID" != 1 ]; then
  die "refusing $PAID: it spends Colab compute units. Only the default T4 without --high-mem runs unless Maadhav approves this run; then pass --yes-paid."
fi
if [ "$DATA" = 1 ]; then
  if ! [[ "$CHUNK_MB" =~ ^[1-9][0-9]*$ ]] || [ "$CHUNK_MB" -gt 50 ]; then
    die "RAKSHAK_COLAB_CHUNK_MB must be a whole number of MiB from 1 to 50 (got '$CHUNK_MB')"
  fi
fi
if [ "$DRY_RUN" != 1 ]; then
  command -v "${COLAB[0]}" >/dev/null 2>&1 || die "the colab CLI is not on PATH; install it with: uv tool install google-colab-cli"
fi

# 1. The runner. build_kernel.py writes only under colab/out/JOB/kernel; it runs in dry-run too.
run_local python3 "$REPO/kaggle/build_kernel.py" --platform colab --job "$JOB" "--job-args=$JOB_ARGS"
if [ "$(wc -c <"$RUNNER")" -ge $((50 * 1024 * 1024)) ]; then
  die "the runner $RUNNER is 50 MiB or more; colab upload is not used for files that large"
fi

# 2. Data bundle (local, also in dry-run; it only writes under colab/out). Built before colab new, so a
#    missing folder or pattern never allocates a VM.
if [ "$DATA" = 1 ]; then
  if [ ! -d "$REPO/data/processed" ]; then
    die "--data: $REPO/data/processed does not exist"
  fi
  if [ "${#DATA_PATTERNS[@]}" -eq 0 ]; then
    mapfile -t DATA_PATTERNS < <(default_data_patterns "$JOB")
    log "data: default patterns for $JOB: ${DATA_PATTERNS[*]}"
  else
    log "data: patterns from --data-files: ${DATA_PATTERNS[*]}"
  fi
  rm -rf "$DATA_OUT"
  run_local python3 "$HERE/data_bundle.py" build "$REPO/data/processed" "$DATA_OUT" \
    "$((CHUNK_MB * 1024 * 1024))" "${DATA_PATTERNS[@]}"
  PARTS=("$DATA_OUT"/rakshak-data.part-*)
fi

report_usage before

# 3. The VM. STOP_PENDING is set before colab new, so a failed allocation is still stopped.
STOP_PENDING=1
NEW_ARGS=(new -s "$SESSION")
if [ "$CPU" != 1 ]; then
  NEW_ARGS+=(--gpu "$GPU")
fi
if [ "$HIGH_MEM" = 1 ]; then
  NEW_ARGS+=(--high-mem)
fi
run "${COLAB[@]}" "${NEW_ARGS[@]}"
if [ "$DRY_RUN" != 1 ]; then
  date +%s >"$OUT/launch_epoch"
  rm -f "$OUT/completion.json"
fi
if [ "$DATA" = 1 ]; then
  # Absolute /content paths only: a relative remote path does not land in /content.
  run "${COLAB[@]}" upload -s "$SESSION" "$DATA_OUT/manifest.json" /content/rakshak-data.manifest.json
  for part in "${PARTS[@]}"; do
    run "${COLAB[@]}" upload -s "$SESSION" "$part" "/content/$(basename "$part")"
  done
  run "${COLAB[@]}" exec -s "$SESSION" -f "$HERE/remote_unpack_data.py" --timeout 900
fi
# The runner is a file on the VM, not a cell run in the kernel: remote_run_job.py starts it in a fresh
# interpreter, so its pip install cannot upgrade numpy under the kernel's live imports.
run "${COLAB[@]}" upload -s "$SESSION" "$RUNNER" /content/rakshak_run.py

# 4. The one blocking run. colab exec's default timeout is 30 s, so the runner's is set explicitly.
EXEC_RC=0
run "${COLAB[@]}" exec -s "$SESSION" -f "$HERE/remote_run_job.py" --timeout "$EXEC_TIMEOUT" || EXEC_RC=$?

# 5. Output. Downloaded even when the runner failed, since a partial run's files are still useful.
PACK_RC=0
run "${COLAB[@]}" exec -s "$SESSION" -f "$HERE/remote_pack_out.py" --timeout 600 || PACK_RC=$?
DOWNLOAD_RC=0
run rm -rf "$OUT/output"
run mkdir -p "$OUT/output"
run "${COLAB[@]}" download -s "$SESSION" /content/rakshak-out.tar.gz "$OUT/output/out.tar.gz" || DOWNLOAD_RC=$?

stop_vm
report_usage after

if [ "$DRY_RUN" = 1 ]; then
  log "launch: dry run complete; no colab command was executed"
  exit 0
fi
if [ "$EXEC_RC" -ne 0 ]; then
  log "launch: colab exec of the runner exited $EXEC_RC; see $LOG and $OUT/output"
  exit "$EXEC_RC"
fi
if [ "$PACK_RC" -ne 0 ] || [ "$DOWNLOAD_RC" -ne 0 ]; then
  log "launch: could not fetch /content/out (pack exit $PACK_RC, download exit $DOWNLOAD_RC)"
  exit 1
fi
log "launch: done. Output tarball: $OUT/output/out.tar.gz (extract with tar -xzf there)"
