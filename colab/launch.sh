#!/usr/bin/env bash
# Usage: colab/launch.sh JOB [--gpu T4|L4|A100|H100 | --cpu] [--high-mem] [--job-args="FLAGS"] [--budget-h H]
#                            [--data | --data-files PATTERN...] [--yes-paid] [--dry-run]
#
# Start phase of one rakshak job on a Google Colab VM. It returns as soon as the runner is detached on
# the VM; it does not wait for the job. Steps:
#   1. build the runner: kaggle/build_kernel.py --platform colab -> colab/out/JOB/kernel/run.py, with
#      a hard wall-clock limit (--budget-h plus 30 min; 11.5 h when --budget-h is not given).
#   2. (--data, or --data-files) select files under data/processed, pack them into an archive, split it
#      into parts of at most RAKSHAK_COLAB_CHUNK_MB MiB, and write a manifest with sha256 sums
#      (colab/data_bundle.py). Then colab upload the manifest and each part, and colab/remote_unpack_data.py
#      joins, verifies and unpacks them into /content/rakshak-data.
#   3. colab new (default GPU T4, or no accelerator with --cpu), then colab upload of the runner to
#      /content/rakshak_run.py.
#   4. one SHORT colab exec of colab/remote_start_job.py, which starts the runner detached on the VM
#      (its own session, output to /content/out/runner.stdout.log, PID in /content/out/runner.pid).
#   5. colab/out/JOB/session.json is written. The VM is NOT stopped on success.
# Then: colab/wait.sh JOB waits for the completion line; colab/finish.sh JOB fetches the output and stops
# the VM. Any failure before step 5 stops the VM (EXIT/INT/TERM trap). Nothing here polls or sleeps.
# Any GPU other than T4, and --high-mem, spend Colab compute units. They are refused unless --yes-paid
# is passed for that run, and Maadhav gives that approval per run. --dry-run prints every colab command
# and runs none of them (the local bundle and runner are still built, under colab/out).
# Everything is logged to colab/out/JOB/launch.log. Auth and config: colab/README.md.
# RAKSHAK_COLAB_AUTH=adc adds --auth adc to every colab call (default: oauth2).
# RAKSHAK_COLAB_CHUNK_MB sets the upload part size in MiB (default 32, at most 50).
set -euo pipefail

usage() {
  echo 'usage: colab/launch.sh JOB [--gpu T4|L4|A100|H100 | --cpu] [--high-mem] [--job-args="FLAGS"] [--budget-h H] [--data | --data-files PATTERN...] [--yes-paid] [--dry-run]' >&2
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
# The runner's hard wall-clock limit: the job's --budget-h plus this margin, or NO_BUDGET_WALL_H when the
# job's budget is not given (Colab free sessions end after about 12 h anyway).
MARGIN_H=0.5
NO_BUDGET_WALL_H=11.5
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
BUDGET_H=""
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
    --budget-h) [ "$#" -ge 2 ] || die "--budget-h needs a value"; BUDGET_H="$2"; shift 2 ;;
    --budget-h=*) BUDGET_H="${1#--budget-h=}"; shift ;;
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
    log "launch: colab stop -s $SESSION did not succeed. If the VM is still listed, stop it with colab/finish.sh $JOB --abort."
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

log "== $(date -u +%FT%TZ) launch $JOB gpu=$GPU cpu=$CPU high_mem=$HIGH_MEM data=$DATA budget_h=${BUDGET_H:-none} dry_run=$DRY_RUN"
log "session: $SESSION (stopped on any failure of this start phase; kept running on success)"
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
# The runner's hard limit in hours. A bad --budget-h is refused before anything is built or allocated.
if [ -n "$BUDGET_H" ]; then
  WALL_H="$(python3 -c '
import sys
try:
    budget = float(sys.argv[1])
except ValueError:
    budget = 0.0
if not budget > 0:
    sys.exit(1)
print("%g" % (budget + float(sys.argv[2])))' "$BUDGET_H" "$MARGIN_H")" || die "--budget-h must be a positive number of hours (got '$BUDGET_H')"
else
  WALL_H="$NO_BUDGET_WALL_H"
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
run_local python3 "$REPO/kaggle/build_kernel.py" --platform colab --job "$JOB" "--job-args=$JOB_ARGS" \
  "--wall-limit-h=$WALL_H"
if [ -n "$BUDGET_H" ]; then
  log "runner: hard wall-clock limit ${WALL_H} h (--budget-h $BUDGET_H h + $MARGIN_H h margin)"
else
  log "runner: hard wall-clock limit ${WALL_H} h (no --budget-h: default limit)"
fi
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

# 3. The VM. STOP_PENDING is set before colab new, so a failed allocation or upload is still stopped.
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
run "${COLAB[@]}" upload -s "$SESSION" "$RUNNER" /content/rakshak_run.py

# 4. Start the runner detached. The cell only starts it and returns (about a second), so no connection
#    has to stay open for the length of the job.
if ! run "${COLAB[@]}" exec -s "$SESSION" -f "$HERE/remote_start_job.py" --timeout 120; then
  log "launch: the runner did not start on $SESSION; the VM is stopped"
  exit 1
fi

# 5. Record the session. From here the VM stays up: the runner is running on it.
if [ "$DRY_RUN" != 1 ]; then
  LAUNCH_EPOCH="$(cat "$OUT/launch_epoch")"
  python3 - "$OUT/session.json" "$SESSION" "$JOB" "$GPU" "$HIGH_MEM" "$LAUNCH_EPOCH" "$WALL_H" \
    "${BUDGET_H:-}" <<'PY'
import json
import sys

path, session, job, accelerator, high_mem, epoch, wall_h, budget_h = sys.argv[1:]
data = {
    'job': job,
    'session': session,
    'accelerator': accelerator,
    'high_mem': high_mem == '1',
    'launch_epoch': int(epoch),
    'budget_h': float(budget_h) if budget_h else None,
    'wall_limit_h': float(wall_h),
    'wall_limit_s': round(float(wall_h) * 3600),
    'state': 'running',
}
with open(path, 'w', encoding='utf-8') as out:
    out.write(json.dumps(data, indent=2) + '\n')
PY
fi
STOP_PENDING=0
if [ "$DRY_RUN" = 1 ]; then
  log "launch: dry run complete; no colab command was executed"
  exit 0
fi
log "launch: runner started detached on $SESSION; the VM keeps running."
log "launch: next: colab/wait.sh $JOB (waits for the completion line, then runs colab/finish.sh $JOB)"
