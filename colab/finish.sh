#!/usr/bin/env bash
# Usage: colab/finish.sh JOB [--abort] [--dry-run]
#
# Fetches the output of a Colab run and always stops its VM. Run it after colab/wait.sh reports the
# completion, or by hand. Reads the session from colab/out/JOB/session.json (written by colab/launch.sh).
#   Without --abort:
#     1. colab exec colab/remote_pack_out.py: packs /content/out into /content/rakshak-out.tar.gz
#     2. colab download: the tarball goes to colab/out/JOB/output/out.tar.gz
#     3. colab stop -s SESSION
#     4. the tarball is extracted in colab/out/JOB/output/ (results: output/out/results.json)
#   With --abort: colab stop -s SESSION only. Nothing is packed or downloaded.
# An EXIT/INT/TERM trap runs the stop again if anything failed before it succeeded. The state in
# session.json becomes "finished" or "aborted" only after a successful stop, so a second run does
# nothing once the VM is stopped, and a failed stop is retried by the next run. --dry-run prints the
# commands and runs none. Nothing here polls or sleeps.
set -euo pipefail

usage() {
  echo 'usage: colab/finish.sh JOB [--abort] [--dry-run]' >&2
}
LOG=""
die() {
  echo "finish: $*" >&2
  if [ -n "$LOG" ] && [ -d "$(dirname "$LOG")" ]; then echo "finish: $*" >>"$LOG"; fi
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
ABORT=0
DRY_RUN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --abort) ABORT=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; die "unknown option: $1" ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/out/$JOB"
LOG="$OUT/finish.log"
SESSION_FILE="$OUT/session.json"
[ -f "$SESSION_FILE" ] || die "no $SESSION_FILE: colab/launch.sh $JOB did not start a session"
mkdir -p "$OUT"
read -r SESSION STATE < <(python3 -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
print(data["session"], data.get("state", "running"))' "$SESSION_FILE") || die "cannot read $SESSION_FILE"
[ -n "$SESSION" ] || die "$SESSION_FILE names no session"
if [ "$STATE" = finished ] || [ "$STATE" = aborted ]; then
  echo "finish: $SESSION is already $STATE; nothing to do"
  exit 0
fi

COLAB=(colab)
if [ -n "${RAKSHAK_COLAB_AUTH:-}" ]; then
  COLAB+=(--auth "$RAKSHAK_COLAB_AUTH")
fi
STOPPED=0

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
# set_state STATE: records the outcome in session.json (not in dry-run).
set_state() {
  [ "$DRY_RUN" = 1 ] && return 0
  python3 - "$SESSION_FILE" "$1" <<'PY'
import json
import sys
import time

path, state = sys.argv[1], sys.argv[2]
with open(path, encoding='utf-8') as f:
    data = json.load(f)
data['state'] = state
data[state + '_epoch'] = int(time.time())
with open(path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(data, indent=2) + '\n')
PY
}
# stop_session: colab stop. A failed stop still counts as done when colab sessions no longer lists the
# session (the VM was reclaimed or already stopped). Returns non-zero only when the session is still up.
stop_session() {
  if run "${COLAB[@]}" stop -s "$SESSION"; then
    STOPPED=1
    return 0
  fi
  if [ "$DRY_RUN" != 1 ] && "${COLAB[@]}" sessions 2>/dev/null | grep -qF "$SESSION"; then
    log "finish: colab stop -s $SESSION failed and the session is still listed. Stop it by hand, then rerun this script."
    return 1
  fi
  log "finish: $SESSION is not listed by colab sessions; treating it as stopped"
  STOPPED=1
  return 0
}
on_exit() {
  local rc=$?
  trap - EXIT
  if [ "$STOPPED" != 1 ] && [ "$DRY_RUN" != 1 ]; then
    if stop_session; then
      set_state "$([ "$ABORT" = 1 ] && echo aborted || echo finished)" || true
    else
      rc=1
    fi
  fi
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

log "== $(date -u +%FT%TZ) finish $JOB session=$SESSION abort=$ABORT dry_run=$DRY_RUN"
if [ "$ABORT" = 1 ]; then
  log "finish: --abort: stopping $SESSION without downloading anything"
  stop_session
  set_state aborted
  STOPPED=1
  run "${COLAB[@]}" usage || true
  exit 0
fi

# 1 and 2: pack and download. A failure here still reaches the stop in step 3.
PACK_RC=0
run "${COLAB[@]}" exec -s "$SESSION" -f "$HERE/remote_pack_out.py" --timeout 600 || PACK_RC=$?
DOWNLOAD_RC=0
if [ "$PACK_RC" = 0 ]; then
  run rm -rf "$OUT/output"
  run mkdir -p "$OUT/output"
  run "${COLAB[@]}" download -s "$SESSION" /content/rakshak-out.tar.gz "$OUT/output/out.tar.gz" || DOWNLOAD_RC=$?
else
  DOWNLOAD_RC=1
fi

# 3: stop. Always attempted; the trap retries it if this fails.
STOP_RC=0
stop_session || STOP_RC=$?
if [ "$STOP_RC" = 0 ]; then
  set_state finished
fi
run "${COLAB[@]}" usage || log "finish: colab usage failed; continuing"

if [ "$STOP_RC" != 0 ]; then
  exit 1
fi
if [ "$DRY_RUN" = 1 ]; then
  log "finish: dry run complete; no colab command was executed"
  exit 0
fi
if [ "$PACK_RC" != 0 ] || [ "$DOWNLOAD_RC" != 0 ]; then
  log "finish: could not fetch /content/out (pack exit $PACK_RC, download exit $DOWNLOAD_RC); the VM is stopped"
  exit 1
fi

# 4: extract. The archive holds out/ (results.json, run.log, runner.stdout.log and the rest).
run tar -xzf "$OUT/output/out.tar.gz" --no-same-owner -C "$OUT/output"
if [ ! -f "$OUT/output/out/results.json" ]; then
  log "finish: the archive has no out/results.json; the runner may have died before writing it"
  exit 1
fi
STATUS="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status"))' "$OUT/output/out/results.json")"
log "finish: done. results: $OUT/output/out/results.json (status $STATUS); VM stopped."
