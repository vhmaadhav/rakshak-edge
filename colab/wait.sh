#!/usr/bin/env bash
# Usage: colab/wait.sh JOB [--timeout SECONDS] [--dry-run]
#
# ONE blocking ntfy subscription for rakshak-colab-JOB (kaggle/wait_event.py --platform colab), then
# colab/finish.sh JOB once the completion line arrives. The caller backgrounds this script; it never
# polls, sleeps or retries. The subscription replays the cached events since the launch, so rerunning
# after a dropped stream costs nothing.
#   exit 0    completion ok or partial: output fetched, VM stopped
#   exit 1    completion fail: output fetched, VM stopped. Or the stream failed: nothing is stopped; rerun.
#   exit 124  no completion within the timeout: nothing is stopped (see the message for the next step)
# --timeout defaults to the run's hard wall-clock limit plus 30 minutes (session.json), so a VM that never
# reports cannot hold this process forever. --dry-run prints the commands and runs none.
set -euo pipefail

usage() {
  echo 'usage: colab/wait.sh JOB [--timeout SECONDS] [--dry-run]' >&2
}
die() {
  echo "wait: $*" >&2
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
TIMEOUT=""
DRY_RUN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --timeout) [ "$#" -ge 2 ] || die "--timeout needs a value"; TIMEOUT="$2"; shift 2 ;;
    --timeout=*) TIMEOUT="${1#--timeout=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; die "unknown option: $1" ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OUT="$HERE/out/$JOB"
SESSION_FILE="$OUT/session.json"
[ -f "$SESSION_FILE" ] || die "no $SESSION_FILE: colab/launch.sh $JOB did not start a session"
WALL_S="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["wall_limit_s"])' "$SESSION_FILE")"
if [ -z "$TIMEOUT" ]; then
  TIMEOUT=$((WALL_S + 1800))
fi
if ! [[ "$TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  die "--timeout must be a whole number of seconds above zero (got '$TIMEOUT')"
fi

if [ "$DRY_RUN" = 1 ]; then
  echo "[dry-run] python3 $REPO/kaggle/wait_event.py --platform colab $JOB --timeout $TIMEOUT"
  echo "[dry-run] $HERE/finish.sh $JOB   (only after a completion line)"
  exit 0
fi

WAIT_RC=0
python3 "$REPO/kaggle/wait_event.py" --platform colab "$JOB" --timeout "$TIMEOUT" || WAIT_RC=$?
case "$WAIT_RC" in
  0) ;;
  124)
    echo "wait: no completion within $TIMEOUT s. Nothing was stopped; the VM may still run." >&2
    echo "wait: next, once: colab sessions. If $(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["session"])' "$SESSION_FILE") is gone or the run is dead: colab/finish.sh $JOB --abort" >&2
    exit 124
    ;;
  *)
    echo "wait: the subscription failed (exit $WAIT_RC); nothing was stopped. Rerun colab/wait.sh $JOB: cached events replay." >&2
    exit 1
    ;;
esac

FINISH_RC=0
"$HERE/finish.sh" "$JOB" || FINISH_RC=$?
if [ "$FINISH_RC" != 0 ]; then
  exit "$FINISH_RC"
fi
STATUS="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["message"].split()[1])' "$OUT/completion.json" 2>/dev/null || echo unknown)"
echo "wait: completion status $STATUS"
if [ "$STATUS" = fail ]; then
  exit 1
fi
