#!/usr/bin/env bash
# Usage: kaggle/launch.sh JOB [build flags]
#
# Builds the job kernel with kaggle/build_kernel.py, then pushes it to Kaggle as maadhavvh/rakshak-JOB.
# The push starts a private Kaggle run; the job runs on Kaggle, never on this machine.
# Build flags (passed through to build_kernel.py):
#   --gpu [t4|p100]          enable a GPU (default t4 when given without a value); CPU when omitted
#   --internet               enable internet (pinned pip install and the completion notification)
#   --datasets OWNER/SLUG... attach Kaggle datasets
#   --models OWNER/MODEL...  attach Kaggle model sources
#   --job-args="FLAGS"       flags for the job (ctx.argv), e.g. --job-args="--hpo 8"
#   --require-clean          refuse while the bundled paths have uncommitted changes
# The Kaggle CLI can exit 0 after an error, so its output is checked for error text and this script exits
# non-zero when it finds any. Writes kaggle/out/latest_launch.json and the launch epoch used by wait_event.sh.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: kaggle/launch.sh JOB [--gpu [t4|p100]] [--internet] [--datasets OWNER/SLUG...] [--models ...] [--require-clean]" >&2
  exit 2
fi
JOB="$1"
shift
if ! [[ "$JOB" =~ ^[a-z][a-z0-9_-]*$ ]]; then
  echo "launch: the job name must start with a letter and use only lowercase letters, digits and hyphens" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLUG="${JOB//_/-}"
KERNEL_DIR="$HERE/out/$JOB/kernel"
LOG="$HERE/out/push-$JOB.log"

python3 "$HERE/build_kernel.py" --job "$JOB" "$@"

mkdir -p "$HERE/out/$JOB"
date +%s > "$HERE/out/$JOB/launch_epoch"
rm -f "$HERE/out/$JOB/completion.json"
python3 - "$HERE/out/latest_launch.json" "$JOB" <<'STAMP'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'job': sys.argv[2]}) + '\n')
STAMP

set +e
kaggle kernels push -p "$KERNEL_DIR" >"$LOG" 2>&1
rc=$?
set -e
cat "$LOG"
if [ "$rc" -ne 0 ]; then
  echo "launch: kaggle kernels push exited $rc" >&2
  exit 1
fi
if grep -qiE 'error|failed|exception|forbidden|unauthori[sz]ed|invalid|not found' "$LOG"; then
  echo "launch: the Kaggle CLI printed error text (see $LOG); treating the push as failed" >&2
  exit 1
fi
echo "launch: pushed maadhavvh/rakshak-$SLUG. Wait once with kaggle/wait_event.sh $JOB, then kaggle/fetch.sh $JOB."
