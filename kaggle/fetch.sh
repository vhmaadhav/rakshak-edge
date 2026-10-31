#!/usr/bin/env bash
# Usage: kaggle/fetch.sh JOB [--run-name NAME]     (or kaggle/fetch.sh JOB-NAME for a run started with --run-name)
#
# Downloads the outputs of maadhavvh/rakshak-JOB (or rakshak-JOB-NAME) once into kaggle/out/JOB/output/
# (kaggle/out/JOB-NAME/output/) and prints the summary from results.json. If completion.json exists in that
# folder, the downloaded results.json must match its job and git label, so a stale or other run is rejected.
set -euo pipefail

usage() {
  echo "usage: kaggle/fetch.sh JOB [--run-name NAME]" >&2
  exit 2
}
if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
  usage
fi
JOB="$1"
shift
RUN_NAME=""
if [ "$#" -gt 0 ]; then
  if [ "$#" -ne 2 ] || [ "$1" != "--run-name" ]; then
    usage
  fi
  RUN_NAME="$2"
  if ! [[ "$RUN_NAME" =~ ^[a-z0-9-]{1,20}$ ]]; then
    echo "fetch: --run-name must be 1 to 20 characters of a-z, 0-9 and hyphen" >&2
    exit 2
  fi
fi
if ! [[ "$JOB" =~ ^[a-z][a-z0-9_-]*$ ]]; then
  echo "fetch: the job name must start with a letter and use only lowercase letters, digits and hyphens" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY="$JOB"
if [ -n "$RUN_NAME" ]; then
  KEY="$JOB-$RUN_NAME"
fi
KERNEL="maadhavvh/rakshak-${KEY//_/-}"
OUT="$HERE/out/$KEY/output"
LOG="$HERE/out/fetch-$KEY.log"
mkdir -p "$OUT"

# Start from an empty folder: the CLI skips files that are newer locally, and a failed download must not
# pass off an older run's results.
rm -rf "$OUT"
mkdir -p "$OUT"
set +e
kaggle kernels output "$KERNEL" -p "$OUT" >"$LOG" 2>&1
rc=$?
set -e
cat "$LOG"
if [ "$rc" -ne 0 ]; then
  echo "fetch: kaggle kernels output exited $rc" >&2
  exit 1
fi
if grep -qiE 'error|failed|exception|forbidden|unauthori[sz]ed|invalid' "$LOG"; then
  echo "fetch: the Kaggle CLI printed error text (see $LOG)" >&2
  exit 1
fi
if [ ! -f "$OUT/results.json" ]; then
  echo "fetch: $OUT/results.json was not downloaded; the run may not have finished" >&2
  exit 1
fi
echo "fetch: $OUT"
python3 - "$OUT/results.json" "$HERE/out/$KEY/completion.json" "$KEY" <<'SUMMARY'
import json
import re
import sys
from pathlib import Path
result = json.loads(Path(sys.argv[1]).read_text())
receipt = Path(sys.argv[2])
key = sys.argv[3]
# results.json names the job (JOB), the folder key is JOB or JOB-NAME for a run started with --run-name.
job = str(result.get('job'))
run = key[len(job) + 1:] if key.startswith(job + '-') else ''
if key != job and not (run and re.fullmatch(r'[a-z0-9-]{1,20}', run)):
    raise SystemExit(f"fetch: results.json is for job {job!r}, not {key!r}")
if receipt.exists():
    expected = json.loads(receipt.read_text())
    message = str(expected.get('message', ''))
    if result.get('git_label') != expected.get('git_label') or result.get('job') != expected.get('job'):
        raise SystemExit('fetch: downloaded output does not match the completion event (stale or other run)')
    # The label is the same for every run of an uncommitted tree, so also compare the tree hash and the
    # outcome that the notification reported.
    tree = re.search(r'\btree=([0-9a-f]+)', message)
    if tree is None or not str(result.get('tree_sha256', '')).startswith(tree.group(1)):
        raise SystemExit('fetch: downloaded output has another tree hash than the completion event '
                         '(stale or other run); the kernel may still be finishing, fetch again later')
    outcome = message.split()[1] if len(message.split()) > 1 else ''
    # The notification shows the status as ok, partial, or fail (an error status is sent as fail).
    notified = result.get('status') if result.get('status') in ('ok', 'partial') else 'fail'
    if outcome not in ('ok', 'partial', 'fail') or notified != outcome:
        raise SystemExit(f"fetch: downloaded status {result.get('status')!r} does not match the event "
                         f"outcome {outcome!r} (stale or other run)")
else:
    print('fetch: no completion receipt; provenance was not checked against the notification')
keys = ('job', 'status', 'error', 'git_label', 'dirty', 'tree_sha256', 'summary', 'total_s')
print(json.dumps({k: result.get(k) for k in keys}, indent=2))
SUMMARY
