#!/usr/bin/env bash
# Usage: kaggle/fetch.sh JOB
#
# Downloads the outputs of maadhavvh/rakshak-JOB once into kaggle/out/JOB/output/ and prints the summary
# from results.json. If kaggle/out/JOB/completion.json exists, the downloaded results.json must match its job
# and git label, so a stale or other run is rejected.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: kaggle/fetch.sh JOB" >&2
  exit 2
fi
JOB="$1"
if ! [[ "$JOB" =~ ^[a-z][a-z0-9_-]*$ ]]; then
  echo "fetch: the job name must start with a letter and use only lowercase letters, digits and hyphens" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL="maadhavvh/rakshak-${JOB//_/-}"
OUT="$HERE/out/$JOB/output"
LOG="$HERE/out/fetch-$JOB.log"
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
python3 - "$OUT/results.json" "$HERE/out/$JOB/completion.json" "$JOB" <<'SUMMARY'
import json
import re
import sys
from pathlib import Path
result = json.loads(Path(sys.argv[1]).read_text())
receipt = Path(sys.argv[2])
job = sys.argv[3]
if result.get('job') != job:
    raise SystemExit(f"fetch: results.json is for job {result.get('job')!r}, not {job!r}")
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
