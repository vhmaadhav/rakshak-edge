#!/usr/bin/env bash
# Usage: kaggle/wait_event.sh [JOB] [--timeout SECONDS]
#
# ONE blocking ntfy subscription that replays cached events since the launch, then returns on the matching
# completion. No polling, no sleep, no retry. Exit 0 = completion event received, 124 = --timeout passed,
# 1 = stream or setup error. The topic URL is never printed. Without JOB, the latest launch is used.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/wait_event.py" "$@"
