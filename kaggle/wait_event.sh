#!/usr/bin/env bash
# Usage: kaggle/wait_event.sh [JOB | JOB-NAME] [--run-name NAME] [--timeout SECONDS]
#
# ONE blocking ntfy subscription that replays cached events since the launch, then returns on the matching
# completion. No polling, no sleep, no retry. Exit 0 = completion event received, 124 = --timeout passed,
# 1 = stream or setup error. The topic URL is never printed. Without JOB, the latest launch is used (with its
# run name, if the launch had --run-name). JOB-NAME and JOB --run-name NAME name the same run.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/wait_event.py" "$@"
