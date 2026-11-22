#!/usr/bin/env bash
# Stop the relay and the Rakshak edge service started by start_service.sh.
#   ~/rakshak-bench/deploy/pi/service/stop_service.sh
# SIGTERM, then wait for each process to exit (tail --pid blocks; no poll loop). SIGKILL after 20 s.
set -euo pipefail

ROOT="${RAKSHAK_BENCH:-$HOME/rakshak-bench}"

stop_one() {
  local pidfile="$1" label="$2" pid
  if [[ ! -f "$pidfile" ]]; then
    echo "stop_service: no $label pid file"
    return 0
  fi
  pid="$(cat "$pidfile")"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "stop_service: $label pid $pid is not running"
    rm -f "$pidfile"
    return 0
  fi
  kill -TERM "$pid"
  if ! timeout 20 tail --pid="$pid" -f /dev/null; then
    echo "stop_service: $label pid $pid still running after 20 s, sending SIGKILL" >&2
    kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$pidfile"
  echo "stop_service: stopped $label pid $pid"
}

stop_one "$ROOT/run/edge_bridge.pid" "relay"
stop_one "$ROOT/run/edge_service.pid" "service"
rm -f "$ROOT/run/edge.sock" "$ROOT"/run/edge.sock.*
