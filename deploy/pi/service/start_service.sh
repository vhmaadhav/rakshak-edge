#!/usr/bin/env bash
# Start the Rakshak edge service in the background, in its own network namespace (no network).
#
#   ~/rakshak-bench/deploy/pi/service/start_service.sh [extra rakshak_edge_service.py args]
#
# Runs on the Pi from the bench root (default ~/rakshak-bench; set RAKSHAK_BENCH to change).
# The service runs as:  unshare -rn sh -c 'ip link set lo up && exec python rakshak_edge_service.py'
# so it has no route and no DNS. It listens on 127.0.0.1:8770 inside that namespace and on the
# Unix socket run/edge.sock (the socket is how the eval client and the page reach it from outside).
# Then socket_bridge.py starts outside the namespace and relays host 127.0.0.1:8770 to the socket.
# No sudo, no systemd unit.
#
# Files (under $ROOT/run): edge_service.pid, edge_service.log (JSON lines), edge.sock,
# edge_bridge.pid, edge_bridge.log, ready.fifo. Readiness is event-driven: the script blocks on a
# FIFO that the service writes once it is listening. Stop with stop_service.sh.
set -euo pipefail

ROOT="${RAKSHAK_BENCH:-$HOME/rakshak-bench}"
RUN="$ROOT/run"
PIDFILE="$RUN/edge_service.pid"
LOG="$RUN/edge_service.log"
FIFO="$RUN/ready.fifo"
SOCK="$RUN/edge.sock"
BRIDGE_PIDFILE="$RUN/edge_bridge.pid"
BRIDGE_LOG="$RUN/edge_bridge.log"
PY="$ROOT/.venv/bin/python"

mkdir -p "$RUN"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "start_service: already running as pid $(cat "$PIDFILE")" >&2
  exit 1
fi
if ! command -v unshare >/dev/null || ! command -v ip >/dev/null; then
  echo "start_service: needs unshare and ip (util-linux, iproute2)" >&2
  exit 1
fi
[[ -x "$PY" ]] || { echo "start_service: missing $PY" >&2; exit 1; }

rm -f "$FIFO"
mkfifo "$FIFO"
cd "$ROOT"
setsid unshare -rn sh -c 'ip link set lo up && exec "$0" "$@"' \
  "$PY" "$ROOT/deploy/pi/service/rakshak_edge_service.py" \
  --root "$ROOT" --log "$LOG" --unix-socket "$SOCK" --ready-fifo "$FIFO" "$@" \
  >>"$LOG" 2>&1 </dev/null &
echo $! >"$PIDFILE"
pid="$(cat "$PIDFILE")"

# Blocks until the service opens the FIFO (listening). Gives up after 300 s.
if ! timeout 300 cat "$FIFO" >/dev/null; then
  echo "start_service: no READY within 300 s; see $LOG" >&2
  kill -TERM "$pid" 2>/dev/null || true
  rm -f "$PIDFILE"
  exit 1
fi
rm -f "$FIFO"

# The relay runs outside the namespace: host 127.0.0.1:8770 -> run/edge.sock. It makes no
# outbound connection. Started only after the service reported READY.
if [[ -f "$BRIDGE_PIDFILE" ]] && kill -0 "$(cat "$BRIDGE_PIDFILE")" 2>/dev/null; then
  echo "start_service: bridge already running as pid $(cat "$BRIDGE_PIDFILE")"
else
  setsid "$PY" "$ROOT/deploy/pi/service/socket_bridge.py" --listen 127.0.0.1:8770 \
    --unix "$SOCK" >>"$BRIDGE_LOG" 2>&1 </dev/null &
  echo $! >"$BRIDGE_PIDFILE"
fi
echo "start_service: service pid $pid in its own network namespace (127.0.0.1:8770 inside it)"
echo "start_service: relay on host 127.0.0.1:8770 -> $SOCK (pid $(cat "$BRIDGE_PIDFILE"))"
echo "start_service: logs $LOG and $BRIDGE_LOG"
