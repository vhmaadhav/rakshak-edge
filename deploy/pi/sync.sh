#!/usr/bin/env bash
# Laptop side: copy the Pi thermal/dashboard files to ~/rakshak-pi on the Pi.
# Only top-level files are sent (no __pycache__ or test code).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
KEY="${PI_KEY:-$HOME/.ssh/pi5}"
HOST="${PI_HOST:-hermoine@pi5.local}"
SSH_OPTS=(-i "$KEY" -o BatchMode=yes -o IdentitiesOnly=yes)

ssh "${SSH_OPTS[@]}" "$HOST" 'mkdir -p ~/rakshak-pi/run'

files=(
  "$HERE"/*.py
  "$HERE"/*.sh
  "$HERE"/*.service
  "$HERE"/README.md
)
scp "${SSH_OPTS[@]}" -q "${files[@]}" "$HOST:~/rakshak-pi/"
echo "synced $(printf '%s\n' "${files[@]}" | xargs -n1 basename | tr '\n' ' ')to $HOST:~/rakshak-pi/"
