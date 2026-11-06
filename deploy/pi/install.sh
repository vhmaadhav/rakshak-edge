#!/usr/bin/env bash
# Install the Rakshak Pi thermal governor and dashboard as systemd services.
# Run once on the Pi with sudo, from the directory synced by sync.sh:
#   cd ~/rakshak-pi && sudo ./install.sh [--keep-bt]
set -euo pipefail

KEEP_BT=0
for arg in "$@"; do
  case "$arg" in
    --keep-bt) KEEP_BT=1 ;;
    -h|--help)
      echo "usage: sudo ./install.sh [--keep-bt]"
      exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ "$(id -u)" -ne 0 ]; then
  echo "install.sh must run as root: sudo ./install.sh" >&2
  exit 1
fi

SRC="$(cd "$(dirname "$0")" && pwd)"
DEST=/opt/rakshak-pi
RUN_USER=hermoine
RUN_HOME="/home/${RUN_USER}"

install -d -m 0755 "$DEST"
for f in rakshak_thermald.py thermal_run.py pi_dashboard.py pi_common.py; do
  install -m 0644 "$SRC/$f" "$DEST/$f"
done
install -d -m 0755 -o "$RUN_USER" -g "$RUN_USER" "$RUN_HOME/rakshak-pi/run"

install -m 0644 "$SRC/rakshak-thermald.service" /etc/systemd/system/rakshak-thermald.service
install -m 0644 "$SRC/rakshak-dashboard.service" /etc/systemd/system/rakshak-dashboard.service
systemctl daemon-reload
systemctl enable --now rakshak-thermald.service rakshak-dashboard.service

if [ "$KEEP_BT" -eq 0 ]; then
  if systemctl list-unit-files bluetooth.service --no-legend | grep -q bluetooth; then
    systemctl disable --now bluetooth.service || true
    echo "bluetooth.service disabled (use --keep-bt to skip)"
  else
    echo "bluetooth.service not present, nothing to disable"
  fi
fi

echo
systemctl --no-pager --lines=5 status rakshak-thermald.service rakshak-dashboard.service || true
echo
echo "dashboard: ssh -i ~/.ssh/pi5 -N -L 8765:127.0.0.1:8765 ${RUN_USER}@pi5.local  then open http://localhost:8765"
