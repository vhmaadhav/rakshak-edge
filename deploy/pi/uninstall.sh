#!/usr/bin/env bash
# Reverse install.sh: stop and remove both services, restore the CPU max frequency.
#   sudo ./uninstall.sh [--leave-bt-off]
set -euo pipefail

LEAVE_BT_OFF=0
for arg in "$@"; do
  case "$arg" in
    --leave-bt-off) LEAVE_BT_OFF=1 ;;
    -h|--help)
      echo "usage: sudo ./uninstall.sh [--leave-bt-off]"
      exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ "$(id -u)" -ne 0 ]; then
  echo "uninstall.sh must run as root: sudo ./uninstall.sh" >&2
  exit 1
fi

systemctl disable --now rakshak-thermald.service rakshak-dashboard.service 2>/dev/null || true
rm -f /etc/systemd/system/rakshak-thermald.service /etc/systemd/system/rakshak-dashboard.service
systemctl daemon-reload

# The governor restores on SIGTERM too; do it again in case it was killed hard.
for policy in /sys/devices/system/cpu/cpufreq/policy*; do
  [ -d "$policy" ] || continue
  if [ -r "$policy/cpuinfo_max_freq" ] && [ -w "$policy/scaling_max_freq" ]; then
    cat "$policy/cpuinfo_max_freq" > "$policy/scaling_max_freq"
  fi
done
echo "scaling_max_freq restored to cpuinfo_max_freq on all policies"

rm -rf /opt/rakshak-pi

if [ "$LEAVE_BT_OFF" -eq 0 ]; then
  if systemctl list-unit-files bluetooth.service --no-legend | grep -q bluetooth; then
    systemctl enable --now bluetooth.service || true
    echo "bluetooth.service re-enabled (use --leave-bt-off to keep it off)"
  fi
fi

echo "uninstalled. Job outputs in ~/rakshak-pi/run were kept."
