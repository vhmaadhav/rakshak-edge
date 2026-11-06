#!/usr/bin/env python3
"""Userspace thermal governor for a Raspberry Pi 5 with no heatsink.

Samples thermal_zone0 every few seconds and steps the cpufreq scaling_max_freq
cap on every policy:

  * mean of the last 3 samples >= 68 C  -> cap down 100 MHz
  * mean of the last 3 samples >= 73 C  -> cap straight to the floor (1500 MHz)
  * mean <= 60 C held for 10 s          -> cap up 100 MHz, up to --ceiling

Steps are at least 6 s apart. A hard drop to the floor is never delayed.
State goes to <state-dir>/thermal.json and transitions to
<state-dir>/thermal_events.log. On exit every policy gets its cpuinfo_max_freq
back. --dry-run decides and logs but never writes sysfs.
"""

from __future__ import annotations

import argparse
import signal
import statistics
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import pi_common as pc

HOT_C = 68.0
HARD_C = 73.0
COOL_C = 60.0
STEP_KHZ = 100_000
FLOOR_MHZ = 1500
DEFAULT_CEILING_MHZ = 2400
MIN_GAP_S = 6.0
COOL_HOLD_S = 10.0
SAMPLE_S = 2.0
WINDOW = 3
SENSOR_FAIL_LIMIT = 3
EVENTS_KEEP = 500


@dataclass(frozen=True)
class Decision:
    action: str  # hold | down | hard_drop | up | fail_safe
    reason: str
    cap_khz: int


class Governor:
    """Pure cap-stepping logic. Time is passed in, so it is testable without sleeping."""

    def __init__(
        self,
        *,
        start_khz: int,
        ceiling_khz: int,
        floor_khz: int = FLOOR_MHZ * 1000,
        step_khz: int = STEP_KHZ,
        hot_c: float = HOT_C,
        hard_c: float = HARD_C,
        cool_c: float = COOL_C,
        cool_hold_s: float = COOL_HOLD_S,
        min_gap_s: float = MIN_GAP_S,
    ) -> None:
        if floor_khz > ceiling_khz:
            raise ValueError("floor must not be above ceiling")
        self.ceiling_khz = ceiling_khz
        self.floor_khz = floor_khz
        self.step_khz = step_khz
        self.hot_c = hot_c
        self.hard_c = hard_c
        self.cool_c = cool_c
        self.cool_hold_s = cool_hold_s
        self.min_gap_s = min_gap_s
        self.cap_khz = min(max(start_khz, floor_khz), ceiling_khz)
        self._last_step: float | None = None
        self._cool_since: float | None = None

    def _gap_ok(self, now: float) -> bool:
        return self._last_step is None or now - self._last_step >= self.min_gap_s

    def _move(self, now: float, new_khz: int, action: str, reason: str) -> Decision:
        self._last_step = now
        self.cap_khz = new_khz
        return Decision(action, reason, new_khz)

    def _hold(self, reason: str) -> Decision:
        return Decision("hold", reason, self.cap_khz)

    def decide(self, now: float, temp_c: float) -> Decision:
        if temp_c >= self.hard_c:
            self._cool_since = None
            if self.cap_khz > self.floor_khz:
                return self._move(
                    now,
                    self.floor_khz,
                    "hard_drop",
                    f"{temp_c:.1f} C >= {self.hard_c:.0f} C, drop to floor",
                )
            return self._hold(f"{temp_c:.1f} C at floor")

        if temp_c >= self.hot_c:
            self._cool_since = None
            if self.cap_khz <= self.floor_khz:
                return self._hold(f"{temp_c:.1f} C at floor")
            if not self._gap_ok(now):
                return self._hold(f"{temp_c:.1f} C, waiting for step gap")
            new = max(self.cap_khz - self.step_khz, self.floor_khz)
            return self._move(now, new, "down", f"{temp_c:.1f} C >= {self.hot_c:.0f} C")

        if temp_c <= self.cool_c:
            if self._cool_since is None:
                self._cool_since = now
            held = now - self._cool_since
            if self.cap_khz >= self.ceiling_khz:
                return self._hold("at ceiling")
            if held >= self.cool_hold_s and self._gap_ok(now):
                new = min(self.cap_khz + self.step_khz, self.ceiling_khz)
                self._cool_since = now  # next step up needs another full hold
                return self._move(
                    now, new, "up", f"{temp_c:.1f} C <= {self.cool_c:.0f} C for {held:.0f}s"
                )
            return self._hold(f"cool for {held:.0f}s, waiting")

        self._cool_since = None
        return self._hold(f"{temp_c:.1f} C in hold band")

    def fail_safe(self, now: float, reason: str) -> Decision:
        self._cool_since = None
        if self.cap_khz > self.floor_khz:
            return self._move(now, self.floor_khz, "fail_safe", reason)
        return self._hold(reason)


class ThermalDaemon:
    def __init__(
        self,
        *,
        sysfs_root: Path,
        state_dir: Path,
        ceiling_mhz: int = DEFAULT_CEILING_MHZ,
        floor_mhz: int = FLOOR_MHZ,
        dry_run: bool = False,
        read_throttled=pc.read_throttled_hex,
    ) -> None:
        self.sysfs_root = Path(sysfs_root)
        self.state_dir = Path(state_dir)
        self.dry_run = dry_run
        self.read_throttled = read_throttled
        self.policies = pc.cpufreq_policies(self.sysfs_root)
        if not self.policies:
            raise RuntimeError(f"no cpufreq policies under {pc.cpufreq_root(self.sysfs_root)}")
        self.temp_path = self.sysfs_root / "class" / "thermal" / "thermal_zone0" / "temp"
        self.state_path = self.state_dir / "thermal.json"
        self.events_path = self.state_dir / "thermal_events.log"
        self.samples: deque[float] = deque(maxlen=WINDOW)
        self.sensor_failures = 0
        self.max_khz = {
            policy: pc.read_int(policy / "cpuinfo_max_freq") or ceiling_mhz * 1000
            for policy in self.policies
        }
        start_khz = pc.read_int(self.policies[0] / "scaling_max_freq") or ceiling_mhz * 1000
        self.gov = Governor(
            start_khz=start_khz,
            ceiling_khz=ceiling_mhz * 1000,
            floor_khz=floor_mhz * 1000,
        )
        self.last_action = "none"
        self.reason = "starting"
        self.running = False
        self.last_temp: float | None = None
        self.last_avg: float | None = None

    # -- sysfs -------------------------------------------------------------

    def _write_cap(self, khz: int) -> list[str]:
        errors: list[str] = []
        for policy in self.policies:
            try:
                (policy / "scaling_max_freq").write_text(str(khz), encoding="utf-8")
            except OSError as exc:
                errors.append(f"{policy.name}: {exc}")
        return errors

    # -- logging -----------------------------------------------------------

    def _event(self, wall: float, text: str) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(wall))
        pc.append_event(self.events_path, f"{stamp} {text}", keep=EVENTS_KEEP)

    def _write_state(self, wall: float) -> None:
        temp_c, avg_c = self.last_temp, self.last_avg
        doc = {
            "ts": wall,
            "temp_c": None if temp_c is None else round(temp_c, 2),
            "avg_temp_c": None if avg_c is None else round(avg_c, 2),
            "cap_mhz": self.gov.cap_khz // 1000,
            "ceiling_mhz": self.gov.ceiling_khz // 1000,
            "floor_mhz": self.gov.floor_khz // 1000,
            "last_action": self.last_action,
            "reason": self.reason,
            "throttled": self.read_throttled(),
            "dry_run": self.dry_run,
            "running": self.running,
            "policies": len(self.policies),
        }
        pc.write_json_atomic(self.state_path, doc)

    # -- lifecycle ---------------------------------------------------------

    def start(self, wall: float | None = None) -> None:
        wall = time.time() if wall is None else wall
        self.running = True
        errors = [] if self.dry_run else self._write_cap(self.gov.cap_khz)
        self.last_action = f"start cap {self.gov.cap_khz // 1000} MHz"
        self.reason = "started"
        text = (
            f"start cap={self.gov.cap_khz // 1000} ceiling={self.gov.ceiling_khz // 1000} "
            f"floor={self.gov.floor_khz // 1000} policies={len(self.policies)} "
            f"dry_run={self.dry_run}"
        )
        if errors:
            text += f" WRITE_ERRORS={errors}"
        self._event(wall, text)
        self._write_state(wall)

    def restore(self, wall: float | None = None) -> None:
        wall = time.time() if wall is None else wall
        self.running = False
        if not self.dry_run:
            for policy in self.policies:
                try:
                    (policy / "scaling_max_freq").write_text(
                        str(self.max_khz[policy]), encoding="utf-8"
                    )
                except OSError as exc:
                    self._event(wall, f"restore failed {policy.name}: {exc}")
        self.last_action = "restored scaling_max_freq to cpuinfo_max_freq"
        self.reason = "daemon stopped"
        self._event(wall, "stop: restored cpuinfo_max_freq on all policies")
        self._write_state(wall)

    # -- one control step --------------------------------------------------

    def tick(self, now: float, temp_c: float | None, wall: float | None = None) -> Decision:
        """Feed one temperature reading (None = unreadable) and act on the decision."""
        wall = time.time() if wall is None else wall
        prev_khz = self.gov.cap_khz
        avg: float | None = None
        if temp_c is None:
            self.sensor_failures += 1
            if self.sensor_failures >= SENSOR_FAIL_LIMIT:
                decision = self.gov.fail_safe(now, "sensor unreadable, failing safe")
            else:
                decision = Decision(
                    "hold", f"sensor read failed ({self.sensor_failures})", prev_khz
                )
        else:
            self.sensor_failures = 0
            self.samples.append(temp_c)
            avg = statistics.fmean(self.samples)
            decision = self.gov.decide(now, avg)

        self.reason = decision.reason
        if decision.action != "hold":
            errors = [] if self.dry_run else self._write_cap(decision.cap_khz)
            text = (
                f"{decision.action} cap {prev_khz // 1000}->{decision.cap_khz // 1000} MHz "
                f"temp={temp_c if temp_c is None else round(temp_c, 1)} "
                f"avg={None if avg is None else round(avg, 1)} reason={decision.reason}"
            )
            if self.dry_run:
                text += " (dry-run)"
            if errors:
                text += f" WRITE_ERRORS={errors}"
            self.last_action = text
            self._event(wall, text)
        self.last_temp = temp_c
        self.last_avg = avg
        self._write_state(wall)
        return decision

    def run(self, interval: float = SAMPLE_S, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        try:
            self.start()
            while not stop.is_set():
                temp = pc.read_temp_c(self.temp_path)
                try:
                    self.tick(time.monotonic(), temp)
                except Exception:  # keep governing even if one tick fails
                    traceback.print_exc(file=sys.stderr)
                stop.wait(interval)
        finally:
            self.restore()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rakshak Pi 5 thermal cap governor")
    parser.add_argument("--ceiling", type=int, default=DEFAULT_CEILING_MHZ, help="max cap, MHz")
    parser.add_argument("--floor", type=int, default=FLOOR_MHZ, help="min cap, MHz")
    parser.add_argument("--interval", type=float, default=SAMPLE_S, help="sample period, s")
    parser.add_argument("--dry-run", action="store_true", help="decide and log, never write sysfs")
    parser.add_argument("--state-dir", type=Path, default=Path("/run/rakshak"))
    parser.add_argument("--sysfs-root", type=Path, default=Path("/sys"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not 0 < args.floor <= args.ceiling:
        parser.error("need 0 < --floor <= --ceiling")
    stop = threading.Event()

    def _request_stop(signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    daemon = ThermalDaemon(
        sysfs_root=args.sysfs_root,
        state_dir=args.state_dir,
        ceiling_mhz=args.ceiling,
        floor_mhz=args.floor,
        dry_run=args.dry_run,
    )
    daemon.run(interval=args.interval, stop=stop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
