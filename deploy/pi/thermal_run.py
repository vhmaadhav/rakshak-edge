#!/usr/bin/env python3
"""Run a job under thermal guard rails, as a normal user (no root).

    python3 thermal_run.py --name NAME [--pause 75 --resume 65 --kill 80] -- cmd args...

The command runs in its own process group under `nice -n 10 ionice -c3`.
While it runs, the temperature is sampled every second:

  * temp >= --pause   -> SIGSTOP the whole group
  * temp <= --resume  -> SIGCONT the group
  * temp >= --kill    -> SIGCONT + SIGTERM, SIGKILL after 5 s, exit 3

Outputs in --run-dir (default ~/rakshak-pi/run):
  current_job.json       live job state (running / paused / done / killed)
  <name>-<ts>.csv        ts,temp,cap,freq,throttled,power_w every sample
  <name>-<ts>.log        combined stdout/stderr of the command
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import pi_common as pc

PAUSE_C = 75.0
RESUME_C = 65.0
KILL_C = 80.0
KILL_GRACE_S = 5.0
SAMPLE_S = 1.0
EXIT_KILLED = 3
TEMP_PATH = Path("/sys/class/thermal/thermal_zone0/temp")
SYSFS_ROOT = Path("/sys")
THERMAL_DIR = Path("/run/rakshak")


def default_run_dir() -> Path:
    return Path.home() / "rakshak-pi" / "run"


class Guard:
    """Threshold logic. Pure, so tests can feed temperatures directly."""

    def __init__(self, pause_c: float, resume_c: float, kill_c: float) -> None:
        if not resume_c < pause_c < kill_c:
            raise ValueError("need resume < pause < kill")
        self.pause_c = pause_c
        self.resume_c = resume_c
        self.kill_c = kill_c

    def decide(self, temp_c: float | None, paused: bool) -> str:
        """Return one of: none, pause, resume, kill."""
        if temp_c is None:
            return "none"
        if temp_c >= self.kill_c:
            return "kill"
        if not paused and temp_c >= self.pause_c:
            return "pause"
        if paused and temp_c <= self.resume_c:
            return "resume"
        return "none"


def wrap_command(cmd: list[str]) -> list[str]:
    prefix: list[str] = []
    if shutil.which("nice"):
        prefix += ["nice", "-n", "10"]
    if shutil.which("ionice"):
        prefix += ["ionice", "-c3"]
    return prefix + cmd


def default_probe() -> dict[str, object]:
    """Snapshot of cap, average frequency, throttle flags and PMIC power."""
    policies = pc.cpufreq_policies(SYSFS_ROOT)
    cap = pc.read_int(policies[0] / "scaling_max_freq") if policies else None
    freqs = [pc.read_int(p / "scaling_cur_freq") for p in policies]
    known = [f for f in freqs if f]
    rails = pc.read_pmic_rails()
    return {
        "cap_mhz": None if cap is None else cap // 1000,
        "freq_mhz": round(sum(known) / len(known) / 1000) if known else None,
        "throttled": pc.read_throttled_hex(),
        "power_w": pc.pmic_power_w(rails),
    }


class Job:
    def __init__(
        self,
        *,
        name: str,
        cmd: list[str],
        guard: Guard,
        run_dir: Path,
        read_temp: Callable[[], float | None] = lambda: pc.read_temp_c(TEMP_PATH),
        probe: Callable[[], dict[str, object]] = default_probe,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        sample_s: float = SAMPLE_S,
        kill_grace_s: float = KILL_GRACE_S,
        echo: BinaryIO | None = None,
    ) -> None:
        self.name = name
        self.cmd = cmd
        self.guard = guard
        self.run_dir = Path(run_dir)
        self.read_temp = read_temp
        self.probe = probe
        self.clock = clock
        self.sleep = sleep
        self.sample_s = sample_s
        self.kill_grace_s = kill_grace_s
        self.echo = echo
        self.safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "job"
        self.stamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_path = self.run_dir / f"{self.safe_name}-{self.stamp}.log"
        self.csv_path = self.run_dir / f"{self.safe_name}-{self.stamp}.csv"
        self.job_path = self.run_dir / "current_job.json"
        self.proc: subprocess.Popen[bytes] | None = None
        self.paused = False
        self.pauses = 0
        self.paused_s = 0.0
        self._pause_started = 0.0
        self.temp: float | None = None
        self.max_temp: float | None = None
        self.throttled_before: str | None = None
        self.throttled_after: str | None = None
        self.exit_code: int | None = None
        self.state = "running"
        self.started_wall = 0.0
        self._t0 = 0.0
        self._stop_requested = threading.Event()
        self._logf: BinaryIO | None = None

    def request_stop(self) -> None:
        """Called from a signal handler: the job is killed on the next loop turn."""
        self._stop_requested.set()

    # -- process control ---------------------------------------------------

    def _alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _signal_group(self, sig: int) -> None:
        if self.proc is None or not self._alive():
            return
        try:
            os.killpg(self.proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _pause(self, now: float) -> None:
        self._signal_group(signal.SIGSTOP)
        self.paused = True
        self.pauses += 1
        self._pause_started = now

    def _resume(self, now: float) -> None:
        self._signal_group(signal.SIGCONT)
        self.paused_s += now - self._pause_started
        self.paused = False

    def _terminate(self, now: float) -> None:
        if self.paused:
            self._resume(now)
        self._signal_group(signal.SIGCONT)
        self._signal_group(signal.SIGTERM)
        if self.proc is None:
            return
        try:
            self.proc.wait(timeout=self.kill_grace_s)
        except subprocess.TimeoutExpired:
            self._signal_group(signal.SIGKILL)
            self.proc.wait()

    # -- output ------------------------------------------------------------

    def _pump(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            if self._logf is not None:
                self._logf.write(chunk)
                self._logf.flush()
            if self.echo is not None:
                try:
                    self.echo.write(chunk)
                    self.echo.flush()
                except (OSError, ValueError):
                    self.echo = None

    def _write_job(self, now: float) -> None:
        paused_s = self.paused_s + (now - self._pause_started if self.paused else 0.0)
        doc = {
            "state": self.state,
            "name": self.name,
            "pid": self.proc.pid if self.proc else None,
            "cmd": " ".join(shlex.quote(part) for part in self.cmd),
            "started": self.started_wall,
            "elapsed_s": round(now - self._t0, 1),
            "temp_c": self.temp,
            "max_temp_c": self.max_temp,
            "pause_at_c": self.guard.pause_c,
            "resume_at_c": self.guard.resume_c,
            "kill_at_c": self.guard.kill_c,
            "paused": self.paused,
            "pauses": self.pauses,
            "paused_s": round(paused_s, 1),
            "throttled_before": self.throttled_before,
            "throttled_after": self.throttled_after,
            "exit_code": self.exit_code,
            "log": str(self.log_path),
            "csv": str(self.csv_path),
            "updated": time.time(),
        }
        pc.write_json_atomic(self.job_path, doc)

    def _csv_row(self, wall: float, extra: dict[str, object]) -> None:
        def cell(key: str) -> str:
            value = extra.get(key)
            return "" if value is None else str(value)

        temp = "" if self.temp is None else f"{self.temp:.1f}"
        row = ",".join(
            [
                f"{wall:.1f}",
                temp,
                cell("cap_mhz"),
                cell("freq_mhz"),
                cell("throttled"),
                cell("power_w"),
            ]
        )
        with open(self.csv_path, "a", encoding="utf-8") as handle:
            handle.write(row + "\n")

    # -- main loop ---------------------------------------------------------

    def run(self) -> int:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._logf = open(self.log_path, "wb")  # noqa: SIM115 - closed in finally
        with open(self.csv_path, "w", encoding="utf-8") as handle:
            handle.write("ts,temp,cap,freq,throttled,power_w\n")
        self.started_wall = time.time()
        self._t0 = self.clock()
        self.throttled_before = self.probe().get("throttled")  # type: ignore[assignment]
        self.proc = subprocess.Popen(
            wrap_command(self.cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            process_group=0,
        )
        reader = threading.Thread(target=self._pump, daemon=True)
        reader.start()
        self._write_job(self.clock())
        outcome = "done"
        try:
            while self.proc.poll() is None:
                now = self.clock()
                if self._stop_requested.is_set():
                    outcome = "killed"
                    self._terminate(now)
                    break
                self.temp = self.read_temp()
                if self.temp is not None:
                    self.max_temp = (
                        self.temp if self.max_temp is None else max(self.max_temp, self.temp)
                    )
                action = self.guard.decide(self.temp, self.paused)
                if action == "kill":
                    outcome = "killed"
                    self._terminate(now)
                    break
                if action == "pause":
                    self._pause(now)
                elif action == "resume":
                    self._resume(now)
                self.state = "paused" if self.paused else "running"
                extra = self.probe()
                self._csv_row(time.time(), extra)
                self._write_job(now)
                self.sleep(self.sample_s)
            rc = self.proc.wait()
            reader.join(timeout=5)
            self.exit_code = EXIT_KILLED if outcome == "killed" else _normalize_rc(rc)
            self.state = outcome
            self.throttled_after = self.probe().get("throttled")  # type: ignore[assignment]
            self._write_job(self.clock())
            return self.exit_code
        finally:
            if self.paused:
                self._resume(self.clock())
            if self.proc.stdout is not None:
                self.proc.stdout.close()
            if self._logf is not None:
                self._logf.close()


def _normalize_rc(rc: int) -> int:
    return 128 - rc if rc < 0 else rc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a command with thermal pause/resume/kill guard rails. "
        "Put the command after `--`."
    )
    parser.add_argument("--name", required=True, help="job name used in output file names")
    parser.add_argument("--pause", type=float, default=PAUSE_C, help="SIGSTOP at this temp C")
    parser.add_argument("--resume", type=float, default=RESUME_C, help="SIGCONT at this temp C")
    parser.add_argument("--kill", type=float, default=KILL_C, help="kill at this temp C")
    parser.add_argument("--run-dir", type=Path, default=None, help="default ~/rakshak-pi/run")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--" not in raw:
        build_arg_parser().error("missing `--` before the command")
    split = raw.index("--")
    parser = build_arg_parser()
    args = parser.parse_args(raw[:split])
    cmd = raw[split + 1 :]
    if not cmd:
        parser.error("no command after `--`")
    try:
        guard = Guard(args.pause, args.resume, args.kill)
    except ValueError as exc:
        parser.error(str(exc))
    run_dir = args.run_dir or default_run_dir()
    job = Job(
        name=args.name,
        cmd=cmd,
        guard=guard,
        run_dir=run_dir,
        echo=sys.stdout.buffer,
    )
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _s, _f: job.request_stop())
    rc = job.run()
    summary = {
        "name": job.name,
        "exit_code": job.exit_code,
        "state": job.state,
        "max_temp_c": job.max_temp,
        "pauses": job.pauses,
        "paused_s": round(job.paused_s, 1),
        "wall_s": round(job.clock() - job._t0, 1),
        "log": str(job.log_path),
        "csv": str(job.csv_path),
    }
    print("thermal_run:", json.dumps(summary), file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
