#!/usr/bin/env python3
"""Live monitoring dashboard for a Raspberry Pi 5 (stdlib only).

Serves one self-contained HTML page plus two JSON endpoints:

  /               the dashboard page (no external assets)
  /api/snapshot   {"latest": sample, "history": [slim samples, last 15 min]}
  /api/stream     Server-Sent Events, one full sample per sampler tick

A background sampler collects a sample every 5 s, or every 1 s while
<run-dir>/current_job.json says running or paused. Bind to 127.0.0.1 and
reach it over an SSH tunnel.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pi_common as pc

HISTORY_S = 15 * 60
IDLE_SAMPLE_S = 5.0
JOB_SAMPLE_S = 1.0
TOP_N = 8
EVENTS_TAIL = 30
LOG_TAIL = 40
IFACE = "wlan0"
GIB = 1024**3
MIB = 1024**2


# -- pure parsers (unit tested) ---------------------------------------------


def parse_proc_stat(text: str | None) -> dict[str, tuple[int, int]]:
    """Map 'cpu', 'cpu0', ... to (idle_jiffies, total_jiffies)."""
    out: dict[str, tuple[int, int]] = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if not parts or not parts[0].startswith("cpu"):
            continue
        try:
            nums = [int(x) for x in parts[1:]]
        except ValueError:
            continue
        if len(nums) < 4:
            continue
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)  # idle + iowait
        total = sum(nums[:8])  # guest time is already inside user/nice
        out[parts[0]] = (idle, total)
    return out


def cpu_percentages(
    prev: dict[str, tuple[int, int]], cur: dict[str, tuple[int, int]]
) -> dict[str, float] | None:
    """Busy percentage per CPU between two /proc/stat readings."""
    if not prev or not cur:
        return None
    result: dict[str, float] = {}
    for name, (idle1, total1) in cur.items():
        if name not in prev:
            continue
        idle0, total0 = prev[name]
        d_total = total1 - total0
        d_idle = idle1 - idle0
        if d_total <= 0:
            result[name] = 0.0
        else:
            pct = 100.0 * (d_total - d_idle) / d_total
            result[name] = round(min(100.0, max(0.0, pct)), 1)
    return result or None


def cpu_summary(percentages: dict[str, float] | None) -> dict | None:
    if percentages is None:
        return None
    cores = sorted(
        (k for k in percentages if k != "cpu"), key=lambda k: int(re.sub(r"\D", "", k) or 0)
    )
    return {"total": percentages.get("cpu"), "cores": [percentages[k] for k in cores]}


def parse_meminfo(text: str | None) -> dict[str, int]:
    """Return /proc/meminfo values in kB, keyed by field name."""
    out: dict[str, int] = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        fields = rest.split()
        if fields and fields[0].isdigit():
            out[key.strip()] = int(fields[0])
    return out


def parse_pid_stat(text: str | None) -> tuple[str, int, int] | None:
    """Return (comm, utime+stime ticks, rss pages) from /proc/<pid>/stat."""
    if not text:
        return None
    lparen = text.find("(")
    rparen = text.rfind(")")
    if lparen < 0 or rparen < 0:
        return None
    name = text[lparen + 1 : rparen]
    fields = text[rparen + 2 :].split()
    # fields[0] is stat field 3 (state); utime=14 -> 11, stime=15 -> 12, rss=24 -> 21
    if len(fields) < 22:
        return None
    try:
        return name, int(fields[11]) + int(fields[12]), int(fields[21])
    except ValueError:
        return None


def process_top(
    prev: dict[int, int],
    cur: dict[int, tuple[str, int, int]],
    dt: float | None,
    clk_tck: int,
    page_size: int,
    n: int = TOP_N,
) -> list[dict]:
    """Top-n processes by CPU% (ticks delta / wall delta), from two readings."""
    if dt is None or dt <= 0:
        return []
    rows = []
    for pid, (name, ticks, rss_pages) in cur.items():
        if pid not in prev:
            continue
        pct = 100.0 * (ticks - prev[pid]) / (clk_tck * dt)
        rows.append(
            {
                "pid": pid,
                "name": name,
                "cpu_pct": round(max(0.0, pct), 1),
                "rss_mb": round(rss_pages * page_size / MIB, 1),
            }
        )
    rows.sort(key=lambda r: (r["cpu_pct"], r["rss_mb"]), reverse=True)
    return rows[:n]


def net_rates(
    prev: tuple[int, int, float] | None, rx: int | None, tx: int | None, mono: float
) -> dict | None:
    """kB/s receive/transmit from two byte counters."""
    if rx is None or tx is None or prev is None:
        return None
    p_rx, p_tx, p_t = prev
    dt = mono - p_t
    if dt <= 0:
        return None
    return {
        "rx_kBps": round((rx - p_rx) / dt / 1024, 1),
        "tx_kBps": round((tx - p_tx) / dt / 1024, 1),
    }


def build_mem(meminfo: dict[str, int]) -> dict | None:
    if "MemTotal" not in meminfo or "MemAvailable" not in meminfo:
        return None
    total = meminfo["MemTotal"] / 1024
    avail = meminfo["MemAvailable"] / 1024
    swap_total = meminfo.get("SwapTotal", 0) / 1024
    swap_free = meminfo.get("SwapFree", 0) / 1024
    return {
        "total_mb": round(total),
        "avail_mb": round(avail),
        "used_mb": round(total - avail),
        "swap_used_mb": round(swap_total - swap_free),
    }


def slim(sample: dict) -> dict:
    """Compact per-tick record kept in the 15-minute history."""
    job = sample.get("job") or {}
    freq = sample.get("freq_mhz") or []
    cpu = sample.get("cpu_pct") or {}
    mem = sample.get("mem") or {}
    return {
        "ts": sample.get("ts"),
        "temp_c": sample.get("temp_c"),
        "cap_mhz": sample.get("cap_mhz"),
        "freq_avg_mhz": round(sum(freq) / len(freq)) if freq else None,
        "power_w": sample.get("power_w"),
        "ext5v_v": sample.get("ext5v_v"),
        "throttled": sample.get("throttled"),
        "cpu_total": cpu.get("total"),
        "mem_used_mb": mem.get("used_mb"),
        "job_state": job.get("state"),
    }


# -- collector ----------------------------------------------------------------


class Collector:
    """Gathers one sample. Holds previous readings so rates and deltas work."""

    def __init__(
        self,
        *,
        sysfs: Path = Path("/sys"),
        proc: Path = Path("/proc"),
        thermal_dir: Path = Path("/run/rakshak"),
        run_dir: Path | None = None,
        iface: str = IFACE,
        disk_path: str = "/",
    ) -> None:
        self.sysfs = Path(sysfs)
        self.proc = Path(proc)
        self.thermal_dir = Path(thermal_dir)
        self.run_dir = Path(run_dir) if run_dir else Path.home() / "rakshak-pi" / "run"
        self.iface = iface
        self.disk_path = disk_path
        self.clk_tck = os.sysconf("SC_CLK_TCK")
        self.page_size = os.sysconf("SC_PAGE_SIZE")
        self._policies = pc.cpufreq_policies(self.sysfs)
        self._prev_cpu = parse_proc_stat(pc.read_text(self.proc / "stat"))
        self._prev_proc: dict[int, int] = {}
        self._prev_mono: float | None = None
        self._prev_net: tuple[int, int, float] | None = None

    def _core_freqs_mhz(self) -> list[int]:
        cores = []
        for path in self.sysfs.glob("devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"):
            khz = pc.read_int(path)
            if khz is not None:
                cores.append((int(re.sub(r"\D", "", path.parents[1].name) or 0), khz // 1000))
        return [mhz for _, mhz in sorted(cores)]

    def _top_processes(self, dt: float | None) -> list[dict]:
        current: dict[int, tuple[str, int, int]] = {}
        for entry in self.proc.iterdir():
            if not entry.name.isdigit():
                continue
            parsed = parse_pid_stat(pc.read_text(entry / "stat"))
            if parsed is not None:
                current[int(entry.name)] = parsed
        rows = process_top(
            self._prev_proc,
            current,
            dt,
            self.clk_tck,
            self.page_size,
        )
        self._prev_proc = {pid: ticks for pid, (_n, ticks, _r) in current.items()}
        return rows

    def _net(self, mono: float) -> dict | None:
        base = self.sysfs / "class" / "net" / self.iface / "statistics"
        rx = pc.read_int(base / "rx_bytes")
        tx = pc.read_int(base / "tx_bytes")
        rate = net_rates(self._prev_net, rx, tx, mono)
        if rx is not None and tx is not None:
            self._prev_net = (rx, tx, mono)
        return rate

    def collect(self, mono: float | None = None, wall: float | None = None) -> dict:
        mono = time.monotonic() if mono is None else mono
        wall = time.time() if wall is None else wall
        dt = None if self._prev_mono is None else mono - self._prev_mono
        self._prev_mono = mono

        policy0 = self._policies[0] if self._policies else None
        rails = pc.read_pmic_rails()
        throttled = pc.read_throttled_hex()
        flags = pc.decode_throttled(throttled)
        cap_khz = pc.read_int(policy0 / "scaling_max_freq") if policy0 else None
        governor = pc.read_text(policy0 / "scaling_governor") if policy0 else None

        cur_cpu = parse_proc_stat(pc.read_text(self.proc / "stat"))
        cpu = cpu_summary(cpu_percentages(self._prev_cpu, cur_cpu))
        self._prev_cpu = cur_cpu

        meminfo = parse_meminfo(pc.read_text(self.proc / "meminfo"))
        mem = build_mem(meminfo)
        swap_used = None if mem is None else mem["swap_used_mb"]

        stat = os.statvfs(self.disk_path)
        disk = {
            "total_gb": round(stat.f_blocks * stat.f_frsize / GIB, 1),
            "used_gb": round((stat.f_blocks - stat.f_bfree) * stat.f_frsize / GIB, 1),
            "free_gb": round(stat.f_bavail * stat.f_frsize / GIB, 1),
        }

        loadavg = pc.read_text(self.proc / "loadavg") or ""
        load = [float(x) for x in loadavg.split()[:3] if _is_float(x)] or None
        uptime_parts = (pc.read_text(self.proc / "uptime") or "").split()
        uptime = float(uptime_parts[0]) if uptime_parts and _is_float(uptime_parts[0]) else None

        thermald_doc = pc.read_json(self.thermal_dir / "thermal.json")
        if thermald_doc is None:
            thermald: object = "not installed"
        else:
            thermald = dict(thermald_doc)
            thermald["age_s"] = round(wall - float(thermald.get("ts", wall)), 1)

        job = pc.read_json(self.run_dir / "current_job.json")
        job_log: list[str] = []
        if job and job.get("log"):
            job_log = pc.tail_lines(self.run_dir / Path(str(job["log"])).name, LOG_TAIL)

        return {
            "ts": wall,
            "temp_c": pc.read_temp_c(self.sysfs / "class" / "thermal" / "thermal_zone0" / "temp"),
            "freq_mhz": self._core_freqs_mhz(),
            "cap_mhz": None if cap_khz is None else cap_khz // 1000,
            "governor": governor,
            "throttled": throttled,
            "throttled_flags": flags,
            "ext5v_v": rails.get("EXT5V_V"),
            "power_w": pc.pmic_power_w(rails),
            "vdd_core": {"v": rails.get("VDD_CORE_V"), "a": rails.get("VDD_CORE_A")},
            "cpu_pct": cpu,
            "loadavg": load,
            "uptime_s": round(uptime) if uptime is not None else None,
            "mem": mem,
            "swap_used_mb": swap_used,
            "disk": disk,
            "net": {self.iface: self._net(mono)},
            "top": self._top_processes(dt),
            "thermald": thermald,
            "thermald_events": pc.tail_lines(self.thermal_dir / "thermal_events.log", EVENTS_TAIL),
            "job": job,
            "job_log": job_log,
        }


def _is_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


# -- sampler ------------------------------------------------------------------


class Sampler:
    """Runs the collector in a thread and keeps the ring buffer and SSE sequence."""

    def __init__(
        self,
        collector: Collector,
        *,
        idle_s: float = IDLE_SAMPLE_S,
        job_s: float = JOB_SAMPLE_S,
        history_s: float = HISTORY_S,
        collect: Callable[[], dict] | None = None,
    ) -> None:
        self._collector = collector
        self._collect = collect or collector.collect
        self._idle_s = idle_s
        self._job_s = job_s
        self._history_s = history_s
        self._cond = threading.Condition()
        self._latest: dict | None = None
        self._seq = 0
        self._history: deque[dict] = deque()
        self._halt = threading.Event()

    def interval(self) -> float:
        job = (self._latest or {}).get("job") or {}
        return self._job_s if job.get("state") in ("running", "paused") else self._idle_s

    def tick(self) -> dict:
        sample = self._collect()
        with self._cond:
            self._latest = sample
            self._seq += 1
            self._history.append(slim(sample))
            cutoff = sample["ts"] - self._history_s
            while self._history and (self._history[0]["ts"] or 0) < cutoff:
                self._history.popleft()
            self._cond.notify_all()
        return sample

    def run(self) -> None:
        while not self._halt.wait(self.interval()):
            try:
                self.tick()
            except Exception:  # keep sampling through transient read errors
                traceback.print_exc(file=sys.stderr)

    @property
    def stopped(self) -> bool:
        return self._halt.is_set()

    def stop(self) -> None:
        self._halt.set()
        with self._cond:
            self._cond.notify_all()

    def snapshot(self) -> dict:
        with self._cond:
            return {"latest": self._latest, "history": list(self._history)}

    def wait_next(self, seq: int, timeout: float) -> tuple[dict | None, int]:
        with self._cond:
            self._cond.wait_for(lambda: self._seq != seq or self._halt.is_set(), timeout=timeout)
            if self._seq != seq and self._latest is not None:
                return self._latest, self._seq
            return None, seq


# -- HTTP ---------------------------------------------------------------------


# -- benchmarks (rbench) ----------------------------------------------------
#
# The Benchmarks tab reads results/index.jsonl and results/<run_id>/result.json, and starts ONE
# `rbench run` at a time through the bench CLI (deploy/pi/bench/bin/rbench) with the bench venv.
# Only 127.0.0.1 / ::1 clients may use /api/bench/*.

BENCH_HOME_DEFAULT = Path.home() / "rakshak-bench"
RUN_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{6}$")
BENCH_LOCK = ".rbench.lock"
BENCH_LIMIT_MAX = 200000
BENCH_THREADS_MAX = 4
BENCH_BODY_MAX = 4096


class BenchError(Exception):
    """An API error with the HTTP status to answer with."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def is_local_client(addr: str) -> bool:
    """True for loopback clients only (the dashboard is also bound to 127.0.0.1)."""
    return addr in ("127.0.0.1", "::1", "::ffff:127.0.0.1")


class BenchControl:
    """Reads bench results and launches one benchmark at a time."""

    def __init__(self, home: Path, python: str | None = None, timeout_s: float = 60.0,
                 wrapper: Path | None = None) -> None:
        self.home = Path(home)
        self.results = self.home / "results"
        self.python = python or str(self.home / ".venv" / "bin" / "python")
        self.wrapper = wrapper or (self.home / "deploy" / "pi" / "bench" / "bin" / "rbench")
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._started: dict[str, Any] | None = None

    # -- reading ----------------------------------------------------------

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        path = self.results / "index.jsonl"
        rows: list[dict[str, Any]] = []
        for raw in (pc.read_text(path) or "").splitlines():
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and RUN_ID_RE.match(str(row.get("run_id", ""))):
                rows.append(row)
        return rows[-limit:][::-1]

    def get_run(self, run_id: str) -> dict[str, Any]:
        if not RUN_ID_RE.match(run_id):
            raise BenchError(400, "bad run id")
        doc = pc.read_json(self.results / run_id / "result.json")
        if doc is None or doc.get("schema") != "rbench/1":
            raise BenchError(404, "no such run")
        return doc

    # -- CLI calls --------------------------------------------------------

    def _cli(self, args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ, RBENCH_HOME=str(self.home))
        return subprocess.run([self.python, str(self.wrapper), *args], capture_output=True,
                              text=True, timeout=timeout, env=env, check=False)

    def options(self) -> dict[str, Any]:
        """Models (with the datasets each can run on) and datasets, from `rbench list`."""
        out: dict[str, Any] = {}
        for what in ("models", "datasets"):
            try:
                done = self._cli(["list", what, "--no-hash", "--json"], timeout=self.timeout_s)
                doc = json.loads(done.stdout or "{}")
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
                raise BenchError(503, f"rbench is not available: {exc}") from exc
            if done.returncode != 0 or what not in doc:
                raise BenchError(503, f"rbench list {what} failed (exit {done.returncode})")
            out[what] = doc[what]
        return {"models": out["models"], "datasets": out["datasets"],
                "limit_default": 300, "limit_max": BENCH_LIMIT_MAX,
                "threads": list(range(1, BENCH_THREADS_MAX + 1))}

    def _lock_held(self) -> bool:
        lock_path = self.results / BENCH_LOCK
        if not lock_path.exists():
            return False
        with open(lock_path, "w") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(handle, fcntl.LOCK_UN)
        return False

    def busy(self) -> bool:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return True
        return self._lock_held()

    def status(self) -> dict[str, Any]:
        return {"busy": self.busy(), "started": self._started}

    def start(self, body: dict[str, Any]) -> dict[str, Any]:
        model = body.get("model")
        dataset = body.get("dataset")
        if not isinstance(model, str) or not isinstance(dataset, str):
            raise BenchError(400, "model and dataset are required")
        limit = body.get("limit", 300)
        threads = body.get("threads", 1)
        if not isinstance(limit, int) or isinstance(limit, bool) or not (
                1 <= limit <= BENCH_LIMIT_MAX):
            raise BenchError(400, f"limit must be an integer in 1..{BENCH_LIMIT_MAX}")
        if not isinstance(threads, int) or isinstance(threads, bool) or not (
                1 <= threads <= BENCH_THREADS_MAX):
            raise BenchError(400, f"threads must be an integer in 1..{BENCH_THREADS_MAX}")
        opts = self.options()
        match = next((m for m in opts["models"] if m.get("id") == model), None)
        if match is None or not match.get("enabled", True):
            raise BenchError(400, f"unknown or disabled model {model!r}")
        if dataset not in (match.get("compatible_datasets") or []):
            raise BenchError(400, f"{dataset!r} is not compatible with {model!r}")
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise BenchError(409, "a benchmark is already running")
            if self._lock_held():
                raise BenchError(409, "a benchmark is already running (lock held)")
            cmd = [self.python, str(self.wrapper), "run", "--model", model, "--dataset", dataset,
                   "--limit", str(limit), "--threads", str(threads), "--json"]
            env = dict(os.environ, RBENCH_HOME=str(self.home))
            self._proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL, start_new_session=True)
            self._started = {"model": model, "dataset": dataset, "limit": limit,
                             "threads": threads, "pid": self._proc.pid,
                             "started": time.time()}
        return dict(self._started)


def _bench_summary(doc: dict[str, Any]) -> dict[str, Any]:
    """The few fields the detail card shows, pulled from a result.json."""
    perf = doc.get("perf") or {}
    lat = perf.get("latency_ms") or {}
    thermal = doc.get("thermal") or {}
    return {
        "run_id": doc.get("run_id"), "status": doc.get("status"), "error": doc.get("error"),
        "model": (doc.get("model") or {}).get("id"),
        "dataset": (doc.get("dataset") or {}).get("id"),
        "n": (doc.get("dataset") or {}).get("n_used"),
        "p50_ms": lat.get("p50"), "p95_ms": lat.get("p95"), "p99_ms": lat.get("p99"),
        "throughput_items_s": perf.get("throughput_items_s"),
        "peak_rss_mb": perf.get("peak_rss_mb"),
        "quality_key": (doc.get("quality") or {}).get("key_metric"),
        "temp_max_c": thermal.get("temp_max_c"),
        "throttled_before": thermal.get("throttled_before"),
        "throttled_after": thermal.get("throttled_after"),
        "pauses": thermal.get("pauses"),
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "rakshak-pi-dashboard/1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return  # keep the journal quiet

    @property
    def sampler(self) -> Sampler:
        return self.server.sampler  # type: ignore[attr-defined,no-any-return]

    @property
    def bench(self) -> BenchControl:
        return self.server.bench  # type: ignore[attr-defined,no-any-return]

    def _json(self, status: int, doc: object) -> None:
        body = json.dumps(doc, default=str).encode("utf-8")
        self._send(status, "application/json", body)

    def _bench_request(self, method: str, path: str) -> None:
        if not is_local_client(self.client_address[0]):
            self._json(403, {"error": "benchmark API is localhost only"})
            return
        try:
            if method == "GET" and path == "/api/bench/runs":
                self._json(200, {"runs": self.bench.list_runs(50)})
            elif method == "GET" and path.startswith("/api/bench/run/"):
                run_id = path[len("/api/bench/run/"):]
                doc = self.bench.get_run(run_id)
                self._json(200, {"summary": _bench_summary(doc), "result": doc})
            elif method == "GET" and path == "/api/bench/options":
                self._json(200, self.bench.options())
            elif method == "GET" and path == "/api/bench/status":
                self._json(200, self.bench.status())
            elif method == "POST" and path == "/api/bench/run":
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0 or length > BENCH_BODY_MAX:
                    raise BenchError(400, "body must be JSON of 1..4096 bytes")
                try:
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise BenchError(400, "body is not valid JSON") from exc
                if not isinstance(body, dict):
                    raise BenchError(400, "body must be a JSON object")
                self._json(202, self.bench.start(body))
            else:
                self._json(404, {"error": "not found"})
        except BenchError as exc:
            self._json(exc.status, {"error": exc.message})

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        path = urlsplit(self.path).path
        if path.startswith("/api/bench/"):
            self._bench_request("POST", path)
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found\n")

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        path = urlsplit(self.path).path
        if path.startswith("/api/bench/"):
            self._bench_request("GET", path)
        elif path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif path == "/api/snapshot":
            body = json.dumps(self.sampler.snapshot(), default=str).encode("utf-8")
            self._send(200, "application/json", body)
        elif path == "/api/stream":
            self._stream()
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found\n")

    def _send(self, status: int, ctype: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        seq = -1
        try:
            while not self.sampler.stopped:
                sample, seq = self.sampler.wait_next(seq, timeout=15.0)
                if sample is None:
                    payload = b": keepalive\n\n"
                else:
                    payload = b"data: " + json.dumps(sample, default=str).encode("utf-8") + b"\n\n"
                self.wfile.write(payload)
                self.wfile.flush()
        except OSError:  # client went away
            return


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], sampler: Sampler,
                 bench: BenchControl | None = None) -> None:
        super().__init__(address, DashboardHandler)
        self.sampler = sampler
        self.bench = bench or BenchControl(BENCH_HOME_DEFAULT)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 monitoring dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--run-dir", type=Path, default=None, help="default ~/rakshak-pi/run")
    parser.add_argument("--thermal-dir", type=Path, default=Path("/run/rakshak"))
    parser.add_argument("--sysfs", type=Path, default=Path("/sys"))
    parser.add_argument("--proc", type=Path, default=Path("/proc"))
    parser.add_argument("--bench-home", type=Path, default=None,
                        help="rbench root (default: RBENCH_HOME or ~/rakshak-bench)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    collector = Collector(
        sysfs=args.sysfs,
        proc=args.proc,
        thermal_dir=args.thermal_dir,
        run_dir=args.run_dir,
    )
    sampler = Sampler(collector)
    sampler.tick()  # first request never sees an empty snapshot
    threading.Thread(target=sampler.run, name="sampler", daemon=True).start()
    bench_home = args.bench_home or Path(os.environ.get("RBENCH_HOME") or BENCH_HOME_DEFAULT)
    server = DashboardServer((args.host, args.port), sampler, BenchControl(bench_home))
    print(f"pi_dashboard listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sampler.stop()
        server.server_close()
    return 0


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rakshak Pi 5</title>
<style>
:root{
  --bg:#f4f6f9; --card:#ffffff; --fg:#141820; --muted:#5b6675; --line:#d8dee6; --grid:#e6eaf0;
  --ok:#17803a; --warn:#a85f00; --bad:#c42b2b; --accent:#2457d6;
  --c-temp:#2457d6; --c-cap:#7b3fc4; --c-power:#0b8080; --c-cpu:#c2410c;
  --banner-bad:#fdeaea; --banner-warn:#fff3d6;
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0e1116; --card:#171b22; --fg:#e6e9ef; --muted:#98a2b0; --line:#2a313c; --grid:#252b35;
    --ok:#3fb950; --warn:#e6a511; --bad:#ff6b63; --accent:#7aa2ff;
    --c-temp:#7aa2ff; --c-cap:#c39bff; --c-power:#5fd4d4; --c-cpu:#ff9a5c;
    --banner-bad:#3d1719; --banner-warn:#3a2c10;
  }
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--bg);color:var(--fg)}
body{font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;overflow-x:hidden}
main{padding:16px;max-width:1200px;margin:0 auto}
header{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;
  justify-content:space-between;margin-bottom:12px}
h1{font-size:18px;margin:0;font-weight:700}
.conn{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:13px}
.dot{width:10px;height:10px;border-radius:50%;background:var(--muted);display:inline-block}
.dot.live{background:var(--ok)} .dot.wait{background:var(--warn)} .dot.down{background:var(--bad)}
.banner{display:none;border-radius:10px;padding:10px 12px;margin-bottom:12px;font-weight:600}
.banner.show{display:block}
.banner.bad{background:var(--banner-bad);color:var(--bad);border:1px solid var(--bad)}
.banner.warn{background:var(--banner-warn);color:var(--warn);border:1px solid var(--warn)}
.tiles{display:grid;
  grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:12px}
.grid2{display:grid;
  grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:12px;margin-bottom:12px}
.card{background:var(--card);border:1px solid var(--line);
  border-radius:12px;padding:12px;min-width:0}
.card h2{font-size:14px;margin:0 0 8px;font-weight:700}
.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.big{font-size:26px;font-weight:700;
  font-variant-numeric:tabular-nums;margin-top:4px;white-space:nowrap}
.sub{color:var(--muted);font-size:12px;margin-top:3px;overflow-wrap:anywhere}
.ok{color:var(--ok)} .warn{color:var(--warn)} .bad{color:var(--bad)}
canvas{width:100%;height:170px;display:block}
.legend{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:12px;color:var(--muted);margin-top:6px}
.sw{display:inline-block;width:14px;height:3px;
  vertical-align:middle;margin-right:5px;border-radius:2px}
.meta{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:13px;margin-bottom:8px}
.pill{display:inline-block;padding:0 8px;border-radius:999px;
  border:1px solid var(--line);font-size:12px;font-weight:700}
pre{font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px;margin:0;
  max-height:260px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}
ol.ev{font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  margin:0;padding:0;list-style:none;max-height:260px;overflow:auto}
ol.ev li{border-bottom:1px dashed var(--line);padding:3px 0;overflow-wrap:anywhere}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:5px 6px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:600}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
td.name{max-width:40vw;overflow:hidden;text-overflow:ellipsis}
.foot{color:var(--muted);font-size:12px;display:flex;flex-wrap:wrap;gap:4px 16px;margin-top:4px}
.empty{color:var(--muted);font-style:italic}
.tabs{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.tab{padding:5px 14px;border-radius:999px;border:1px solid var(--line);color:var(--fg);
text-decoration:none;font-weight:600;font-size:13px}
.tab.on{background:var(--accent);border-color:var(--accent);color:#fff}
.bform label{display:flex;flex-direction:column;gap:4px;font-size:12px;color:var(--muted);
margin-bottom:8px}
.bform select,.bform input,.bform button{font:inherit;padding:6px 8px;border-radius:8px;
border:1px solid var(--line);background:var(--card);color:var(--fg);max-width:100%}
.bform button{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600;
cursor:pointer}
.bform button:disabled{opacity:.5;cursor:default}
tr.click{cursor:pointer} tr.click:hover{background:var(--grid)}
[hidden]{display:none !important}
</style>
</head>
<body>
<main>
<header>
  <h1>Rakshak Pi 5 <span class="sub" id="host"></span></h1>
  <div class="conn">
    <span class="dot" id="dot"></span><span id="conn">connecting&hellip;</span>
  </div>
</header>

<div class="banner" id="thr" role="alert"></div>

<nav class="tabs" aria-label="views">
  <a class="tab on" id="tab-live" href="#live">Live</a>
  <a class="tab" id="tab-bench" href="#bench">Benchmarks</a>
</nav>

<div id="view-live">
<section class="tiles">
  <div class="card">
    <div class="label">Temperature</div>
    <div class="big" id="t-temp">--</div>
    <div class="sub" id="t-temp-sub">no cooler fitted</div>
  </div>
  <div class="card">
    <div class="label">CPU freq / cap</div>
    <div class="big" id="t-freq">--</div>
    <div class="sub" id="t-freq-sub"></div>
  </div>
  <div class="card">
    <div class="label">Power (PMIC est.)</div>
    <div class="big" id="t-pow">--</div>
    <div class="sub" id="t-ext"></div>
  </div>
  <div class="card">
    <div class="label">Throttle</div>
    <div class="big" id="t-thr">--</div>
    <div class="sub" id="t-thr-sub"></div>
  </div>
  <div class="card">
    <div class="label">CPU</div>
    <div class="big" id="t-cpu">--</div>
    <div class="sub" id="t-cpu-sub"></div>
  </div>
  <div class="card">
    <div class="label">RAM</div>
    <div class="big" id="t-ram">--</div>
    <div class="sub" id="t-ram-sub"></div>
  </div>
</section>

<section class="grid2">
  <div class="card">
    <h2>Temperature and cap</h2>
    <canvas id="c-temp"></canvas>
    <div class="legend">
      <span><i class="sw" style="background:var(--c-temp)"></i>temp &deg;C (left)</span>
      <span><i class="sw" style="background:var(--c-cap)"></i>cap MHz (right)</span>
      <span>dashed: 60 / 68 / 75 / 80 &deg;C</span>
    </div>
  </div>
  <div class="card">
    <h2>Power estimate</h2>
    <canvas id="c-pow"></canvas>
    <div class="legend">
      <span><i class="sw" style="background:var(--c-power)"></i>W from PMIC rails</span>
    </div>
  </div>
  <div class="card">
    <h2>CPU busy %</h2>
    <canvas id="c-cpu"></canvas>
    <div class="legend">
      <span><i class="sw" style="background:var(--c-cpu)"></i>all cores</span>
    </div>
  </div>
</section>

<section class="grid2">
  <div class="card">
    <h2>Current job</h2>
    <div class="meta" id="job-meta"><span class="empty">no job</span></div>
    <pre id="job-log"></pre>
  </div>
  <div class="card">
    <h2>Thermald</h2>
    <div class="meta" id="td-meta"></div>
    <ol class="ev" id="td-ev"></ol>
  </div>
</section>

<section class="card">
  <h2>Top processes <span class="sub">(CPU % of one core)</span></h2>
  <div class="scroll">
    <table>
      <thead><tr>
        <th>PID</th><th>Name</th><th class="n">CPU %</th><th class="n">RSS MB</th>
      </tr></thead>
      <tbody id="procs"></tbody>
    </table>
  </div>
</section>

<div class="foot" id="foot"></div>
</div>

<div id="view-bench" hidden>
<section class="grid2">
  <div class="card">
    <h2>Run a benchmark</h2>
    <form class="bform" id="bench-form">
      <label>Model <select id="bf-model"></select></label>
      <label>Dataset <select id="bf-dataset"></select></label>
      <label>Rows <input id="bf-limit" type="number" min="1" max="200000" value="300"></label>
      <label>Threads <select id="bf-threads"></select></label>
      <button type="submit" id="bf-submit">Start run</button>
    </form>
    <div class="sub" id="bf-note">Runs one benchmark at a time, under the thermal guard
      (pause 75 &deg;C, resume 65 &deg;C, kill 80 &deg;C).</div>
  </div>
  <div class="card">
    <h2>Running benchmark</h2>
    <div class="meta" id="bl-meta"><span class="empty">no benchmark running</span></div>
    <pre id="bl-log"></pre>
  </div>
</section>

<section class="card">
  <h2>Recent runs <span class="sub">(click a row for details)</span></h2>
  <div class="scroll">
    <table>
      <thead><tr>
        <th>Run</th><th>Status</th><th>Model</th><th>Dataset</th><th class="n">n</th>
        <th class="n">p50 ms</th><th class="n">p95 ms</th><th>Key metric</th>
        <th class="n">Max &deg;C</th><th>Throttled after</th>
      </tr></thead>
      <tbody id="bench-runs"></tbody>
    </table>
  </div>
  <div class="sub" id="bench-empty"></div>
</section>

<section class="card" id="bd-card" hidden>
  <h2>Run <span id="bd-id"></span></h2>
  <div class="meta" id="bd-meta"></div>
  <pre id="bd-json"></pre>
</section>
</div>

</main>
<script>
(function () {
  "use strict";
  var HIST_S = 900;
  var hist = [];
  var latest = null;
  var es = null;
  var lastMsg = 0;
  var stateText = "connecting";

  function $(id) { return document.getElementById(id); }
  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  function isNum(v) { return typeof v === "number" && isFinite(v); }
  function fmt(v, d) { return isNum(v) ? v.toFixed(d) : "--"; }
  function setText(id, text) { var el = $(id); if (el) el.textContent = text; }
  function setClass(id, cls) { var el = $(id); if (el) el.className = cls; }
  function tempClass(t) {
    if (!isNum(t)) return "";
    return t >= 70 ? "bad" : (t >= 60 ? "warn" : "ok");
  }
  function clock(t) {
    var opts = { hour: "2-digit", minute: "2-digit", second: "2-digit" };
    return new Date(t * 1000).toLocaleTimeString([], opts);
  }

  function slimOf(s) {
    var cpu = s.cpu_pct || {};
    var freq = s.freq_mhz || [];
    var avg = freq.length ? freq.reduce(function (a, b) { return a + b; }, 0) / freq.length : null;
    return { ts: s.ts, temp_c: s.temp_c, cap_mhz: s.cap_mhz, freq_avg_mhz: avg,
             power_w: s.power_w, cpu_total: cpu.total };
  }

  function pushSample(s) {
    latest = s;
    var rec = slimOf(s);
    var last = hist[hist.length - 1];
    if (!last || last.ts !== rec.ts) { hist.push(rec); }
    var cut = s.ts - HIST_S;
    while (hist.length && hist[0].ts < cut) { hist.shift(); }
  }

  function setConn(text, cls) {
    stateText = text;
    setText("conn", text);
    setClass("dot", "dot " + cls);
  }

  function renderTiles(s) {
    setText("host", s.governor ? "governor " + s.governor : "");
    setText("t-temp", isNum(s.temp_c) ? fmt(s.temp_c, 1) + " °C" : "--");
    setClass("t-temp", "big " + tempClass(s.temp_c));
    var td = s.thermald;
    var tsub = "passive, no cooler";
    if (td && typeof td === "object" && td.reason) { tsub += " · " + td.reason; }
    setText("t-temp-sub", tsub);

    var freqs = s.freq_mhz || [];
    var sum = freqs.reduce(function (a, b) { return a + b; }, 0);
    var avg = freqs.length ? Math.round(sum / freqs.length) : null;
    setText("t-freq", isNum(avg) ? avg + " MHz" : "--");
    var capTxt = isNum(s.cap_mhz) ? "cap " + s.cap_mhz + " MHz" : "cap --";
    setText("t-freq-sub", capTxt + (freqs.length ? " · cores " + freqs.join(" / ") : ""));

    setText("t-pow", isNum(s.power_w) ? fmt(s.power_w, 2) + " W" : "--");
    var ext = s.ext5v_v;
    setText("t-ext", isNum(ext) ? "EXT5V " + fmt(ext, 2) + " V" : "EXT5V --");
    setClass("t-ext", "sub " + (isNum(ext) && ext < 4.85 ? "bad" : ""));

    var flags = s.throttled_flags || { now: [], occurred: [] };
    var thr = $("thr");
    var nowList = flags.now || [];
    var occList = flags.occurred || [];
    if (nowList.length) {
      setText("t-thr", "NOW");
      setClass("t-thr", "big bad");
      thr.className = "banner show bad";
      thr.textContent = "Throttling NOW: " + nowList.join(", ");
    } else if (occList.length) {
      setText("t-thr", "PAST");
      setClass("t-thr", "big warn");
      thr.className = "banner show warn";
      thr.textContent = "Occurred since boot: " + occList.join(", ");
    } else {
      setText("t-thr", s.throttled ? "OK" : "--");
      setClass("t-thr", "big ok");
      thr.className = "banner";
      thr.textContent = "";
    }
    setText("t-thr-sub", (s.throttled ? "throttled=" + s.throttled : "vcgencmd unavailable") +
            (nowList.length + occList.length ? "" : ""));

    var cpu = s.cpu_pct || {};
    setText("t-cpu", isNum(cpu.total) ? fmt(cpu.total, 0) + " %" : "--");
    setText("t-cpu-sub", (cpu.cores || []).map(function (c) { return fmt(c, 0); }).join(" · "));

    var mem = s.mem || {};
    setText("t-ram", isNum(mem.used_mb) ? mem.used_mb + " MB" : "--");
    setText("t-ram-sub", isNum(mem.avail_mb) ? mem.avail_mb + " MB free of " + mem.total_mb +
            " · swap " + (isNum(s.swap_used_mb) ? s.swap_used_mb : "--") + " MB" : "");
  }

  function renderJob(s) {
    var job = s.job;
    var meta = $("job-meta");
    meta.textContent = "";
    var pre = $("job-log");
    if (!job) {
      meta.innerHTML = '<span class="empty">no job</span>';
      pre.textContent = "";
      return;
    }
    function add(text, cls) {
      var span = document.createElement("span");
      span.textContent = text;
      if (cls) span.className = cls;
      meta.appendChild(span);
    }
    var pill = document.createElement("span");
    var cls = job.state === "paused" ? "warn" :
      (job.state === "killed" ? "bad" : (job.state === "done" ? "ok" : ""));
    pill.className = "pill " + cls;
    pill.textContent = String(job.state || "?").toUpperCase();
    meta.appendChild(pill);
    add(String(job.name || ""));
    add("elapsed " + fmt(job.elapsed_s || 0, 0) + " s");
    add("temp " + fmt(job.temp_c, 1) + " °C");
    add("max " + fmt(job.max_temp_c, 1) + " °C");
    add("pauses " + (job.pauses || 0) + " (" + fmt(job.paused_s || 0, 0) + " s)");
    add("exit " + (job.exit_code === null || job.exit_code === undefined ? "--" : job.exit_code));
    add("thr " + (job.throttled_before || "--") + " → " + (job.throttled_after || "--"));
    pre.textContent = (s.job_log || []).join("\n") || "(log empty)";
  }

  function renderThermald(s) {
    var td = s.thermald;
    var meta = $("td-meta");
    var list = $("td-ev");
    meta.textContent = "";
    list.textContent = "";
    if (!td || typeof td !== "object") {
      var none = document.createElement("span");
      none.className = "empty";
      none.textContent = "not installed";
      meta.appendChild(none);
    } else {
      var items = [
        "cap " + (isNum(td.cap_mhz) ? td.cap_mhz : "--") + " MHz",
        "range " + (td.floor_mhz || "--") + "–" + (td.ceiling_mhz || "--"),
        td.running ? "running" : "stopped",
        td.dry_run ? "DRY RUN" : null,
        td.age_s > 30 ? "stale " + Math.round(td.age_s) + " s" : null,
        td.reason ? td.reason : null
      ];
      items.forEach(function (text) {
        if (!text) return;
        var span = document.createElement("span");
        span.textContent = text;
        meta.appendChild(span);
      });
    }
    var events = s.thermald_events || [];
    if (!events.length) {
      var li = document.createElement("li");
      li.className = "empty";
      li.textContent = "no events yet";
      list.appendChild(li);
    }
    events.slice().reverse().forEach(function (line) {
      var li = document.createElement("li");
      li.textContent = line;
      list.appendChild(li);
    });
  }

  function renderProcs(s) {
    var body = $("procs");
    body.textContent = "";
    (s.top || []).forEach(function (p) {
      var tr = document.createElement("tr");
      [[String(p.pid), ""], [String(p.name), "name"],
       [fmt(p.cpu_pct, 1), "n"], [fmt(p.rss_mb, 1), "n"]]
        .forEach(function (pair) {
          var td = document.createElement("td");
          td.textContent = pair[0];
          if (pair[1]) td.className = pair[1];
          tr.appendChild(td);
        });
      body.appendChild(tr);
    });
  }

  function renderFoot(s) {
    var parts = [];
    if (s.disk) parts.push("disk free " + fmt(s.disk.free_gb, 1) +
      " GB of " + fmt(s.disk.total_gb, 1));
    if (s.loadavg) parts.push("load " +
      s.loadavg.map(function (x) { return fmt(x, 2); }).join(" "));
    if (isNum(s.uptime_s)) parts.push("up " + Math.floor(s.uptime_s / 3600) + "h " +
      Math.floor(s.uptime_s % 3600 / 60) + "m");
    var net = s.net || {};
    Object.keys(net).forEach(function (k) {
      if (net[k]) parts.push(k + " rx " + fmt(net[k].rx_kBps, 1) +
        " tx " + fmt(net[k].tx_kBps, 1) + " kB/s");
    });
    if (s.vdd_core && isNum(s.vdd_core.v)) parts.push("VDD_CORE " + fmt(s.vdd_core.v, 3) + " V");
    setText("foot", parts.join("  ·  "));
  }

  function niceTicks(lo, hi, count) {
    var step = Math.max(1, Math.ceil((hi - lo) / count));
    var out = [];
    for (var v = lo; v <= hi + 1e-9; v += step) out.push(v);
    return out;
  }

  function drawChart(id, cfg) {
    var cv = $(id);
    if (!cv || !hist.length) return;
    var dpr = window.devicePixelRatio || 1;
    var w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    var padL = 40, padR = cfg.right ? 42 : 10, padT = 8, padB = 20;
    var pw = w - padL - padR, ph = h - padT - padB;
    var t0 = hist[0].ts, t1 = Math.max(hist[hist.length - 1].ts, t0 + 30);
    function X(t) { return padL + (t - t0) / (t1 - t0) * pw; }
    function Y(v, r) { return padT + (1 - (v - r[0]) / (r[1] - r[0])) * ph; }

    ctx.font = "11px system-ui, sans-serif";
    ctx.strokeStyle = css("--grid");
    ctx.fillStyle = css("--muted");
    ctx.lineWidth = 1;
    cfg.yTicks.forEach(function (tick) {
      var y = Y(tick, cfg.yl);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(String(tick), padL - 6, y);
    });
    if (cfg.right) {
      cfg.rTicks.forEach(function (tick) {
        ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText(String(tick), w - padR + 5, Y(tick, cfg.yr));
      });
    }
    (cfg.refs || []).forEach(function (ref) {
      var y = Y(ref.v, cfg.yl);
      ctx.save();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = css(ref.color);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      ctx.restore();
      ctx.fillStyle = css(ref.color);
      ctx.textAlign = "left"; ctx.textBaseline = "bottom";
      ctx.fillText(ref.v + "°", padL + 3, y - 1);
    });
    cfg.series.forEach(function (s) {
      var range = s.axis === "r" ? cfg.yr : cfg.yl;
      ctx.strokeStyle = css(s.color);
      ctx.lineWidth = 2;
      ctx.beginPath();
      var pen = false;
      hist.forEach(function (p) {
        var v = p[s.key];
        if (!isNum(v)) { pen = false; return; }
        var x = X(p.ts), y = Y(v, range);
        if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; }
      });
      ctx.stroke();
    });
    ctx.fillStyle = css("--muted");
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText(clock(t0), padL, h - 16);
    ctx.textAlign = "right";
    ctx.fillText(clock(t1), w - padR, h - 16);
  }

  function drawAll() {
    var temps = hist.map(function (p) { return p.temp_c; }).filter(isNum);
    var maxT = temps.length ? Math.max.apply(null, temps) : 60;
    drawChart("c-temp", {
      right: true,
      yl: [40, Math.max(85, Math.ceil(maxT + 2))],
      yr: [1400, 2500],
      yTicks: niceTicks(40, Math.max(85, Math.ceil(maxT + 2)), 4),
      rTicks: [1500, 2000, 2400],
      refs: [
        { v: 60, color: "--ok" }, { v: 68, color: "--warn" },
        { v: 75, color: "--bad" }, { v: 80, color: "--bad" }
      ],
      series: [
        { key: "temp_c", color: "--c-temp", axis: "l" },
        { key: "cap_mhz", color: "--c-cap", axis: "r" }
      ]
    });
    var powers = hist.map(function (p) { return p.power_w; }).filter(isNum);
    var maxP = powers.length ? Math.max.apply(null, powers) : 5;
    var pTop = Math.max(5, Math.ceil(maxP * 1.1));
    drawChart("c-pow", {
      yl: [0, pTop], yTicks: niceTicks(0, pTop, 4),
      series: [{ key: "power_w", color: "--c-power", axis: "l" }]
    });
    drawChart("c-cpu", {
      yl: [0, 100], yTicks: [0, 25, 50, 75, 100],
      series: [{ key: "cpu_total", color: "--c-cpu", axis: "l" }]
    });
  }

  // Draw directly: requestAnimationFrame is suspended in background/hidden tabs.
  function scheduleDraw() {
    drawAll();
  }

  // -- Benchmarks tab ------------------------------------------------------
  var benchOpts = null;
  var benchRunning = false;
  var benchWasRunning = false;

  function api(path, opts) {
    return fetch(path, Object.assign({ cache: "no-store" }, opts || {}))
      .then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (d) {
          return { ok: r.ok, status: r.status, data: d };
        });
      });
  }

  function showView() {
    var bench = location.hash === "#bench";
    $("view-live").hidden = bench;
    $("view-bench").hidden = !bench;
    $("tab-live").classList.toggle("on", !bench);
    $("tab-bench").classList.toggle("on", bench);
    if (bench) { loadBenchOptions(); loadRuns(); }
  }

  function fillSelect(sel, values, keep) {
    var prev = keep || sel.value;
    sel.textContent = "";
    values.forEach(function (v) {
      var o = document.createElement("option");
      o.value = v.value;
      o.textContent = v.label;
      sel.appendChild(o);
    });
    if (prev && values.some(function (v) { return v.value === prev; })) sel.value = prev;
  }

  function datasetsFor(modelId) {
    var m = (benchOpts && benchOpts.models || []).find(function (x) { return x.id === modelId; });
    return m ? (m.compatible_datasets || []) : [];
  }

  function refillDatasets() {
    var model = $("bf-model").value;
    var ids = datasetsFor(model);
    fillSelect($("bf-dataset"), ids.map(function (d) { return { value: d, label: d }; }));
    var submit = $("bf-submit");
    submit.disabled = benchRunning || ids.length === 0;
  }

  function loadBenchOptions() {
    if (benchOpts) return;
    api("/api/bench/options").then(function (r) {
      if (!r.ok) {
        setText("bf-note", "Benchmarks unavailable: " + (r.data.error || r.status));
        return;
      }
      benchOpts = r.data;
      var models = (benchOpts.models || []).filter(function (m) { return m.enabled; });
      fillSelect($("bf-model"), models.map(function (m) {
        return { value: m.id, label: m.id + " (" + m.status + ")" };
      }));
      var threads = (benchOpts.threads || [1, 2, 3, 4]).map(function (t) {
        return { value: String(t), label: String(t) };
      });
      fillSelect($("bf-threads"), threads, "1");
      refillDatasets();
    });
  }

  function keyText(k) {
    if (!k || typeof k.value !== "number") return k && k.name ? k.name : "--";
    return k.name + " " + k.value.toFixed(4);
  }

  function loadRuns() {
    api("/api/bench/runs").then(function (r) {
      var body = $("bench-runs");
      body.textContent = "";
      var rows = (r.data && r.data.runs) || [];
      var empty = rows.length ? "" : (r.ok ? "No benchmark runs yet." : "Could not load runs.");
      setText("bench-empty", empty);
      rows.forEach(function (row) {
        var tr = document.createElement("tr");
        tr.className = "click";
        var cells = [
          row.run_id, row.status, row.model, row.dataset, fmt(row.n, 0), fmt(row.p50_ms, 2),
          fmt(row.p95_ms, 2), keyText({ name: row.key_metric, value: row.key_value }),
          fmt(row.max_temp_c, 1), row.throttled_after || "--"
        ];
        cells.forEach(function (text, i) {
          var td = document.createElement("td");
          td.textContent = String(text);
          if (i === 4 || i === 5 || i === 6 || i === 8) td.className = "n";
          if (i === 1 && text !== "ok") td.className = "bad";
          tr.appendChild(td);
        });
        tr.addEventListener("click", function () { loadRun(row.run_id); });
        body.appendChild(tr);
      });
    });
  }

  function loadRun(runId) {
    api("/api/bench/run/" + encodeURIComponent(runId)).then(function (r) {
      if (!r.ok) { setText("bd-id", runId + " (" + (r.data.error || r.status) + ")"); return; }
      var s = r.data.summary;
      $("bd-card").hidden = false;
      setText("bd-id", s.run_id);
      var meta = $("bd-meta");
      meta.textContent = "";
      [s.status, s.model, s.dataset, "n " + s.n, "p50 " + fmt(s.p50_ms, 3) + " ms",
        "p95 " + fmt(s.p95_ms, 3) + " ms", "p99 " + fmt(s.p99_ms, 3) + " ms",
        keyText(s.quality_key), "max " + fmt(s.temp_max_c, 1) + " °C",
        "pauses " + (s.pauses || 0),
        "throttled " + (s.throttled_before || "--") + " → " + (s.throttled_after || "--")
      ].forEach(function (text) {
        var span = document.createElement("span");
        span.textContent = String(text);
        meta.appendChild(span);
      });
      $("bd-json").textContent = JSON.stringify(r.data.result, null, 2);
    });
  }

  $("bench-form").addEventListener("submit", function (ev) {
    ev.preventDefault();
    var body = {
      model: $("bf-model").value,
      dataset: $("bf-dataset").value,
      limit: parseInt($("bf-limit").value, 10),
      threads: parseInt($("bf-threads").value, 10)
    };
    setText("bf-note", "Starting " + body.model + " on " + body.dataset + " ...");
    api("/api/bench/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (r) {
      if (r.ok) {
        setText("bf-note", "Started " + body.model + " on " + body.dataset + " (" + body.limit +
          " rows). Live status is on the right.");
        benchRunning = true;
        refillDatasets();
      } else {
        var prefix = r.status === 409 ? "Busy: " : "Not started: ";
        setText("bf-note", prefix + (r.data.error || r.status));
      }
    });
  });

  $("bf-model").addEventListener("change", refillDatasets);

  // Live status of a benchmark: the job panel's state (rbench runs under the thermal guard).
  function renderBenchLive(s) {
    var job = s.job;
    var running = !!job && (job.state === "running" || job.state === "paused") &&
      String(job.name || "").indexOf("rbench-") === 0;
    if (benchWasRunning && !running) loadRuns();
    benchWasRunning = running;
    benchRunning = running;
    if (!job || String(job.name || "").indexOf("rbench-") !== 0) {
      if (!running && $("bf-submit")) {
        $("bf-submit").disabled = !benchOpts || datasetsFor($("bf-model").value).length === 0;
      }
      return;
    }
    var meta = $("bl-meta");
    meta.textContent = "";
    var pill = document.createElement("span");
    var tone = { paused: "warn", killed: "bad", done: "ok" }[job.state] || "";
    pill.className = "pill " + tone;
    pill.textContent = String(job.state || "?").toUpperCase();
    meta.appendChild(pill);
    [String(job.name), "elapsed " + fmt(job.elapsed_s || 0, 0) + " s",
      "temp " + fmt(job.temp_c, 1) + " °C", "max " + fmt(job.max_temp_c, 1) + " °C",
      "pauses " + (job.pauses || 0)].forEach(function (text) {
      var span = document.createElement("span");
      span.textContent = text;
      meta.appendChild(span);
    });
    $("bl-log").textContent = (s.job_log || []).join("\n") || "(log empty)";
    if (running && $("bf-submit")) $("bf-submit").disabled = true;
  }

  function renderAll() {
    if (!latest) return;
    renderTiles(latest);
    renderJob(latest);
    renderThermald(latest);
    renderProcs(latest);
    renderFoot(latest);
    renderBenchLive(latest);
    scheduleDraw();
  }

  function onSample(s) {
    lastMsg = Date.now();
    pushSample(s);
    renderAll();
    setConn("live", "live");
  }

  function connect() {
    if (es) es.close();
    setConn("connecting", "wait");
    es = new EventSource("/api/stream");
    es.onopen = function () { setConn("live", "live"); };
    es.onmessage = function (ev) {
      try {
        onSample(JSON.parse(ev.data));
      } catch (e) {
        console.error("dashboard: bad frame or render error", e);
      }
    };
    es.onerror = function () {
      if (es.readyState === EventSource.CLOSED) {
        setConn("disconnected, retrying", "down");
        setTimeout(connect, 3000);
      } else {
        setConn("reconnecting", "wait");
      }
    };
  }

  function loadSnapshot() {
    return fetch("/api/snapshot", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        hist = (d.history || []).slice();
        if (d.latest) { latest = d.latest; lastMsg = Date.now(); }
        renderAll();
      })
      .catch(function () { /* stream will fill in */ });
  }

  setInterval(function () {
    if (!lastMsg) return;
    var age = Math.round((Date.now() - lastMsg) / 1000);
    var suffix = age > 15 ? " (no data for " + age + " s)" : "";
    setText("conn", stateText + (lastMsg ? " · updated " + age + " s ago" : "") + suffix);
  }, 1000);

  window.addEventListener("resize", scheduleDraw);
  window.addEventListener("hashchange", showView);
  showView();
  loadSnapshot().then(connect);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
