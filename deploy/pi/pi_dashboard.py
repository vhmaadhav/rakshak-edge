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
import json
import os
import re
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "rakshak-pi-dashboard/1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return  # keep the journal quiet

    @property
    def sampler(self) -> Sampler:
        return self.server.sampler  # type: ignore[attr-defined,no-any-return]

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        path = urlsplit(self.path).path
        if path == "/":
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

    def __init__(self, address: tuple[str, int], sampler: Sampler) -> None:
        super().__init__(address, DashboardHandler)
        self.sampler = sampler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 monitoring dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--run-dir", type=Path, default=None, help="default ~/rakshak-pi/run")
    parser.add_argument("--thermal-dir", type=Path, default=Path("/run/rakshak"))
    parser.add_argument("--sysfs", type=Path, default=Path("/sys"))
    parser.add_argument("--proc", type=Path, default=Path("/proc"))
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
    server = DashboardServer((args.host, args.port), sampler)
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

  function renderAll() {
    if (!latest) return;
    renderTiles(latest);
    renderJob(latest);
    renderThermald(latest);
    renderProcs(latest);
    renderFoot(latest);
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
  loadSnapshot().then(connect);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
