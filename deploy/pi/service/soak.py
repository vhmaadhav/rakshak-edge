#!/usr/bin/env python3
"""Sustained-load soak for the resident edge service (run on the Pi, outside its namespace).

    soak.py run --threads N [--requests 100]   # back-to-back POSTs over the Unix socket
    soak.py cool [--below 55]                  # blocks until the SoC is below the limit, then exits

``run`` cycles the 40 eval rows (eval_service.py select) with no idle gap, so request k uses row
k % 40. Per request it records latency (total and LLM), the SoC temperature, the throttle flags, the
PMIC power, the energy, the cap and ARM clock read before and after the request, and any 503. A
background sampler reads the SoC, throttle flags, ARM clock and cpufreq cap once a second.

Safety: the sampler aborts the soak (and stops the service with stop_service.sh) when the SoC is
at or above ABORT_TEMP_C, when throttled is not 0x0, or when the temperature cannot be read for
three samples in a row. A 503 from the service (SoC at or above its 75 C gate) is recorded and the
same row is retried after RETRY_S seconds; it does not count toward the accepted requests.

Output (on the Pi): results/soak/t<threads>/requests.jsonl, samples.jsonl and summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [
    str(HERE),
    str(REPO / "src"),
    str(REPO / "deploy" / "pi" / "bench"),
    str(REPO / "deploy" / "pi"),
]

import eval_service as ev  # noqa: E402
import pi_common as pc  # noqa: E402
from rbench.telemetry import read_temp_now  # noqa: E402

ROOT = ev.ROOT
ROWS = ev.ROWS
ABORT_TEMP_C = 79.0
RETRY_S = 3.0
SAMPLE_S = 1.0
TEMP_MISSING_LIMIT = 3
CAP_PATH = Path("/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq")
STOP_SCRIPT = ROOT / "deploy" / "pi" / "service" / "stop_service.sh"
_ARM_RE = re.compile(r"frequency\(\d+\)=(\d+)")


def arm_mhz() -> float | None:
    text = pc.run_vcgencmd(["measure_clock", "arm"])
    m = _ARM_RE.search(text or "")
    return round(int(m.group(1)) / 1e6, 1) if m else None


def cap_mhz() -> float | None:
    khz = pc.read_int(CAP_PATH)
    return None if khz is None else round(khz / 1000.0, 1)


def snapshot() -> dict[str, Any]:
    return {
        "ts": round(time.time(), 3),
        "temp_c": read_temp_now(),
        "throttled": pc.read_throttled_hex(),
        "arm_mhz": arm_mhz(),
        "cap_mhz": cap_mhz(),
    }


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)


def _dist(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50": _pct(values, 0.5),
        "p95": _pct(values, 0.95),
        "mean": round(sum(values) / len(values), 3) if values else None,
        "max": round(max(values), 3) if values else None,
    }


class Sampler(threading.Thread):
    """One snapshot per SAMPLE_S, written to samples.jsonl. Aborts the soak on a limit."""

    def __init__(self, path: Path, abort: threading.Event, reason: list[str]) -> None:
        super().__init__(name="soak-sampler", daemon=True)
        self.path = path
        self.abort = abort
        self.reason = reason
        self.stop_evt = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.missing = 0

    def run(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            while not self.stop_evt.is_set():
                snap = snapshot()
                self.samples.append(snap)
                handle.write(json.dumps(snap) + "\n")
                handle.flush()
                self._check(snap)
                if self.abort.is_set():
                    return
                self.stop_evt.wait(SAMPLE_S)

    def _check(self, snap: dict[str, Any]) -> None:
        temp, thr = snap["temp_c"], snap["throttled"]
        self.missing = self.missing + 1 if temp is None else 0
        why = None
        if temp is not None and temp >= ABORT_TEMP_C:
            why = f"SoC {temp} C >= {ABORT_TEMP_C} C"
        elif thr is not None and thr != "0x0":
            why = f"throttled {thr} != 0x0"
        elif self.missing >= TEMP_MISSING_LIMIT:
            why = "temperature unreadable for 3 samples"
        if why is None:
            return
        self.reason.append(why)
        self.abort.set()
        subprocess.run(["sh", str(STOP_SCRIPT)], check=False, timeout=60)


def _record(
    seq: int,
    accepted: int,
    row: dict[str, Any],
    status: int,
    body: dict[str, Any],
    t0: float,
    t_req: float,
    t_end: float,
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "seq": seq,
        "accepted_index": accepted,
        "row_id": row["row_id"],
        "group": row["group"],
        "http": status,
        "t_start_s": round(t_req - t0, 3),
        "t_end_s": round(t_end - t0, 3),
        "cap_before_mhz": before["cap_mhz"],
        "cap_after_mhz": after["cap_mhz"],
        "arm_before_mhz": before["arm_mhz"],
        "arm_after_mhz": after["arm_mhz"],
    }
    if status != 200:
        rec["error"] = body.get("error")
        rec["temp_c"] = body.get("temp_c")
        return rec
    dev = body["device"]
    calls = body["llm_calls"]
    # Cached calls decoded nothing: they are counted, but not in latency, tokens or decode time.
    live = [c for c in calls if not c.get("cached")]
    new_tokens = sum(float(c.get("new_tokens") or 0) for c in live)
    decode_s = (
        sum(max(float(c.get("wall_ms") or 0) - float(c.get("ttft_ms") or 0), 0.0) for c in live)
        / 1000.0
    )
    rec.update(
        {
            "risk": body["decision"]["risk"],
            "mode": body["advisory"]["mode"],
            "path": body["advisory"].get("path"),
            "total_ms": body["timings_ms"]["total"],
            "llm_total_ms": body["timings_ms"]["llm_total"],
            "llm_calls": len(calls),
            "llm_calls_cached": len(calls) - len(live),
            "llm_call_ms": [round(float(c.get("wall_ms") or 0), 1) for c in live],
            "new_tokens": new_tokens,
            "decode_s": round(decode_s, 3),
            "temp_start_c": dev["start"]["temp_c"],
            "temp_end_c": dev["end"]["temp_c"],
            "throttled_start": dev["start"]["throttled"],
            "throttled_end": dev["end"]["throttled"],
            "power_start_w": dev["start"]["power_w"],
            "power_end_w": dev["end"]["power_w"],
            "energy_j": dev["energy"]["energy_j"],
            "mean_w": dev["energy"]["mean_w"],
        }
    )
    return rec


def run(args: argparse.Namespace) -> int:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))["rows"]
    out = Path(args.out) if args.out else ROOT / "results" / "soak" / f"t{args.threads}"
    out.mkdir(parents=True, exist_ok=True)
    health = ev._get("/health")
    if health.get("threads") != args.threads:
        print(
            f"soak: service reports threads={health.get('threads')}, expected {args.threads}",
            file=sys.stderr,
        )
        return 2
    abort = threading.Event()
    reason: list[str] = []
    sampler = Sampler(out / "samples.jsonl", abort, reason)
    sampler.start()
    records: list[dict[str, Any]] = []
    done_s: list[float] = []  # t_end of each accepted request, seconds from t0
    accepted = 0
    attempts = 0
    t0 = time.perf_counter()
    error = None
    with (out / "requests.jsonl").open("w", encoding="utf-8") as handle:
        while accepted < args.requests and not abort.is_set():
            if attempts >= args.max_attempts:
                error = f"stopped after {attempts} attempts"
                break
            row = rows[accepted % len(rows)]
            attempts += 1
            before = snapshot()
            t_req = time.perf_counter()
            try:
                status, body = ev._post(row["text"])
            except Exception as exc:  # the service went away (for example the abort stopped it)
                error = f"request failed: {exc!r}"
                break
            t_end = time.perf_counter()
            after = snapshot()
            rec = _record(attempts, accepted, row, status, body, t0, t_req, t_end, before, after)
            records.append(rec)
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            if status == 200:
                accepted += 1
                done_s.append(t_end - t0)
                print(
                    f"{accepted:3d} {rec['risk']:9s} {rec['mode']:8s} "
                    f"total={rec['total_ms']:8.0f}ms llm={rec['llm_total_ms']:8.0f}ms "
                    f"T={rec['temp_end_c']} cap={after['cap_mhz']}",
                    flush=True,
                )
            elif status == 503:
                time.sleep(RETRY_S)
            else:
                error = f"HTTP {status}: {body.get('error')}"
                break
    sampler.stop_evt.set()
    sampler.join(timeout=10)
    summary = summarise(
        args.threads,
        args.requests,
        records,
        sampler.samples,
        done_s,
        accepted,
        attempts,
        health,
        reason,
        error,
        time.perf_counter() - t0,
    )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k in ("accepted", "refused_503", "aborted", "error")
            }
        )
    )
    return 0 if accepted == args.requests and not reason and error is None else 1


def summarise(
    threads: int,
    target: int,
    records: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    done_s: list[float],
    accepted: int,
    attempts: int,
    health: dict[str, Any],
    reason: list[str],
    error: str | None,
    wall_s: float,
) -> dict[str, Any]:
    ok = [r for r in records if r["http"] == 200]
    refused = [r for r in records if r["http"] == 503]
    called = [r for r in ok if r["llm_calls"] > 0]
    call_ms = [c for r in called for c in r["llm_call_ms"]]
    decode_s = sum(r["decode_s"] for r in called)
    tokens = sum(r["new_tokens"] for r in called)
    temps = [s["temp_c"] for s in samples if s["temp_c"] is not None]
    temps += [r["temp_end_c"] for r in ok if r["temp_end_c"] is not None]
    caps = [s["cap_mhz"] for s in samples if s["cap_mhz"] is not None]
    cap0 = caps[0] if caps else None
    first_cut = next(
        (
            s["ts"] - samples[0]["ts"]
            for s in samples
            if s["cap_mhz"] is not None and cap0 is not None and s["cap_mhz"] < cap0
        ),
        None,
    )
    rate_last50 = None
    if len(done_s) >= 51:
        rate_last50 = round(50 / (done_s[-1] - done_s[-51]) * 60, 2)
    # 503 gate: time from the first refusal to the next accepted request
    recovery_s = None
    first503 = next((r for r in records if r["http"] == 503), None)
    if first503 is not None:
        nxt = next((r for r in records if r["seq"] > first503["seq"] and r["http"] == 200), None)
        recovery_s = None if nxt is None else round(nxt["t_end_s"] - first503["t_start_s"], 1)
    return {
        "threads": threads,
        "target_accepted": target,
        "accepted": accepted,
        "attempts": attempts,
        "refused_503": len(refused),
        "refused_temps_c": [r.get("temp_c") for r in refused],
        "recovery_after_503_s": recovery_s,
        "aborted": bool(reason),
        "abort_reason": reason,
        "error": error,
        "wall_s": round(wall_s, 1),
        "requests_per_min_all": round(accepted / wall_s * 60, 2) if wall_s > 0 else None,
        "requests_per_min_last50": rate_last50,
        "total_ms": _dist([r["total_ms"] for r in ok]),
        "llm_per_advisory_ms": _dist([r["llm_total_ms"] for r in called]),
        "llm_per_call_ms": _dist(call_ms),
        "llm_calls_cached": sum(r["llm_calls_cached"] for r in ok),
        "decode_tok_s": round(tokens / decode_s, 2) if decode_s > 0 else None,
        "energy_j_per_advisory": _dist([r["energy_j"] for r in ok]),
        "energy_j_per_llm_advisory": _dist([r["energy_j"] for r in called]),
        "max_temp_c": max(temps) if temps else None,
        "max_temp_end_c_request": max(
            (r["temp_end_c"] for r in ok if r["temp_end_c"] is not None), default=None
        ),
        "throttled_seen": sorted({s["throttled"] for s in samples if s["throttled"] is not None}),
        "cap_start_mhz": cap0,
        "cap_min_mhz": min(caps) if caps else None,
        "first_cap_reduction_s": None if first_cut is None else round(first_cut, 1),
        "arm_min_mhz": min(
            (s["arm_mhz"] for s in samples if s["arm_mhz"] is not None), default=None
        ),
        "mode_counts": _count([r["mode"] for r in ok]),
        "service_threads_reported": health.get("threads"),
        "service_pid": health.get("pid"),
    }


def _count(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def cool(args: argparse.Namespace) -> int:
    """Block until the SoC is below ``--below``, one line per reading. Exit 0 when cool."""
    deadline = time.time() + args.max_wait
    while time.time() < deadline:
        temp = read_temp_now()
        print(f"{time.strftime('%H:%M:%S')} temp={temp}", flush=True)
        if temp is not None and temp < args.below:
            print(f"cool: {temp} C below {args.below} C", flush=True)
            return 0
        time.sleep(args.interval)
    print("cool: timed out", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, required=True, help="must match the running service")
    r.add_argument("--requests", type=int, default=100)
    r.add_argument("--max-attempts", type=int, default=400)
    r.add_argument("--out", default=None)
    c = sub.add_parser("cool")
    c.add_argument("--below", type=float, default=55.0)
    c.add_argument("--interval", type=float, default=5.0)
    c.add_argument("--max-wait", type=float, default=3600.0)
    args = p.parse_args(argv)
    return run(args) if args.command == "run" else cool(args)


if __name__ == "__main__":
    raise SystemExit(main())
