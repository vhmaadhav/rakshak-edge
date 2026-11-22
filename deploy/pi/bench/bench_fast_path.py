#!/usr/bin/env python3
"""Benchmark and verdict-parity check for the fast path: OLD (sync, one lock) against NEW.

OLD  = EdgeAnalyzer.analyse, the sync path, called under one lock like the service did.
NEW  = verdict-first: EdgeAnalyzer.analyse_verdict (no lock), the batched scorer, and the table.

The explainer LLM is a stub here (instant), because the Llama .pte is not on the laptop. So OLD
time-to-verdict is the pre-LLM cost plus the template assembly; the LLM time is NOT in these
numbers. The Pi audit's LLM time (p50 989 ms, from the service log) is reported next to them.

The detector is the ONNX SMS student. Without the trained model on this machine, the default is
the stand-in graph (deploy/pi/bench/standin_onnx.py): the code path is real, the scores are not
the trained model's. Pass --onnx PATH to bench the real model.

Verdict parity: for every message in the parity sample, the decision fields of OLD, NEW single,
NEW batch and NEW with table must be equal (assert). The flip count is the number of mismatches.

Usage (from the worktree):  deploy/pi/bench/bench_fast_path.py --out PATH.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import platform
import random
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [
    str(REPO / "src"),
    str(HERE),
    str(REPO / "deploy" / "pi" / "service"),
    str(REPO / "deploy" / "pi"),
]

from build_explanation_table import build_table, load_samples  # noqa: E402
from edge_pipeline import EdgeAnalyzer  # noqa: E402
from explanation_table import ExplanationTable  # noqa: E402
from rakshak_edge_service import _DetectorAdapter  # noqa: E402
from rbench.runners.onnx import SmsRunner  # noqa: E402
from standin_onnx import build_standin  # noqa: E402

THRESHOLD = 0.35127517580986023  # SYSTEM_V1 (docs/SYSTEM_V1.md), frozen
PI_LLM_P50_MS = 989.0  # audit: Pi p50 time-to-verdict (LLM is ~95% of it), from the audit notes
PI_LLM_P95_MS = 1887.0


class StubLLM:
    """The explainer stand-in: instant, returns a fixed continuation (no model is loaded)."""

    def generate(self, prompt, max_new_tokens=64, stop_when=None):
        tail = " asks for action on [[SLOT]]." if "[[SLOT]]" in prompt else " asks for action."
        return tail, {"wall_ms": 0.0, "stop_hit": True}


class StubTableRunner:
    """Builds the stand-in table (the real table is built on the Pi)."""

    def generate(self, prompt, max_new_tokens=64, stop_when=None):
        return StubLLM().generate(prompt, max_new_tokens, stop_when)


def decision_fields(result: dict[str, Any]) -> dict[str, Any]:
    d = result["decision"]
    return {
        "risk": d["risk"],
        "action": d["action"],
        "score": d["score"],
        "scam_state": d["scam_state"],
        "detector_only": d["detector_only"],
        "evidence": [(e["id"], e["type"], e["weight"]) for e in result["evidence"]],
        "remediation": result["remediation"],
    }


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 3)


def timed(fn, items) -> list[float]:
    out = []
    for item in items:
        t0 = time.perf_counter()
        fn(item)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def throughput(total_s: float, n: int) -> float:
    return round(n / total_s, 1) if total_s > 0 else float("inf")


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--parquet", default=str(REPO / "data" / "processed" / "sms.parquet"))
    p.add_argument("--onnx", default=None, help="the SMS student ONNX (default: the stand-in)")
    p.add_argument("--workdir", required=True, help="scratch directory (stand-in, table)")
    p.add_argument("--timing-n", type=int, default=300, help="messages for latency and throughput")
    p.add_argument("--parity-n", type=int, default=2000, help="messages for verdict parity")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    import pandas as pd

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.parquet, columns=["text"])
    pool = df["text"].astype(str).drop_duplicates().tolist()
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    need = max(args.timing_n, args.parity_n)
    messages = pool[:need]

    onnx = Path(args.onnx) if args.onnx else build_standin(workdir / "standin_sms_student.onnx")
    runner = SmsRunner()
    runner.load(onnx, 1, {})
    adapter = _DetectorAdapter(runner, batch_onnx=False)  # the default: bit-identical to single
    table_doc = build_table(StubTableRunner(), load_samples(), built="bench")
    table = ExplanationTable(table_doc, source="stand-in (bench)")

    old = EdgeAnalyzer(adapter, StubLLM(), THRESHOLD)
    new = EdgeAnalyzer(adapter, StubLLM(), THRESHOLD)
    new_table = EdgeAnalyzer(adapter, StubLLM(), THRESHOLD, explanation_table=table)
    old_lock = threading.Lock()

    timing = messages[: args.timing_n]
    # Warm the ONNX session and the caches once for each path (not measured).
    for t in timing[:20]:
        old.analyse(t)
        new.analyse_verdict(t)
        new_table.analyse_verdict(t)

    # (i) time-to-verdict, one message at a time.
    old_ms = timed(lambda t: old.analyse(t), timing)
    new_ms = timed(lambda t: new.analyse_verdict(t), timing)
    table_ms = timed(lambda t: new_table.analyse_verdict(t), timing)
    table_hits = sum(1 for t in timing if new_table.analyse_verdict(t)[1] is None)

    # (ii) throughput, single message at a time, batched, and concurrent.
    t0 = time.perf_counter()
    for t in timing:
        old.analyse(t)
    old_single_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for t in timing:
        new.analyse_verdict(t)
    new_single_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(0, len(timing), args.batch):
        new.analyse_verdict_batch(timing[i : i + args.batch])
    new_batch_s = time.perf_counter() - t0

    def old_locked(text):
        with old_lock:
            return old.analyse(text)

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        t0 = time.perf_counter()
        list(ex.map(old_locked, timing))
        old_conc_s = time.perf_counter() - t0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        t0 = time.perf_counter()
        list(ex.map(lambda t: new.analyse_verdict(t), timing))
        new_conc_s = time.perf_counter() - t0

    # Verdict parity over the parity sample.
    parity = messages[: args.parity_n]
    flips = []
    flips_batch_onnx = 0
    old_dec = []
    for t in parity:
        old_dec.append(decision_fields(old.analyse(t)))
    new_single = [decision_fields(new.analyse_verdict(t)[0]) for t in parity]
    new_batch = []
    for i in range(0, len(parity), args.batch):
        new_batch.extend(
            decision_fields(r) for r, _ in new.analyse_verdict_batch(parity[i : i + args.batch])
        )
    new_table_dec = [decision_fields(new_table.analyse_verdict(t)[0]) for t in parity]
    for i, (o, a, b, c) in enumerate(
        zip(old_dec, new_single, new_batch, new_table_dec, strict=False)
    ):
        if not (o == a == b == c):
            flips.append(i)
    assert not flips, f"verdict mismatch on {len(flips)} messages, first at index {flips[0]}"

    batched_adapter = _DetectorAdapter(runner, batch_onnx=True)
    batched_an = EdgeAnalyzer(batched_adapter, StubLLM(), THRESHOLD)
    for i in range(0, len(parity), args.batch):
        for k, (r, _) in enumerate(batched_an.analyse_verdict_batch(parity[i : i + args.batch])):
            if decision_fields(r) != old_dec[i + k]:
                flips_batch_onnx += 1

    result = {
        "hardware": {
            "cpu": cpu_model(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "onnx": str(onnx),
        "onnx_is_standin": args.onnx is None,
        "table": table.stats(),
        "threshold": THRESHOLD,
        "timing_messages": len(timing),
        "parity": {
            "messages": len(parity),
            "verdict_flips_old_vs_new_single_batch_table": len(flips),
            "verdict_flips_batched_onnx_vs_old": flips_batch_onnx,
        },
        "time_to_verdict_ms": {
            "old_sync_llm_stubbed": {"p50": pct(old_ms, 0.5), "p95": pct(old_ms, 0.95)},
            "new_verdict_first": {"p50": pct(new_ms, 0.5), "p95": pct(new_ms, 0.95)},
            "new_verdict_first_table_hit_e2e": {
                "p50": pct(table_ms, 0.5),
                "p95": pct(table_ms, 0.95),
            },
            "table_hits_of_timing_messages": table_hits,
            "pi_audit_old_llm_path": {"p50": PI_LLM_P50_MS, "p95": PI_LLM_P95_MS},
        },
        "msgs_per_s": {
            "old_single": throughput(old_single_s, len(timing)),
            "new_single": throughput(new_single_s, len(timing)),
            "new_batched": throughput(new_batch_s, len(timing)),
            "old_concurrent_locked": throughput(old_conc_s, len(timing)),
            "new_concurrent_unlocked": throughput(new_conc_s, len(timing)),
            "concurrency": args.concurrency,
            "batch": args.batch,
        },
        "mean_ms": {
            "old": round(statistics.fmean(old_ms), 3),
            "new": round(statistics.fmean(new_ms), 3),
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
