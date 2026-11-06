#!/usr/bin/env python3
"""Fix A prompt trial on the Pi: the default slot prompt and up to three describe variants, run
directly through rbench.slot (no service), on a small held-out set.

    prompt_trial.py trial [--rows 10] [--variants slot,describe,...]   # writes prompt_trial/
    prompt_trial.py scan-detector-only [--max 3000]     # Fix B: a message the student flags
                                                        # with no rule hit

The trial rows come from the same seeded sampler as the eval (worker._ela_rows, seed 0) but are
drawn from a larger pool and skip every row in the 40-row eval (results/edge_eval/rows.json), so the
choice of variant does not use the rows the final numbers are reported on. Only rows with rule
evidence and a risk other than LOW are used (LOW never reaches the explainer in the service).
Each variant runs with the edge guards (G1, G2) exactly as the service does.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = Path.home() / "rakshak-bench"
sys.path[:0] = [
    str(HERE),
    str(REPO / "src"),
    str(REPO / "deploy" / "pi" / "bench"),
    str(REPO / "deploy" / "pi"),
]

from edge_device import read_snapshot  # noqa: E402

EVAL_DIR = ROOT / "results" / "edge_eval"
ROWS = EVAL_DIR / "rows.json"
OUT = EVAL_DIR / "prompt_trial"
SEED = 0
POOL = 80
TEMP_STOP_C = 72.0  # stop the trial before the service's 75 C refusal


def _p50(values: list[float]) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    return round(xs[math.floor((len(xs) - 1) * 0.5)], 1)


def _load():
    from rakshak_edge_service import (
        ONNX_PATH,
        PTE_PATH,
        TOKENIZER_PATH,
        _DetectorAdapter,
        load_models,
    )

    from rakshak.canonicalize import canonicalize
    from rakshak.evidence import build_package
    from rakshak.rules import run_rules

    detector_runner, llm, threshold, info = load_models(
        ROOT, 2, ROOT / ONNX_PATH, ROOT / PTE_PATH, ROOT / TOKENIZER_PATH
    )
    detector = _DetectorAdapter(detector_runner)

    def package_for(text: str, state_fn):
        canon = canonicalize(text)
        ev = run_rules(canon)
        p = detector.score(text)
        state = state_fn(p, threshold, ev)
        return build_package(canon, ev, state), p

    return detector, llm, threshold, info, package_for


def trial(rows: int, variants: list[str]) -> int:
    from edge_pipeline import scam_state_for
    from rbench.registry import load_registry
    from rbench.slot import generate_slot_advisory
    from rbench.worker import _ela_rows

    from rakshak.evidence import Risk

    detector, llm, threshold, info, package_for = _load()
    reg = load_registry(ROOT / "deploy" / "pi" / "bench" / "registry.toml")
    ds = reg.dataset("sms_ela_campaign_test")
    eval_ids = {r["row_id"] for r in json.loads(ROWS.read_text(encoding="utf-8"))["rows"]}
    pool = _ela_rows(ROOT, ds, POOL, SEED).items
    chosen = []
    for item in pool:
        if item["row_id"] in eval_ids:
            continue
        pkg, p = package_for(item["text"], scam_state_for)
        if not pkg.evidence or pkg.risk is Risk.LOW:
            continue
        chosen.append((item, pkg, p))
        if len(chosen) >= rows:
            break
    print(f"trial rows: {len(chosen)} (pool {len(pool)}, excluded eval rows {len(eval_ids)})")
    OUT.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        name = "slot_default" if variant == "slot" else variant
        records[name] = []
        for item, pkg, p in chosen:
            temp = read_snapshot().get("temp_c")
            if temp is not None and temp >= TEMP_STOP_C:
                print(f"stopping at {temp} C", file=sys.stderr)
                break
            gen = generate_slot_advisory(
                llm,
                pkg,
                max_new_tokens=20,
                guards=True,
                variant=None if variant == "slot" else variant,
            )
            pts = json.loads(gen["raw_output"])["points"] if gen["raw_output"] else []
            checks = gen["point_checks"]
            records[name].append(
                {
                    "row_id": item["row_id"],
                    "risk": pkg.risk.value,
                    "types": [e.type for e in pkg.evidence],
                    "score": round(p, 4),
                    "points": [
                        {
                            "type": e.type,
                            "raw": pts[i] if pts else None,
                            "status": gen["point_status"][i],
                            "ok": bool(checks[i][1]),
                            "problems": gen["point_problems"][i],
                        }
                        for i, e in enumerate(pkg.evidence)
                    ],
                    "g2_rejected": [c["sentence"] for c in gen["g2_rejected"]],
                    "prompt_tokens": [c.get("prompt_tokens") for c in gen["calls"]],
                    "wall_ms": [c.get("wall_ms") for c in gen["calls"]],
                    "final_ok": gen["final_verifier_ok"],
                }
            )
            print(
                f"{name:14s} {item['row_id'][:8]} {pkg.risk.value:6s} "
                f"raw {sum(1 for c in checks if c[1])}/{len(checks)}",
                flush=True,
            )
    summary = {}
    for name, recs in records.items():
        pts = [q for r in recs for q in r["points"]]
        advs = len(recs)
        g2 = sorted({s for r in recs for s in r["g2_rejected"]})
        ptoks = [t for r in recs for t in r["prompt_tokens"] if t]
        walls = [w for r in recs for w in r["wall_ms"] if w]
        summary[name] = {
            "advisories": advs,
            "points": len(pts),
            "points_raw_pass": sum(1 for q in pts if q["ok"]),
            "advisories_raw_pass": sum(1 for r in recs if all(q["ok"] for q in r["points"])),
            "empty_or_unterminated": sum(1 for q in pts if q["status"] != "ok"),
            "g2_rejected_points": sum(len(r["g2_rejected"]) for r in recs),
            "g2_distinct_sentences": g2,
            "prompt_tokens_mean": round(sum(ptoks) / len(ptoks), 1) if ptoks else None,
            "call_wall_p50_ms": _p50(walls),
            "final_ok": sum(1 for r in recs if r["final_ok"]),
        }
    (OUT / "records.json").write_text(json.dumps(records, indent=1, ensure_ascii=False))
    (OUT / "summary.json").write_text(
        json.dumps({"llm": info, "variants": summary}, indent=1, ensure_ascii=False)
    )
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


def scan_detector_only(max_rows: int) -> int:
    """Fix B: test-split messages the student flags (score >= threshold), no rule evidence."""
    from rbench.datasets import load_rows
    from rbench.registry import load_registry

    _detector, _llm, threshold, _info, package_for = _load()
    reg = load_registry(ROOT / "deploy" / "pi" / "bench" / "registry.toml")
    ds = reg.dataset("sms_ela_campaign_test")
    items = load_rows(ROOT, ds, None, SEED).items
    hits = []
    scanned = 0
    for item in items[:max_rows]:
        scanned += 1
        pkg, p = package_for(item["text"], lambda pr, th, ev: "scam" if pr >= th else "benign")
        if p >= threshold and not pkg.evidence:
            hits.append(
                {
                    "row_id": item["row_id"],
                    "label": item["label"],
                    "score": round(p, 4),
                    "risk_before_fix_b": pkg.risk.value,
                    "text": item["text"][:300],
                }
            )
            if len(hits) >= 5:
                break
    print(
        json.dumps(
            {"scanned": scanned, "threshold": threshold, "hits": hits}, indent=1, ensure_ascii=False
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trial")
    t.add_argument("--rows", type=int, default=10)
    t.add_argument("--variants", default="slot,describe,describe_kw,describe_plain")
    s = sub.add_parser("scan-detector-only")
    s.add_argument("--max", type=int, default=3000)
    args = p.parse_args(argv)
    if args.cmd == "trial":
        return trial(args.rows, [v.strip() for v in args.variants.split(",") if v.strip()])
    return scan_detector_only(args.max)


if __name__ == "__main__":
    raise SystemExit(main())
