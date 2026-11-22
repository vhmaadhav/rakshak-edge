#!/usr/bin/env python3
"""Explainer speed bench: the shipped explainer (baseline) against the candidates, one .pte.

    python deploy/pi/bench/explainer_speed.py run --root ROOT --model PTE --tokenizer TOK \\
        --out DIR [--reps 3] [--threads 3] [--limit 30] [--seed 0] [--probe]

    ROOT is a directory with ``data/processed/sms.parquet`` and ``data/processed/splits/``
    (the registry paths), e.g. the Pi's ``~/rakshak-bench`` or a symlinked copy on the VM.
    Writes DIR/metrics.json (the table and the gate verdicts) and DIR/advisories.jsonl (one line
    per advisory and arm, with the accounting counts and the sentences).

The rows are the seeded evidence rows of the edge eval (rbench.worker._ela_rows, the same sampler
as eval_service.select), so they match the Pi's evidence group. The 10 benign LOW rows are not
run: the G1 guard skips the LLM for them in every arm, so they cannot change a gate.

Arms run in blocks (all rows of one arm, then the next arm). A block keeps the system-turn prefix
in the runner's KV cache, as the service does. The arm order rotates across repeats. The
tight-cap value is the one statistic taken from the baseline: the p95 of terminated baseline
outputs (rule in docs/EXPLAINER_SPEED_PLAN.md). Gates are computed here from the pre-registered
rules, not by hand.

Greedy decoding; the numbers in metrics.json are the same on every machine except the wall times.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "src"), str(REPO / "deploy" / "pi" / "service")]

from rbench.explainer_candidates import (  # noqa: E402
    BASELINE,
    CANDIDATES,
    AccountingGenerator,
    run_explainer,
    summarise_accounting,
)
from rbench.registry import load_registry  # noqa: E402

REGISTRY = HERE / "registry.toml"
DATASET = "sms_ela_campaign_test"
# Candidates that use the tight cap (rule: p95 of terminated baseline outputs).
TIGHT_CAP_ARMS = ("c1_tight", "c2_batch", "c3_trim")
# Pre-registered gate thresholds (docs/EXPLAINER_SPEED_PLAN.md).
E4_RATIO_MAX = 0.70
E5_BOOT = 2000
# Derived Pi projection base: the shipped direct-runner total p50 on the Pi (2026-11-11,
# clean power, docs/EDGE_PIPELINE.md). Used only to label a projection, never as a measurement.
PI_BASELINE_TOTAL_P50_MS = 989.0


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def tight_cap(baseline_records: list[dict[str, Any]]) -> dict[str, Any]:
    """The tight cap: ceil of the p95 of new_tokens over baseline calls with status ok."""
    lengths = [
        int(c["new_tokens"])
        for rec in baseline_records
        for c in rec["calls"]
        if c.get("status") == "ok" and c.get("new_tokens") is not None
    ]
    p95 = _pct([float(x) for x in lengths], 0.95)
    return {
        "rule": "ceil(p95, linear interpolation) of new_tokens over baseline calls with status ok",
        "n_terminated_calls": len(lengths),
        "p95": p95,
        "cap": int(math.ceil(p95)) if p95 is not None else None,
        "max_terminated": max(lengths) if lengths else None,
    }


def _term_status(res: dict[str, Any]) -> list[str]:
    """Termination status per point: the sentence status before cleanup (ok, unterminated, ...)."""
    calls = res["calls"]
    if calls and "point_statuses" in calls[0]:
        return list(calls[0]["point_statuses"])
    return [c.get("status") for c in calls]


def record_for(
    cand: str,
    rep: int,
    row: dict[str, Any],
    llm_package,
    res: dict[str, Any],
    log: list[dict[str, Any]],
    wall_s: float,
) -> dict[str, Any]:
    calls = res["calls"]
    acct = summarise_accounting(log)
    return {
        "candidate": cand,
        "rep": rep,
        "row_id": row["row_id"],
        "types": [e.type for e in llm_package.evidence],
        "n_points": len(llm_package.evidence),
        "skipped": bool(res["skipped"]),
        "point_checks": [bool(ok) for _, ok in res["point_checks"]],
        "point_checks_types": [t for t, _ in res["point_checks"]],
        "term_status": _term_status(res),
        "point_status": list(res["point_status"]),
        "final_verifier_ok": bool(res["final_verifier_ok"]),
        "llm_wall_ms": round(sum(float(c.get("wall_ms") or 0.0) for c in calls), 3),
        "advisory_wall_ms": round(wall_s * 1000.0, 3),
        "accounting": acct,
        "calls": [
            {
                "status": c.get("status"),
                "new_tokens": c.get("new_tokens"),
                "prompt_tokens": c.get("prompt_tokens"),
                "stop_reason": c.get("stop_reason"),
                "generated": c.get("generated"),
            }
            for c in calls
        ],
        "output": res["output"],
        "raw_output": res["raw_output"],
    }


def run_arm_block(
    runner,
    cand: str,
    rep: int,
    rows: list[dict[str, Any]],
    packages: list,
    cap: int | None,
    out_records: list[dict[str, Any]],
) -> None:
    for row, llm_package in zip(rows, packages, strict=True):
        gen = AccountingGenerator(runner)
        t0 = time.perf_counter()
        res = run_explainer(
            gen,
            llm_package,
            "en",
            candidate=cand,
            cap=cap if cand in TIGHT_CAP_ARMS else None,
            guards=True,
            cleanup=True,
        )
        wall = time.perf_counter() - t0
        out_records.append(record_for(cand, rep, row, llm_package, res, gen.take(), wall))


def _arm_metrics(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts from rep 0 (the greedy outputs are the same across repeats; checked below)."""
    rep0 = [r for r in recs if r["rep"] == recs[0]["rep"]]
    points = sum(r["n_points"] for r in rep0)
    raw_pass = sum(sum(r["point_checks"]) for r in rep0)
    unterminated = sum(
        1 for r in rep0 for s in r["term_status"] if s not in ("ok", None)
    )
    stops: dict[str, int] = {}
    for r in rep0:
        for c in r["calls"]:
            key = str(c.get("stop_reason") or "unknown")
            stops[key] = stops.get(key, 0) + 1
    n_adv = len(rep0)

    def mean(key: str) -> float:
        return round(sum(r["accounting"][key] for r in rep0) / n_adv, 3)

    called = [r for r in recs if r["calls"]]
    llm_walls = [r["llm_wall_ms"] for r in called]
    adv_walls_all = [r["advisory_wall_ms"] for r in recs]
    # Per-repeat p50 (to see drift between repeats), pooled p50 and p95 over all repeats.
    per_rep = {}
    for rep in sorted({r["rep"] for r in called}):
        vals = [r["llm_wall_ms"] for r in called if r["rep"] == rep]
        per_rep[str(rep)] = round(statistics.median(vals), 3)
    return {
        "advisories": n_adv,
        "points": points,
        "points_ok_raw_pass": raw_pass,
        "unterminated_points": unterminated,
        "final_contract_ok": sum(1 for r in rep0 if r["final_verifier_ok"]),
        "calls_per_advisory": round(sum(r["accounting"]["calls"] for r in rep0) / n_adv, 3),
        "prompt_tokens_per_advisory": mean("prompt_tokens"),
        "reused_prefix_tokens_per_advisory": mean("reused_prefix_tokens"),
        "prefill_steady_per_advisory": mean("prefill_steady"),
        "prefill_observed_per_advisory": mean("prefill_observed"),
        "decode_tokens_per_advisory": mean("decode_tokens"),
        "tokens_processed_steady_per_advisory": mean("tokens_processed_steady"),
        "forward_calls_steady_per_advisory": mean("forward_calls_steady"),
        "forward_calls_observed_per_advisory": mean("forward_calls"),
        "stop_reasons_calls": stops,
        "llm_wall_ms_p50": round(_pct(llm_walls, 0.5), 3),
        "llm_wall_ms_p95": round(_pct(llm_walls, 0.95), 3),
        "advisories_llm_called": len(called),
        "advisories_g1_skipped": n_adv - sum(1 for r in rep0 if r["calls"]),
        "advisory_wall_ms_p50_all_including_skipped": round(_pct(adv_walls_all, 0.5), 3),
        "llm_wall_ms_p50_per_repeat": per_rep,
        "repeats": len({r["rep"] for r in recs}),
        "outputs_identical_across_repeats": all(
            r["output"] == next(x["output"] for x in rep0 if x["row_id"] == r["row_id"])
            for r in recs
        ),
    }


def _bootstrap_ratio(
    arm: list[dict[str, Any]], base: list[dict[str, Any]], seed: int = 0
) -> tuple[float | None, float | None, float | None]:
    """Median ratio arm/baseline (LLM wall, pooled over repeats) with a 95% bootstrap CI.

    Resamples rows (clusters of repeats) independently for the two arms.
    """
    by_row_a: dict[str, list[float]] = {}
    by_row_b: dict[str, list[float]] = {}
    for r in arm:
        if r["calls"]:
            by_row_a.setdefault(r["row_id"], []).append(r["llm_wall_ms"])
    for r in base:
        if r["calls"]:
            by_row_b.setdefault(r["row_id"], []).append(r["llm_wall_ms"])
    ids_a, ids_b = sorted(by_row_a), sorted(by_row_b)
    if not ids_a or not ids_b:
        return None, None, None
    point = statistics.median([v for vs in by_row_a.values() for v in vs]) / statistics.median(
        [v for vs in by_row_b.values() for v in vs]
    )
    rng = random.Random(seed)
    ratios = []
    for _ in range(E5_BOOT):
        sa = [v for _ in ids_a for v in by_row_a[rng.choice(ids_a)]]
        sb = [v for _ in ids_b for v in by_row_b[rng.choice(ids_b)]]
        den = statistics.median(sb)
        if den > 0 and sa:
            ratios.append(statistics.median(sa) / den)
    if len(ratios) < E5_BOOT // 2:
        return point, None, None
    ratios.sort()
    n = len(ratios)
    return point, ratios[int(0.025 * n)], ratios[min(int(0.975 * n), n - 1)]


def verdicts(arm: dict[str, Any], base: dict[str, Any], ratio: dict[str, Any]) -> dict[str, Any]:
    tok_ratio = (
        arm["tokens_processed_steady_per_advisory"] / base["tokens_processed_steady_per_advisory"]
        if base["tokens_processed_steady_per_advisory"]
        else None
    )
    fwd_ratio = (
        arm["forward_calls_steady_per_advisory"] / base["forward_calls_steady_per_advisory"]
        if base["forward_calls_steady_per_advisory"]
        else None
    )
    e4 = bool(
        (tok_ratio is not None and tok_ratio <= E4_RATIO_MAX)
        or (fwd_ratio is not None and fwd_ratio <= E4_RATIO_MAX)
    )
    e5 = bool(
        ratio["point"] is not None
        and ratio["ci_high"] is not None
        and ratio["point"] < 1.0
        and ratio["ci_high"] < 1.0
    )
    gates = {
        "E1_unterminated_zero": arm["unterminated_points"] == 0,
        "E2_raw_pass_not_lower": arm["points_ok_raw_pass"] >= base["points_ok_raw_pass"],
        "E3_final_contract_all": arm["final_contract_ok"] == arm["advisories"],
        "E4_tokens_or_calls_down_30pct": e4,
        "E5_x86_p50_down_ci_excludes_1": e5,
    }
    return {
        "gates": gates,
        "passed_all": all(gates.values()),
        "tokens_ratio": None if tok_ratio is None else round(tok_ratio, 4),
        "forward_calls_ratio": None if fwd_ratio is None else round(fwd_ratio, 4),
    }


def summarise(records: list[dict[str, Any]], cap: dict[str, Any]) -> dict[str, Any]:
    arms = []
    for r in records:
        if r["candidate"] not in arms:
            arms.append(r["candidate"])
    by_arm = {a: [r for r in records if r["candidate"] == a] for a in arms}
    base_recs = by_arm[BASELINE]
    base = _arm_metrics(base_recs)
    out: dict[str, Any] = {"baseline": {"candidate": BASELINE, **base}, "arms": {}}
    for a in arms:
        if a == BASELINE:
            continue
        m = _arm_metrics(by_arm[a])
        point, lo, hi = _bootstrap_ratio(by_arm[a], base_recs)
        ratio = {"point": point, "ci_low": lo, "ci_high": hi}
        out["arms"][a] = {
            "candidate": a,
            "description": CANDIDATES[a].description,
            "cap": cap["cap"] if a in TIGHT_CAP_ARMS else None,
            **m,
            "llm_wall_ratio_vs_baseline_p50": ratio,
            "pi_projection_p50_ms_derived": None
            if point is None
            else round(PI_BASELINE_TOTAL_P50_MS * point, 1),
            **verdicts(m, base, ratio),
        }
    out["baseline"]["pi_projection_note"] = (
        "pi_projection_p50_ms_derived is PI_BASELINE_TOTAL_P50_MS times the x86 ratio: an "
        "estimate, not a Pi measurement"
    )
    return out


def probe_model(path: Path) -> dict[str, Any]:
    """Output shapes of the .pte forward for a 1-token and a 4-token input (PLD feasibility).

    Loads its own copy of the model and frees it before the runner loads. A forward output with
    only the last position (shape [1, vocab]) means prompt-lookup verification is not possible
    without re-export (docs/EXPLAINER_SPEED_RESULTS.md).
    """
    import gc

    import torch
    from executorch.extension.pybindings import portable_lib as pl

    from rbench.runners.pte_llm import _preload_kernels

    _preload_kernels()  # the Llama custom ops must be registered before the method loads
    model = pl._load_for_executorch(str(path))
    out: dict[str, Any] = {
        "eos_ids": [int(x) for x in model.run_method("get_eos_ids", ())],
        "max_context_len": int(model.run_method("get_max_context_len", ())[0]),
    }
    o1 = model.forward((torch.tensor([[1]]), torch.tensor([0])))
    out["forward_1_token_shape"] = list(o1[0].shape)
    o4 = model.forward((torch.tensor([[1, 2, 3, 4]]), torch.tensor([1])))
    out["forward_4_token_shape"] = list(o4[0].shape)
    out["per_position_logits"] = bool(len(o4[0].shape) >= 2 and o4[0].shape[-2] == 4)
    del model
    gc.collect()
    return out


def build_rows(root: Path, limit: int, seed: int) -> tuple[list[dict[str, Any]], list]:
    from rakshak.ela_eval import package_for_sms
    from rbench.worker import _ela_rows

    from edge_pipeline import collapse_by_type

    reg = load_registry(REGISTRY)
    ds = reg.dataset(DATASET)
    items = _ela_rows(root, ds, limit, seed).items
    rows, packages = [], []
    for item in items:
        _canon, package = package_for_sms(item["text"])
        rows.append(
            {
                "row_id": str(item.get("row_id", item.get("id"))),
                "text": item["text"],
                "label": int(item.get("label_binary", item.get("label", -1))),
            }
        )
        packages.append(collapse_by_type(package))
    return rows, packages


def run(args: argparse.Namespace) -> int:
    from rbench.runners import make_runner

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    model = Path(args.model)
    probe = probe_model(model) if args.probe else None
    rows, packages = build_rows(root, args.limit, args.seed)
    runner = make_runner("pte_llm_direct")
    runner.load(model, args.threads, {"tokenizer": Path(args.tokenizer)})
    arms = [a.strip() for a in args.candidates.split(",") if a.strip()]
    if BASELINE not in arms:
        arms.insert(0, BASELINE)
    unknown = [a for a in arms if a not in CANDIDATES]
    if unknown:
        raise SystemExit(f"unknown candidates: {unknown}")
    records: list[dict[str, Any]] = []
    cap: dict[str, Any] | None = None
    t_start = time.time()
    for rep in range(args.reps):
        # Repeat 0 starts with the baseline (it sets the tight cap); later repeats rotate the order.
        k = 0 if rep == 0 else rep % len(arms)
        order = arms[k:] + arms[:k]
        for cand in order:
            if cand in TIGHT_CAP_ARMS and cap is None:
                base_recs = [r for r in records if r["candidate"] == BASELINE]
                cap = tight_cap(base_recs)
                print(f"tight cap from baseline: {cap}", flush=True)
            block: list[dict[str, Any]] = []
            run_arm_block(
                runner,
                cand,
                rep,
                rows,
                packages,
                cap["cap"] if cap and cand in TIGHT_CAP_ARMS else None,
                block,
            )
            records.extend(block)
            med = statistics.median([r["llm_wall_ms"] for r in block])
            print(f"rep {rep} {cand}: n={len(block)} llm_wall_p50={med:.1f} ms", flush=True)
    if cap is None:
        cap = tight_cap([r for r in records if r["candidate"] == BASELINE])
    summary = summarise(records, cap)
    summary["meta"] = {
        "model": model.name,
        "root_rows_seed": args.seed,
        "rows": len(rows),
        "reps": args.reps,
        "threads": args.threads,
        "wall_seconds": round(time.time() - t_start, 1),
        "probe": probe,
        "runner_threads_reported": runner.threads_reported,
        "explainer_cache": "off (not used by this bench)",
        "tight_cap": cap,
        "uses_gate_rows_for_cap": True,
        "gate_rows": len(rows),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    with (out / "advisories.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {out / 'metrics.json'}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the arms and write metrics.json")
    r.add_argument("--root", required=True, help="dir with data/processed (registry paths)")
    r.add_argument("--model", required=True, help="path to the SpinQuant INT4 .pte")
    r.add_argument("--tokenizer", required=True, help="path to tokenizer.model")
    r.add_argument("--out", required=True)
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--threads", type=int, default=3)
    r.add_argument("--limit", type=int, default=30)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--candidates", default=",".join(CANDIDATES))
    r.add_argument("--probe", action="store_true", help="record the .pte output shapes first")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        return run(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
