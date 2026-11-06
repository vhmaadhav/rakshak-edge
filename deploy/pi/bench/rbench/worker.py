"""The benchmark worker: loads one model and one dataset, times one item at a time, and writes
``worker.json`` and ``items.jsonl`` into the run directory.

It runs as a child of the thermal guard (see rbench/guard.py), so it never talks to the
thermal governor itself. Exit codes: 0 ok, 2 bad usage or incompatible pair, 3 missing
artifact or dependency, 5 runtime error.

    python -m rbench.worker --root ~/rakshak-bench --model sms_student --dataset sms_campaign_test \
        --limit 300 --threads 1 --warmup 10 --seed 0 --out ~/rakshak-bench/results/<run_id>
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from rbench import EXIT_MISSING, EXIT_OK, EXIT_RUNTIME, EXIT_USAGE
from rbench import metrics as M
from rbench.errors import ArtifactError, BenchError, MissingDependency
from rbench.registry import DEFAULT_REGISTRY as REGISTRY_PATH
from rbench.registry import (
    IncompatibleError,
    Model,
    RegistryError,
    check_compatible,
    file_sha256,
    load_registry,
)


def _extra_paths(root: Path, model: Model) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if model.side_scaler:
        out["side_scaler"] = root / model.side_scaler
    for extra in model.extras:
        out[extra.role] = root / extra.path
    return out


def verify_artifacts(
    root: Path,
    model: Model,
    dataset_path: Path | None,
    dataset_sha: str,
    split_sha: str,
    split_path: Path | None,
) -> dict[str, Any]:
    """Check every model file and the dataset file against their pins. Raises ArtifactError."""
    report: dict[str, Any] = {"files": []}
    for f in model.files():
        full = root / f.path
        if not full.is_file():
            raise ArtifactError(
                f"model file missing: {full} (see `rbench fetch {model.id}`)"
                if model.hf
                else f"model file missing: {full}"
            )
        actual = file_sha256(full)
        if f.sha256 and actual != f.sha256:
            raise ArtifactError(f"sha256 mismatch for {full}: expected {f.sha256}, got {actual}")
        report["files"].append(
            {"role": f.role, "path": f.path, "sha256": actual, "pinned": bool(f.sha256)}
        )
    if dataset_path is not None:
        if not dataset_path.is_file():
            raise ArtifactError(f"dataset file missing: {dataset_path}")
        if dataset_sha and file_sha256(dataset_path) != dataset_sha:
            raise ArtifactError(f"sha256 mismatch for dataset {dataset_path}")
    if split_path is not None:
        if not split_path.is_file():
            raise ArtifactError(f"split file missing: {split_path}")
        if split_sha and file_sha256(split_path) != split_sha:
            raise ArtifactError(f"sha256 mismatch for split {split_path}")
    report["model_sha256"] = report["files"][0]["sha256"]
    return report


def _peak_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)


def _latency_stats(values_ms: list[float]) -> dict[str, Any]:
    return M.percentiles(values_ms)


def _quality_labelled(items: list[dict], scores: list[float], threshold: float | None) -> dict:
    labels = [it["label"] for it in items]
    out: dict[str, Any] = {
        "n_pos": int(sum(1 for y in labels if y == 1)),
        "n_neg": int(sum(1 for y in labels if y == 0)),
        "roc_auc": M.roc_auc(labels, scores),
        "average_precision": M.average_precision(labels, scores),
        "threshold": threshold,
    }
    if threshold is not None:
        out["at_threshold"] = M.threshold_metrics(labels, scores, threshold)
        out["key_metric"] = {"name": "f1", "value": out["at_threshold"]["f1"]}
    else:
        out["key_metric"] = {"name": "average_precision", "value": out["average_precision"]}
    return out


def _quality_groundlm(
    items: list[dict], scores: list[float], threshold: float | None, notes: str
) -> dict:
    """FPR on policy-negatives: every row is a policy-negative (label 0)."""
    if threshold is None:
        return {"task": "groundlm_fpr", "status": "no threshold", "key_metric": None}
    fp = sum(1 for s in scores if s >= threshold)
    block = M.fpr_block(fp, len(scores))
    return {
        "task": "groundlm_fpr",
        "threshold": threshold,
        "fpr": block,
        "n_scored": len(scores),
        "denominator_notes": notes,
        "key_metric": {"name": "fpr", "value": block["fpr"]},
    }


def _parity(items: list[dict], model: Model, root: Path) -> dict | None:
    """Compare the ONNX logits with the logits the job stored for the same rows."""
    if not model.parity_logits or not model.parity_column:
        return None
    path = root / model.parity_logits
    if not path.is_file():
        return {"skipped": f"parity file missing: {path}"}
    import pandas as pd

    table = pd.read_parquet(path, columns=["id", model.parity_column])
    stored = dict(
        zip(table["id"].astype(str), table[model.parity_column].astype(float), strict=True)
    )
    pairs = [
        (it["logit"], stored[it["row_id"]])
        for it in items
        if it.get("logit") is not None and it["row_id"] in stored
    ]
    if not pairs:
        return {"n": 0, "note": "no measured row is in the stored logits"}
    diffs = [abs(a - b) for a, b in pairs]
    return {
        "n": len(pairs),
        "column": model.parity_column,
        "max_abs_logit_diff": round(float(max(diffs)), 6),
        "mean_abs_logit_diff": round(float(sum(diffs) / len(diffs)), 6),
    }


def run_onnx(root: Path, model: Model, dataset, args, out: Path) -> dict[str, Any]:
    from rbench.datasets import load_rows, synthetic_features
    from rbench.env import capture
    from rbench.runners import make_runner

    t_load = time.perf_counter()
    runner = make_runner(model.kind)
    extras = _extra_paths(root, model)
    runner.load(root / model.path, args.threads, extras)
    load_s = time.perf_counter() - t_load

    t_data = time.perf_counter()
    if dataset.synthetic:
        rows = synthetic_features(args.limit, args.seed)
    else:
        rows = load_rows(root, dataset, args.limit, args.seed)
    data_load_s = time.perf_counter() - t_data
    items = rows.items

    warmup = min(args.warmup, len(items))
    for item in items[:warmup]:
        runner.run(runner.make_feed(item))

    records: list[dict[str, Any]] = []
    lat_ms: list[float] = []
    inf_ms: list[float] = []
    for index, item in enumerate(items):
        t0 = time.perf_counter()
        feed = runner.make_feed(item)
        t1 = time.perf_counter()
        outputs = runner.run(feed)
        t2 = time.perf_counter()
        score, logit = runner.score(outputs)
        lat_ms.append((t2 - t0) * 1000.0)
        inf_ms.append((t2 - t1) * 1000.0)
        records.append(
            {
                "i": index,
                "row_id": item["row_id"],
                "label": item["label"],
                "score": round(score, 8),
                "logit": None if logit != logit else round(logit, 6),  # NaN -> None
                "latency_ms": round(lat_ms[-1], 4),
                "infer_ms": round(inf_ms[-1], 4),
            }
        )

    thr = model.threshold
    scores = [r["score"] for r in records]
    quality: dict[str, Any]
    if dataset.task == "nids_latency":
        quality = {
            "task": "nids_latency",
            "status": "not measured",
            "reason": "synthetic latency-only inputs",
            "key_metric": None,
        }
    elif dataset.task == "groundlm_fpr":
        quality = _quality_groundlm(items, scores, thr, rows.denominator_notes)
    else:
        quality = _quality_labelled(items, scores, thr)
    parity = _parity(records, model, root) if model.kind == "onnx_url" else None

    total_s = sum(lat_ms) / 1000.0
    perf = {
        "load_s": round(load_s, 4),
        "data_load_s": round(data_load_s, 4),
        "latency_ms": _latency_stats(lat_ms),
        "infer_ms": _latency_stats(inf_ms),
        "throughput_items_s": round(len(lat_ms) / total_s, 3) if total_s > 0 else None,
        "peak_rss_mb": _peak_rss_mb(),
        "latency_scope": (
            "featurise + onnxruntime run, one item at a time"
            if model.kind != "onnx_nids"
            else "onnxruntime run only"
        ),
        "warmup_items": warmup,
    }
    with (out / "items.jsonl").open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, separators=(",", ":")) + "\n")
    return {
        "perf": perf,
        "quality": quality,
        "parity": parity,
        "llm": None,
        "env_extra": capture(root),
        "dataset_info": {
            "n_file": rows.n_file,
            "n_filtered": rows.n_filtered,
            "n_used": rows.n_used,
            "notes": rows.denominator_notes,
        },
    }


def _ela_rows(root: Path, dataset, limit: int | None, seed: int):
    """Seeded random rows that have at least one evidence item.

    The rules hit only a minority of the campaign SMS test (about 8.5% in a 200-row probe), and
    a row with no evidence tests only the "No facts" path. So the sample is drawn from the rows
    the rules fire on. The scan count and hit rate go into the denominator notes.
    """
    import numpy as np

    from rakshak.ela_eval import package_for_sms
    from rbench.datasets import Rows, load_rows

    base = load_rows(root, dataset, None, seed)
    order = np.random.default_rng(seed).permutation(len(base.items))
    kept: list[dict[str, Any]] = []
    scanned = 0
    for idx in order:
        if limit is not None and len(kept) >= limit:
            break
        item = base.items[int(idx)]
        scanned += 1
        if len(package_for_sms(item["text"])[1].evidence) > 0:
            kept.append(item)
    hit_rate = round(len(kept) / scanned, 4) if scanned else None
    note = (
        f"{base.denominator_notes}; ELA sample: {len(kept)} rows with >=1 rule evidence item "
        f"drawn in seed-{seed} order after scanning {scanned} rows (hit rate {hit_rate})"
    )
    return Rows(
        items=kept,
        n_file=base.n_file,
        n_filtered=base.n_filtered,
        n_used=len(kept),
        denominator_notes=note,
    )


SLOT_MODES = ("slot", "slot_kw")  # one generate call per evidence item; the code writes the JSON


def _skips_llm(package) -> bool:
    """G1 test for one package (verifier.skip_llm_for_risk)."""
    from rakshak.verifier import skip_llm_for_risk

    return skip_llm_for_risk(package)


def run_llm(root: Path, model: Model, dataset, args, out: Path) -> dict[str, Any]:
    """LLM path: greedy generation, then JSON-validity and timing rollups.

    For ELA datasets the mode comes from ``args.ela_mode``. ``points`` makes one call per
    advisory and the model writes the whole JSON. ``slot`` makes one call per evidence item
    (one sentence each) and the code writes the JSON (rbench/slot.py).
    """
    from rbench.datasets import load_rows
    from rbench.env import capture
    from rbench.runners import make_runner

    t_load = time.perf_counter()
    runner = make_runner(model.kind)
    runner.load(root / model.path, args.threads, _extra_paths(root, model))
    load_s = time.perf_counter() - t_load
    t_data = time.perf_counter()
    if dataset.task == "ela_prompts":
        rows = _ela_rows(root, dataset, args.limit, args.seed)
    else:
        rows = load_rows(root, dataset, args.limit, args.seed)
    data_load_s = time.perf_counter() - t_data
    records: list[dict[str, Any]] = []
    walls: list[float] = []
    ttfts: list[float] = []
    prefill: list[float] = []
    decode: list[float] = []
    new_tokens: list[int] = []
    prompt_tokens_adv: list[int] = []
    calls_adv: list[int] = []
    nonempty = 0
    json_ok = 0
    ela = dataset.task == "ela_prompts"
    ela_mode = getattr(args, "ela_mode", "points") if ela else None
    ela_ok = 0
    ela_modes: dict[str, int] = {}
    points_n = points_pass = 0
    final_ok = 0
    slot_status: dict[str, int] = {}
    slot_stop_hits = slot_calls = 0
    slot_prompt_tokens: list[int] = []
    # Guards (G1, G2; see verifier.py). Off unless --guards, so existing runs are unchanged.
    guards = bool(getattr(args, "guards", False)) and ela
    llm_skipped = 0
    ela_ok_llm = 0
    g2_rejected: list[dict[str, Any]] = []
    for index, item in enumerate(rows.items):
        package = None
        prompt = item["text"]
        extra: dict[str, Any] = {}
        skipped = False
        if ela:
            from rakshak.ela_eval import package_for_sms

            _canon, package = package_for_sms(item["text"])
        if ela and (ela_mode in SLOT_MODES or (guards and _skips_llm(package))):
            from rbench.slot import SLOT_MAX_NEW_TOKENS, generate_slot_advisory

            res = generate_slot_advisory(
                runner,
                package,
                max_new_tokens=SLOT_MAX_NEW_TOKENS,
                keyword_prefix=(ela_mode == "slot_kw"),
                guards=guards,
            )
            text = res["output"]
            skipped = bool(res["skipped"])
            adv = res["adv"]
            calls = res["calls"]
            timing = _aggregate_calls(calls)
            for call in calls:
                slot_calls += 1
                slot_stop_hits += bool(call["stop_hit"])
                slot_status[call["status"]] = slot_status.get(call["status"], 0) + 1
                if call["prompt_tokens"] is not None:
                    slot_prompt_tokens.append(int(call["prompt_tokens"]))
            extra = {
                "raw_output": res["raw_output"],
                "point_status": res["point_status"],
                "point_sources": res["point_sources"],
                "point_checks": res["point_checks"],
                "prefixes": res["prefixes"],
                "call_records": calls,
                "llm_skipped_low_risk": skipped,
                "g2_rejected": res["g2_rejected"],
            }
            g2_rejected.extend({"i": index, **r} for r in res["g2_rejected"])
            final_ok += bool(res["final_verifier_ok"])
            prompt_chars = sum(len(c["prompt"]) for c in calls)
        else:
            if ela:
                # The repo's own ELA v2 construction: rules -> evidence package -> points prompt.
                from rakshak.ela_eval import ela_prompt

                prompt = ela_prompt(package)
            text, timing = runner.generate(prompt, max_new_tokens=args.max_new_tokens)
            if package is not None:
                from rakshak.ela_eval import score_advisory

                adv = score_advisory(text, package, risk=package.risk if guards else None)
                prompt_chars = len(prompt)
            calls = None
        if skipped:
            # No explainer call: the advisory is the template. Kept out of the raw LLM rates.
            llm_skipped += 1
            adv = {
                "mode": "template",
                "verifier_ok": False,
                "points_n": 0,
                "points_pass_n": 0,
                "skipped": True,
            }
            timing = timing | {
                "mode": "template",
                "verifier_ok": None,
                "calls": 0,
                "prompt_chars": 0,
            }
        if package is not None:
            ela_modes[adv["mode"]] = ela_modes.get(adv["mode"], 0) + 1
            if not skipped:
                ela_ok += bool(adv["verifier_ok"])
                ela_ok_llm += bool(adv["verifier_ok"])
                points_n += int(adv["points_n"])
                points_pass += int(adv["points_pass_n"])
                timing = timing | {
                    "verifier_ok": adv["verifier_ok"],
                    "mode": adv["mode"],
                    "json_ok_ela": adv["json_ok"],
                    "points_pass_n": adv["points_pass_n"],
                    "points_n": adv["points_n"],
                    "prompt_chars": prompt_chars,
                }
        walls.append(timing["wall_ms"])
        if timing.get("ttft_ms") is not None:
            ttfts.append(timing["ttft_ms"])
        if timing.get("prefill_tok_s") is not None:
            prefill.append(timing["prefill_tok_s"])
        if timing.get("decode_tok_s") is not None:
            decode.append(timing["decode_tok_s"])
        if timing.get("new_tokens") is not None:
            new_tokens.append(int(timing["new_tokens"]))
        if timing.get("prompt_tokens") is not None:
            prompt_tokens_adv.append(int(timing["prompt_tokens"]))
        calls_adv.append(int(timing.get("calls", 1)))
        nonempty += bool(text.strip())
        parsed = _json_parses(text)
        json_ok += parsed
        records.append(
            {
                "i": index,
                "row_id": item["row_id"],
                "output": text,
                "json_ok": parsed,
                **timing,
                **extra,
            }
        )
    with (out / "items.jsonl").open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")
    n = max(len(records), 1)

    def mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 3) if xs else None

    llm = {
        "ttft_ms": mean(ttfts),
        "prefill_tok_s": mean(prefill),
        "decode_tok_s": mean(decode),
        "new_tokens_mean": mean([float(x) for x in new_tokens]),
        "nonempty_rate": round(nonempty / n, 4),
        "json_parse_rate": round(json_ok / n, 4),
        # verifier_pass_rate and template_fallback_rate need the ELA evidence packages,
        # which this dataset does not carry (see registry: ela_generations).
        # Share of advisories whose raw explainer output passed check_points as a whole.
        "verifier_pass_rate": round(ela_ok / n, 4) if ela else None,
        # Guards on: G1 skips the explainer for LOW risk. The raw rate is over the advisories
        # that called the explainer; the template fallback rate is over all advisories.
        "guards": guards,
        "llm_skipped_low_risk": llm_skipped if guards else None,
        "llm_advisories": (len(records) - llm_skipped) if guards else None,
        "verifier_pass_rate_llm": (
            round(ela_ok_llm / (len(records) - llm_skipped), 4)
            if guards and len(records) > llm_skipped
            else None
        ),
        "g2_rejected_n": len(g2_rejected) if guards else None,
        "g2_rejected": g2_rejected if guards else None,
        # Share of advisories where any part fell back to the template (mode != "llm").
        "template_fallback_rate": (round(1 - ela_modes.get("llm", 0) / n, 4) if ela else None),
        "ela_modes": ela_modes or None,
        "ela_mode": ela_mode,
        "calls_per_advisory_mean": mean([float(x) for x in calls_adv]) if ela else None,
        "prompt_tokens_per_advisory_mean": mean([float(x) for x in prompt_tokens_adv]),
        "prompt_tokens_per_call_mean": mean([float(x) for x in slot_prompt_tokens]),
        "point_pass_rate": round(points_pass / points_n, 4) if ela and points_n else None,
        "point_template_rate": (round(1 - points_pass / points_n, 4) if ela and points_n else None),
        "final_verifier_pass_rate": round(final_ok / n, 4) if ela_mode in SLOT_MODES else None,
        "slot_status_counts": slot_status or None,
        "slot_stop_hit_rate": round(slot_stop_hits / slot_calls, 4) if slot_calls else None,
        "slot_max_new_tokens": _slot_budget() if ela_mode in SLOT_MODES else None,
        # Runtime facts: the model load (method load, inside the warmup) and the threadpool size
        # the runtime reports after the request. Both are None for runners without them.
        "model_load_ms": getattr(runner, "load_ms", None),
        "threads_set": getattr(runner, "threads_set", None),
        "threads_reported": getattr(runner, "threads_reported", None),
        "warmup_prompts": 1 if getattr(runner, "load_ms", None) is not None else 0,
    }
    perf = {
        "load_s": round(load_s, 4),
        "data_load_s": round(data_load_s, 4),
        "latency_ms": M.percentiles(walls),
        "infer_ms": M.percentiles(walls),
        "throughput_items_s": round(len(walls) / (sum(walls) / 1000.0), 4) if walls else None,
        "peak_rss_mb": _peak_rss_mb(),
        "latency_scope": (
            "whole advisory: one generate call per evidence item, then assembly"
            if ela_mode in SLOT_MODES
            else "whole generation per prompt (greedy, max_new_tokens fixed)"
        ),
        "warmup_items": 0,
    }
    return {
        "perf": perf,
        "quality": {
            "task": "ela_json",
            "key_metric": {"name": "json_parse_rate", "value": llm["json_parse_rate"]},
        },
        "parity": None,
        "llm": llm,
        "env_extra": capture(root),
        "dataset_info": {
            "n_file": rows.n_file,
            "n_filtered": rows.n_filtered,
            "n_used": rows.n_used,
            "notes": rows.denominator_notes,
        },
    }


def _slot_budget() -> int:
    from rbench.slot import SLOT_MAX_NEW_TOKENS

    return SLOT_MAX_NEW_TOKENS


def _aggregate_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-advisory timing from its generate calls: wall and tokens summed, rates averaged."""

    def total(key: str) -> int | None:
        vals = [int(c[key]) for c in calls if c.get(key) is not None]
        return sum(vals) if vals else None

    def avg(key: str) -> float | None:
        vals = [float(c[key]) for c in calls if c.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    ttft = avg("ttft_ms")
    return {
        "wall_ms": round(sum(float(c["wall_ms"] or 0.0) for c in calls), 3),
        "new_tokens": total("new_tokens"),
        "prompt_tokens": total("prompt_tokens"),
        "ttft_ms": ttft,
        "prefill_tok_s": avg("prefill_tok_s"),
        "decode_tok_s": avg("decode_tok_s"),
        "calls": len(calls),
    }


def _json_parses(text: str) -> bool:
    try:
        from rakshak.verifier import extract_json_object

        return extract_json_object(text) is not None
    except Exception:  # noqa: BLE001 - a broken verifier import must not hide the output
        try:
            json.loads(text)
            return True
        except ValueError:
            return False


def execute(args: argparse.Namespace) -> dict[str, Any]:
    """Run one benchmark and return the worker document. Raises BenchError subclasses."""
    root = Path(args.root)
    reg = load_registry(args.registry or REGISTRY_PATH)
    model = reg.model(args.model)
    dataset = reg.dataset(args.dataset)
    if not model.enabled:
        raise IncompatibleError(f"model {model.id} is disabled: {model.notes}")
    if not dataset.enabled:
        raise IncompatibleError(f"dataset {dataset.id} is disabled: {dataset.notes}")
    check_compatible(model, dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset_path = None if dataset.synthetic else root / dataset.path
    split_path = root / dataset.split_file if dataset.split_file else None
    artifacts = verify_artifacts(
        root, model, dataset_path, dataset.sha256, dataset.split_sha256, split_path
    )
    if model.kind == "pte_llm":
        body = run_llm(root, model, dataset, args, out)
    else:
        body = run_onnx(root, model, dataset, args, out)
    info = body.pop("dataset_info")
    env = body.pop("env_extra")
    return {
        "status": "ok",
        "error": None,
        "model": {
            "id": model.id,
            "kind": model.kind,
            "task": model.task,
            "path": model.path,
            "sha256": model.sha256 or artifacts["model_sha256"],
            "threshold": model.threshold,
            "notes": model.notes,
            "files": artifacts["files"],
        },
        "dataset": {
            "id": dataset.id,
            "task": dataset.task,
            "path": dataset.path,
            "sha256": dataset.sha256,
            "synthetic": dataset.synthetic,
            "n_file": info["n_file"],
            "n_filtered": info["n_filtered"],
            "n_used": info["n_used"],
            "denominator_notes": info["notes"],
            "sampling": {"limit": args.limit, "seed": args.seed},
        },
        "params": {
            "limit": args.limit,
            "threads": args.threads,
            "warmup": args.warmup,
            "seed": args.seed,
            "max_new_tokens": getattr(args, "max_new_tokens", None),
            "ela_mode": getattr(args, "ela_mode", None),
        },
        "env": env,
        "perf": body["perf"],
        "llm": body["llm"],
        "quality": body["quality"],
        "parity": body["parity"],
        "items_file": "items.jsonl",
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rbench.worker", description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True)
    p.add_argument("--registry", default=None)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--ela-mode",
        choices=("points", "slot", "slot_kw"),
        default="points",
        help="ELA contract: points (model writes the JSON), slot (one sentence per item) or "
        "slot_kw (slot with the evidence keyword already in the assistant prefix)",
    )
    p.add_argument(
        "--guards",
        action="store_true",
        help="edge guards: G1 (no explainer call for LOW risk) and G2 (attribution check)",
    )
    p.add_argument("--out", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.threads < 1 or args.warmup < 0 or (args.limit is not None and args.limit < 1):
        print("rbench.worker: threads >= 1, warmup >= 0, limit >= 1", file=sys.stderr)
        return EXIT_USAGE
    try:
        doc = execute(args)
    except (IncompatibleError, RegistryError, KeyError) as exc:
        print(f"rbench: {exc}", file=sys.stderr)
        _write_error(out, args, str(exc).strip("'\""), "usage")
        return EXIT_USAGE
    except (MissingDependency, ArtifactError) as exc:
        print(f"rbench: {exc}", file=sys.stderr)
        _write_error(out, args, str(exc), "missing")
        return EXIT_MISSING
    except BenchError as exc:
        print(f"rbench: {exc}", file=sys.stderr)
        _write_error(out, args, str(exc), "runtime")
        return EXIT_RUNTIME
    except Exception as exc:  # noqa: BLE001 - every unexpected failure is a runtime error
        traceback.print_exc()
        _write_error(out, args, f"{type(exc).__name__}: {exc}", "runtime")
        return EXIT_RUNTIME
    (out / "worker.json").write_text(
        json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(f"rbench.worker: ok, {doc['dataset']['n_used']} items", file=sys.stderr)
    return EXIT_OK


def _write_error(out: Path, args: argparse.Namespace, message: str, category: str) -> None:
    doc = {
        "status": "error",
        "error": message,
        "error_category": category,
        "params": {
            "limit": args.limit,
            "threads": args.threads,
            "warmup": args.warmup,
            "seed": args.seed,
        },
    }
    (out / "worker.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
