"""Results on disk: results/<run_id>/ folders, the results/index.jsonl log, and comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rbench import SCHEMA


def results_root(bench_root: Path) -> Path:
    return bench_root / "results"


def append_index(results: Path, line: dict[str, Any]) -> None:
    results.mkdir(parents=True, exist_ok=True)
    with (results / "index.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True, default=str) + "\n")


def read_index(results: Path) -> list[dict[str, Any]]:
    path = results / "index.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_result(results: Path, run_id: str) -> dict[str, Any]:
    if "/" in run_id or run_id in ("", ".", ".."):
        raise ValueError(f"bad run id {run_id!r}")
    path = results / run_id / "result.json"
    if not path.is_file():
        raise FileNotFoundError(f"no result for run {run_id!r} at {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not an {SCHEMA} result")
    return doc


# Metrics compared between two runs, as dotted paths into result.json.
COMPARED = (
    "perf.latency_ms.p50",
    "perf.latency_ms.p95",
    "perf.latency_ms.p99",
    "perf.infer_ms.p50",
    "perf.infer_ms.p95",
    "perf.throughput_items_s",
    "perf.peak_rss_mb",
    "perf.load_s",
    "thermal.temp_max_c",
    "thermal.paused_s",
    "quality.at_threshold.f1",
    "quality.at_threshold.precision",
    "quality.at_threshold.recall",
    "quality.at_threshold.fpr",
    "quality.roc_auc",
    "quality.average_precision",
    "quality.fpr.fpr",
    "llm.ttft_ms",
    "llm.prefill_tok_s",
    "llm.decode_tok_s",
    "llm.json_parse_rate",
)


def _dig(doc: dict[str, Any], dotted: str) -> Any:
    cur: Any = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Side-by-side comparison of two results with deltas (b minus a) and comparability notes."""
    reasons: list[str] = []
    da, db = a.get("dataset") or {}, b.get("dataset") or {}
    if da.get("id") != db.get("id"):
        reasons.append(f"different datasets ({da.get('id')} vs {db.get('id')})")
    if da.get("sha256") != db.get("sha256"):
        reasons.append("dataset sha256 differs")
    if da.get("n_used") != db.get("n_used"):
        reasons.append(f"different row counts ({da.get('n_used')} vs {db.get('n_used')})")
    ma, mb = a.get("model") or {}, b.get("model") or {}
    if ma.get("id") != mb.get("id"):
        reasons.append(f"different models ({ma.get('id')} vs {mb.get('id')})")
    if (a.get("params") or {}).get("threads") != (b.get("params") or {}).get("threads"):
        reasons.append("different thread counts (latency is not directly comparable)")
    if a.get("status") != "ok" or b.get("status") != "ok":
        reasons.append("a run is not status ok")
    metrics: dict[str, Any] = {}
    for key in COMPARED:
        va, vb = _dig(a, key), _dig(b, key)
        if va is None and vb is None:
            continue
        delta = None
        pct = None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = round(float(vb) - float(va), 6)
            pct = round(100.0 * delta / float(va), 2) if va else None
        metrics[key] = {"a": va, "b": vb, "delta": delta, "delta_pct": pct}
    return {
        "a": a.get("run_id"),
        "b": b.get("run_id"),
        "comparable": not any(
            r.startswith(("different datasets", "dataset sha256", "different row counts"))
            for r in reasons
        ),
        "reasons": reasons,
        "metrics": metrics,
    }
