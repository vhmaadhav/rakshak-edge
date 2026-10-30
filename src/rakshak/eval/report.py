"""Write ``metrics.json`` and a markdown report with one table per split.

Input shape: ``{"title", "meta", "splits": {split_name: {model_key: result}}}``, where each
``result`` is a ``protocol.evaluate_scores`` dict, optionally with a ``robustness`` entry.
NaN and infinity are written as null so the JSON is strict.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

METRICS_FILE = "metrics.json"
REPORT_FILE = "report.md"


def sanitize(obj):
    """Make an object JSON-safe: numpy scalars to Python, NaN and inf to None, tuples to lists."""
    if isinstance(obj, Mapping):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(result: Mapping, key: str) -> str:
    interval = result.get("bootstrap", {}).get(key)
    if not interval:
        return ""
    return f" [{_fmt(interval[0])}, {_fmt(interval[1])}]"


def _targets(results: Mapping[str, Mapping]) -> list[str]:
    found: set[float] = set()
    for per_model in results.values():
        for result in per_model.values():
            for key in result.get("threshold_free", {}).get("recall_at_cal_fpr", {}):
                found.add(float(key))
    return [f"{t:g}" for t in sorted(found)]


def _row(key: str, result: Mapping, targets: list[str]) -> str:
    op = result["operating_points"]["max_f1"]
    tf = result["threshold_free"]
    cells = [
        key,
        _fmt(op["f1"]) + _ci(result, "f1"),
        _fmt(op["precision"]),
        _fmt(op["recall"]),
        _fmt(op["fpr"]),
        _fmt(op["mcc"]),
        _fmt(tf["pr_auc"]) + _ci(result, "pr_auc"),
        _fmt(tf["roc_auc"]),
    ]
    for t in targets:
        entry = tf["recall_at_cal_fpr"].get(t)
        cells.append("n/a" if entry is None else _fmt(entry["test_recall"]))
    cells += [_fmt(tf["ece"]), _fmt(tf["brier"])]
    rob = result.get("robustness")
    if rob and rob.get("summary"):
        cells += [_fmt(rob["summary"]["mean_recall_under_attack"]),
                  _fmt(rob["summary"]["mean_attack_success_rate"])]
    else:
        cells += ["n/a", "n/a"]
    return "| " + " | ".join(cells) + " |"


def render_markdown(payload: Mapping) -> str:
    results = payload["splits"]
    targets = _targets(results)
    header = ["Model", "F1 (max-F1 thr, 95% CI)", "Precision", "Recall", "FPR", "MCC",
              "PR-AUC (95% CI)", "ROC-AUC"]
    header += [f"Recall @ cal FPR<={float(t) * 100:g}%" for t in targets]
    header += ["ECE", "Brier", "Robust recall (mean)", "ASR (mean)"]
    lines = [f"# {payload.get('title', 'Rakshak evaluation')}", ""]
    for key, value in (payload.get("meta") or {}).items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "Thresholds are chosen on the calibration split only. Recall at FPR uses the threshold "
        "that gives FPR <= target on cal, measured on test. Bootstrap intervals are stratified "
        "and hold the threshold fixed.",
        "",
    ]
    for split, per_model in results.items():
        lines += [f"## {split}", ""]
        sizes = next(iter(per_model.values()))["sizes"] if per_model else {}
        if sizes:
            lines.append(
                f"Test: {sizes['test']} rows ({sizes['test_pos']} scam, "
                f"{sizes['test_neg']} benign); "
                f"cal: {sizes['cal']} rows ({sizes['cal_pos']} scam)."
            )
            lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for model_key, result in per_model.items():
            lines.append(_row(model_key, result, targets))
        lines.append("")
    return "\n".join(lines)


def write_report(payload: Mapping, out_dir: Path) -> dict[str, Path]:
    """Write metrics.json and report.md into ``out_dir``. Returns the two paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / METRICS_FILE
    tmp = metrics_path.with_name(metrics_path.name + ".tmp")
    tmp.write_text(json.dumps(sanitize(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(metrics_path)
    report_path = out_dir / REPORT_FILE
    report_path.write_text(render_markdown(payload), encoding="utf-8")
    return {"metrics": metrics_path, "report": report_path}
