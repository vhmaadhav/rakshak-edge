"""result.json schema "rbench/1": required keys and a validator used by the tests and the
dashboard before it shows a run."""

from __future__ import annotations

import datetime as dt
import secrets
from typing import Any

from rbench import SCHEMA

REQUIRED_KEYS: tuple[str, ...] = (
    "schema",
    "run_id",
    "ts",
    "status",
    "error",
    "model",
    "dataset",
    "params",
    "env",
    "perf",
    "llm",
    "quality",
    "thermal",
)
STATUSES = ("ok", "error", "thermal_killed")


def new_run_id(now: dt.datetime | None = None) -> str:
    """Sortable id: UTC timestamp plus a random suffix, e.g. 20261106T120102Z-3f9a1c."""
    now = now or dt.datetime.now(dt.UTC)
    return now.strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


def utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_result(doc: dict[str, Any]) -> list[str]:
    """Return a list of problems (empty when the document is a valid rbench/1 result)."""
    problems: list[str] = []
    if not isinstance(doc, dict):
        return ["result is not an object"]
    if doc.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}")
    for key in REQUIRED_KEYS:
        if key not in doc:
            problems.append(f"missing key {key!r}")
    if doc.get("status") not in STATUSES:
        problems.append(f"status must be one of {STATUSES}")
    for block in ("model", "dataset", "params", "env", "perf", "thermal"):
        if block in doc and not isinstance(doc[block], dict):
            problems.append(f"{block} must be an object")
    if isinstance(doc.get("dataset"), dict):
        for key in ("id", "sha256", "n_used", "denominator_notes"):
            if key not in doc["dataset"]:
                problems.append(f"dataset.{key} missing")
    if isinstance(doc.get("perf"), dict) and doc.get("status") == "ok":
        for key in ("load_s", "latency_ms", "throughput_items_s", "peak_rss_mb"):
            if key not in doc["perf"]:
                problems.append(f"perf.{key} missing")
    return problems


def index_line(doc: dict[str, Any], result_dir: str) -> dict[str, Any]:
    """One compact row for index.jsonl (what `results` and the dashboard table show)."""
    perf = doc.get("perf") or {}
    lat = perf.get("latency_ms") or {}
    quality = doc.get("quality") or {}
    thermal = doc.get("thermal") or {}
    key_metric = quality.get("key_metric") or {}
    return {
        "run_id": doc.get("run_id"),
        "ts": doc.get("ts"),
        "name": doc.get("name"),
        "status": doc.get("status"),
        "model": (doc.get("model") or {}).get("id"),
        "dataset": (doc.get("dataset") or {}).get("id"),
        "n": (doc.get("dataset") or {}).get("n_used"),
        "threads": (doc.get("params") or {}).get("threads"),
        "p50_ms": lat.get("p50"),
        "p95_ms": lat.get("p95"),
        "throughput_items_s": perf.get("throughput_items_s"),
        "key_metric": key_metric.get("name"),
        "key_value": key_metric.get("value"),
        "max_temp_c": thermal.get("temp_max_c"),
        "throttled_before": thermal.get("throttled_before"),
        "throttled_after": thermal.get("throttled_after"),
        "error": doc.get("error"),
        "dir": result_dir,
    }
