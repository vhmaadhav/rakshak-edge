"""Chunk ledger: a record of what was exported, validated and merged, per job and chunk.

Cloud credit is billed differently from the API, so there is no dollar arithmetic here. The
ledger is an append-only JSON Lines file at ``data/llm/ledger.jsonl``. Each entry is one chunk
event with the job, the labelling model, the number of messages, and start and finish times.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LEDGER_NAME = "ledger.jsonl"
STATUSES = ("exported", "validated_ok", "validated_failed", "missing")


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def ledger_path(root: Path) -> Path:
    return Path(root) / LEDGER_NAME


def record_chunk(
    root: Path,
    *,
    job: str,
    model: str,
    chunk: str,
    messages: int,
    started: str | None,
    finished: str | None,
    status: str,
) -> dict[str, Any]:
    """Append one chunk event and return the entry that was written."""
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    entry = {
        "job": job,
        "model": model,
        "chunk": chunk,
        "messages": messages,
        "started": started,
        "finished": finished,
        "status": status,
    }
    path = ledger_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def read_ledger(root: Path) -> list[dict[str, Any]]:
    path = ledger_path(root)
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            entries.append(json.loads(line))
    return entries


def latest_per_chunk(entries: list[dict[str, Any]], job: str) -> dict[str, dict[str, Any]]:
    """Latest entry for each chunk of one job (later lines win)."""
    latest: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if entry["job"] == job:
            latest[entry["chunk"]] = entry
    return latest


def exported_start(root: Path, job: str, chunk: str) -> str | None:
    """Start time of the most recent export of a chunk, if any."""
    start = None
    for entry in read_ledger(root):
        if entry["job"] == job and entry["chunk"] == chunk and entry["status"] == "exported":
            start = entry["started"]
    return start
