"""Export a job to task chunks and a brief for cloud labelling sessions.

Each chunk holds at most ``CHUNK_SIZE`` messages as JSON Lines with only ``id`` and ``text``.
No label, source, campaign or split field is written, so a session never sees metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rakshak.llm import budget, jobs, prompts

CHUNK_SIZE = 500
TASKS_DIR = "tasks"
RESULTS_DIR = "results"


def dump_line(row: dict[str, str]) -> str:
    """One JSON object on one line. U+2028, U+2029 and U+0085 are escaped because Python's
    str.splitlines() treats them as line breaks, which would break line counts."""
    text = json.dumps(row, ensure_ascii=False)
    return text.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029").replace("\x85", "\\u0085")


def job_dir(root: Path, job: str) -> Path:
    return Path(root) / job


def export_job(
    job: str,
    root: Path,
    frame: pd.DataFrame,
    parts: jobs.Partitions,
    chunk_size: int = CHUNK_SIZE,
) -> list[Path]:
    """Write tasks/chunk_XXXX.jsonl and BRIEF.md for one job, and log each chunk as exported."""
    spec = jobs.get_job(job)
    return export_spec(spec, root, frame, parts, chunk_size)


def export_spec(
    spec: jobs.JobSpec,
    root: Path,
    frame: pd.DataFrame,
    parts: jobs.Partitions,
    chunk_size: int = CHUNK_SIZE,
) -> list[Path]:
    """Use the same chunk/ledger contract with an explicit open-model job spec."""
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    ids = spec.select(frame, parts)
    if len(set(ids)) != len(ids):
        raise ValueError(f"job {spec.name}: selected ids are not unique")
    text_by_id = dict(zip(frame["id"].astype(str), frame["text"], strict=True))

    jdir = job_dir(root, spec.name)
    tasks = jdir / TASKS_DIR
    tasks.mkdir(parents=True, exist_ok=True)
    for stale in tasks.glob("chunk_*.jsonl"):
        stale.unlink()

    started = budget.now_iso()
    written: list[Path] = []
    for index, start in enumerate(range(0, len(ids), chunk_size)):
        part = ids[start : start + chunk_size]
        name = f"chunk_{index:04d}.jsonl"
        path = tasks / name
        with path.open("w", encoding="utf-8") as fh:
            for rid in part:
                row = {"id": rid, "text": str(text_by_id[rid] or "")}
                fh.write(dump_line(row) + "\n")
        budget.record_chunk(
            root,
            job=spec.name,
            model=spec.labeller,
            chunk=name,
            messages=len(part),
            started=started,
            finished=None,
            status="exported",
        )
        written.append(path)

    brief = prompts.render_brief(spec.name, spec.kind, spec.instructions, spec.schema_name)
    (jdir / "BRIEF.md").write_text(brief, encoding="utf-8")
    return written
