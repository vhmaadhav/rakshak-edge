"""Validate returned result files against their chunk, and merge valid chunks into one table.

For every chunk the checks are: every line is valid JSON, every line is an object with an ``id``
that belongs to the chunk, no id is repeated, no chunk id is missing, each line matches the
schema, and (annotation schemas only) each evidence span is a verbatim substring of its message.
A chunk with any problem is reported and is left out of the merge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from rakshak.llm import budget, chunks, jobs
from rakshak.llm.schemas import SCHEMAS, SPAN_SCHEMAS, validate, with_id

MAX_ERRORS_KEPT = 20


@dataclass
class ChunkReport:
    chunk: str
    n_expected: int
    n_found: int
    missing: bool = False
    error_count: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error_count == 0

    def add(self, message: str) -> None:
        self.error_count += 1
        if len(self.errors) < MAX_ERRORS_KEPT:
            self.errors.append(message)


def _read_jsonl(path: Path) -> list[tuple[int, str]]:
    return [
        (i, line)
        for i, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1)
        if line.strip()
    ]


def check_chunk(
    tasks_file: Path, results_file: Path, schema_name: str
) -> tuple[ChunkReport, dict[str, dict[str, Any]]]:
    """Check one result file. Returns the report and the valid rows keyed by id."""
    texts: dict[str, str] = {}
    for _, line in _read_jsonl(tasks_file):
        row = json.loads(line)
        texts[row["id"]] = row["text"]
    report = ChunkReport(chunk=tasks_file.name, n_expected=len(texts), n_found=0)
    rows: dict[str, dict[str, Any]] = {}

    if not results_file.exists():
        report.missing = True
        report.add(f"results file missing: {results_file}")
        return report, rows

    line_schema = with_id(SCHEMAS[schema_name])
    for lineno, line in _read_jsonl(results_file):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            report.add(f"line {lineno}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("id"), str):
            report.add(f"line {lineno}: not an object with a string 'id'")
            continue
        rid = obj["id"]
        if rid not in texts:
            report.add(f"line {lineno}: id {rid!r} is not in this chunk")
            continue
        if rid in rows:
            report.add(f"line {lineno}: duplicate id {rid!r}")
            continue
        problems = validate(obj, line_schema)
        if not problems and schema_name in SPAN_SCHEMAS:
            for span in obj["evidence_spans"]:
                if span not in texts[rid]:
                    problems.append(f"$.evidence_spans: not verbatim: {span[:40]!r}")
        if problems:
            for problem in problems:
                report.add(f"line {lineno} id {rid!r}: {problem}")
            continue
        rows[rid] = obj

    report.n_found = len(rows)
    missing = sorted(set(texts) - set(rows))
    if missing:
        report.add(f"{len(missing)} ids have no valid row, first: {missing[0]!r}")
    return report, rows


def check_job(job: str, root: Path) -> list[tuple[ChunkReport, dict[str, dict[str, Any]]]]:
    """Check every exported chunk of a job, in chunk order."""
    spec = jobs.get_job(job)
    tasks_dir = chunks.job_dir(root, spec.name) / chunks.TASKS_DIR
    task_files = sorted(tasks_dir.glob("chunk_*.jsonl")) if tasks_dir.exists() else []
    if not task_files:
        raise FileNotFoundError(f"no exported chunks for job {spec.name}; run export first")
    results_dir = chunks.job_dir(root, spec.name) / chunks.RESULTS_DIR
    return [check_chunk(task, results_dir / task.name, spec.schema_name) for task in task_files]


def validate_job(job: str, root: Path) -> list[ChunkReport]:
    """Check a job and append one ledger event per chunk."""
    spec = jobs.get_job(job)
    reports = []
    for report, _ in check_job(job, root):
        if report.missing:
            status = "missing"
        else:
            status = "validated_ok" if report.ok else "validated_failed"
        budget.record_chunk(
            root,
            job=spec.name,
            model=spec.labeller,
            chunk=report.chunk,
            messages=report.n_expected,
            started=budget.exported_start(root, spec.name, report.chunk),
            finished=budget.now_iso(),
            status=status,
        )
        reports.append(report)
    return reports


def merge_job(job: str, root: Path) -> tuple[Path, int, list[str]]:
    """Write results.parquet from the valid chunks only. Returns path, row count, skipped chunks."""
    spec = jobs.get_job(job)
    fields = list(SCHEMAS[spec.schema_name]["properties"])
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for report, rows in check_job(job, root):
        if not report.ok:
            skipped.append(report.chunk)
            continue
        for rid, obj in rows.items():
            record: dict[str, Any] = {"id": rid, "chunk": report.chunk}
            for name in fields:
                value = obj.get(name)
                if name == "evidence_spans":
                    value = json.dumps(value, ensure_ascii=False)
                record[name] = value
            records.append(record)
    out = chunks.job_dir(root, spec.name) / "results.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = ["id", "chunk", *fields]
    pd.DataFrame(records, columns=columns).to_parquet(out, index=False)
    return out, len(records), skipped
