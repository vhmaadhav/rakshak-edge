"""Command line for the chunked labelling pipeline.

    python -m rakshak.llm.cli export JOB     write task chunks and BRIEF.md
    python -m rakshak.llm.cli validate JOB   check result files, log each chunk
    python -m rakshak.llm.cli merge JOB      write results.parquet from valid chunks
    python -m rakshak.llm.cli status JOB     summarise the ledger and the chunk files

No command calls a model API. Exit codes: 0 ok, 1 some chunk failed or is missing, 2 refused.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

from rakshak.llm import budget, chunks, jobs, validate

DEFAULT_ROOT = jobs.PROJECT_ROOT / "data" / "llm"


def _status(root: Path, job: str) -> int:
    spec = jobs.get_job(job)
    jdir = chunks.job_dir(root, spec.name)
    task_files = sorted((jdir / chunks.TASKS_DIR).glob("chunk_*.jsonl"))
    result_files = sorted((jdir / chunks.RESULTS_DIR).glob("chunk_*.jsonl"))
    latest = budget.latest_per_chunk(budget.read_ledger(root), spec.name)
    by_status: dict[str, list[str]] = {}
    for chunk, entry in latest.items():
        by_status.setdefault(entry["status"], []).append(chunk)
    messages = sum(entry["messages"] for entry in latest.values())

    print(f"job {spec.name}: {spec.description}")
    print(f"labeller: {spec.labeller}")
    print(f"exported chunks: {len(task_files)}, messages in ledger: {messages}")
    print(f"result files present: {len(result_files)}")
    for status in budget.STATUSES:
        chunk_list = sorted(by_status.get(status, []))
        if chunk_list:
            print(
                f"  {status}: {len(chunk_list)} ({', '.join(chunk_list[:5])}"
                f"{' ...' if len(chunk_list) > 5 else ''})"
            )
    merged = jdir / "results.parquet"
    if merged.exists():
        print(f"merged rows: {pq.ParquetFile(merged).metadata.num_rows} ({merged})")
    else:
        print("merged rows: none (run merge after validating)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rakshak.llm.cli")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="data/llm directory")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "validate", "merge", "status"):
        sub.add_parser(name).add_argument("job", type=str.upper)
    args = parser.parse_args(argv)

    try:
        spec = jobs.get_job(args.job)
    except (jobs.JobDisabledError, jobs.JobUnavailableError) as exc:
        print(exc, file=sys.stderr)
        return 2

    root = Path(args.root)
    if args.command == "status":
        return _status(root, spec.name)

    if args.command == "export":
        frame, parts = jobs.load_inputs()
        written = chunks.export_job(spec.name, root, frame, parts)
        messages = sum(
            1 for path in written for line in path.read_text(encoding="utf-8").split("\n") if line
        )
        print(
            f"exported {len(written)} chunks, {messages} messages, to "
            f"{chunks.job_dir(root, spec.name) / chunks.TASKS_DIR}"
        )
        print(f"brief: {chunks.job_dir(root, spec.name) / 'BRIEF.md'}")
        return 0

    if args.command == "validate":
        reports = validate.validate_job(spec.name, root)
        failed = [r for r in reports if not r.ok]
        for report in reports:
            if report.ok:
                print(f"{report.chunk} ok {report.n_found}/{report.n_expected}")
            else:
                print(
                    f"{report.chunk} FAIL {report.n_found}/{report.n_expected} "
                    f"({report.error_count} problems)"
                )
                for message in report.errors[:5]:
                    print(f"    {message}")
        print(f"{len(reports) - len(failed)}/{len(reports)} chunks ok")
        return 0 if not failed else 1

    if args.command == "merge":
        out, n_rows, skipped = validate.merge_job(spec.name, root)
        print(f"merged {n_rows} rows into {out}")
        if skipped:
            print(f"skipped {len(skipped)} chunks that failed validation: {', '.join(skipped)}")
            return 1
        return 0

    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
