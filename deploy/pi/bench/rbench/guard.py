"""Run one benchmark under the thermal guard and assemble its result.json.

The worker is the child of thermal_run.Job (pause 75 C, resume 65 C, kill 80 C). The guard's
per-second CSV becomes the thermal block. A lock file keeps two benchmarks from running at once.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rbench import EXIT_MISSING, EXIT_OK, EXIT_RUNTIME, EXIT_THERMAL, EXIT_USAGE, SCHEMA, store
from rbench import schema as S
from rbench.env import capture
from rbench.registry import PACKAGE_DIR
from rbench.telemetry import read_rows, summarise

LOCK_NAME = ".rbench.lock"
GUARD_PAUSE_C = 75.0
GUARD_RESUME_C = 65.0
GUARD_KILL_C = 80.0


class BusyError(RuntimeError):
    """Another benchmark holds the lock."""


def default_job_run_dir() -> Path:
    env = os.environ.get("RBENCH_JOB_RUN_DIR")
    return Path(env) if env else Path.home() / "rakshak-pi" / "run"


@dataclass
class RunSpec:
    model: str
    dataset: str
    limit: int | None
    threads: int
    warmup: int
    seed: int
    name: str
    max_new_tokens: int = 64
    ela_mode: str = "points"
    guards: bool = False


def _exit_for(status: str, worker_rc: int | None) -> int:
    if status == "ok":
        return EXIT_OK
    if status == "thermal_killed":
        return EXIT_THERMAL
    if worker_rc in (EXIT_USAGE, EXIT_MISSING):
        return worker_rc
    return EXIT_RUNTIME


def run_guarded(
    bench_root: Path,
    registry_path: Path,
    spec: RunSpec,
    *,
    job_run_dir: Path | None = None,
    job_factory=None,
) -> dict[str, Any]:
    """Run the worker under the guard. Returns the result document (also written to disk)."""
    from thermal_run import Guard, Job  # deploy/pi on sys.path (the wrapper sets it)

    results = store.results_root(bench_root)
    results.mkdir(parents=True, exist_ok=True)
    lock = (results / LOCK_NAME).open("w")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BusyError("another benchmark is running (results/.rbench.lock is held)") from exc
        run_id = S.new_run_id()
        out = results / run_id
        out.mkdir(parents=True)
        cmd = [
            sys.executable,
            "-m",
            "rbench.worker",
            "--root",
            str(bench_root),
            "--registry",
            str(registry_path),
            "--model",
            spec.model,
            "--dataset",
            spec.dataset,
            "--threads",
            str(spec.threads),
            "--warmup",
            str(spec.warmup),
            "--seed",
            str(spec.seed),
            "--max-new-tokens",
            str(spec.max_new_tokens),
            "--ela-mode",
            spec.ela_mode,
            "--out",
            str(out),
        ]
        if spec.guards:
            cmd.append("--guards")
        if spec.limit is not None:
            cmd += ["--limit", str(spec.limit)]
        env_path = [str(PACKAGE_DIR), str(PACKAGE_DIR.parent), str(bench_root / "src")]
        os.environ["PYTHONPATH"] = os.pathsep.join(
            env_path + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        )
        guard = Guard(GUARD_PAUSE_C, GUARD_RESUME_C, GUARD_KILL_C)
        run_dir = job_run_dir or default_job_run_dir()
        safe = f"rbench-{spec.model}-{spec.dataset}"
        factory = job_factory or Job
        job = factory(name=safe, cmd=cmd, guard=guard, run_dir=run_dir)
        started = time.monotonic()
        rc = job.run()
        wall_s = time.monotonic() - started
        worker_doc = _read_worker(out)
        state = job.state
        csv_src = Path(job.csv_path)
        log_src = Path(job.log_path)
        if csv_src.is_file():
            shutil.copy(csv_src, out / "telemetry.csv")
        if log_src.is_file():
            shutil.copy(log_src, out / "run.log")
        thermal = summarise(
            read_rows(csv_src),
            throttled_before=job.throttled_before,
            throttled_after=job.throttled_after,
            pauses=job.pauses,
            paused_s=job.paused_s,
            kill_temp_c=guard.kill_c if state == "killed" else None,
        )
        doc = _compose(bench_root, spec, run_id, worker_doc, thermal, state, rc, wall_s, out)
        (out / "result.json").write_text(
            json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        store.append_index(results, S.index_line(doc, str(out)))
        doc["exit_code"] = _exit_for(doc["status"], rc if state != "killed" else None)
        return doc
    finally:
        lock.close()


def _read_worker(out: Path) -> dict[str, Any] | None:
    path = out / "worker.json"
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return doc if isinstance(doc, dict) else None


def _compose(
    bench_root: Path,
    spec: RunSpec,
    run_id: str,
    worker: dict[str, Any] | None,
    thermal: dict[str, Any],
    state: str,
    rc: int,
    wall_s: float,
    out: Path,
) -> dict[str, Any]:
    if state == "killed":
        status = "thermal_killed"
        error = f"thermal guard killed the run at {GUARD_KILL_C:.0f} C"
    elif worker and worker.get("status") == "ok":
        status = "ok"
        error = None
    else:
        status = "error"
        error = (worker or {}).get("error") or f"worker exited with code {rc}; see run.log"
    worker = worker or {}
    doc = {
        "schema": SCHEMA,
        "run_id": run_id,
        "ts": S.utc_now_iso(),
        "name": spec.name,
        "status": status,
        "error": error,
        "model": worker.get("model") or {"id": spec.model},
        "dataset": worker.get("dataset")
        or {"id": spec.dataset, "sha256": None, "n_used": None, "denominator_notes": ""},
        "params": worker.get("params")
        or {"limit": spec.limit, "threads": spec.threads, "warmup": spec.warmup, "seed": spec.seed},
        "env": worker.get("env") or capture(bench_root),
        "perf": worker.get("perf") or {},
        "llm": worker.get("llm"),
        "quality": worker.get("quality") or {},
        "parity": worker.get("parity"),
        "thermal": thermal,
        "wall_s": round(wall_s, 2),
        "items_file": "items.jsonl" if (out / "items.jsonl").is_file() else None,
        "telemetry_file": "telemetry.csv",
        "log_file": "run.log",
    }
    problems = S.validate_result(doc) if status == "ok" else []
    if problems:
        doc["status"] = "error"
        doc["error"] = "result failed schema check: " + "; ".join(problems)
    return doc
