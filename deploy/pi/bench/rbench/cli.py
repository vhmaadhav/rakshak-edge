"""`rbench` command line. Stable exit codes and --json output for agents.

Exit codes: 0 ok, 2 bad usage or incompatible pair (or disabled entry), 3 missing artifact or
dependency, 4 thermal guard kill, 5 runtime error (including a held bench lock).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from rbench import EXIT_MISSING, EXIT_OK, EXIT_RUNTIME, EXIT_USAGE, __version__, store
from rbench.registry import (
    DEFAULT_REGISTRY,
    PACKAGE_DIR,
    IncompatibleError,
    RegistryError,
    check_compatible,
    compatible_datasets,
    dataset_state,
    load_registry,
    model_state,
    sync_manifest,
)


def default_root() -> Path:
    env = os.environ.get("RBENCH_HOME")
    return Path(env) if env else PACKAGE_DIR.parents[2]


def _emit(args: argparse.Namespace, payload: Any, text: str) -> None:
    if args.json:
        print(json.dumps(payload, indent=1, sort_keys=True, default=str))
    else:
        print(text)


def _fail(args: argparse.Namespace, code: int, message: str) -> int:
    if args.json:
        print(json.dumps({"error": message, "exit_code": code}, indent=1))
    else:
        print(f"rbench: {message}", file=sys.stderr)
    return code


def cmd_list(args: argparse.Namespace) -> int:
    reg = load_registry(args.registry)
    hash_it = not args.no_hash
    if args.what == "models":
        rows = []
        for m in reg.models.values():
            state = model_state(args.root, m, hash_it=hash_it)
            rows.append(
                {
                    "id": m.id,
                    "kind": m.kind,
                    "task": m.task,
                    "enabled": m.enabled,
                    "status": state["status"],
                    "threshold": m.threshold,
                    "compatible_datasets": compatible_datasets(reg, m),
                    "notes": m.notes,
                }
            )
        text = "\n".join(
            f"{r['id']:<28} {r['kind']:<10} {r['status']:<24} "
            f"datasets: {', '.join(r['compatible_datasets']) or '-'}"
            for r in rows
        )
        _emit(args, {"models": rows}, text)
    else:
        rows = []
        for d in reg.datasets.values():
            state = dataset_state(args.root, d, hash_it=hash_it)
            rows.append(
                {
                    "id": d.id,
                    "task": d.task,
                    "enabled": d.enabled,
                    "status": state["status"],
                    "synthetic": d.synthetic,
                    "notes": d.notes,
                }
            )
        text = "\n".join(f"{r['id']:<22} {r['task']:<14} {r['status']:<24}" for r in rows)
        _emit(args, {"datasets": rows}, text)
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    try:
        reg = load_registry(args.registry)
        model = reg.model(args.model)
        dataset = reg.dataset(args.dataset)
    except RegistryError as exc:
        return _fail(args, EXIT_USAGE, str(exc))
    except KeyError as exc:
        return _fail(args, EXIT_USAGE, str(exc).strip("'\""))
    if args.limit is not None and args.limit < 1:
        return _fail(args, EXIT_USAGE, "--limit must be >= 1")
    if args.threads < 1 or args.warmup < 0:
        return _fail(args, EXIT_USAGE, "--threads must be >= 1 and --warmup >= 0")
    if not model.enabled:
        return _fail(args, EXIT_USAGE, f"model {model.id} is disabled: {model.notes}")
    if not dataset.enabled:
        return _fail(args, EXIT_USAGE, f"dataset {dataset.id} is disabled: {dataset.notes}")
    try:
        check_compatible(model, dataset)
    except IncompatibleError as exc:
        return _fail(args, EXIT_USAGE, str(exc))
    for entry in model.files():
        if not (args.root / entry.path).is_file():
            hint = f" (run `rbench fetch {model.id}`)" if model.hf else ""
            return _fail(args, EXIT_MISSING, f"missing model file {entry.path}{hint}")
    if not dataset.synthetic and not (args.root / dataset.path).is_file():
        return _fail(args, EXIT_MISSING, f"missing dataset file {dataset.path}")

    from rbench.guard import BusyError, RunSpec, run_guarded

    spec = RunSpec(
        model=model.id,
        dataset=dataset.id,
        limit=args.limit,
        threads=args.threads,
        warmup=args.warmup,
        seed=args.seed,
        name=args.name or "",
        max_new_tokens=args.max_new_tokens,
        ela_mode=args.ela_mode,
        guards=args.guards,
    )
    try:
        doc = run_guarded(
            args.root,
            Path(args.registry),
            spec,
            job_run_dir=Path(args.job_run_dir) if args.job_run_dir else None,
        )
    except BusyError as exc:
        return _fail(args, EXIT_RUNTIME, str(exc))
    code = int(doc.pop("exit_code"))
    if args.json:
        print(json.dumps(doc, indent=1, sort_keys=True, default=str))
    else:
        print(_summary_line(doc))
        print(f"result: {args.root / 'results' / doc['run_id'] / 'result.json'}", file=sys.stderr)
    return code


def _summary_line(doc: dict[str, Any]) -> str:
    perf = doc.get("perf") or {}
    lat = perf.get("latency_ms") or {}
    thermal = doc.get("thermal") or {}
    quality = doc.get("quality") or {}
    key = quality.get("key_metric") or {}
    qtxt = (
        f"{key['name']}={key['value']:.4f}"
        if isinstance(key.get("value"), float)
        else (f"{key['name']}={key['value']}" if key else "quality: not measured")
    )
    return (
        f"{doc['run_id']} {doc['status']}: {doc['model']['id']} on {doc['dataset']['id']} "
        f"(n={doc['dataset'].get('n_used')}) p50 {lat.get('p50')} ms, p95 {lat.get('p95')} "
        f"ms, {qtxt}; max temp {thermal.get('temp_max_c')} C, throttled before "
        f"{thermal.get('throttled_before')} after {thermal.get('throttled_after')}"
        + (f"; error: {doc['error']}" if doc.get("error") else "")
    )


def cmd_results(args: argparse.Namespace) -> int:
    rows = store.read_index(store.results_root(args.root))
    if args.last:
        rows = rows[-args.last :]
    lines = [
        f"{r.get('run_id')} {str(r.get('status')):<15} {str(r.get('model')):<26} "
        f"{str(r.get('dataset')):<22} n={r.get('n')} p50={r.get('p50_ms')} "
        f"p95={r.get('p95_ms')} temp_max={r.get('max_temp_c')} "
        f"thr={r.get('throttled_after')}"
        for r in rows
    ]
    _emit(args, {"runs": rows}, "\n".join(lines) or "no runs yet")
    return EXIT_OK


def cmd_show(args: argparse.Namespace) -> int:
    try:
        doc = store.load_result(store.results_root(args.root), args.run_id)
    except (FileNotFoundError, ValueError) as exc:
        return _fail(args, EXIT_USAGE, str(exc))
    _emit(args, doc, json.dumps(doc, indent=1, sort_keys=True, default=str))
    return EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    results = store.results_root(args.root)
    try:
        a = store.load_result(results, args.run_a)
        b = store.load_result(results, args.run_b)
    except (FileNotFoundError, ValueError) as exc:
        return _fail(args, EXIT_USAGE, str(exc))
    report = store.compare(a, b)
    lines = [f"compare {report['a']} vs {report['b']} (comparable: {report['comparable']})"]
    lines += [f"  note: {r}" for r in report["reasons"]]
    for key, row in report["metrics"].items():
        lines.append(f"  {key:<34} {row['a']!s:>12} -> {row['b']!s:<12} delta {row['delta']}")
    _emit(args, report, "\n".join(lines))
    return EXIT_OK


def cmd_manifest(args: argparse.Namespace) -> int:
    """Files sync_bench.sh copies from the repo to the Pi (enabled entries with a source)."""
    try:
        reg = load_registry(args.registry)
    except RegistryError as exc:
        return _fail(args, EXIT_USAGE, str(exc))
    rows = sync_manifest(reg)
    text = "\n".join(
        "\t".join(r[k] for k in ("owner", "role", "source", "path", "sha256")) for r in rows
    )
    _emit(args, {"files": rows}, text)
    return EXIT_OK


def cmd_fetch(args: argparse.Namespace) -> int:
    from rbench.errors import ArtifactError, MissingDependency
    from rbench.fetch import FetchRefused, fetch_model

    try:
        reg = load_registry(args.registry)
        model = reg.model(args.model)
    except (RegistryError, KeyError) as exc:
        return _fail(args, EXIT_USAGE, str(exc).strip("'\""))
    try:
        result = fetch_model(args.root, model)
    except FetchRefused as exc:
        return _fail(args, EXIT_USAGE, str(exc))
    except ArtifactError as exc:
        return _fail(args, EXIT_MISSING, str(exc))
    except MissingDependency as exc:
        return _fail(args, EXIT_MISSING, str(exc))
    text = f"{model.id}: {result['status']} (downloaded: {result['downloaded'] or 'nothing'})"
    _emit(args, result, text)
    return EXIT_OK


def cmd_doctor(args: argparse.Namespace) -> int:
    from rbench.env import doctor

    report = doctor(args.root)
    reg_error = None
    states: list[dict[str, Any]] = []
    try:
        reg = load_registry(args.registry)
        states = [model_state(args.root, m, hash_it=not args.no_hash) for m in reg.models.values()]
        states += [
            dataset_state(args.root, d, hash_it=not args.no_hash) for d in reg.datasets.values()
        ]
    except RegistryError as exc:
        reg_error = str(exc)
    report["registry"] = {"path": str(args.registry), "error": reg_error}
    report["artifacts"] = states
    report["cli_version"] = __version__
    lines = [
        f"python {report['python']} ({report['python_executable']})",
        "imports: "
        + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in report["imports"].items()),
        "versions: " + ", ".join(f"{k}={v}" for k, v in report["versions"].items() if v),
        f"cpus {report['threads_cpu_count']}, ram {report['ram_total_mb']} MB, "
        f"temp {report['temp_c']} C, throttled {report['throttled']}",
        f"bench root {report['bench_root']}, git {report['git_sha']}",
    ]
    if reg_error:
        lines.append(f"registry error: {reg_error}")
    for st in states:
        lines.append(f"  {st['id']:<30} {st['status']}")
    missing_deps = not (report["imports"]["onnxruntime"] and report["imports"]["pandas"])
    _emit(args, report, "\n".join(lines))
    if reg_error:
        return EXIT_USAGE
    return EXIT_MISSING if missing_deps else EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="machine-readable output")
    common.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    common.add_argument(
        "--root",
        type=Path,
        default=default_root(),
        help="bench root (default: RBENCH_HOME or the repo root)",
    )
    p = argparse.ArgumentParser(
        prog="rbench", parents=[common], description="Benchmark Rakshak models on the Pi 5."
    )
    p.add_argument("--version", action="version", version=f"rbench {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", parents=[common], help="list models or datasets")
    s.add_argument("what", choices=["models", "datasets"])
    s.add_argument("--no-hash", action="store_true", help="skip sha256 (fast, no pin check)")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("run", parents=[common], help="run one model on one dataset")
    s.add_argument("--model", required=True)
    s.add_argument("--dataset", required=True)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--warmup", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--name", default="")
    s.add_argument("--max-new-tokens", type=int, default=64)
    s.add_argument(
        "--ela-mode",
        choices=("points", "slot", "slot_kw"),
        default="points",
        help="ELA contract: points (model writes the JSON), slot (one sentence per item) or "
        "slot_kw (slot with the evidence keyword already in the assistant prefix)",
    )
    s.add_argument(
        "--guards",
        action="store_true",
        help="edge guards: G1 (no explainer call for LOW risk) and G2 (attribution check)",
    )
    s.add_argument(
        "--job-run-dir",
        default=None,
        help="thermal job state dir (default ~/rakshak-pi/run, the dashboard's)",
    )
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("results", parents=[common], help="list recent runs")
    s.add_argument("--last", type=int, default=20)
    s.set_defaults(func=cmd_results)

    s = sub.add_parser("show", parents=[common], help="print one run's result.json")
    s.add_argument("run_id")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("compare", parents=[common], help="compare two runs")
    s.add_argument("run_a")
    s.add_argument("run_b")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("manifest", parents=[common], help="repo-to-Pi file list (for sync)")
    s.set_defaults(func=cmd_manifest)

    s = sub.add_parser("fetch", parents=[common], help="download a pinned prebuilt .pte")
    s.add_argument("model")
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("doctor", parents=[common], help="check the machine and artifacts")
    s.add_argument("--no-hash", action="store_true")
    s.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
