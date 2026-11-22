"""Verdict-first serving with the LLM in its own process (round 2).

Layout, for ``--response-mode verdict_first`` (the default ``--llm-process 1``):

- one LLM process (``llm_main``): loads the .pte and the detector once, runs the explainer jobs
  from a queue and writes each finished explanation to a shared store. It is niced
  (``--llm-nice``, default 10) and may be pinned (``--llm-cpus``), and it is the only process that
  runs the LLM. Its placement is set before any model thread exists, so every thread it creates
  (the ExecuTorch pool included) inherits it. A background explainer no longer competes with the
  verdict path for CPU time.
- ``--workers N`` verdict processes (``http_main``): each loads the detector with ONNX intra-op
  threads (``--onnx-threads``, default 1) and serves the same HTTP API on one port, shared by
  SO_REUSEPORT. ``N = 1`` runs the server inside the supervisor process instead.
- the explanation store is a Manager dict, so GET /explanation/<id> answers from any worker.

Nothing here runs on import. The spawn context is used for every child, so no child inherits a
half-initialised ONNX Runtime or LLM from the parent. Children exit with the supervisor
(PR_SET_PDEATHSIG).
"""

from __future__ import annotations

import collections
import ctypes
import importlib
import json
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PR_SET_PDEATHSIG = 1


def _die_with_parent() -> None:
    """Ask the kernel to send SIGTERM when the supervisor dies (Linux only; ignored elsewhere)."""
    try:
        ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except (OSError, AttributeError):
        pass


def _set_placement(nice: int, cpus: list[int] | None) -> None:
    """Lower the priority and fix the CPUs of the calling process, before its threads start.

    On Linux, nice and affinity are per thread, and a new thread inherits both from its creator,
    so setting them on the main thread of this process before the models load covers them all.
    """
    if nice:
        os.nice(nice)
    if cpus:
        os.sched_setaffinity(0, set(cpus))


def _emit(log_path: str | None, record: dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    if not log_path:
        print(line, file=sys.stderr, flush=True)
        return
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _engine_modules():
    """The service module, imported by name (its directory is on sys.path)."""
    return importlib.import_module("rakshak_edge_service")


def llm_main(cfg: dict[str, Any], jobs: Any, store: Any, ready: Any) -> None:
    """The explainer process: loads the LLM, then answers jobs until it gets None."""
    _die_with_parent()
    try:
        _set_placement(int(cfg["llm_nice"]), cfg.get("llm_cpus"))
        svc = _engine_modules()
        from edge_pipeline import EdgeAnalyzer
        from explainer_cache import CachedGenerator
        from rakshak_edge_service import (
            ONNX_PATH,
            PTE_PATH,
            TOKENIZER_PATH,
            URL_MODEL_ID,
            _DetectorAdapter,
        )

        root = Path(cfg["root"])
        detector_runner, llm, threshold, info = svc.load_models(
            root,
            int(cfg["llm_threads"]),
            root / ONNX_PATH,
            root / PTE_PATH,
            root / TOKENIZER_PATH,
            llm_runner=cfg["llm_runner"],
        )
        cache = int(cfg["explainer_cache"])
        explainer = CachedGenerator(llm, cache) if cache > 0 else llm
        url_scorer, url_threshold, url_info = svc.load_url_model(root)
        info[URL_MODEL_ID] = url_info
        analyzer = EdgeAnalyzer(
            _DetectorAdapter(detector_runner, batch_onnx=bool(cfg["batch_onnx"])),
            explainer,
            threshold,
            detector_only_mode=cfg["detector_only_mode"],
            hinglish_mode=cfg["hinglish_mode"],
            url_scorer=url_scorer,
            url_threshold=url_threshold,
        )
        info["placement"] = {"nice": int(cfg["llm_nice"]), "cpus": cfg.get("llm_cpus")}
        store["_llm"] = {"status": "ready", "pid": os.getpid(), "models": info}
        ready.put(("llm", os.getpid(), None))
        _job_loop(analyzer, jobs, store)
    except Exception as exc:  # noqa: BLE001 - reported to the supervisor, which exits
        ready.put(("error", "llm", repr(exc)))
        raise


def _job_loop(analyzer: Any, jobs: Any, store: Any, kept: int | None = None) -> None:
    """Run each explainer job: the full answer for one message, written under its id."""
    from rakshak_edge_service import EXPLANATIONS_KEPT

    cap = EXPLANATIONS_KEPT if kept is None else kept
    done: collections.deque[str] = collections.deque()
    while True:
        job = jobs.get()
        if job is None:
            return
        eid, text, lang = job
        queued = (store.get(eid) or {}).get("queued_at")
        started = time.time()
        try:
            t0 = time.perf_counter()
            full = analyzer.analyse(text, lang=lang)
            entry = {
                "status": "ready",
                "advisory": full["advisory"],
                "verifier": full["verifier"],
                "llm_calls": full["llm_calls"],
                "timings_ms": full["timings_ms"],
                "wall_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            }
        except Exception as exc:  # noqa: BLE001 - keep serving; the caller sees the error status
            entry = {"status": "error", "detail": str(exc)}
        # Queue timing, recorded by the process that ran the job: e2e = finished - queued.
        entry.update(queued_at=queued, started_at=started, finished_at=time.time())
        store[eid] = entry
        done.append(eid)
        while len(done) > cap:
            store.pop(done.popleft(), None)


def build_verdict_service(cfg: dict[str, Any], store: Any, jobs: Any):
    """The verdict-side service of one process: detector (ONNX, ``onnx_threads``), the URL model,
    the NIDS model, the explanation table, and the Service with the shared store and job queue.
    Returns (service, the servers' index.html bytes, the detector info dict)."""
    svc = _engine_modules()
    from edge_pipeline import EdgeAnalyzer, check_url
    from explanation_table import ExplanationTable
    from nids_pipeline import NidsAnalyzer
    from rakshak_edge_service import (
        DETECTOR_MODEL_ID,
        NET_MODEL_ID,
        ONNX_PATH,
        URL_MODEL_ID,
        Service,
        _DetectorAdapter,
        netns_report,
    )
    from rbench.registry import load_registry, model_state
    from rbench.runners.onnx import SmsRunner

    root = Path(cfg["root"])
    reg = load_registry(root / "deploy" / "pi" / "bench" / "registry.toml")
    info: dict[str, Any] = {}
    sms_model = reg.model(DETECTOR_MODEL_ID)
    state = model_state(root, sms_model, hash_it=True)
    info[DETECTOR_MODEL_ID] = {"status": state.get("status")}
    if state.get("status") != "ready":
        raise SystemExit(f"model {sms_model.id} is not ready: {state}")
    detector = SmsRunner()
    t0 = time.perf_counter()
    detector.load(root / ONNX_PATH, int(cfg["onnx_threads"]), {})
    info[DETECTOR_MODEL_ID]["load_s"] = round(time.perf_counter() - t0, 3)
    threshold = float(sms_model.threshold)
    url_scorer, url_threshold, url_info = svc.load_url_model(root)
    info[URL_MODEL_ID] = url_info
    table_path = cfg.get("explanation_table")
    table = ExplanationTable.load(table_path) if table_path else None
    analyzer = EdgeAnalyzer(
        _DetectorAdapter(detector, batch_onnx=bool(cfg["batch_onnx"])),
        None,  # the LLM lives in the explainer process; the verdict path never calls it
        threshold,
        detector_only_mode=cfg["detector_only_mode"],
        hinglish_mode=cfg["hinglish_mode"],
        url_scorer=url_scorer,
        url_threshold=url_threshold,
        explanation_table=table,
    )
    nids_scorer, nids_info = svc.load_nids(root)
    info[NET_MODEL_ID] = nids_info
    nids = NidsAnalyzer(
        nids_scorer,
        alert=cfg["nids_alert"],
        uncertain=cfg["nids_uncertain"],
        scratch=root / "run" / "nids_uploads",
        flow_source=cfg["nids_flow_source"],
        model_mode=cfg["net_model_mode"],
    )
    log_path = cfg.get("log")
    service = Service(
        analyzer,
        threads=int(cfg["onnx_threads"]),
        temp_limit=float(cfg["temp_limit"]),
        log_path=Path(log_path) if log_path else None,
        models=info,
        detector_only_mode=cfg["detector_only_mode"],
        hinglish_mode=cfg["hinglish_mode"],
        nids=nids,
        explainer_cache=None,
        llm_runner=cfg["llm_runner"],
        url_check=lambda text: check_url(text, url_scorer, url_threshold),
        response_mode="verdict_first",
        jobs=jobs,
        store=store,
        device_ttl=float(cfg.get("device_ttl", 1.0)),
    )
    service.network = netns_report()
    index_html = (HERE / "index.html").read_bytes()
    return service, index_html, info


def http_main(cfg: dict[str, Any], store: Any, jobs: Any, ready: Any, index: int) -> None:
    """One verdict worker process: its own detector, the shared port (SO_REUSEPORT)."""
    _die_with_parent()
    try:
        service, index_html, _info = build_verdict_service(cfg, store, jobs)
        server = _make_http_server(service, cfg, index_html)
        unix_server = _maybe_unix(service, cfg, index_html, index)
        ready.put(("http", os.getpid(), index))
        _serve(server, unix_server)
    except SystemExit as exc:
        ready.put(("error", f"http{index}", str(exc)))
        raise
    except Exception as exc:  # noqa: BLE001
        ready.put(("error", f"http{index}", repr(exc)))
        raise


def _make_http_server(service: Any, cfg: dict[str, Any], index_html: bytes):
    from rakshak_edge_service import serve

    return serve(
        service,
        cfg["host"],
        int(cfg["port"]),
        index_html,
        reuse_port=True,
    )


def worker_socket_path(base: str, index: int, workers: int) -> str:
    """One Unix socket per verdict worker: ``<base>.<index>`` when there are several workers, so
    the relay (socket_bridge.py) can spread connections over them; ``<base>`` for one worker."""
    return base if workers <= 1 else f"{base}.{index}"


def _maybe_unix(service: Any, cfg: dict[str, Any], index_html: bytes, index: int):
    """This worker's Unix socket (see worker_socket_path)."""
    base = cfg.get("unix_socket")
    if not base:
        return None
    from rakshak_edge_service import serve_unix

    path = worker_socket_path(base, index, int(cfg.get("workers") or 1))
    server = serve_unix(service, path, index_html)
    server.rakshak_path = path  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, name="unix", daemon=True).start()
    return server


def _serve(server: Any, unix_server: Any) -> None:
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if unix_server is not None:
            unix_server.server_close()
            try:
                os.unlink(unix_server.rakshak_path)  # a stopped worker leaves no socket behind
            except OSError:
                pass


def cfg_from_args(args, root: Path) -> dict[str, Any]:
    """The picklable configuration every process of the verdict-first service is started with."""
    cpus = None
    if args.llm_cpus:
        cpus = [int(c) for c in args.llm_cpus.split(",") if c.strip() != ""]
    return {
        "root": str(root),
        "host": args.host,
        "port": args.port,
        "unix_socket": args.unix_socket,
        "log": args.log,
        "temp_limit": args.temp_limit,
        "llm_threads": args.threads,
        "onnx_threads": args.onnx_threads,
        "llm_runner": args.llm_runner,
        "llm_nice": args.llm_nice,
        "llm_cpus": cpus,
        "explainer_cache": args.explainer_cache,
        "batch_onnx": args.batch_onnx,
        "detector_only_mode": args.detector_only_mode,
        "hinglish_mode": args.hinglish_mode,
        "explanation_table": args.explanation_table,
        "nids_alert": args.nids_alert,
        "nids_uncertain": args.nids_uncertain,
        "nids_flow_source": args.nids_flow_source,
        "net_model_mode": args.net_model_mode,
        "workers": args.workers,
        "ready_fifo": args.ready_fifo,
        "device_ttl": args.device_ttl,
    }


def serve_verdict(args, root: Path) -> int:
    """Supervisor: starts the explainer process and the verdict workers, waits until every one is
    loaded, signals the ready FIFO, and serves until it is stopped."""
    from rakshak_edge_service import _signal_ready

    cfg = cfg_from_args(args, root)
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    store = manager.dict()
    jobs = ctx.Queue()
    ready = ctx.Queue()
    procs: list[Any] = []
    # SIGTERM (stop_service.sh) unwinds through the finally block, so the children and the
    # Manager process are stopped too, instead of being left behind.
    signal.signal(signal.SIGTERM, _raise_exit)
    try:
        llm = ctx.Process(target=llm_main, args=(cfg, jobs, store, ready), name="rakshak-llm")
        llm.start()
        procs.append(llm)
        workers = max(int(args.workers), 1)
        if workers == 1:
            # One verdict worker: it serves in this process; the LLM process is the other one.
            service, index_html, _info = build_verdict_service(cfg, store, jobs)
            server = _make_http_server(service, cfg, index_html)
            unix_server = _maybe_unix(service, cfg, index_html, 0)
            waited = _wait_ready(ready, 1, procs)
            _announce(cfg, workers, os.getpid(), waited, store)
            _signal_ready(args.ready_fifo)
            _serve(server, unix_server)
            return 0
        for i in range(workers):
            p = ctx.Process(
                target=http_main, args=(cfg, store, jobs, ready, i), name=f"rakshak-http{i}"
            )
            p.start()
            procs.append(p)
        waited = _wait_ready(ready, 1 + workers, procs)
        _announce(cfg, workers, os.getpid(), waited, store)
        _signal_ready(args.ready_fifo)
        for p in procs:
            p.join()
        return 0
    except (KeyboardInterrupt, SystemExit):
        return 0
    finally:
        _stop(procs, jobs)
        manager.shutdown()


def _raise_exit(signum: int, frame: Any) -> None:
    raise SystemExit(0)


def _wait_ready(ready: Any, expected: int, procs: list[Any]) -> list[tuple]:
    """Blocks until ``expected`` processes report ready. A child that dies without reporting ends
    the wait with an error, and so does a child that reports one."""
    import queue as _queue

    seen: list[tuple] = []
    while len(seen) < expected:
        try:
            msg = ready.get(timeout=2.0)
        except _queue.Empty:
            dead = [p.name for p in procs if not p.is_alive() and p.exitcode not in (None, 0)]
            if dead:
                raise SystemExit(f"verdict service: {dead} exited before it was ready") from None
            continue
        if msg[0] == "error":
            raise SystemExit(f"verdict service: {msg[1]} failed: {msg[2]}")
        seen.append(msg)
    return seen


def _announce(cfg: dict[str, Any], workers: int, pid: int, ready_msgs: list, store: Any) -> None:
    _emit(
        cfg.get("log"),
        {
            "event": "ready",
            "pid": pid,
            "host": cfg["host"],
            "port": cfg["port"],
            "unix_socket": cfg.get("unix_socket"),
            "workers": workers,
            "onnx_threads": cfg["onnx_threads"],
            "llm_threads": cfg["llm_threads"],
            "llm_nice": cfg["llm_nice"],
            "llm_cpus": cfg["llm_cpus"],
            "response_mode": "verdict_first",
            "processes": [{"role": r, "pid": p} for r, p, *_ in ready_msgs],
            "llm": store.get("_llm"),
        },
    )


def _stop(procs: list[Any], jobs: Any) -> None:
    """Stops the processes: the explainer finishes its queue position and exits on None."""
    try:
        jobs.put(None)
    except (OSError, ValueError):
        pass
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=10)
