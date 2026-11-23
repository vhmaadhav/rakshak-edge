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
import queue
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
        from edge_decision import DEFAULT_VERIFY_THRESHOLD
        from edge_pipeline import EdgeAnalyzer
        from explainer_cache import CachedGenerator
        from rakshak_edge_service import (
            ONNX_PATH,
            PTE_PATH,
            TOKENIZER_PATH,
            URL_MODEL_ID,
            _DetectorAdapter,
        )
        from rbench.explainer_candidates import BASELINE

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
        url_scorer, url_threshold, url_info = svc.load_url_model(
            root, hybrid=bool(cfg.get("url_hybrid", False))
        )
        info[URL_MODEL_ID] = url_info
        analyzer = EdgeAnalyzer(
            _DetectorAdapter(detector_runner, batch_onnx=bool(cfg["batch_onnx"])),
            explainer,
            threshold,
            detector_only_mode=cfg["detector_only_mode"],
            verify_threshold=cfg.get("verify_threshold", DEFAULT_VERIFY_THRESHOLD),
            tx_verify_threshold=cfg.get("tx_verify_threshold"),
            hinglish_mode=cfg["hinglish_mode"],
            url_scorer=url_scorer,
            url_threshold=url_threshold,
            explainer_candidate=cfg.get("explainer_candidate", BASELINE),
            explainer_cap=cfg.get("explainer_cap"),
        )
        info["placement"] = {"nice": int(cfg["llm_nice"]), "cpus": cfg.get("llm_cpus")}
        store["_llm"] = {"status": "ready", "pid": os.getpid(), "models": info}
        ready.put(("llm", os.getpid(), None))
        _job_loop(
            analyzer,
            jobs,
            store,
            kept=cfg.get("explanations_kept"),
            ttl=cfg.get("explain_ttl"),
        )
    except Exception as exc:  # noqa: BLE001 - reported to the supervisor, which exits
        ready.put(("error", "llm", repr(exc)))
        raise


def _job_loop(
    analyzer: Any,
    jobs: Any,
    store: Any,
    kept: int | None = None,
    ttl: float | None = None,
    clock: Any = time.time,
) -> None:
    """Run each explainer job: the full answer for one message, written under its id.

    A job that is still queued ``ttl`` seconds after it was queued is not run: its entry becomes
    ``expired``, and the verdict (already answered with the template) stands. A finished entry is
    dropped after ``kept`` later jobs or once it is older than ``ttl``, whichever comes first.
    Job counts are written to ``store["_jobs"]``. Keys starting with "_" are not explanations.
    """
    from rakshak_edge_service import EXPLAIN_TTL_S, EXPLANATIONS_KEPT

    cap = EXPLANATIONS_KEPT if kept is None else int(kept)
    max_age = EXPLAIN_TTL_S if ttl is None else float(ttl)
    done: collections.deque[tuple[str, float]] = collections.deque()
    counts = {"completed": 0, "failed": 0, "expired": 0}
    while True:
        job = jobs.get()
        if job is None:
            return
        eid, text, lang = job
        queued = (store.get(eid) or {}).get("queued_at")
        if queued is not None and clock() - queued > max_age:
            entry: dict[str, Any] = {
                "status": "expired",
                "explanation_source": "template",
                "reason": "expired",
            }
            counts["expired"] += 1
            started = None
        else:
            started = clock()
            try:
                t0 = time.perf_counter()
                full = analyzer.analyse(text, lang=lang)
                entry = {
                    "status": "ready",
                    "explanation_source": "llm",
                    "advisory": full["advisory"],
                    "verifier": full["verifier"],
                    "llm_calls": full["llm_calls"],
                    "timings_ms": full["timings_ms"],
                    "wall_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                }
                counts["completed"] += 1
            except Exception as exc:  # noqa: BLE001 - keep serving; the caller sees the error
                entry = {"status": "error", "detail": str(exc)}
                counts["failed"] += 1
        # Queue timing, recorded by the process that ran the job: e2e = finished - queued.
        finished = clock()
        entry.update(queued_at=queued, started_at=started, finished_at=finished)
        store[eid] = entry
        done.append((eid, finished))
        while done and (len(done) > cap or finished - done[0][1] > max_age):
            store.pop(done.popleft()[0], None)
        store["_jobs"] = dict(counts)


def _expire_stale(store: Any, ttl: float, clock: Any = time.time) -> int:
    """Marks pending entries older than ``ttl`` as expired. Run after the explainer process is
    restarted: a job it was running when it died never finishes; its client gets
    the template."""
    now = clock()
    swept = 0
    for key in list(store.keys()):
        if str(key).startswith("_"):
            continue
        entry = store.get(key) or {}
        if entry.get("status") == "pending" and now - entry.get("queued_at", now) > ttl:
            store[key] = {
                **entry,
                "status": "expired",
                "explanation_source": "template",
                "reason": "expired",
                "finished_at": now,
            }
            swept += 1
    return swept


def _limits(cfg: dict[str, Any]):
    """The request limits of the HTTP servers, from the picklable configuration."""
    from rakshak_edge_service import RequestLimits

    return RequestLimits(
        max_body_bytes=int(cfg["max_body_bytes"]),
        read_timeout_s=float(cfg["read_timeout"]),
    )


class ThreadWorker:
    """A worker thread with the surface the supervisor needs (start, is_alive, exitcode, join).
    ``exitcode`` is 0 when the target returns and 1 when it raises; the exception goes to
    ``on_error``. A thread cannot be stopped from outside, so it is a daemon."""

    def __init__(self, target: Any, args: tuple = (), name: str = "worker", on_error: Any = None):
        self.name = name
        self.exitcode: int | None = None
        self.error: str | None = None
        self._target = target
        self._args = tuple(args)
        self._on_error = on_error
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def terminate(self) -> None:
        """Nothing to do: a daemon thread ends with the process."""

    def _run(self) -> None:
        try:
            self._target(*self._args)
            self.exitcode = 0
        except BaseException as exc:  # noqa: BLE001 - the supervisor decides what happens next
            self.exitcode = 1
            self.error = repr(exc)
            if self._on_error is not None:
                self._on_error(exc)


class WorkerSupervisor:
    """Keeps one worker alive. ``spawn`` returns a new, unstarted worker (a multiprocessing.Process
    or a ThreadWorker). ``poll`` restarts the worker when it died with a failure, and logs and
    counts the death and the restart. A worker that fails more than ``max_restarts`` times is
    marked failed and is not restarted again. A worker that returns normally
    (exitcode 0) is done."""

    def __init__(self, name: str, spawn: Any, *, log: Any = None, max_restarts: int = 5):
        self.name = name
        self._spawn = spawn
        self._log = log or (lambda record: None)
        self.max_restarts = int(max_restarts)
        self.restarts = 0
        self.failed = False
        self.worker: Any = None

    def start(self) -> WorkerSupervisor:
        self.worker = self._spawn()
        self.worker.start()
        return self

    def poll(self) -> bool:
        """Restarts the worker if it died with a failure. Returns True when a restart happened."""
        worker = self.worker
        if self.failed or worker is None or worker.is_alive() or worker.exitcode == 0:
            return False
        self._log(
            {
                "event": "worker_died",
                "worker": self.name,
                "exitcode": worker.exitcode,
                "restarts": self.restarts,
                "ts": time.time(),
            }
        )
        if self.restarts >= self.max_restarts:
            self.failed = True
            self._log({"event": "worker_failed", "worker": self.name, "ts": time.time()})
            return False
        self.restarts += 1
        self.worker = self._spawn()
        self.worker.start()
        self._log(
            {
                "event": "worker_restarted",
                "worker": self.name,
                "restarts": self.restarts,
                "ts": time.time(),
            }
        )
        return True


def _watch(
    supervisors: list,
    stop: threading.Event,
    *,
    on_restart: Any,
    on_failed: Any,
    interval: float = 0.5,
) -> None:
    """Polls the supervisors until ``stop`` is set. A supervisor that has failed for good ends
    the watch: ``on_failed`` runs once, and ``stop`` is set."""
    while not stop.wait(interval):
        for sup in supervisors:
            if sup.poll():
                on_restart(sup)
            if sup.failed:
                on_failed(sup)
                stop.set()
                return


def build_verdict_service(cfg: dict[str, Any], store: Any, jobs: Any, counters: Any = None):
    """The verdict-side service of one process: detector (ONNX, ``onnx_threads``), the URL model,
    the NIDS model, the explanation table, and the Service with the shared store and job queue.
    Returns (service, the servers' index.html bytes, the detector info dict)."""
    svc = _engine_modules()
    from edge_decision import DEFAULT_VERIFY_THRESHOLD
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
    url_scorer, url_threshold, url_info = svc.load_url_model(
        root, hybrid=bool(cfg.get("url_hybrid", False))
    )
    info[URL_MODEL_ID] = url_info
    table_path = cfg.get("explanation_table")
    table = ExplanationTable.load(table_path) if table_path else None
    analyzer = EdgeAnalyzer(
        _DetectorAdapter(detector, batch_onnx=bool(cfg["batch_onnx"])),
        None,  # the LLM lives in the explainer process; the verdict path never calls it
        threshold,
        detector_only_mode=cfg["detector_only_mode"],
        verify_threshold=cfg.get("verify_threshold", DEFAULT_VERIFY_THRESHOLD),
        tx_verify_threshold=cfg.get("tx_verify_threshold"),
        hinglish_mode=cfg["hinglish_mode"],
        url_scorer=url_scorer,
        url_threshold=url_threshold,
        explanation_table=table,
    )
    nids_scorer, nids_info = svc.load_nids(root)
    info[NET_MODEL_ID] = nids_info
    slow_s = int(cfg.get("nids_dns_slow_window_s") or 0)
    nids = NidsAnalyzer(
        nids_scorer,
        alert=cfg["nids_alert"],
        uncertain=cfg["nids_uncertain"],
        scratch=root / "run" / "nids_uploads",
        flow_source=cfg["nids_flow_source"],
        model_mode=cfg["net_model_mode"],
        dns_min_distinct=int(cfg.get("nids_dns_min_distinct") or 1),
        dns_slow_window_ms=(slow_s * 1000 if slow_s > 0 else None),
    )
    log_path = cfg.get("log")
    service = Service(
        analyzer,
        threads=int(cfg["onnx_threads"]),
        temp_limit=float(cfg["temp_limit"]),
        log_path=Path(log_path) if log_path else None,
        models=info,
        detector_only_mode=cfg["detector_only_mode"],
        verify_threshold=cfg.get("verify_threshold", DEFAULT_VERIFY_THRESHOLD),
        hinglish_mode=cfg["hinglish_mode"],
        nids=nids,
        explainer_cache=None,
        llm_runner=cfg["llm_runner"],
        url_check=lambda text: check_url(
            text, url_scorer, url_threshold, high_threshold=cfg.get("url_high_threshold")
        ),
        response_mode="verdict_first",
        response_mode_source=cfg.get("response_mode_source", "flag"),
        jobs=jobs,
        store=store,
        counters=counters,
        device_ttl=float(cfg.get("device_ttl", 1.0)),
        explain_queue_depth=int(cfg["explain_queue_depth"]),
        explain_ttl=float(cfg["explain_ttl"]),
        explanations_kept=int(cfg["explanations_kept"]),
    )
    service.network = netns_report()
    index_html = (HERE / "index.html").read_bytes()
    return service, index_html, info


def http_main(
    cfg: dict[str, Any], store: Any, jobs: Any, ready: Any, index: int, counters: Any = None
) -> None:
    """One verdict worker process: its own detector, the shared port (SO_REUSEPORT)."""
    _die_with_parent()
    try:
        service, index_html, _info = build_verdict_service(cfg, store, jobs, counters)
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
        limits=_limits(cfg),
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
    server = serve_unix(service, path, index_html, limits=_limits(cfg))
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
        "verify_threshold": args.verify_threshold,
        "tx_verify_threshold": getattr(args, "tx_verify_threshold", None),
        "url_hybrid": bool(getattr(args, "url_hybrid", False)),
        "url_high_threshold": args.url_high_threshold,
        "nids_dns_min_distinct": getattr(args, "nids_dns_min_distinct", 1),
        "nids_dns_slow_window_s": getattr(args, "nids_dns_slow_window_s", 0),
        "hinglish_mode": args.hinglish_mode,
        "explainer_candidate": args.explainer_candidate,
        "explainer_cap": args.explainer_cap,
        "explanation_table": args.explanation_table,
        "nids_alert": args.nids_alert,
        "nids_uncertain": args.nids_uncertain,
        "nids_flow_source": args.nids_flow_source,
        "net_model_mode": args.net_model_mode,
        "workers": args.workers,
        "ready_fifo": args.ready_fifo,
        "device_ttl": args.device_ttl,
        "response_mode_source": args.response_mode_source,
        "max_body_bytes": args.max_body_bytes,
        "read_timeout": args.read_timeout,
        "explain_queue_depth": args.explain_queue_depth,
        "explain_ttl": args.explain_ttl,
        "explanations_kept": args.explanations_kept,
    }


def serve_verdict(args, root: Path) -> int:
    """Supervisor: starts the explainer process and the verdict workers, waits until every one is
    loaded, signals the ready FIFO, and serves until it is stopped."""
    from rakshak_edge_service import SharedCounters, _signal_ready

    cfg = cfg_from_args(args, root)
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    store = manager.dict()
    counters = SharedCounters(ctx=ctx)  # service-wide counters, shared by every process
    # Bounded: at most explain_queue_depth jobs wait for the explainer; a put beyond that is
    # refused and the answer falls back to the template (see Service._schedule_explanation).
    jobs = ctx.Queue(maxsize=int(cfg["explain_queue_depth"]))
    ready = ctx.Queue()
    stop = threading.Event()
    failed = {"value": False}
    supervisors: list[WorkerSupervisor] = []

    def log(record: dict[str, Any]) -> None:
        _emit(cfg.get("log"), record)

    def on_restart(sup: WorkerSupervisor) -> None:
        if sup.name == "llm":
            swept = _expire_stale(store, float(cfg["explain_ttl"]))
            log({"event": "explainer_jobs_swept", "expired": swept, "ts": time.time()})
        store["_workers"] = _worker_counts(supervisors)

    def on_failed(sup: WorkerSupervisor) -> None:
        failed["value"] = True
        store["_workers"] = _worker_counts(supervisors)
        log({"event": "service_stopping", "reason": f"{sup.name} failed", "ts": time.time()})
        os.kill(os.getpid(), signal.SIGTERM)  # unwinds through the finally block below

    def llm_spawn():
        return ctx.Process(target=llm_main, args=(cfg, jobs, store, ready), name="rakshak-llm")

    def http_spawn(i: int):
        return lambda: ctx.Process(
            target=http_main, args=(cfg, store, jobs, ready, i, counters), name=f"rakshak-http{i}"
        )

    # SIGTERM (stop_service.sh) unwinds through the finally block, so the children and the
    # Manager process are stopped too, instead of being left behind.
    signal.signal(signal.SIGTERM, _raise_exit)
    status = 0
    try:
        supervisors.append(WorkerSupervisor("llm", llm_spawn, log=log).start())
        workers = max(int(args.workers), 1)
        if workers == 1:
            # One verdict worker: it serves in this process; the LLM process is the other one.
            service, index_html, _info = build_verdict_service(cfg, store, jobs, counters)
            server = _make_http_server(service, cfg, index_html)
            unix_server = _maybe_unix(service, cfg, index_html, 0)
            waited = _wait_ready(ready, 1, _procs(supervisors))
            _announce(cfg, workers, os.getpid(), waited, store)
            _signal_ready(args.ready_fifo)
            threading.Thread(
                target=_watch,
                args=(supervisors, stop),
                kwargs={"on_restart": on_restart, "on_failed": on_failed},
                name="supervisor",
                daemon=True,
            ).start()
            _serve(server, unix_server)
        else:
            for i in range(workers):
                supervisors.append(WorkerSupervisor(f"http{i}", http_spawn(i), log=log).start())
            waited = _wait_ready(ready, 1 + workers, _procs(supervisors))
            _announce(cfg, workers, os.getpid(), waited, store)
            _signal_ready(args.ready_fifo)
            _watch(supervisors, stop, on_restart=on_restart, on_failed=on_failed)
    except KeyboardInterrupt:
        pass
    except SystemExit as exc:
        if isinstance(exc.code, str):  # a start-up failure: say why, and exit non-zero
            log({"event": "fatal", "error": exc.code, "ts": time.time()})
            status = 1
    finally:
        stop.set()
        _stop(_procs(supervisors), jobs)
        manager.shutdown()
    return 1 if failed["value"] else status


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
            "response_mode_source": cfg.get("response_mode_source"),
            "explain_queue_depth": cfg["explain_queue_depth"],
            "explain_ttl_s": cfg["explain_ttl"],
            "max_body_bytes": cfg["max_body_bytes"],
            "read_timeout_s": cfg["read_timeout"],
            "processes": [{"role": r, "pid": p} for r, p, *_ in ready_msgs],
            "llm": store.get("_llm"),
        },
    )


def _worker_counts(supervisors: list[WorkerSupervisor]) -> dict[str, dict[str, Any]]:
    """Restarts and failure state of each supervised worker, for GET /health."""
    return {s.name: {"restarts": s.restarts, "failed": s.failed} for s in supervisors}


def _procs(supervisors: list[WorkerSupervisor]) -> list[Any]:
    """The current worker of each supervisor (a restarted worker replaces the old one)."""
    return [s.worker for s in supervisors if s.worker is not None]


def _stop(procs: list[Any], jobs: Any) -> None:
    """Stops the processes: the explainer finishes its queue position and exits on None."""
    try:
        jobs.put(None, timeout=5.0)  # the queue is bounded: do not wait forever behind a full one
    except (OSError, ValueError, queue.Full):
        pass
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=10)
