#!/usr/bin/env python3
"""Rakshak edge service: one resident process that loads the SMS student and the Llama .pte once.

    POST /analyze   {"sms": "..."}  -> decision, evidence, advisory, remediation, timings
    POST /analyze_url {"url": "..."} -> URL score, domain rules, verdict, remediation (no explainer)
    POST /analyze_flows  raw pcap body, or {"flows": [...]} -> per-host alerts, remediation, timings
    GET  /health                    -> models, threads, RSS, device state
    GET  /                          -> a minimal page to paste an SMS and see the stages

Detector-only flags (score >= threshold, no rule evidence) follow ``--detector-only-mode`` (env
RAKSHAK_DETECTOR_ONLY_MODE): "quiet" (default) or "verify" (see docs/EDGE_PIPELINE.md, Fix B).
Hinglish SMS (detect_lang) are answered in Hinglish; ``--hinglish-mode`` (env RAKSHAK_HINGLISH_MODE)
is "template" (verifier templates, no LLM) or "llm" (Hinglish slot prompt). The request may set
``"lang": "auto" | "en" | "hinglish"`` (default auto). ``--explainer-cache N`` (env
RAKSHAK_EXPLAINER_CACHE, default 256) memoises explainer sentences by exact prompt; 0 turns it off,
and latency benchmarks must use 0 (docs/EDGE_PIPELINE.md, "Explainer cache").
``--llm-runner`` (env RAKSHAK_LLM_RUNNER) defaults to ``direct`` (PteDirectRunner with prefix reuse)
since 2026-11-11; ``runner`` is the previous TextLLMRunner path, kept for comparison
(docs/EDGE_PIPELINE.md, "Prefix reuse (direct runner)").

Stdlib HTTP on 127.0.0.1 only. In sync mode requests run one at a time (a lock). In
verdict_first mode the verdict path takes no lock and the explainer runs as a background job under
that lock (--response-mode, --explanation-table, POST /analyze_batch, GET /explanation/<id>).
Before each request the SoC temperature is read: at or above ``--temp-limit`` (75 C) the
service answers 503 "cooling" and does not run the explainer. Each request is logged as one
JSON line (stderr and ``--log``).

Start and stop: deploy/pi/service/start_service.sh and stop_service.sh (they run the service in
``unshare -rn``, so it has no network). ``--ready-fifo`` is set by start_service.sh only.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import socket
import socketserver
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent import futures as cf
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [
    str(HERE),
    str(REPO / "src"),
    str(REPO / "deploy" / "pi" / "bench"),
    str(REPO / "deploy" / "pi"),
]

from edge_decision import (  # noqa: E402
    DEFAULT_DETECTOR_ONLY_MODE,
    DEFAULT_VERIFY_THRESHOLD,
    DETECTOR_ONLY_MODES,
    URL_ACTIVE_MODES,
)
from edge_device import EnergyWindow, read_snapshot  # noqa: E402
from edge_pipeline import (  # noqa: E402
    DEFAULT_HINGLISH_MODE,
    HINGLISH_MODES,
    LANG_REQUESTS,
    check_url,
)
from nids_pipeline import (  # noqa: E402
    FLOW_SOURCES,
    MAX_BODY_BYTES,
    NET_ALERT_THRESHOLD,
    NET_MODEL_ID,
    NET_MODEL_MODE,
    NET_MODEL_MODES,
    NET_UNCERTAIN_THRESHOLD,
    FlowRequestError,
    NidsAnalyzer,
)
from rbench.explainer_candidates import BASELINE, CANDIDATES  # noqa: E402

DEFAULT_PORT = 8770
DEFAULT_HOST = "127.0.0.1"
# Response mode (docs/EDGE_PIPELINE.md, "Verdict first"). sync: /analyze returns the verdict and
# the explainer text together, after the LLM (the default; unchanged). verdict_first: /analyze
# returns the verdict at once and the explainer text is a table lookup or a background job,
# read from GET /explanation/<id>.
RESPONSE_MODES = ("sync", "verdict_first")
DEFAULT_RESPONSE_MODE = "sync"
VERDICT_DEVICE_TTL_S = 1.0  # verdict path: temperature re-read at most this often
MAX_BATCH = 1024  # messages per POST /analyze_batch (answered in chunks of BATCH_CHUNK)
BATCH_CHUNK = 128
EXPLANATIONS_KEPT = 4096  # finished explanations kept for GET /explanation/<id> (the store's cap)
# Explainer jobs (verdict_first). At most EXPLAIN_QUEUE_DEPTH jobs wait for the LLM; a job that
# would wait behind a full queue is refused, and its answer is the template, with the reason. A job
# that waits longer than EXPLAIN_TTL_S, or a finished explanation older than that, is expired.
EXPLAIN_QUEUE_DEPTH = 64
EXPLAIN_TTL_S = 120.0
# A POST body must arrive within this many seconds (408 otherwise).
DEFAULT_READ_TIMEOUT_S = 10.0
EGRESS_PROBE = ("1.1.1.1", 443)
DEFAULT_TEMP_LIMIT = 75.0
MAX_SMS_CHARS = 2000
MAX_URL_CHARS = 2000
URL_MODEL_ID = "url_student"
ONNX_PATH = "models/sms_student/best_student.onnx"
PTE_PATH = "models/llama32_1b_spinquant/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte"
TOKENIZER_PATH = "models/llama32_1b_spinquant/tokenizer.model"
DETECTOR_MODEL_ID = "sms_student"
LLM_MODEL_ID = "llama32_1b_spinquant"
NIDS_ONNX_PATH = "models/nids/student_cse_cic_ids2018_int8.onnx"


@dataclass(frozen=True)
class RequestLimits:
    """Bounds on what the HTTP layer accepts, checked before parsing. ``max_body_bytes`` is the
    cap on every POST body; the default is the flow path's cap (nids_pipeline.MAX_BODY_BYTES), which
    is also large enough for a full batch of MAX_BATCH messages of MAX_SMS_CHARS characters."""

    max_body_bytes: int = MAX_BODY_BYTES
    read_timeout_s: float = DEFAULT_READ_TIMEOUT_S
    max_batch: int = MAX_BATCH
    max_sms_chars: int = MAX_SMS_CHARS
    max_url_chars: int = MAX_URL_CHARS


def netns_report(probe: bool = True) -> dict[str, Any]:
    """What this process can see of the network: its interfaces and routes (its own namespace,
    from /proc/self/net) and one outbound connect attempt. Under unshare -rn the only interface
    is lo, there is no route, and the probe fails with "Network is unreachable"."""
    ifaces: list[str] = []
    routes = 0
    try:
        with open("/proc/self/net/dev", encoding="utf-8") as handle:
            ifaces = [line.split(":")[0].strip() for line in handle if ":" in line]
        with open("/proc/self/net/route", encoding="utf-8") as handle:
            routes = max(sum(1 for _ in handle) - 1, 0)
    except OSError as exc:
        return {"error": repr(exc)}
    out: dict[str, Any] = {"interfaces": ifaces, "routes": routes}
    if probe:
        try:
            with socket.create_connection(EGRESS_PROBE, timeout=2):
                out["egress_probe"] = "connected (network is NOT isolated)"
        except OSError as exc:
            out["egress_probe"] = f"failed: {exc.strerror or exc}"
    return out


def rss_mb(which: str = "VmRSS") -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(which + ":"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        return None
    return None


# Service-wide counters. Every verdict process of one service adds to the same shared memory
# (a lock-protected array), so /health reports exact totals whichever process answers.
COUNTER_SLOTS = (
    "http_requests_total",
    "batch_requests_total",
    "messages_total",
    "overloaded",
    "expired",
    "refused_cooling",
)
ENDPOINT_SLOTS = (
    "/",
    "/health",
    "/analyze",
    "/analyze_batch",
    "/analyze_url",
    "/analyze_flows",
    "/explanation",
    "other",
)


class SharedCounters:
    """Counters in shared memory. Pass ``ctx`` (the spawn context of the supervisor) so the array
    can be handed to child processes; without it a spawn-context array is made here. The array's
    own lock makes each increment atomic across processes."""

    def __init__(self, ctx: Any = None, array: Any = None) -> None:
        if array is None:
            ctx = ctx or mp.get_context("spawn")
            array = ctx.Array("q", len(COUNTER_SLOTS) + len(ENDPOINT_SLOTS), lock=True)
        self._array = array

    def add(self, name: str, n: int = 1) -> None:
        i = COUNTER_SLOTS.index(name)
        with self._array.get_lock():
            self._array[i] += n

    def add_endpoint(self, endpoint: str) -> None:
        """One HTTP request on ``endpoint`` (unknown names count as "other")."""
        if endpoint not in ENDPOINT_SLOTS:
            endpoint = "other"
        j = len(COUNTER_SLOTS) + ENDPOINT_SLOTS.index(endpoint)
        with self._array.get_lock():
            self._array[COUNTER_SLOTS.index("http_requests_total")] += 1
            self._array[j] += 1

    def get(self, name: str) -> int:
        with self._array.get_lock():
            return int(self._array[COUNTER_SLOTS.index(name)])

    def snapshot(self) -> dict[str, Any]:
        with self._array.get_lock():
            values = [int(v) for v in self._array[:]]
        out: dict[str, Any] = {name: values[i] for i, name in enumerate(COUNTER_SLOTS)}
        base = len(COUNTER_SLOTS)
        out["http_requests_by_endpoint"] = {
            ep: values[base + k] for k, ep in enumerate(ENDPOINT_SLOTS) if values[base + k]
        }
        return out


class Service:
    """Loaded models and the request path. HTTP is a thin layer over this (see Handler)."""

    def __init__(
        self,
        analyzer: Any,
        *,
        threads: int,
        temp_limit: float,
        log_path: Path | None,
        models: dict[str, Any],
        detector_only_mode: str = DEFAULT_DETECTOR_ONLY_MODE,
        verify_threshold: float = DEFAULT_VERIFY_THRESHOLD,
        hinglish_mode: str = DEFAULT_HINGLISH_MODE,
        device_fn=read_snapshot,
        energy_factory=EnergyWindow,
        clock=time.time,
        nids: NidsAnalyzer | None = None,
        explainer_cache: Any | None = None,
        llm_runner: str = "runner",
        url_check: Any | None = None,
        response_mode: str = DEFAULT_RESPONSE_MODE,
        jobs: Any | None = None,
        store: Any | None = None,
        verdict_device_fn=None,
        device_ttl: float = VERDICT_DEVICE_TTL_S,
        response_mode_source: str = "default",
        explain_queue_depth: int = EXPLAIN_QUEUE_DEPTH,
        explain_ttl: float = EXPLAIN_TTL_S,
        explanations_kept: int = EXPLANATIONS_KEPT,
        counters: SharedCounters | None = None,
    ) -> None:
        """``jobs`` and ``store`` (verdict-first, edge_workers.py): explainer jobs go to the one
        process that holds the LLM, and finished explanations are read from ``store`` (a mapping
        shared with that process). Without them the explainer runs in this process, as before."""
        if response_mode not in RESPONSE_MODES:
            raise ValueError(
                f"response_mode must be one of {RESPONSE_MODES}, got {response_mode!r}"
            )
        self.response_mode = response_mode
        self.response_mode_source = response_mode_source  # "flag", "env" or "default"
        self.explain_queue_depth = int(explain_queue_depth)
        self.explain_ttl = float(explain_ttl)
        self.explanations_kept = int(explanations_kept)
        self.analyzer = analyzer
        self.nids = nids
        self.url_check = url_check
        self.explainer_cache = explainer_cache
        self.llm_runner = llm_runner
        self.threads = threads
        self.temp_limit = temp_limit
        self.log_path = log_path
        self.models = models
        self.detector_only_mode = detector_only_mode
        self.verify_threshold = float(verify_threshold)
        self.hinglish_mode = hinglish_mode
        self.device_fn = device_fn
        self.energy_factory = energy_factory
        self.clock = clock
        self.jobs = jobs
        self.store = store
        # The verdict path reads the SoC temperature (the cooling check) at most once per
        # ``device_ttl`` seconds: the full snapshot costs three vcgencmd subprocesses, about 30 ms,
        # and that was the whole HTTP overhead of round 1. The sync path still reads everything.
        if verdict_device_fn is None:
            if device_fn is read_snapshot:  # the default: temperature only, see above
                verdict_device_fn = lambda: read_snapshot(  # noqa: E731
                    throttled_fn=lambda: None, power_fn=lambda: None
                )
            else:  # a caller's own device reader is used as it is
                verdict_device_fn = device_fn
        self.verdict_device_fn = verdict_device_fn
        self.device_ttl = float(device_ttl)
        self._device_cache: tuple[float, dict[str, Any]] | None = None
        self._device_lock = threading.Lock()
        self.lock = threading.Lock()  # the LLM, and the sync path: one explainer at a time
        self._stats_lock = threading.Lock()
        self._explanations_lock = threading.Lock()
        self._explanations: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._explain_pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="explain")
        self._explain_futures: list[cf.Future] = []
        self.started = clock()
        self.network: dict[str, Any] = {}
        # Counters (health), shared by every verdict process of the service (see SharedCounters).
        # One HTTP request is counted by the handler, one batch by analyse_batch, one message per
        # message scored (/analyze: 1; /analyze_batch: its items).
        self.counters = counters if counters is not None else SharedCounters()
        self._log_lock = threading.Lock()
        # One handle for the life of the process (opening the file per line cost a syscall round
        # trip per message). Line-buffered, so each record is on disk when its request returns.
        self._log_handle = (
            open(self.log_path, "a", encoding="utf-8", buffering=1) if log_path else None
        )

    @property
    def refused(self) -> int:
        """Service-wide count of cooling refusals, backed by SharedCounters."""
        return self.counters.get("refused_cooling")

    def _log(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if self._log_handle is None:
            print(line, file=sys.stderr, flush=True)
            return
        with self._log_lock:
            self._log_handle.write(line + "\n")

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "pid": os.getpid(),
            "uptime_s": round(self.clock() - self.started, 1),
            "threads": self.threads,
            "temp_limit_c": self.temp_limit,
            "models": self.models,
            "detector_only_mode": self.detector_only_mode,
            "verify_threshold": self.verify_threshold,
            "url_model": {
                # Diagnostic only (no per-request field in /analyze). Quiet mode never scores
                # URLs for /analyze; balanced adds corroborated URLs; /analyze_url works always.
                "enabled": bool((self.models.get(URL_MODEL_ID) or {}).get("enabled")),
                "threshold": (self.models.get(URL_MODEL_ID) or {}).get("threshold"),
                "evidence_visible": self.detector_only_mode in URL_ACTIVE_MODES,
            },
            "hinglish_mode": self.hinglish_mode,
            "response_mode": self.response_mode,
            # Explicit: the default is sync; the source says whether a flag, the environment
            # (RAKSHAK_RESPONSE_MODE) or the default chose this mode.
            "response_mode_config": {
                "mode": self.response_mode,
                "source": self.response_mode_source,
                "default": DEFAULT_RESPONSE_MODE,
                "flag": "--response-mode",
                "env": "RAKSHAK_RESPONSE_MODE",
            },
            "explanation_table": (
                self.analyzer.explanation_table.stats()
                if getattr(self.analyzer, "explanation_table", None) is not None
                else None
            ),
            "llm_runner": self.llm_runner,
            "llm": self.store.get("_llm") if self.store is not None else None,
            **self._request_counts(),
            "refused_cooling": self.refused,
            "explain_jobs": self._explain_jobs_health(),
            "workers": self.store.get("_workers") if self.store is not None else None,
            "rss_mb": rss_mb(),
            "rss_peak_mb": rss_mb("VmHWM"),
            "device": self.device_fn(),
            "network": self.network,
            "explainer_cache": (
                self.explainer_cache.stats()
                if self.explainer_cache is not None
                else {"capacity": 0, "size": 0, "hits": 0, "misses": 0}
            ),
        }

    # ----------------------------------------------------------------- counters
    def count_http(self, endpoint: str) -> None:
        """One call per HTTP request, made by the handler whatever the answer."""
        self.counters.add_endpoint(endpoint)

    def _request_counts(self) -> dict[str, Any]:
        """The service-wide request counters for /health. ``requests`` is a deprecated alias of
        http_requests_total, kept so existing readers keep working."""
        snap = self.counters.snapshot()
        return {
            "http_requests_total": snap["http_requests_total"],
            "http_requests_by_endpoint": snap["http_requests_by_endpoint"],
            "batch_requests_total": snap["batch_requests_total"],
            "messages_total": snap["messages_total"],
            "requests": snap["http_requests_total"],
        }

    def _count_messages(self, n: int) -> None:
        """Messages scored: one for /analyze, one per item of /analyze_batch."""
        self.counters.add("messages_total", n)

    def _overloaded(self, request_id: str) -> dict[str, Any]:
        """The explainer job queue is full. The answer is the template, and it says why."""
        self.counters.add("overloaded")
        self._log(
            {
                "event": "explainer_fallback",
                "request_id": request_id,
                "reason": "overload",
                "queue_depth": self.explain_queue_depth,
                "ts": round(self.clock(), 3),
            }
        )
        return {
            "status": "fallback",
            "id": None,
            "explanation_source": "template",
            "reason": "overload",
        }

    def _explain_jobs_health(self) -> dict[str, Any]:
        """The explainer job queue and store. ``overloaded`` and ``expired`` count this process;
        ``llm`` holds the explainer process's own counts (completed, failed, expired)."""
        return {
            "mode": "llm_process" if self.jobs is not None else "in_process",
            "queue_depth": self.explain_queue_depth,
            "ttl_s": self.explain_ttl,
            "store_max": self.explanations_kept,
            "overloaded": self.counters.get("overloaded"),
            "expired": self.counters.get("expired"),
            "llm": self.store.get("_jobs") if self.store is not None else None,
        }

    def analyse(self, text: str, lang: str = "auto") -> tuple[int, dict[str, Any]]:
        """Returns (HTTP status, body). Serialised by the lock."""
        request_id = uuid.uuid4().hex[:12]
        with self.lock:
            start = self.device_fn()
            refusal = self._cooling_refusal(request_id, start)
            if refusal is not None:
                return refusal
            energy = self.energy_factory()
            energy.start()
            t0 = time.perf_counter()
            try:
                result = self.analyzer.analyse(text, lang=lang)
            finally:
                energy_doc = energy.stop()
            wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            end = self.device_fn()
            self._count_messages(1)
            body = {
                "request_id": request_id,
                "input_chars": len(text),
                **result,
                "device": {
                    "start": start,
                    "end": end,
                    "energy": energy_doc,
                    "rss_mb": rss_mb(),
                    "rss_peak_mb": rss_mb("VmHWM"),
                },
                "wall_ms": wall_ms,
            }
            self._log(
                {
                    "event": "analyse",
                    "request_id": request_id,
                    "ts": round(self.clock(), 3),
                    "risk": result["decision"]["risk"],
                    "action": result["decision"]["action"],
                    "mode": result["advisory"]["mode"],
                    "path": result["advisory"].get("path"),
                    "detector_only": result["decision"].get("detector_only"),
                    "lang_detected": result["decision"].get("lang_detected"),
                    "lang": result["advisory"].get("lang"),
                    "total_ms": result["timings_ms"]["total"],
                    "llm_total_ms": result["timings_ms"]["llm_total"],
                    "llm_cached": result["timings_ms"].get("llm_cached", 0),
                    "temp_start_c": start.get("temp_c"),
                    "temp_end_c": end.get("temp_c"),
                    "energy_j": energy_doc["energy_j"],
                    "rss_mb": body["device"]["rss_mb"],
                }
            )
            return 200, body

    # ------------------------------------------------------- verdict-first path
    def _verdict_device(self) -> dict[str, Any]:
        """The device snapshot for the verdict path: a cached temperature (see __init__)."""
        now = time.monotonic()
        with self._device_lock:
            if self._device_cache is None or now - self._device_cache[0] >= self.device_ttl:
                self._device_cache = (now, self.verdict_device_fn())
            return dict(self._device_cache[1])

    def analyse_verdict(self, text: str, lang: str = "auto") -> tuple[int, dict[str, Any]]:
        """POST /analyze in verdict-first mode. The verdict is returned without the LLM. The
        explainer text is in the answer when a table covers the message, and otherwise comes from
        a background job (``explanation.status`` is "pending", read it at GET /explanation/<id>).
        The verdict path takes no lock; only the explainer does."""
        request_id = uuid.uuid4().hex[:12]
        start = self._verdict_device()
        t0 = time.perf_counter()
        result, pending = self.analyzer.analyse_verdict(text, lang=lang)
        self._count_messages(1)
        body = self._verdict_item(request_id, len(text), result, pending, start, t0)
        return 200, body

    def analyse_batch(self, texts: list[str], lang: str = "auto") -> tuple[int, dict[str, Any]]:
        """POST /analyze_batch: N messages in one detector call. Each item is a verdict-first
        result (see analyse_verdict) and equals the single-message result."""
        request_id = uuid.uuid4().hex[:12]
        start = self._verdict_device()
        t0 = time.perf_counter()
        items = []
        for c0 in range(0, len(texts), BATCH_CHUNK):
            chunk = texts[c0 : c0 + BATCH_CHUNK]
            pairs = self.analyzer.analyse_verdict_batch(chunk, lang=lang)
            for j, (text, (result, pending)) in enumerate(zip(chunk, pairs, strict=True)):
                i = c0 + j
                item = self._verdict_item(
                    f"{request_id}-{i}",
                    len(text),
                    result,
                    pending,
                    start,
                    None,
                    log=False,
                    include_device=False,
                )
                item["index"] = i
                items.append(item)
        wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        self.counters.add("batch_requests_total")
        self.counters.add("messages_total", len(texts))
        self._log(
            {
                "event": "analyse_batch",
                "request_id": request_id,
                "ts": round(self.clock(), 3),
                "count": len(texts),
                "wall_ms": wall_ms,
                "temp_start_c": start.get("temp_c"),
            }
        )
        return 200, {
            "request_id": request_id,
            "count": len(items),
            "items": items,
            "wall_ms": wall_ms,
            "device": {"start": start, "rss_mb": rss_mb(), "rss_peak_mb": rss_mb("VmHWM")},
        }

    def _verdict_item(
        self,
        request_id: str,
        input_chars: int,
        result: dict[str, Any],
        pending: Any,
        start: dict[str, Any] | None,
        t0: float | None,
        *,
        log: bool = True,
        include_device: bool = True,
    ) -> dict[str, Any]:
        explanation = self._schedule_explanation(request_id, pending, start)
        wall_ms = None if t0 is None else round((time.perf_counter() - t0) * 1000.0, 3)
        body = {
            "request_id": request_id,
            "input_chars": input_chars,
            **result,
            "explanation": explanation,
            "wall_ms": wall_ms,
        }
        if include_device and start is not None:
            body["device"] = {"start": start, "rss_mb": rss_mb(), "rss_peak_mb": rss_mb("VmHWM")}
        if not log:
            return body
        self._log(
            {
                "event": "analyse_verdict",
                "request_id": request_id,
                "ts": round(self.clock(), 3),
                "risk": result["decision"]["risk"],
                "action": result["decision"]["action"],
                "mode": result["advisory"]["mode"],
                "path": result["advisory"].get("path"),
                "detector_only": result["decision"].get("detector_only"),
                "lang": result["advisory"].get("lang"),
                "total_ms": result["timings_ms"]["total"],
                "explanation": explanation["status"],
                "wall_ms": wall_ms,
            }
        )
        return body

    def _schedule_explanation(
        self, request_id: str, pending: Any, start: dict[str, Any] | None
    ) -> dict[str, Any]:
        """ready when the answer is final; pending (with an id) when the explainer job is queued;
        fallback when the job queue is full (the template answers, with the reason); refused_cooling
        when the SoC is at the limit (the explainer does not run). The verdict is the same in every
        case: it is computed before this is called."""
        if pending is None:
            return {"status": "ready", "id": None}
        temp = (start or {}).get("temp_c")
        if temp is not None and temp >= self.temp_limit:
            self.counters.add("refused_cooling")
            self._log(
                {
                    "event": "refused_cooling",
                    "request_id": request_id,
                    "temp_c": temp,
                    "limit_c": self.temp_limit,
                    "ts": round(self.clock(), 3),
                }
            )
            return {"status": "refused_cooling", "id": None}
        eid = uuid.uuid4().hex[:16]
        now = time.time()
        if self.jobs is not None:
            # The explainer process holds the LLM: it recomputes the message and writes the entry.
            # The queue is bounded: a put beyond its depth is refused, and the entry is withdrawn.
            self.store[eid] = {"status": "pending", "queued_at": now}
            try:
                self.jobs.put_nowait((eid, pending.text, pending.lang_route))
            except queue.Full:
                self.store.pop(eid, None)
                return self._overloaded(request_id)
            return {
                "status": "pending",
                "id": eid,
                "url": f"/explanation/{eid}",
                "explanation_source": "llm",
            }
        with self._explanations_lock:
            # One explainer thread: the running job and the waiting ones are the backlog.
            self._explain_futures = [f for f in self._explain_futures if not f.done()]
            if len(self._explain_futures) >= self.explain_queue_depth + 1:
                return self._overloaded(request_id)
            self._explanations[eid] = {"status": "pending", "queued_at": now}
            while len(self._explanations) > self.explanations_kept:
                self._explanations.popitem(last=False)
            self._explain_futures.append(
                self._explain_pool.submit(self._run_explanation, eid, pending, now)
            )
        return {
            "status": "pending",
            "id": eid,
            "url": f"/explanation/{eid}",
            "explanation_source": "llm",
        }

    def _run_explanation(self, eid: str, pending: Any, queued: float) -> None:
        """The background explainer job. It holds self.lock, so the LLM is never used by two
        requests at once; the verdict path does not wait for it. A job that waited longer than
        the TTL is not run: its entry is expired and the template answer stands."""
        if time.time() - queued > self.explain_ttl:
            entry: dict[str, Any] = {
                "status": "expired",
                "explanation_source": "template",
                "reason": "expired",
            }
            self.counters.add("expired")
        else:
            try:
                with self.lock:
                    t0 = time.perf_counter()
                    full = self.analyzer.explain_pending(pending)
                    wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                entry = {
                    "status": "ready",
                    "explanation_source": "llm",
                    "advisory": full["advisory"],
                    "verifier": full["verifier"],
                    "llm_calls": full["llm_calls"],
                    "timings_ms": full["timings_ms"],
                    "wall_ms": wall_ms,
                }
            except Exception as exc:  # noqa: BLE001 - keep serving; the caller sees status error
                self._log({"event": "error", "explanation_id": eid, "error": repr(exc)})
                entry = {"status": "error", "detail": str(exc)}
        entry.update(queued_at=queued, finished_at=time.time())
        with self._explanations_lock:
            if eid in self._explanations:
                self._explanations[eid] = entry

    def _stale(self, entry: dict[str, Any]) -> bool:
        """A pending job queued more than the TTL ago, or a finished one older than the TTL."""
        now = time.time()
        if entry.get("status") == "pending":
            queued = entry.get("queued_at")
            return queued is not None and now - float(queued) > self.explain_ttl
        finished = entry.get("finished_at")
        return finished is not None and now - float(finished) > self.explain_ttl

    def _view(self, eid: str, entry: dict[str, Any]) -> dict[str, Any]:
        """What GET /explanation/<id> says about an entry. A stale entry reads as expired: the
        verdict was answered with the template, and that stays the explanation."""
        if self._stale(entry):
            return {
                "id": eid,
                "status": "expired",
                "explanation_source": "template",
                "reason": "expired",
                "queued_at": entry.get("queued_at"),
            }
        return {"id": eid, **entry}

    def explanation(self, eid: str, wait: float = 0.0) -> tuple[int, dict[str, Any]]:
        """GET /explanation/<id>. ``wait`` (seconds, query ``?wait=``) holds the answer until the
        job has finished, for benchmarks; the default answers at once."""
        if not eid or eid.startswith("_"):
            return 404, {"error": "unknown or expired explanation id", "id": eid}
        clamped_wait = max(0.0, min(wait, 900.0)) if math.isfinite(wait) else 0.0
        deadline = time.monotonic() + clamped_wait
        if self.store is not None:
            entry = self.store.get(eid)
            while (
                entry is not None
                and entry.get("status") == "pending"
                and not self._stale(entry)
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
                entry = self.store.get(eid)
            if entry is None:
                return 404, {"error": "unknown or expired explanation id", "id": eid}
            return 200, self._view(eid, dict(entry))
        with self._explanations_lock:
            raw_entry = self._explanations.get(eid)
            entry = dict(raw_entry) if raw_entry is not None else None
        while (
            entry is not None
            and entry.get("status") == "pending"
            and not self._stale(entry)
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
            with self._explanations_lock:
                raw_entry = self._explanations.get(eid)
                entry = dict(raw_entry) if raw_entry is not None else None
        if entry is None:
            return 404, {"error": "unknown or expired explanation id", "id": eid}
        return 200, self._view(eid, entry)

    def wait_explanations(self) -> None:
        """Block until every queued explainer job has finished (tests and benchmarks)."""
        cf.wait(list(self._explain_futures))

    def _cooling_refusal(self, request_id: str, start: dict[str, Any]) -> tuple[int, dict] | None:
        """503 "cooling" when the SoC is at or above the limit (the caller holds the lock)."""
        temp = start.get("temp_c")
        if temp is not None and temp >= self.temp_limit:
            self.counters.add("refused_cooling")
            self._log(
                {
                    "event": "refused_cooling",
                    "request_id": request_id,
                    "temp_c": temp,
                    "limit_c": self.temp_limit,
                    "ts": round(self.clock(), 3),
                }
            )
            return 503, {
                "error": "cooling",
                "request_id": request_id,
                "temp_c": temp,
                "limit_c": self.temp_limit,
                "detail": "SoC at or above the limit; explainer not run",
            }
        return None

    def analyse_url(self, url: str) -> tuple[int, dict[str, Any]]:
        """POST /analyze_url. Same lock and 503 cooling refusal as analyse()."""
        request_id = uuid.uuid4().hex[:12]
        if self.url_check is None:
            return 503, {"error": "url check not loaded", "request_id": request_id}
        with self.lock:
            start = self.device_fn()
            refusal = self._cooling_refusal(request_id, start)
            if refusal is not None:
                return refusal
            energy = self.energy_factory()
            energy.start()
            t0 = time.perf_counter()
            try:
                result = self.url_check(url)
            finally:
                energy_doc = energy.stop()
            wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            end = self.device_fn()
            body = {
                "request_id": request_id,
                "input_chars": len(url),
                **result,
                "device": {
                    "start": start,
                    "end": end,
                    "energy": energy_doc,
                    "rss_mb": rss_mb(),
                    "rss_peak_mb": rss_mb("VmHWM"),
                },
                "wall_ms": wall_ms,
            }
            self._log(
                {
                    "event": "analyse_url",
                    "request_id": request_id,
                    "ts": round(self.clock(), 3),
                    "verdict": result["verdict"],
                    "risk": result["risk"],
                    "action": result["action"],
                    "url_fired": result["url_model"]["fired"],
                    "total_ms": result["timings_ms"]["total"],
                    "wall_ms": wall_ms,
                    "temp_start_c": start.get("temp_c"),
                    "temp_end_c": end.get("temp_c"),
                    "energy_j": energy_doc["energy_j"],
                }
            )
            return 200, body

    def analyse_flows(self, raw: bytes) -> tuple[int, dict[str, Any]]:
        """Network path: a pcap or JSON flows body. Same 503 cooling refusal as analyse()."""
        request_id = uuid.uuid4().hex[:12]
        if self.nids is None:
            return 503, {"error": "network model not loaded", "request_id": request_id}
        with self.lock:
            start = self.device_fn()
            refusal = self._cooling_refusal(request_id, start)
            if refusal is not None:
                return refusal
            energy = self.energy_factory()
            energy.start()
            t0 = time.perf_counter()
            try:
                result = self.nids.analyse_body(raw)
            finally:
                energy_doc = energy.stop()
            wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            end = self.device_fn()
            body = {
                "request_id": request_id,
                "input_bytes": len(raw),
                **result,
                "device": {
                    "start": start,
                    "end": end,
                    "energy": energy_doc,
                    "rss_mb": rss_mb(),
                    "rss_peak_mb": rss_mb("VmHWM"),
                },
                "wall_ms": wall_ms,
            }
            self._log(
                {
                    "event": "analyse_flows",
                    "request_id": request_id,
                    "ts": round(self.clock(), 3),
                    "flow_source": result["flow_source"],
                    "flows": result["flows_processed"],
                    "alerts": result["decision"]["alerts"],
                    "risk": result["decision"]["risk"],
                    "total_ms": result["timings_ms"]["total"],
                    "wall_ms": wall_ms,
                    "temp_start_c": start.get("temp_c"),
                    "temp_end_c": end.get("temp_c"),
                    "energy_j": energy_doc["energy_j"],
                    "rss_mb": body["device"]["rss_mb"],
                }
            )
            return 200, body


def tcp_nodelay(sock: socket.socket) -> None:
    """Turn off Nagle on a TCP connection. A Unix socket has no such option; that is ignored."""
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass


ENDPOINTS = ("/", "/health", "/analyze", "/analyze_batch", "/analyze_url", "/analyze_flows")


def _endpoint(route: str) -> str:
    """The metrics name of a route. Unknown paths share one name, so the counts stay bounded."""
    if route in ENDPOINTS:
        return route
    if route.startswith("/explanation/"):
        return "/explanation"
    return "other"


def make_handler(service: Service, index_html: bytes, limits: RequestLimits | None = None):
    """The HTTP handler of ``service``. A POST body is checked before it is read: Content-Length
    must be present (411), a non-negative integer (400) and within ``limits.max_body_bytes`` (413).
    The body must then arrive within ``limits.read_timeout_s`` (408). Only then is it parsed (400)
    and its fields validated (400)."""
    limits = limits or RequestLimits()

    class Handler(BaseHTTPRequestHandler):
        server_version = "rakshak-edge/1"
        # Keep-alive: one connection serves many requests (HTTP/1.0 closed it after every reply,
        # and each request paid a new connect, a new relay thread and a new Unix connect).
        protocol_version = "HTTP/1.1"
        # Buffered writer: headers and body leave in one write, flushed after the request. Two
        # unbuffered writes let Nagle hold the body until the client's delayed ACK (about 40 ms).
        wbufsize = -1
        # The socket timeout for reads on this connection: an idle keep-alive connection that
        # sends nothing for this long is closed. A body has its own deadline (see _read_body).
        timeout = limits.read_timeout_s

        def setup(self):
            super().setup()
            tcp_nodelay(self.connection)

        def log_message(self, fmt, *args):  # keep stderr for the JSON lines only
            return

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, doc: dict[str, Any]) -> None:
            self._send(
                status,
                json.dumps(doc, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _reject(self, status: int, doc: dict[str, Any]) -> None:
            """An error answer that ends the connection: the unread rest of the body is dropped."""
            self.close_connection = True
            self._json(status, doc)

        def _content_length(self, *, empty_ok: bool) -> int | None:
            """The Content-Length, checked before any body byte is read. Answers 411 (missing, or
            zero when ``empty_ok`` is False), 400 (not a non-negative integer) or 413 (over the
            cap), and returns None after answering."""
            value = self.headers.get("Content-Length")
            if value is None:
                self._reject(411, {"error": "Content-Length is required"})
                return None
            try:
                length = int(value.strip())
            except ValueError:
                length = -1
            if length < 0:
                self._reject(400, {"error": "Content-Length must be a non-negative integer"})
                return None
            if length == 0 and not empty_ok:
                self._reject(411, {"error": "Content-Length is required (pcap or JSON body)"})
                return None
            if length > limits.max_body_bytes:
                self._reject(413, {"error": f"body is larger than {limits.max_body_bytes} bytes"})
                return None
            return length

        def _read_body(self, length: int) -> bytes | None:
            """Reads ``length`` bytes. The whole body must arrive within ``read_timeout_s``: each
            read waits no longer than what is left of that deadline, so a slow sender cannot hold
            the connection by dripping bytes. 408 on the deadline, 400 when the client closes
            early. Returns None after answering."""
            deadline = time.monotonic() + limits.read_timeout_s
            out = bytearray()
            try:
                while len(out) < length:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TimeoutError("body deadline passed")
                    self.connection.settimeout(left)
                    chunk = self.rfile.read1(min(65536, length - len(out)))
                    if not chunk:
                        break
                    out += chunk
            except TimeoutError:  # socket.timeout is the same class since Python 3.10
                self._reject(
                    408, {"error": f"request body not received within {limits.read_timeout_s:g} s"}
                )
                return None
            finally:
                try:
                    self.connection.settimeout(self.timeout)
                except OSError:
                    pass
            if len(out) != length:
                self._reject(400, {"error": "body is shorter than Content-Length"})
                return None
            return bytes(out)

        def _body(self, *, empty_ok: bool) -> bytes | None:
            """The checked body of a POST, or None after the error answer has been sent."""
            length = self._content_length(empty_ok=empty_ok)
            if length is None:
                return None
            return self._read_body(length)

        def _parsed(self, raw: bytes) -> tuple[bool, Any]:
            """(True, the JSON document), or (False, None) after answering 400."""
            try:
                return True, json.loads(raw.decode("utf-8") or "{}")
            except (ValueError, UnicodeDecodeError, RecursionError):
                self._json(400, {"error": "body must be JSON"})
                return False, None

        def do_GET(self):  # noqa: N802 - stdlib name
            path = self.path.split("?", 1)[0]
            service.count_http(_endpoint(path))
            if path == "/":
                self._send(200, index_html, "text/html; charset=utf-8")
            elif path == "/health":
                self._json(200, service.health())
            elif path.startswith("/explanation/"):
                wait = 0.0
                query = self.path.partition("?")[2]
                for part in query.split("&"):
                    if part.startswith("wait="):
                        try:
                            wait = float(part[len("wait=") :])
                        except ValueError:
                            wait = 0.0
                status, doc = service.explanation(path[len("/explanation/") :], wait)
                self._json(status, doc)
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802 - stdlib name
            route = self.path.split("?", 1)[0]
            service.count_http(_endpoint(route))
            if route == "/analyze_flows":
                self._analyse_flows()
            elif route == "/analyze_url":
                self._analyse_url()
            elif route == "/analyze_batch":
                self._analyse_batch()
            elif route == "/analyze":
                self._analyse_sms()
            elif self._body(empty_ok=True) is not None:  # read it, so the connection stays in step
                self._json(404, {"error": "not found"})

        def _analyse_sms(self) -> None:
            raw = self._body(empty_ok=True)
            if raw is None:
                return
            ok, doc = self._parsed(raw)
            if not ok:
                return
            sms = doc.get("sms") if isinstance(doc, dict) else None
            if not isinstance(sms, str) or not sms.strip():
                self._json(400, {"error": "field 'sms' must be a non-empty string"})
                return
            if len(sms) > limits.max_sms_chars:
                self._json(400, {"error": f"sms is longer than {limits.max_sms_chars} characters"})
                return
            lang = doc.get("lang", "auto")
            if lang not in LANG_REQUESTS:
                self._json(400, {"error": f"field 'lang' must be one of {list(LANG_REQUESTS)}"})
                return
            try:
                if service.response_mode == "verdict_first":
                    status, body = service.analyse_verdict(sms, lang=lang)
                else:
                    status, body = service.analyse(sms, lang=lang)
            except Exception as exc:  # noqa: BLE001 - report, keep serving
                service._log(
                    {"event": "error", "error": repr(exc), "trace": traceback.format_exc(limit=6)}
                )
                self._json(500, {"error": "analysis failed", "detail": str(exc)})
                return
            self._json(status, body)

        def _analyse_batch(self) -> None:
            raw = self._body(empty_ok=True)
            if raw is None:
                return
            ok, doc = self._parsed(raw)
            if not ok:
                return
            texts = doc.get("sms") if isinstance(doc, dict) else None
            if (
                not isinstance(texts, list)
                or not 1 <= len(texts) <= limits.max_batch
                or not all(isinstance(t, str) and t.strip() for t in texts)
            ):
                self._json(
                    400,
                    {
                        "error": "field 'sms' must be a list of 1 to "
                        f"{limits.max_batch} non-empty strings"
                    },
                )
                return
            if any(len(t) > limits.max_sms_chars for t in texts):
                self._json(400, {"error": f"sms is longer than {limits.max_sms_chars} characters"})
                return
            lang = doc.get("lang", "auto")
            if lang not in LANG_REQUESTS:
                self._json(400, {"error": f"field 'lang' must be one of {list(LANG_REQUESTS)}"})
                return
            try:
                status, body = service.analyse_batch(texts, lang=lang)
            except Exception as exc:  # noqa: BLE001 - report, keep serving
                service._log(
                    {"event": "error", "error": repr(exc), "trace": traceback.format_exc(limit=6)}
                )
                self._json(500, {"error": "batch analysis failed", "detail": str(exc)})
                return
            self._json(status, body)

        def _analyse_url(self) -> None:
            raw = self._body(empty_ok=True)
            if raw is None:
                return
            ok, doc = self._parsed(raw)
            if not ok:
                return
            url = doc.get("url") if isinstance(doc, dict) else None
            if not isinstance(url, str) or not url.strip():
                self._json(400, {"error": "field 'url' must be a non-empty string"})
                return
            if len(url) > limits.max_url_chars:
                self._json(400, {"error": f"url is longer than {limits.max_url_chars} characters"})
                return
            try:
                status, body = service.analyse_url(url)
            except Exception as exc:  # noqa: BLE001 - report, keep serving
                service._log(
                    {"event": "error", "error": repr(exc), "trace": traceback.format_exc(limit=6)}
                )
                self._json(500, {"error": "url analysis failed", "detail": str(exc)})
                return
            self._json(status, body)

        def _analyse_flows(self) -> None:
            raw = self._body(empty_ok=False)
            if raw is None:
                return
            try:
                status, body = service.analyse_flows(raw)
            except FlowRequestError as exc:
                self._json(400, {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001 - report, keep serving
                service._log(
                    {"event": "error", "error": repr(exc), "trace": traceback.format_exc(limit=6)}
                )
                self._json(500, {"error": "flow analysis failed", "detail": str(exc)})
                return
            self._json(status, body)

    return Handler


class ReusePortHTTPServer(ThreadingHTTPServer):
    """A TCP server whose port several worker processes can share (SO_REUSEPORT): the kernel
    spreads new connections across the processes that listen on it."""

    allow_reuse_port = True
    daemon_threads = True


def serve(
    service: Service,
    host: str,
    port: int,
    index_html: bytes,
    *,
    reuse_port: bool = False,
    limits: RequestLimits | None = None,
) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise SystemExit("refusing to bind a non-loopback address; the service is local only")
    cls = ReusePortHTTPServer if reuse_port else ThreadingHTTPServer
    server = cls((host, port), make_handler(service, index_html, limits))
    server.daemon_threads = True
    return server


class ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def serve_unix(
    service: Service, path: str, index_html: bytes, limits: RequestLimits | None = None
) -> ThreadingUnixServer:
    """The same HTTP API on a Unix socket. Unix sockets are files, so they work across network
    namespaces: the eval client and socket_bridge.py reach the service this way."""
    if os.path.exists(path):
        os.unlink(path)
    server = ThreadingUnixServer(path, make_handler(service, index_html, limits))
    os.chmod(path, 0o600)
    return server


def _signal_ready(fifo: str | None) -> None:
    """Tell start_service.sh the server is listening. A blocking open runs on a daemon thread, so
    nothing waits if no reader exists (for example when started by hand without a FIFO)."""
    if not fifo:
        return

    def writer() -> None:
        with open(fifo, "w", encoding="utf-8") as handle:
            handle.write("READY\n")

    threading.Thread(target=writer, name="ready-fifo", daemon=True).start()


def load_nids(root: Path):
    """The network student (INT8 ONNX, cse_cic_ids2018). Checks the pin and the model contract."""
    from rbench.registry import load_registry, model_state

    from rakshak.nids.pi_monitor import OnnxScorer

    reg = load_registry(root / "deploy" / "pi" / "bench" / "registry.toml")
    model = reg.model(NET_MODEL_ID)
    state = model_state(root, model, hash_it=True)
    if state.get("status") != "ready":
        raise SystemExit(f"model {model.id} is not ready: {state}")
    t0 = time.perf_counter()
    scorer = OnnxScorer(str(root / NIDS_ONNX_PATH))
    info = {"status": state.get("status"), "load_s": round(time.perf_counter() - t0, 3)}
    return scorer, info


def load_url_model(root: Path):
    """The URL student (url_student) through the same registry check as the SMS model.

    Returns (scorer or None, threshold or None, info). The path is OFF unless the registry has a
    threshold for url_student and the model and its side scaler are ready (every pinned file
    hashed). Off means no URL evidence is ever added and /analyze is unchanged. The reason is
    kept in info and logged at startup.
    """
    from rbench.registry import load_registry, model_state

    from rakshak.fusion.url_expert import FrozenURLExpert

    reg = load_registry(root / "deploy" / "pi" / "bench" / "registry.toml")
    model = reg.model(URL_MODEL_ID)
    info: dict[str, Any] = {"enabled": False, "threshold": model.threshold}
    if not model.enabled:
        reason = "registry entry is disabled"
    elif model.threshold is None:
        reason = "no threshold for url_student in registry.toml; URL path disabled"
    else:
        state = model_state(root, model, hash_it=True)
        info["status"] = state.get("status")
        if state.get("status") != "ready":
            reason = f"model not ready: {state.get('status')}"
        else:
            t0 = time.perf_counter()
            scorer = FrozenURLExpert((root / model.path).parent)
            info["enabled"] = True
            info["load_s"] = round(time.perf_counter() - t0, 3)
            return scorer, float(model.threshold), info
    info["reason"] = reason
    return None, None, info


LLM_RUNNERS = ("runner", "direct", "direct-noreuse")


def load_models(
    root: Path,
    threads: int,
    onnx: Path,
    pte: Path,
    tokenizer: Path,
    llm_runner: str = "runner",
):
    """Load the SMS student (ONNX) and the Llama runner once. Returns (analyzer parts, info).

    llm_runner: runner (TextLLMRunner, reset per prompt), direct (PteDirectRunner with prefix
    reuse) or direct-noreuse (PteDirectRunner, whole prompt every call; for A/B measurement).
    """
    from rbench.registry import load_registry, model_state
    from rbench.runners import make_runner
    from rbench.runners.onnx import SmsRunner

    if llm_runner not in LLM_RUNNERS:
        raise ValueError(f"llm_runner must be one of {LLM_RUNNERS}, got {llm_runner!r}")

    reg = load_registry(root / "deploy" / "pi" / "bench" / "registry.toml")
    sms_model = reg.model(DETECTOR_MODEL_ID)
    llm_model = reg.model(LLM_MODEL_ID)
    info: dict[str, Any] = {}
    for model in (sms_model, llm_model):
        state = model_state(root, model, hash_it=True)
        info[model.id] = {"status": state.get("status")}
        if state.get("status") != "ready":
            raise SystemExit(f"model {model.id} is not ready: {state}")
    t0 = time.perf_counter()
    detector_runner = SmsRunner()
    detector_runner.load(onnx, threads, {})
    info[DETECTOR_MODEL_ID]["load_s"] = round(time.perf_counter() - t0, 3)
    t1 = time.perf_counter()
    if llm_runner == "runner":
        llm = make_runner("pte_llm")
    elif llm_runner == "direct":
        llm = make_runner("pte_llm_direct")
    else:
        from rbench.runners.pte_direct import PteDirectRunner

        llm = PteDirectRunner(reuse_prefix=False)
    llm.load(pte, threads, {"tokenizer": tokenizer})
    info[LLM_MODEL_ID]["load_s"] = round(time.perf_counter() - t1, 3)
    info[LLM_MODEL_ID]["model_load_ms"] = getattr(llm, "load_ms", None)
    info[LLM_MODEL_ID]["threads_reported"] = getattr(llm, "threads_reported", None)
    info[LLM_MODEL_ID]["runner"] = llm_runner
    return detector_runner, llm, float(sms_model.threshold), info


class _DetectorAdapter:
    """Wraps rbench's SmsRunner as the pipeline's Detector (probability of scam).

    ``batch_onnx`` False (default): a batch is featurised once, then each row runs through ONNX
    on its own. A batched session.run is not bit-identical to single-row runs (float32 accumulation
    order differs by about 1e-7 on the stand-in model), and that can move a score across the
    threshold. True: one session.run for the whole batch (--batch-onnx), for use after the
    real-model flip check has passed.
    """

    def __init__(self, runner, batch_onnx: bool = False) -> None:
        self.runner = runner
        self.batch_onnx = bool(batch_onnx)

    def score(self, text: str) -> float:
        feed = self.runner.make_feed({"text": text})
        prob, _logit = self.runner.score(self.runner.run(feed))
        return float(prob)

    def score_prepared(self, texts: list[str], canons: list, canon_rules: list) -> list[float]:
        """score_many with the canonicalisation and canonical rule evidence already computed for
        each text (the verdict path computes them once per message)."""
        return self.score_many(texts, canons=canons, canon_rules=canon_rules)

    def score_many(
        self,
        texts: list[str],
        *,
        canons: list | None = None,
        canon_rules: list | None = None,
    ) -> list[float]:
        """One featurisation for the whole list. The ONNX runs: one batched run when batch_onnx,
        otherwise one run per row (bit-identical to score)."""
        if not texts:
            return []
        feed = self.runner.make_feed_batch(
            [{"text": t} for t in texts], canons=canons, canon_rules=canon_rules
        )
        if self.batch_onnx:
            return [float(p) for p in self.runner.score_batch(self.runner.run(feed))]
        out = []
        for i in range(len(texts)):
            row = {name: value[i : i + 1] for name, value in feed.items()}
            prob, _logit = self.runner.score(self.runner.run(row))
            out.append(float(prob))
        return out


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def _env_cache_default() -> int:
    raw = os.environ.get("RAKSHAK_EXPLAINER_CACHE", "256")
    try:
        return _non_negative_int(raw)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(f"RAKSHAK_EXPLAINER_CACHE: {exc}") from exc


def _env_llm_runner_default() -> str:
    raw = os.environ.get("RAKSHAK_LLM_RUNNER", "direct")
    if raw not in LLM_RUNNERS:
        raise SystemExit(
            f"RAKSHAK_LLM_RUNNER: expected one of {', '.join(LLM_RUNNERS)}, got {raw!r}"
        )
    return raw


def _resolve_response_mode(flag: str | None) -> tuple[str, str]:
    """The response mode and where it came from: --response-mode, then RAKSHAK_RESPONSE_MODE, then
    the default. The default is sync, as before; only the source is new (health shows it)."""
    if flag is not None:
        return flag, "flag"
    raw = os.environ.get("RAKSHAK_RESPONSE_MODE", "")
    if raw == "":
        return DEFAULT_RESPONSE_MODE, "default"
    if raw not in RESPONSE_MODES:
        raise SystemExit(
            f"RAKSHAK_RESPONSE_MODE: expected one of {', '.join(RESPONSE_MODES)}, got {raw!r}"
        )
    return raw, "env"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default=str(Path.home() / "rakshak-bench"))
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--temp-limit", type=float, default=DEFAULT_TEMP_LIMIT)
    p.add_argument("--log", default=None, help="JSON-lines request log (also written to stderr)")
    p.add_argument(
        "--unix-socket",
        default=None,
        help="also serve the same API on this Unix socket (for socket_bridge.py)",
    )
    p.add_argument("--ready-fifo", default=None, help="written once when listening (start script)")
    p.add_argument(
        "--hinglish-mode",
        choices=HINGLISH_MODES,
        default=os.environ.get("RAKSHAK_HINGLISH_MODE", DEFAULT_HINGLISH_MODE),
        help="template (default): Hinglish advisories from the verifier's Hinglish templates, "
        "no LLM call; llm: the Hinglish slot prompt through Llama. Env RAKSHAK_HINGLISH_MODE.",
    )
    p.add_argument(
        "--detector-only-mode",
        choices=DETECTOR_ONLY_MODES,
        default=os.environ.get("RAKSHAK_DETECTOR_ONLY_MODE", DEFAULT_DETECTOR_ONLY_MODE),
        help="quiet (default): detector-only flags stay LOW/IGNORE; balanced: high-confidence "
        "detector-only flags (>= --verify-threshold) and corroborated URL hits produce "
        "UNCERTAIN/VERIFY_INDEPENDENTLY; verify: every detector-only flag produces "
        "UNCERTAIN/VERIFY_INDEPENDENTLY. Env RAKSHAK_DETECTOR_ONLY_MODE.",
    )
    p.add_argument(
        "--verify-threshold",
        type=float,
        default=float(os.environ.get("RAKSHAK_VERIFY_THRESHOLD", str(DEFAULT_VERIFY_THRESHOLD))),
        help="balanced mode only: student probability at or above which a detector-only flag "
        "escalates to UNCERTAIN / VERIFY_INDEPENDENTLY. Env RAKSHAK_VERIFY_THRESHOLD.",
    )
    p.add_argument(
        "--url-high-threshold",
        type=float,
        default=(
            float(os.environ["RAKSHAK_URL_HIGH_THRESHOLD"])
            if os.environ.get("RAKSHAK_URL_HIGH_THRESHOLD")
            else None
        ),
        help="POST /analyze_url: optional high-confidence threshold for verdict='phishing' "
        "(scores between url threshold and this value return verdict='suspicious', reducing "
        "out-of-domain false alarms). Env RAKSHAK_URL_HIGH_THRESHOLD.",
    )
    p.add_argument(
        "--nids-alert",
        type=float,
        default=NET_ALERT_THRESHOLD,
        help="network: attack score at or above which a flow is ALERT (calibrated)",
    )
    p.add_argument(
        "--nids-uncertain",
        type=float,
        default=NET_UNCERTAIN_THRESHOLD,
        help="network: attack score at or above which a flow is UNCERTAIN (calibrated)",
    )
    p.add_argument(
        "--nids-flow-source",
        choices=FLOW_SOURCES,
        default="auto",
        help="network: pcap to flows by nfstream (auto, when installed) or stdlib",
    )
    p.add_argument(
        "--net-model-mode",
        choices=NET_MODEL_MODES,
        default=os.environ.get("RAKSHAK_NET_MODEL_MODE", NET_MODEL_MODE),
        help="network: quiet (rules decide; a model-only flag is advice to verify) or alert "
        "(the 2026-11-06 behaviour, for comparison). Env RAKSHAK_NET_MODEL_MODE.",
    )
    p.add_argument(
        "--response-mode",
        choices=RESPONSE_MODES,
        default=None,
        help="sync (the default when neither this flag nor RAKSHAK_RESPONSE_MODE is set): "
        "/analyze waits for the explainer; verdict_first: /analyze returns the verdict at once "
        "and the explainer comes from a table or GET /explanation/<id>. The mode in use and its "
        "source are in GET /health (response_mode_config).",
    )
    p.add_argument(
        "--batch-onnx",
        action="store_true",
        default=os.environ.get("RAKSHAK_BATCH_ONNX", "") == "1",
        help="score a /analyze_batch list with one ONNX run (faster; not bit-identical to single "
        "runs, so use only after the real-model flip check). Env RAKSHAK_BATCH_ONNX=1.",
    )
    p.add_argument(
        "--explanation-table",
        default=os.environ.get("RAKSHAK_EXPLANATION_TABLE") or None,
        help="built explanation table (deploy/pi/bench/build_explanation_table.py); absent: every "
        "explainer sentence is a background job. Env RAKSHAK_EXPLANATION_TABLE.",
    )
    p.add_argument(
        "--explainer-candidate",
        choices=list(CANDIDATES),
        default=os.environ.get("RAKSHAK_EXPLAINER_CANDIDATE", BASELINE),
        help="English explainer path: baseline (shipped) or a speed candidate from "
        "docs/EXPLAINER_SPEED_PLAN.md. Not the default unless docs/EXPLAINER_SPEED_RESULTS.md "
        "says the gates passed. Env RAKSHAK_EXPLAINER_CANDIDATE.",
    )
    p.add_argument(
        "--explainer-cap",
        type=_non_negative_int,
        default=None,
        help="per-call token cap for the candidate (tight caps are set from the baseline, see "
        "docs/EXPLAINER_SPEED_RESULTS.md); unset uses the candidate's own cap.",
    )
    p.add_argument(
        "--explainer-cache",
        type=_non_negative_int,
        default=_env_cache_default(),
        help="an LRU of explainer sentences keyed by the exact prompt; 0 turns it off; use 0 "
        "for latency benchmarks. Env RAKSHAK_EXPLAINER_CACHE, default 256.",
    )
    p.add_argument(
        "--llm-runner",
        choices=LLM_RUNNERS,
        default=_env_llm_runner_default(),
        help="explainer runner: direct (own loop on the .pte with prefix reuse; default since "
        "2026-11-11); runner (TextLLMRunner, reset per prompt; the previous path, kept for "
        "comparison); direct-noreuse (own loop, whole prompt each call; for A/B). Env "
        "RAKSHAK_LLM_RUNNER, default direct.",
    )
    p.add_argument(
        "--llm-process",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="verdict_first only: run the explainer in its own process (edge_workers.py), niced "
        "and optionally pinned. --no-llm-process keeps the explainer in the verdict process "
        "(the round-1 layout, for comparison).",
    )
    p.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="verdict_first only: verdict processes sharing the port (SO_REUSEPORT), each with "
        "its own detector; 1 serves in the supervisor process.",
    )
    p.add_argument(
        "--onnx-threads",
        type=_positive_int,
        default=1,
        help="ONNX intra-op threads of each verdict process (verdict_first; default 1).",
    )
    p.add_argument(
        "--device-ttl",
        type=float,
        default=1.0,
        help="verdict_first: seconds between SoC temperature reads on the verdict path (the "
        "cooling check). Default 1.0.",
    )
    p.add_argument(
        "--llm-nice",
        type=_non_negative_int,
        default=10,
        help="verdict_first: nice increment of the explainer process (0 = none). Default 10.",
    )
    p.add_argument(
        "--llm-cpus",
        default=None,
        help="verdict_first: comma-separated CPUs for the explainer process, e.g. 3 "
        "(default: any).",
    )
    p.add_argument(
        "--max-body-bytes",
        type=_positive_int,
        default=MAX_BODY_BYTES,
        help="cap on every POST body (/analyze, /analyze_batch, /analyze_url, /analyze_flows); "
        "larger is 413, a missing Content-Length is 411. Default: the flow cap, 20 MiB.",
    )
    p.add_argument(
        "--read-timeout",
        type=float,
        default=DEFAULT_READ_TIMEOUT_S,
        help="seconds a POST body may take to arrive; slower is 408. Default 10.",
    )
    p.add_argument(
        "--explain-queue-depth",
        type=_positive_int,
        default=EXPLAIN_QUEUE_DEPTH,
        help="verdict_first: explainer jobs that may wait for the LLM. A job beyond this is "
        "refused and answered with the template (reason overload). Default 64.",
    )
    p.add_argument(
        "--explain-ttl",
        type=float,
        default=EXPLAIN_TTL_S,
        help="verdict_first: seconds a job may wait, and a finished explanation may be kept, "
        "before it expires (answered with the template, reason expired). Default 120.",
    )
    p.add_argument(
        "--explanations-kept",
        type=_positive_int,
        default=EXPLANATIONS_KEPT,
        help="finished explanations kept for GET /explanation/<id> (the store's size cap). "
        "Default 4096.",
    )
    args = p.parse_args(argv)
    args.response_mode, args.response_mode_source = _resolve_response_mode(args.response_mode)
    limits = RequestLimits(max_body_bytes=args.max_body_bytes, read_timeout_s=args.read_timeout)
    root = Path(args.root)
    if args.response_mode == "verdict_first" and args.llm_process:
        from edge_workers import serve_verdict

        return serve_verdict(args, root)
    from edge_pipeline import EdgeAnalyzer
    from explainer_cache import CachedGenerator

    detector_runner, llm, threshold, info = load_models(
        root,
        args.threads,
        root / ONNX_PATH,
        root / PTE_PATH,
        root / TOKENIZER_PATH,
        llm_runner=args.llm_runner,
    )
    explainer_cache = (
        CachedGenerator(llm, args.explainer_cache) if args.explainer_cache > 0 else None
    )
    url_scorer, url_threshold, url_info = load_url_model(root)
    info[URL_MODEL_ID] = url_info
    from explanation_table import ExplanationTable

    table = ExplanationTable.load(args.explanation_table) if args.explanation_table else None
    analyzer = EdgeAnalyzer(
        _DetectorAdapter(detector_runner, batch_onnx=args.batch_onnx),
        explainer_cache if explainer_cache is not None else llm,
        threshold,
        detector_only_mode=args.detector_only_mode,
        verify_threshold=args.verify_threshold,
        hinglish_mode=args.hinglish_mode,
        url_scorer=url_scorer,
        url_threshold=url_threshold,
        explanation_table=table,
        explainer_candidate=args.explainer_candidate,
        explainer_cap=args.explainer_cap,
    )
    try:
        nids_scorer, nids_info = load_nids(root)
    except ValueError as exc:
        raise SystemExit(f"network model contract mismatch: {exc}") from exc
    info[NET_MODEL_ID] = nids_info
    nids = NidsAnalyzer(
        nids_scorer,
        alert=args.nids_alert,
        uncertain=args.nids_uncertain,
        scratch=root / "run" / "nids_uploads",
        flow_source=args.nids_flow_source,
        model_mode=args.net_model_mode,
    )
    service = Service(
        analyzer,
        threads=args.threads,
        temp_limit=args.temp_limit,
        log_path=Path(args.log) if args.log else None,
        models=info,
        detector_only_mode=args.detector_only_mode,
        verify_threshold=args.verify_threshold,
        hinglish_mode=args.hinglish_mode,
        nids=nids,
        explainer_cache=explainer_cache,
        llm_runner=args.llm_runner,
        url_check=lambda text: check_url(
            text, url_scorer, url_threshold, high_threshold=args.url_high_threshold
        ),
        response_mode=args.response_mode,
        response_mode_source=args.response_mode_source,
        explain_queue_depth=args.explain_queue_depth,
        explain_ttl=args.explain_ttl,
        explanations_kept=args.explanations_kept,
    )
    index_html = (HERE / "index.html").read_bytes()
    service.network = netns_report()
    server = serve(service, args.host, args.port, index_html, limits=limits)
    unix_server = None
    if args.unix_socket:
        unix_server = serve_unix(service, args.unix_socket, index_html, limits)
        threading.Thread(target=unix_server.serve_forever, name="unix", daemon=True).start()
    service._log(
        {
            "event": "ready",
            "pid": os.getpid(),
            "host": args.host,
            "port": args.port,
            "unix_socket": args.unix_socket,
            "threads": args.threads,
            "models": info,
            "threshold": threshold,
            "detector_only_mode": args.detector_only_mode,
            "hinglish_mode": args.hinglish_mode,
            "explainer_cache": args.explainer_cache,
            "llm_runner": args.llm_runner,
            "temp_limit_c": args.temp_limit,
            "response_mode": args.response_mode,
            "response_mode_source": args.response_mode_source,
            "max_body_bytes": limits.max_body_bytes,
            "read_timeout_s": limits.read_timeout_s,
            "network": service.network,
        }
    )
    _signal_ready(args.ready_fifo)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if unix_server is not None:
            unix_server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
