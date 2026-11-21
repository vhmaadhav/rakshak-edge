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

Stdlib HTTP on 127.0.0.1 only. Requests run one at a time (a lock). Before each request the
SoC temperature is read: at or above ``--temp-limit`` (75 C) the service answers 503 "cooling" and
does not run the explainer. Each request is logged as one JSON line (stderr and ``--log``).

Start and stop: deploy/pi/service/start_service.sh and stop_service.sh (they run the service in
``unshare -rn``, so it has no network). ``--ready-fifo`` is set by start_service.sh only.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import sys
import threading
import time
import traceback
import uuid
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
    DETECTOR_ONLY_MODES,
    URL_EVIDENCE_MODE,
)
from edge_device import EnergyWindow, read_snapshot  # noqa: E402
from rbench.explainer_candidates import BASELINE, CANDIDATES  # noqa: E402
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

DEFAULT_PORT = 8770
DEFAULT_HOST = "127.0.0.1"
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
        hinglish_mode: str = DEFAULT_HINGLISH_MODE,
        device_fn=read_snapshot,
        energy_factory=EnergyWindow,
        clock=time.time,
        nids: NidsAnalyzer | None = None,
        explainer_cache: Any | None = None,
        llm_runner: str = "runner",
        url_check: Any | None = None,
    ) -> None:
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
        self.hinglish_mode = hinglish_mode
        self.device_fn = device_fn
        self.energy_factory = energy_factory
        self.clock = clock
        self.lock = threading.Lock()
        self.started = clock()
        self.network: dict[str, Any] = {}
        self.requests = 0
        self.refused = 0
        self._log_lock = threading.Lock()

    def _log(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if self.log_path is None:
            print(line, file=sys.stderr, flush=True)
            return
        with self._log_lock, open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "pid": os.getpid(),
            "uptime_s": round(self.clock() - self.started, 1),
            "threads": self.threads,
            "temp_limit_c": self.temp_limit,
            "models": self.models,
            "detector_only_mode": self.detector_only_mode,
            "url_model": {
                # Diagnostic only (no per-request field in /analyze). Quiet mode never scores
                # URLs for /analyze; /analyze_url works in every mode.
                "enabled": bool((self.models.get(URL_MODEL_ID) or {}).get("enabled")),
                "threshold": (self.models.get(URL_MODEL_ID) or {}).get("threshold"),
                "evidence_visible": self.detector_only_mode == URL_EVIDENCE_MODE,
            },
            "hinglish_mode": self.hinglish_mode,
            "llm_runner": self.llm_runner,
            "requests": self.requests,
            "refused_cooling": self.refused,
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
            self.requests += 1
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

    def _cooling_refusal(self, request_id: str, start: dict[str, Any]) -> tuple[int, dict] | None:
        """503 "cooling" when the SoC is at or above the limit (the caller holds the lock)."""
        temp = start.get("temp_c")
        if temp is not None and temp >= self.temp_limit:
            self.refused += 1
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
            self.requests += 1
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
            self.requests += 1
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


def make_handler(service: Service, index_html: bytes):
    class Handler(BaseHTTPRequestHandler):
        server_version = "rakshak-edge/1"

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

        def do_GET(self):  # noqa: N802 - stdlib name
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(200, index_html, "text/html; charset=utf-8")
            elif path == "/health":
                self._json(200, service.health())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802 - stdlib name
            route = self.path.split("?", 1)[0]
            if route == "/analyze_flows":
                self._analyse_flows()
                return
            if route == "/analyze_url":
                self._analyse_url()
                return
            if route != "/analyze":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length > 0 else b""
                doc = json.loads(raw.decode("utf-8") or "{}")
            except (ValueError, UnicodeDecodeError):
                self._json(400, {"error": "body must be JSON"})
                return
            sms = doc.get("sms") if isinstance(doc, dict) else None
            if not isinstance(sms, str) or not sms.strip():
                self._json(400, {"error": "field 'sms' must be a non-empty string"})
                return
            if len(sms) > MAX_SMS_CHARS:
                self._json(400, {"error": f"sms is longer than {MAX_SMS_CHARS} characters"})
                return
            lang = doc.get("lang", "auto")
            if lang not in LANG_REQUESTS:
                self._json(400, {"error": f"field 'lang' must be one of {list(LANG_REQUESTS)}"})
                return
            try:
                status, body = service.analyse(sms, lang=lang)
            except Exception as exc:  # noqa: BLE001 - report, keep serving
                service._log(
                    {"event": "error", "error": repr(exc), "trace": traceback.format_exc(limit=6)}
                )
                self._json(500, {"error": "analysis failed", "detail": str(exc)})
                return
            self._json(status, body)

        def _analyse_url(self) -> None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length > 0 else b""
                doc = json.loads(raw.decode("utf-8") or "{}")
            except (ValueError, UnicodeDecodeError):
                self._json(400, {"error": "body must be JSON"})
                return
            url = doc.get("url") if isinstance(doc, dict) else None
            if not isinstance(url, str) or not url.strip():
                self._json(400, {"error": "field 'url' must be a non-empty string"})
                return
            if len(url) > MAX_URL_CHARS:
                self._json(400, {"error": f"url is longer than {MAX_URL_CHARS} characters"})
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
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                self._json(411, {"error": "Content-Length is required (pcap or JSON body)"})
                return
            if length > MAX_BODY_BYTES:
                self._json(413, {"error": f"body is larger than {MAX_BODY_BYTES} bytes"})
                return
            raw = self.rfile.read(length)
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


def serve(service: Service, host: str, port: int, index_html: bytes) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise SystemExit("refusing to bind a non-loopback address; the service is local only")
    server = ThreadingHTTPServer((host, port), make_handler(service, index_html))
    server.daemon_threads = True
    return server


class ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def serve_unix(service: Service, path: str, index_html: bytes) -> ThreadingUnixServer:
    """The same HTTP API on a Unix socket. Unix sockets are files, so they work across network
    namespaces: the eval client and socket_bridge.py reach the service this way."""
    if os.path.exists(path):
        os.unlink(path)
    server = ThreadingUnixServer(path, make_handler(service, index_html))
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
    """Wraps rbench's SmsRunner as the pipeline's Detector (probability of scam)."""

    def __init__(self, runner) -> None:
        self.runner = runner

    def score(self, text: str) -> float:
        feed = self.runner.make_feed({"text": text})
        prob, _logit = self.runner.score(self.runner.run(feed))
        return float(prob)


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
        help="quiet (default): a detector-only flag gets the no-warning path; "
        "verify: UNCERTAIN / VERIFY_INDEPENDENTLY notice. Env RAKSHAK_DETECTOR_ONLY_MODE.",
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
        "--explainer-candidate",
        choices=list(CANDIDATES),
        default=os.environ.get("RAKSHAK_EXPLAINER_CANDIDATE", BASELINE),
        help="English explainer path: baseline (shipped) or a speed candidate from "
        "docs/EXPLAINER_SPEED_PLAN.md. Not the default unless docs/EXPLAINER_SPEED_RESULTS.md "
        "says the gates passed. Env RAKSHAK_EXPLAINER_CANDIDATE.",
    )
    parser.add_argument(
        "--explainer-cap",
        type=_non_negative_int,
        default=None,
        help="per-call token cap for the candidate (tight caps are set from the baseline, see "
        "docs/EXPLAINER_SPEED_RESULTS.md); unset uses the candidate's own cap.",
    )
    parser.add_argument(
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
    args = p.parse_args(argv)
    root = Path(args.root)
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
    analyzer = EdgeAnalyzer(
        _DetectorAdapter(detector_runner),
        explainer_cache if explainer_cache is not None else llm,
        threshold,
        detector_only_mode=args.detector_only_mode,
        hinglish_mode=args.hinglish_mode,
        url_scorer=url_scorer,
        url_threshold=url_threshold,
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
        hinglish_mode=args.hinglish_mode,
        nids=nids,
        explainer_cache=explainer_cache,
        llm_runner=args.llm_runner,
        url_check=lambda text: check_url(text, url_scorer, url_threshold),
    )
    index_html = (HERE / "index.html").read_bytes()
    service.network = netns_report()
    server = serve(service, args.host, args.port, index_html)
    unix_server = None
    if args.unix_socket:
        unix_server = serve_unix(service, args.unix_socket, index_html)
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
