#!/usr/bin/env python3
"""Rakshak edge service: one resident process that loads the SMS student and the Llama .pte once.

    POST /analyze   {"sms": "..."}  -> decision, evidence, advisory, per-stage timings, device state
    GET  /health                    -> models, threads, RSS, device state
    GET  /                          -> a minimal page to paste an SMS and see the stages

Detector-only flags (score >= threshold, no rule evidence) follow ``--detector-only-mode`` (env
RAKSHAK_DETECTOR_ONLY_MODE): "quiet" (default) or "verify" (see docs/EDGE_PIPELINE.md, Fix B).
Hinglish SMS (detect_lang) are answered in Hinglish; ``--hinglish-mode`` (env RAKSHAK_HINGLISH_MODE)
is "template" (verifier templates, no LLM) or "llm" (Hinglish slot prompt). The request may set
``"lang": "auto" | "en" | "hinglish"`` (default auto).

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

from edge_decision import DEFAULT_DETECTOR_ONLY_MODE, DETECTOR_ONLY_MODES  # noqa: E402
from edge_device import EnergyWindow, read_snapshot  # noqa: E402
from edge_pipeline import (  # noqa: E402
    DEFAULT_HINGLISH_MODE,
    HINGLISH_MODES,
    LANG_REQUESTS,
)

DEFAULT_PORT = 8770
DEFAULT_HOST = "127.0.0.1"
EGRESS_PROBE = ("1.1.1.1", 443)
DEFAULT_TEMP_LIMIT = 75.0
MAX_SMS_CHARS = 2000
ONNX_PATH = "models/sms_student/best_student.onnx"
PTE_PATH = "models/llama32_1b_spinquant/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte"
TOKENIZER_PATH = "models/llama32_1b_spinquant/tokenizer.model"
DETECTOR_MODEL_ID = "sms_student"
LLM_MODEL_ID = "llama32_1b_spinquant"


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
    ) -> None:
        self.analyzer = analyzer
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
            "hinglish_mode": self.hinglish_mode,
            "requests": self.requests,
            "refused_cooling": self.refused,
            "rss_mb": rss_mb(),
            "rss_peak_mb": rss_mb("VmHWM"),
            "device": self.device_fn(),
            "network": self.network,
        }

    def analyse(self, text: str, lang: str = "auto") -> tuple[int, dict[str, Any]]:
        """Returns (HTTP status, body). Serialised by the lock."""
        request_id = uuid.uuid4().hex[:12]
        with self.lock:
            start = self.device_fn()
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
            if self.path.split("?", 1)[0] != "/analyze":
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


def load_models(root: Path, threads: int, onnx: Path, pte: Path, tokenizer: Path):
    """Load the SMS student (ONNX) and the Llama runner once. Returns (analyzer parts, info)."""
    from rbench.registry import load_registry, model_state
    from rbench.runners import make_runner
    from rbench.runners.onnx import SmsRunner

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
    llm = make_runner("pte_llm")
    llm.load(pte, threads, {"tokenizer": tokenizer})
    info[LLM_MODEL_ID]["load_s"] = round(time.perf_counter() - t1, 3)
    info[LLM_MODEL_ID]["model_load_ms"] = getattr(llm, "load_ms", None)
    info[LLM_MODEL_ID]["threads_reported"] = getattr(llm, "threads_reported", None)
    return detector_runner, llm, float(sms_model.threshold), info


class _DetectorAdapter:
    """Wraps rbench's SmsRunner as the pipeline's Detector (probability of scam)."""

    def __init__(self, runner) -> None:
        self.runner = runner

    def score(self, text: str) -> float:
        feed = self.runner.make_feed({"text": text})
        prob, _logit = self.runner.score(self.runner.run(feed))
        return float(prob)


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
    args = p.parse_args(argv)
    root = Path(args.root)
    from edge_pipeline import EdgeAnalyzer

    detector_runner, llm, threshold, info = load_models(
        root, args.threads, root / ONNX_PATH, root / PTE_PATH, root / TOKENIZER_PATH
    )
    analyzer = EdgeAnalyzer(
        _DetectorAdapter(detector_runner),
        llm,
        threshold,
        detector_only_mode=args.detector_only_mode,
        hinglish_mode=args.hinglish_mode,
    )
    service = Service(
        analyzer,
        threads=args.threads,
        temp_limit=args.temp_limit,
        log_path=Path(args.log) if args.log else None,
        models=info,
        detector_only_mode=args.detector_only_mode,
        hinglish_mode=args.hinglish_mode,
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
