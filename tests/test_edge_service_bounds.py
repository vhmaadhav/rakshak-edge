"""Bounds and accounting of the edge service: exact request, batch and message counters; input
bounds checked before the body is parsed (411/400/413/408); a bounded explainer job queue and
store with template fallback (reason overload or expired); the explicit response-mode source; a
worker that dies is restarted and counted. The HTTP tests bind 127.0.0.1 and skip if that is not
allowed here. No model is loaded: the analyser is a fake with the same surface."""

from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import queue
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "deploy" / "pi" / "service"
BENCH = REPO / "deploy" / "pi" / "bench"
for p in (SERVICE, REPO / "src", BENCH, REPO / "deploy" / "pi"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import edge_workers  # noqa: E402
import rakshak_edge_service as svc  # noqa: E402


class FakeAnalyzer:
    """The analyser's verdict-first surface: every message gets a verdict; ``pending`` says whether
    its explainer job is still to run (a message the table does not cover, with a score over the
    threshold). ``explain_pending`` is the in-process explainer."""

    def __init__(self, pending: bool = True):
        self.pending = pending

    def analyse_verdict(self, text, lang="auto"):
        result = {
            "decision": {
                "risk": "MEDIUM",
                "action": "WARN",
                "detector_only": False,
                "score": 0.9,
                "scam_state": "scam",
            },
            "timings_ms": {"total": 0.0, "llm_total": 0.0},
            "advisory": {"mode": "template", "path": "pending", "lang": "en", "points": []},
            "evidence": [],
            "remediation": {},
            "verifier": {},
        }
        pending = SimpleNamespace(text=text, lang_route="en") if self.pending else None
        return result, pending

    def analyse_verdict_batch(self, texts, lang="auto"):
        return [self.analyse_verdict(t, lang) for t in texts]

    def explain_pending(self, pending):
        return {
            "advisory": {"mode": "llm", "path": "explainer"},
            "verifier": {},
            "llm_calls": [],
            "timings_ms": {},
        }


def _svc(analyzer=None, *, jobs=None, store=None, depth=64, ttl=120.0, counters=None):
    return svc.Service(
        analyzer or FakeAnalyzer(),
        threads=1,
        temp_limit=75.0,
        log_path=None,
        models={},
        device_fn=lambda: {"temp_c": 40.0},
        response_mode="verdict_first",
        jobs=jobs,
        store=store,
        explain_queue_depth=depth,
        explain_ttl=ttl,
        counters=counters,
    )


@contextlib.contextmanager
def live_server(service, limits=None):
    """A real server on 127.0.0.1 (an ephemeral port). Skips the test when binding is refused."""
    try:
        srv = svc.serve(service, "127.0.0.1", 0, b"<html></html>", limits=limits)
    except OSError as exc:
        pytest.skip(f"cannot bind a loopback socket in this environment: {exc}")
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", srv
    finally:
        srv.shutdown()
        srv.server_close()


def _post(base, route, doc):
    req = urllib.request.Request(
        base + route,
        data=json.dumps(doc).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def _raw(port, head: bytes, body: bytes = b"", *, half_close: bool = False) -> int:
    """Sends a request by hand and returns its status. ``Connection: close`` is in ``head``, so the
    server closes after answering and the reply can be read to the end."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(head + body)
        if half_close:
            sock.shutdown(socket.SHUT_WR)
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    return int(data.split(b" ", 2)[1])


def _head(route, *, length=None, extra=b"", method=b"POST"):
    lines = [
        method + b" " + route.encode() + b" HTTP/1.1",
        b"Host: 127.0.0.1",
        b"Content-Type: application/json",
        b"Connection: close",
    ]
    if length is not None:
        lines.append(b"Content-Length: " + str(length).encode())
    return b"\r\n".join(lines) + b"\r\n" + extra + b"\r\n"


# --- exact counters ---------------------------------------------------------------------------


def test_one_single_and_a_batch_of_six_count_two_requests_one_batch_seven_messages():
    service = _svc(FakeAnalyzer(pending=False))
    with live_server(service) as (base, _):
        status_single, _ = _post(base, "/analyze", {"sms": "hello there"})
        status_batch, body = _post(base, "/analyze_batch", {"sms": [f"msg {i}" for i in range(6)]})
    assert status_single == 200 and status_batch == 200 and body["count"] == 6
    health = service.health()
    assert health["http_requests_total"] == 2
    assert health["batch_requests_total"] == 1
    assert health["messages_total"] == 7
    assert health["http_requests_by_endpoint"] == {"/analyze": 1, "/analyze_batch": 1}


def test_a_batch_of_n_is_n_messages_not_2n():
    service = _svc(FakeAnalyzer(pending=False))
    service.analyse_batch([f"m{i}" for i in range(9)])
    health = service.health()
    assert health["messages_total"] == 9
    assert health["batch_requests_total"] == 1
    assert health["http_requests_total"] == 0  # no HTTP request was made


def test_url_and_unknown_routes_count_as_http_requests_not_messages():
    service = _svc(FakeAnalyzer(pending=False))
    with live_server(service) as (base, _):
        urllib.request.urlopen(base + "/health", timeout=10).read()
        assert _raw(int(base.rsplit(":", 1)[1]), _head("/nope", length=2), b"{}") == 404
    health = service.health()
    assert health["messages_total"] == 0
    assert health["http_requests_by_endpoint"].get("other") == 1


# --- input bounds, checked before the body is parsed ------------------------------------------


@pytest.fixture()
def bounded():
    limits = svc.RequestLimits(
        max_body_bytes=256, read_timeout_s=0.5, max_batch=3, max_sms_chars=20
    )
    service = _svc(FakeAnalyzer(pending=False))
    with live_server(service, limits) as (base, _):
        yield int(base.rsplit(":", 1)[1])


def test_a_post_without_content_length_is_411(bounded):
    assert _raw(bounded, _head("/analyze")) == 411


@pytest.mark.parametrize("value", [b"abc", b"-5", b"1.5"])
def test_an_invalid_content_length_is_400(bounded, value):
    assert _raw(bounded, _head("/analyze", extra=b"Content-Length: " + value + b"\r\n")) == 400


def test_a_body_over_the_cap_is_413_and_is_not_read(bounded):
    # Only the header is sent: the refusal comes from the declared length alone.
    assert _raw(bounded, _head("/analyze_batch", length=257)) == 413


def test_a_body_that_does_not_arrive_in_time_is_408(bounded):
    head = _head("/analyze", length=100)
    assert _raw(bounded, head, b"0123456789") == 408  # 10 of 100 bytes, then silence


def test_a_body_shorter_than_its_content_length_is_400(bounded):
    assert _raw(bounded, _head("/analyze", length=50), b"0123456789", half_close=True) == 400


def test_invalid_json_is_400(bounded):
    body = b"{not json"
    assert _raw(bounded, _head("/analyze", length=len(body)), body) == 400


def test_sms_longer_than_the_cap_is_400_and_a_short_one_is_200(bounded):
    long_body = json.dumps({"sms": "x" * 21}).encode()
    assert _raw(bounded, _head("/analyze", length=len(long_body)), long_body) == 400
    ok_body = json.dumps({"sms": "x" * 20}).encode()
    assert _raw(bounded, _head("/analyze", length=len(ok_body)), ok_body) == 200


def test_a_batch_over_the_item_cap_is_400(bounded):
    body = json.dumps({"sms": ["a", "b", "c", "d"]}).encode()
    assert _raw(bounded, _head("/analyze_batch", length=len(body)), body) == 400


def test_a_batch_with_an_overlong_item_or_no_items_is_400(bounded):
    for doc in ({"sms": ["ok", "y" * 21]}, {"sms": []}):
        body = json.dumps(doc).encode()
        assert _raw(bounded, _head("/analyze_batch", length=len(body)), body) == 400


def test_url_and_flows_routes_need_a_content_length_too(bounded):
    assert _raw(bounded, _head("/analyze_url")) == 411
    assert _raw(bounded, _head("/analyze_flows", length=0)) == 411


def test_the_default_cap_is_the_flow_cap_and_the_server_binds_loopback_only():
    from nids_pipeline import MAX_BODY_BYTES

    assert svc.RequestLimits().max_body_bytes == MAX_BODY_BYTES
    assert svc.DEFAULT_HOST == "127.0.0.1"
    with pytest.raises(SystemExit):
        svc.serve(_svc(FakeAnalyzer(pending=False)), "0.0.0.0", 0, b"")


# --- bounded explainer job queue and store ---------------------------------------------------


def test_a_full_queue_answers_the_template_with_the_reason_and_keeps_the_verdict():
    jobs: queue.Queue = queue.Queue(maxsize=2)
    store: dict = {}
    service = _svc(FakeAnalyzer(), jobs=jobs, store=store, depth=2)
    answers = [service.analyse_verdict(f"msg {i}")[1] for i in range(5)]
    pending = [a for a in answers if a["explanation"]["status"] == "pending"]
    fallback = [a for a in answers if a["explanation"]["status"] == "fallback"]
    assert len(pending) == 2 and len(fallback) == 3
    assert jobs.qsize() == 2, "the queue must stay at its bound"
    assert len([k for k in store if not k.startswith("_")]) == 2, "a refused job leaves no entry"
    for answer in fallback:
        assert answer["explanation"] == {
            "status": "fallback",
            "id": None,
            "explanation_source": "template",
            "reason": "overload",
        }
        assert answer["decision"]["risk"] == "MEDIUM"  # the verdict is always there
        assert answer["advisory"]["mode"] == "template"
    assert service.health()["explain_jobs"]["overloaded"] == 3
    assert service.health()["explain_jobs"]["queue_depth"] == 2


def test_a_slow_explainer_drains_the_queue_and_the_accepted_jobs_finish():
    jobs: queue.Queue = queue.Queue(maxsize=2)
    store: dict = {}
    service = _svc(FakeAnalyzer(), jobs=jobs, store=store, depth=2)
    started, release = threading.Event(), threading.Event()

    class SlowLLM:
        def analyse(self, text, lang="auto"):
            started.set()
            release.wait(timeout=10)  # the fake LLM is slow until the test lets it finish
            return {
                "advisory": {"mode": "llm", "path": "explainer"},
                "verifier": {},
                "llm_calls": [],
                "timings_ms": {},
            }

    first = service.analyse_verdict("m0")[1]["explanation"]
    worker = threading.Thread(
        target=edge_workers._job_loop, args=(SlowLLM(), jobs, store), daemon=True
    )
    worker.start()
    assert started.wait(timeout=10)  # the first job is with the LLM; one slot is free again
    second = service.analyse_verdict("m1")[1]["explanation"]
    third = service.analyse_verdict("m2")[1]["explanation"]
    refused = service.analyse_verdict("m3")[1]["explanation"]
    assert [first["status"], second["status"], third["status"]] == ["pending"] * 3
    assert refused["status"] == "fallback" and refused["reason"] == "overload"
    release.set()
    jobs.put(None)
    worker.join(timeout=10)
    for answer in (first, second, third):
        code, done = service.explanation(answer["id"])
        assert code == 200 and done["status"] == "ready"
        assert done["explanation_source"] == "llm"
    assert store["_jobs"] == {"completed": 3, "failed": 0, "expired": 0}


def test_a_job_that_waited_past_the_ttl_is_expired_and_the_llm_is_not_run():
    store = {"old": {"status": "pending", "queued_at": time.time() - 1000}}
    jobs: queue.Queue = queue.Queue()
    jobs.put(("old", "text", "en"))
    jobs.put(None)

    class MustNotRun:
        def analyse(self, text, lang="auto"):
            raise AssertionError("an expired job must not reach the LLM")

    edge_workers._job_loop(MustNotRun(), jobs, store, ttl=60.0)
    assert store["old"]["status"] == "expired" and store["old"]["reason"] == "expired"
    assert store["old"]["explanation_source"] == "template"
    assert store["_jobs"] == {"completed": 0, "failed": 0, "expired": 1}


def test_a_pending_answer_read_after_the_ttl_is_expired_with_the_template_as_source():
    store = {"x": {"status": "pending", "queued_at": time.time() - 1000}}
    service = _svc(FakeAnalyzer(), jobs=queue.Queue(), store=store, ttl=60.0)
    code, doc = service.explanation("x")
    assert code == 200 and doc["status"] == "expired"
    assert doc["explanation_source"] == "template" and doc["reason"] == "expired"


def test_the_store_is_bounded_by_count_and_by_age():
    now = [1000.0]
    store: dict = {}
    jobs: queue.Queue = queue.Queue()

    class Step:
        def analyse(self, text, lang="auto"):
            if text == "late":
                now[0] = 1500.0  # time passes while this job runs
            return {"advisory": {}, "verifier": {}, "llm_calls": [], "timings_ms": {}}

    jobs.put(("a", "early", "en"))
    jobs.put(("b", "late", "en"))
    jobs.put(None)
    edge_workers._job_loop(Step(), jobs, store, kept=100, ttl=120.0, clock=lambda: now[0])
    assert "a" not in store, "a finished explanation older than the TTL is dropped"
    assert "b" in store


def test_in_process_explainer_refuses_beyond_its_backlog_and_says_so():
    gate = threading.Event()

    class Blocking(FakeAnalyzer):
        def explain_pending(self, pending):
            gate.wait(timeout=10)
            return super().explain_pending(pending)

    service = _svc(Blocking(), depth=1)  # no jobs queue: the explainer runs in this process
    a1 = service.analyse_verdict("m1")[1]["explanation"]
    a2 = service.analyse_verdict("m2")[1]["explanation"]
    a3 = service.analyse_verdict("m3")[1]["explanation"]
    assert a1["status"] == "pending" and a2["status"] == "pending"
    assert a3["status"] == "fallback" and a3["reason"] == "overload"
    gate.set()
    service.wait_explanations()
    code, done = service.explanation(a1["id"])
    assert code == 200 and done["status"] == "ready"
    assert service.health()["explain_jobs"]["mode"] == "in_process"


# --- explicit response mode -------------------------------------------------------------------


def test_the_response_mode_default_is_sync_and_its_source_is_visible(monkeypatch):
    monkeypatch.delenv("RAKSHAK_RESPONSE_MODE", raising=False)
    assert svc.DEFAULT_RESPONSE_MODE == "sync"
    assert svc._resolve_response_mode(None) == ("sync", "default")
    assert svc._resolve_response_mode("verdict_first") == ("verdict_first", "flag")
    monkeypatch.setenv("RAKSHAK_RESPONSE_MODE", "verdict_first")
    assert svc._resolve_response_mode(None) == ("verdict_first", "env")
    monkeypatch.setenv("RAKSHAK_RESPONSE_MODE", "fast")
    with pytest.raises(SystemExit):
        svc._resolve_response_mode(None)


def test_health_shows_the_mode_its_source_and_the_explainer_limits():
    service = svc.Service(
        FakeAnalyzer(pending=False),
        threads=1,
        temp_limit=75.0,
        log_path=None,
        models={},
        device_fn=lambda: {"temp_c": 40.0},
        response_mode="sync",
        response_mode_source="default",
    )
    cfg = service.health()["response_mode_config"]
    assert cfg["mode"] == "sync" and cfg["source"] == "default" and cfg["default"] == "sync"
    assert service.health()["explain_jobs"]["queue_depth"] == svc.EXPLAIN_QUEUE_DEPTH


# --- worker restart ---------------------------------------------------------------------------


def test_a_worker_that_raises_once_is_logged_counted_and_restarted():
    calls: list[int] = []
    finished = threading.Event()

    def target():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("worker lost its model once")
        finished.set()

    events: list[dict] = []
    sup = edge_workers.WorkerSupervisor(
        "probe", lambda: edge_workers.ThreadWorker(target, name="probe"), log=events.append
    ).start()
    sup.worker.join(timeout=10)
    assert sup.poll() is True, "a worker that died with a failure is restarted"
    sup.worker.join(timeout=10)
    assert finished.wait(timeout=10), "the restarted worker does its work"
    assert sup.poll() is False, "a worker that returned normally is not restarted"
    assert sup.restarts == 1 and not sup.failed
    assert [e["event"] for e in events] == ["worker_died", "worker_restarted"]
    assert events[0]["exitcode"] == 1


def test_a_worker_that_keeps_failing_is_marked_failed_after_max_restarts():
    def target():
        raise RuntimeError("always")

    events: list[dict] = []
    sup = edge_workers.WorkerSupervisor(
        "probe",
        lambda: edge_workers.ThreadWorker(target, name="probe"),
        log=events.append,
        max_restarts=2,
    ).start()
    for _ in range(3):  # the first run and two restarts
        sup.worker.join(timeout=10)
        sup.poll()
    assert sup.failed and sup.restarts == 2
    assert events[-1]["event"] == "worker_failed"


def test_the_watch_stops_when_a_worker_fails_for_good():
    def target():
        raise RuntimeError("always")

    failed: list = []
    restarted: list = []
    stop = threading.Event()
    sup = edge_workers.WorkerSupervisor(
        "probe",
        lambda: edge_workers.ThreadWorker(target, name="probe"),
        log=lambda record: None,
        max_restarts=1,
    ).start()
    edge_workers._watch(
        [sup], stop, on_restart=restarted.append, on_failed=failed.append, interval=0.001
    )
    assert stop.is_set() and failed == [sup] and len(restarted) == 1


# --- the bench's telemetry guard --------------------------------------------------------------


def test_encoder_bench_disables_telemetry_before_it_creates_a_session():
    src = (BENCH / "encoder_bench.py").read_text(encoding="utf-8")
    guard = 'if hasattr(ort, "disable_telemetry_events"):'
    assert guard in src and "ort.disable_telemetry_events()" in src
    assert src.index(guard) < src.index("ort.InferenceSession(")


# --- service-wide counters across worker processes --------------------------------------------


def _traffic_worker(counters, rounds):
    """One verdict process's traffic, on its own Service but the shared counters."""
    service = _svc(FakeAnalyzer(pending=False), counters=counters)
    for _ in range(rounds):
        service.count_http("/analyze")
        service.analyse_batch(["a", "b", "c"])
    service.counters.add("overloaded")


def test_worker_processes_report_exact_service_wide_totals():
    ctx = mp.get_context("spawn")
    counters = svc.SharedCounters(ctx=ctx)
    procs = [ctx.Process(target=_traffic_worker, args=(counters, 5)) for _ in range(2)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)
    assert [proc.exitcode for proc in procs] == [0, 0]
    health = _svc(FakeAnalyzer(pending=False), counters=counters).health()
    assert health["http_requests_total"] == 10  # 2 processes x 5 requests
    assert health["requests"] == health["http_requests_total"]  # deprecated alias
    assert health["http_requests_by_endpoint"] == {"/analyze": 10}
    assert health["batch_requests_total"] == 10
    assert health["messages_total"] == 30  # 2 x 5 batches x 3 messages
    assert health["explain_jobs"]["overloaded"] == 2
    assert health["explain_jobs"]["expired"] == 0


def test_batch_cooling_refuses_explainer_jobs_and_shares_refused_counter():
    jobs: queue.Queue = queue.Queue(maxsize=8)
    store: dict = {}
    counters = svc.SharedCounters()
    hot_service = svc.Service(
        FakeAnalyzer(pending=True),
        threads=1,
        temp_limit=75.0,
        log_path=None,
        models={},
        device_fn=lambda: {"temp_c": 81.0},
        response_mode="verdict_first",
        jobs=jobs,
        store=store,
        counters=counters,
    )
    observer = _svc(FakeAnalyzer(pending=False), counters=counters)

    status, body = hot_service.analyse_batch(["m0", "m1", "m2"])
    assert status == 200 and body["count"] == 3
    assert all(
        item["explanation"] == {"status": "refused_cooling", "id": None} for item in body["items"]
    )
    assert all("device" not in item for item in body["items"])
    assert jobs.empty(), "hot batch must not enqueue explainer jobs"
    assert hot_service.refused == 3
    assert observer.health()["refused_cooling"] == 3


def test_explanation_endpoint_blocks_internal_store_keys():
    store = {
        "_llm": {"status": "ready", "pid": 1234},
        "_jobs": {"completed": 1},
        "_workers": {"llm": {"restarts": 0, "failed": False}},
    }
    service = _svc(FakeAnalyzer(), jobs=queue.Queue(), store=store)
    for key in ("", "_llm", "_jobs", "_workers"):
        code, doc = service.explanation(key)
        assert code == 404 and "unknown" in doc["error"]


def test_in_process_explanation_honors_wait_and_handles_non_finite_wait():
    started, release = threading.Event(), threading.Event()

    class BlockingAnalyzer(FakeAnalyzer):
        def explain_pending(self, pending):
            started.set()
            release.wait(timeout=10)
            return super().explain_pending(pending)

    service = _svc(BlockingAnalyzer(pending=True))
    status, verdict = service.analyse_verdict("msg")
    assert status == 200
    eid = verdict["explanation"]["id"]
    assert started.wait(timeout=5)

    # Non-finite wait does not block or raise.
    code, pending_doc = service.explanation(eid, wait=float("nan"))
    assert code == 200 and pending_doc["status"] == "pending"

    def finish_soon():
        time.sleep(0.08)
        release.set()

    threading.Thread(target=finish_soon, daemon=True).start()
    code, done = service.explanation(eid, wait=5.0)
    assert code == 200 and done["status"] == "ready"
    service.wait_explanations()


def test_keep_alive_socket_timeout_is_restored_after_slow_post():
    limits = svc.RequestLimits(max_body_bytes=1024, read_timeout_s=0.4)
    service = _svc(FakeAnalyzer(pending=False))
    with live_server(service, limits) as (base, _):
        port = int(base.rsplit(":", 1)[1])
        body = json.dumps({"sms": "hello"}).encode("utf-8")
        head = (
            b"POST /analyze HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Connection: keep-alive\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
        )
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            # First request uses ~75% of read_timeout_s (leaves ~0.1s on the body deadline).
            sock.sendall(head + body[:3])
            time.sleep(0.30)
            sock.sendall(body[3:])
            resp1 = sock.recv(65536)
            assert resp1.startswith(b"HTTP/1.1 200 ")

            # Pause longer than the ~0.1s residual deadline, but well under read_timeout_s (0.4s).
            time.sleep(0.20)
            head2 = head.replace(b"Connection: keep-alive", b"Connection: close")
            sock.sendall(head2 + body)
            resp2 = sock.recv(65536)
            assert resp2.startswith(b"HTTP/1.1 200 ")


def test_health_reports_policy_hybrid_and_nids_config(tmp_path):
    from nids_pipeline import NidsAnalyzer

    nids = NidsAnalyzer(
        lambda x: [[1.0, 0.0, 0.0, 0.0, 0.0] for _ in x],
        scratch=tmp_path / "uploads",
        flow_source="stdlib",
        dns_min_distinct=3,
        dns_slow_window_ms=300_000,
    )
    analyzer = FakeAnalyzer(pending=False)
    analyzer.tx_verify_threshold = 0.999
    service = svc.Service(
        analyzer,
        threads=1,
        temp_limit=75.0,
        log_path=None,
        models={
            svc.URL_MODEL_ID: {
                "enabled": True,
                "threshold": 0.6308,
                "hybrid": True,
                "domain_intel_version": "url-domain-intel-v1",
            }
        },
        detector_only_mode="balanced",
        verify_threshold=0.9566,
        device_fn=lambda: {"temp_c": 42.0},
        nids=nids,
    )
    health = service.health()
    assert health["detector_only_mode"] == "balanced"
    assert health["verify_threshold"] == pytest.approx(0.9566)
    assert health["tx_verify_threshold"] == pytest.approx(0.999)
    assert health["url_model"]["hybrid"] is True
    assert health["url_model"]["domain_intel_version"] == "url-domain-intel-v1"
    assert health["nids"] == {
        "model_mode": "quiet",
        "dns_min_distinct": 3,
        "dns_slow_window_s": 300,
    }


def test_socket_bridge_connect_timeout_and_connection_cap(tmp_path):
    import socket_bridge

    # Dead/missing socket respects short connect timeout and returns None without hanging.
    t0 = time.monotonic()
    assert socket_bridge._connect(str(tmp_path / "missing.sock"), timeout=0.05) is None
    assert time.monotonic() - t0 < 1.0

    # Semaphore release is guaranteed cleanly even if client fails before upstream connects.
    sem = threading.BoundedSemaphore(1)
    assert sem.acquire(blocking=False)
    s1, s2 = socket.socketpair()
    s1.close()
    s2.close()
    socket_bridge._handle(s1, str(tmp_path / "missing.sock"), sem, connect_timeout=0.05)
    # Semaphore must have been released by _handle's finally block.
    assert sem.acquire(blocking=False)
    sem.release()

    with pytest.raises(ValueError, match="max_connections"):
        socket_bridge.serve("127.0.0.1", 0, str(tmp_path / "missing.sock"), max_connections=0)
