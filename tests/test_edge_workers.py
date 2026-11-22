"""Round 2 serving: keep-alive HTTP, batch chunking, the explainer job queue, and the precomputed
canonicalisation path (one canonicalise and one rule pass per message). No model is loaded."""

from __future__ import annotations

import http.client
import queue
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "deploy" / "pi" / "service"
BENCH = REPO / "deploy" / "pi" / "bench"
for p in (SERVICE, REPO / "src", BENCH, REPO / "deploy" / "pi"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import edge_pipeline  # noqa: E402
import edge_workers  # noqa: E402
import rakshak_edge_service as svc  # noqa: E402
from edge_pipeline import EdgeAnalyzer  # noqa: E402

from rakshak.canonicalize import canonicalize  # noqa: E402
from rakshak.models import features  # noqa: E402
from rakshak.rules import run_rules  # noqa: E402

THRESHOLD = 0.35127517580986023
SCAM = (
    "URGENT: your account is blocked. Verify KYC at http://sbi-kyc.in/update now or lose access."
)
BENIGN = "Dinner at 8 tonight? Bring the blue folder, thanks."


class ConstDetector:
    """A detector that returns a fixed probability for every text (score only)."""

    def __init__(self, p: float = 0.1):
        self.p = p
        self.calls = 0

    def score(self, text: str) -> float:
        self.calls += 1
        return self.p


class PreparedDetector(ConstDetector):
    """Also accepts precomputed canon and rules, as the SMS adapter does."""

    def score_prepared(self, texts, canons, rules):
        assert len(texts) == len(canons) == len(rules)
        return [self.p for _ in texts]


class StubService:
    """The handler's whole surface, without models."""

    response_mode = "verdict_first"

    def __init__(self):
        self.seen: list[str] = []

    def _log(self, record):
        return

    def health(self):
        return {"status": "ok"}

    def count_http(self, endpoint):
        return

    def explanation(self, eid):
        return 404, {"error": "unknown", "id": eid}

    def analyse_verdict(self, text, lang="auto"):
        self.seen.append(text)
        return 200, {"echo": text}

    def analyse_batch(self, texts, lang="auto"):
        return 200, {"count": len(texts)}


@pytest.fixture()
def server():
    service = StubService()
    srv = svc.serve(service, "127.0.0.1", 0, b"<html></html>")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv, service
    srv.shutdown()
    srv.server_close()


def test_one_connection_serves_many_requests(server):
    """HTTP/1.1 keep-alive: the second and third requests reuse the first connection."""
    srv, service = server
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    import json

    sock_ids = []
    for text in ("one", "two", "three"):
        conn.request(
            "POST", "/analyze", json.dumps({"sms": text}), {"Content-Type": "application/json"}
        )
        resp = conn.getresponse()
        body = json.loads(resp.read())
        assert resp.status == 200 and body["echo"] == text
        assert resp.version == 11  # HTTP/1.1
        sock_ids.append(id(conn.sock))
    assert len(set(sock_ids)) == 1, "the connection was reopened between requests"
    assert service.seen == ["one", "two", "three"]
    conn.close()


def test_unknown_route_with_a_body_keeps_the_connection_in_step(server):
    """A POST that is not answered by a route still reads its body, so the next request parses."""
    import json

    srv, _ = server
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    conn.request("POST", "/nope", "x" * 500, {"Content-Type": "text/plain"})
    assert conn.getresponse().read() is not None
    conn.request(
        "POST", "/analyze", json.dumps({"sms": "after"}), {"Content-Type": "application/json"}
    )
    resp = conn.getresponse()
    assert resp.status == 200 and json.loads(resp.read())["echo"] == "after"
    conn.close()


def test_batch_cap_is_1024_and_the_batch_is_answered_in_chunks():
    assert svc.MAX_BATCH == 1024
    assert svc.BATCH_CHUNK <= 256
    chunks: list[int] = []

    class Recorder:
        def analyse_verdict_batch(self, texts, lang="auto"):
            chunks.append(len(texts))
            return [({"decision": {"risk": "LOW", "action": "IGNORE", "detector_only": True},
                      "timings_ms": {"total": 0.0}, "advisory": {"mode": "template"},
                      "evidence": [], "remediation": {}, "verifier": {}}, None) for _ in texts]

    service = svc.Service(
        Recorder(),
        threads=1,
        temp_limit=75.0,
        log_path=None,
        models={},
        device_fn=lambda: {"temp_c": 40.0},
        response_mode="verdict_first",
        jobs=queue.Queue(),
        store={},
    )
    _status, body = service.analyse_batch([f"msg {i}" for i in range(1000)])
    assert body["count"] == 1000
    assert chunks == [svc.BATCH_CHUNK] * (1000 // svc.BATCH_CHUNK) + [1000 % svc.BATCH_CHUNK]
    assert [item["index"] for item in body["items"]] == list(range(1000))


def test_pending_explanation_goes_to_the_job_queue_and_comes_back_from_the_store():
    jobs: queue.Queue = queue.Queue()
    store: dict = {}
    analyzer = EdgeAnalyzer(ConstDetector(0.1), None, THRESHOLD)  # quiet, no table -> pending
    service = svc.Service(
        analyzer,
        threads=1,
        temp_limit=75.0,
        log_path=None,
        models={},
        device_fn=lambda: {"temp_c": 40.0},
        response_mode="verdict_first",
        jobs=jobs,
        store=store,
    )
    # A scam-like message has rule evidence and a score above threshold: its answer is pending.
    detector = ConstDetector(0.99)
    service.analyzer.detector = detector
    status, body = service.analyse_verdict(SCAM, lang="en")
    assert status == 200 and body["explanation"]["status"] == "pending"
    eid = body["explanation"]["id"]
    assert store[eid]["status"] == "pending" and "queued_at" in store[eid]
    text, lang = jobs.get()[1:]
    assert text == SCAM and lang == "en"

    class FullAnalyzer:
        def analyse(self, text, lang="auto"):
            return {
                "advisory": {"mode": "llm"},
                "verifier": {"final_contract_ok": True},
                "llm_calls": [],
                "timings_ms": {"total": 1.0},
            }

    job_q: queue.Queue = queue.Queue()
    job_q.put((eid, SCAM, "en"))
    job_q.put(None)
    edge_workers._job_loop(FullAnalyzer(), job_q, store)
    code, doc = service.explanation(eid)
    assert code == 200 and doc["status"] == "ready" and doc["advisory"]["mode"] == "llm"


def test_job_loop_keeps_only_the_last_explanations():
    store: dict = {}
    q: queue.Queue = queue.Queue()
    for i in range(5):
        q.put((f"id{i}", "x", "en"))
    q.put(None)

    class Quick:
        def analyse(self, text, lang="auto"):
            return {"advisory": {}, "verifier": {}, "llm_calls": [], "timings_ms": {}}

    edge_workers._job_loop(Quick(), q, store, kept=2)
    # Keys starting with "_" are the job counts, not explanations.
    assert sorted(k for k in store if not k.startswith("_")) == ["id3", "id4"]


def test_featurize_with_precomputed_canon_and_rules_is_identical_and_skips_recompute(monkeypatch):
    texts = [SCAM, BENIGN, "", "Pay ₹500 to 9876543210 via UPI collect now"]
    fresh = features.featurize(texts, buckets=None, raw_side=False)

    canons = [canonicalize(t) for t in texts]
    rules = [run_rules(c) for c in canons]
    calls = {"n": 0}
    real = features.canonicalize

    def counting(text):
        calls["n"] += 1
        return real(text)

    monkeypatch.setattr(features, "canonicalize", counting)
    reused = features.featurize(
        texts, buckets=None, canons=canons, canon_rules=rules, raw_side=False
    )
    assert calls["n"] == 0
    for name in ("raw_bytes", "canon_bytes", "side_canon", "side_raw"):
        assert np.array_equal(getattr(fresh, name), getattr(reused, name)), name


def test_featurize_default_is_unchanged_with_the_raw_side():
    texts = [SCAM, BENIGN]
    default = features.featurize(texts)
    full = features.featurize(texts, raw_side=True)
    assert np.array_equal(default.side_raw, full.side_raw)
    assert default.side_raw.any()


@pytest.mark.parametrize("text", [SCAM, BENIGN, "Your OTP is 4455 do not share"])
def test_verdict_is_the_same_with_and_without_the_precomputed_path(text):
    plain = EdgeAnalyzer(PreparedDetector(0.6), None, THRESHOLD).analyse_verdict(text)[0]
    canon = canonicalize(text)
    prepared = EdgeAnalyzer(PreparedDetector(0.6), None, THRESHOLD).analyse_verdict(
        text, canon=canon, base_rules=run_rules(canon)
    )[0]
    assert plain["decision"] == prepared["decision"]
    assert plain["evidence"] == prepared["evidence"]
    assert plain["remediation"] == prepared["remediation"]


def test_single_message_canonicalises_once_with_a_prepared_detector(monkeypatch):
    calls = {"n": 0}
    real = edge_pipeline.canonicalize

    def counting(text):
        calls["n"] += 1
        return real(text)

    monkeypatch.setattr(edge_pipeline, "canonicalize", counting)
    analyzer = EdgeAnalyzer(PreparedDetector(0.2), None, THRESHOLD)
    analyzer.analyse_verdict(SCAM)
    assert calls["n"] == 1


def test_run_rules_runs_once_per_message_on_the_verdict_path(monkeypatch):
    calls = {"n": 0}
    real = edge_pipeline.run_rules

    def counting(canon):
        calls["n"] += 1
        return real(canon)

    monkeypatch.setattr(edge_pipeline, "run_rules", counting)
    analyzer = EdgeAnalyzer(PreparedDetector(0.2), None, THRESHOLD)
    analyzer.analyse_verdict(SCAM)
    assert calls["n"] == 1
