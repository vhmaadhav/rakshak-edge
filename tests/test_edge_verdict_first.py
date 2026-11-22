"""Verdict-first and batched scoring (fast-path steps 1, 3): the verdict is the same as the sync
path, the explainer runs later or from the table, the verdict path takes no lock, and the batch
scores are the single-message scores. The stand-in ONNX model (deploy/pi/bench/standin_onnx.py) is
used where a real session is needed; it is not the trained model."""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "deploy" / "pi" / "service"
BENCH = REPO / "deploy" / "pi" / "bench"
for p in (SERVICE, REPO / "src", BENCH, REPO / "deploy" / "pi"):
    sys.path.insert(0, str(p))

import build_explanation_table as bt  # noqa: E402
import rakshak_edge_service as svc  # noqa: E402
from conftest import load_fixture  # noqa: E402
from edge_pipeline import EdgeAnalyzer  # noqa: E402
from explanation_table import ExplanationTable  # noqa: E402

from rakshak.canonicalize import canonicalize  # noqa: E402
from rakshak.evidence import Risk  # noqa: E402
from rakshak.prompts import fact_slot  # noqa: E402
from rakshak.rules import run_rules  # noqa: E402

MEDIUM_TEXT = "Congrats! You have won the 1st Prize. For Registration please Contact PAYTM Helpline"
HIGH_TEXT = "Dear customer share your OTP to avoid account suspension"
LOW_TEXT = "Hey it's Sam from twitter Could I ask an urgent favour please?"
THRESHOLD = 0.35
MESSAGES = [LOW_TEXT, MEDIUM_TEXT, HIGH_TEXT, "Hello, the meeting moved to 4pm"]


class FakeDetector:
    """Score by text: a dict of probabilities, default 0.1. Has score_many as the real one does."""

    def __init__(self, scores: dict[str, float] | None = None, default: float = 0.1):
        self.scores = scores or {}
        self.default = default
        self.calls = 0
        self.batch_calls = 0

    def score(self, text: str) -> float:
        self.calls += 1
        return self.scores.get(text, self.default)

    def score_many(self, texts):
        self.batch_calls += 1
        return [self.score(t) for t in texts]


class FakeLLM:
    def __init__(self, sentence: str = " may be a scam because it claims a prize."):
        self.sentence = sentence
        self.calls = 0

    def generate(self, prompt, max_new_tokens=64, stop_when=None):
        self.calls += 1
        if stop_when is not None:
            stop_when(self.sentence)
        return self.sentence, {"wall_ms": 1.0, "stop_hit": True}


class TableRunner:
    """A stub explainer for the builder: keyword-bearing sentence with the placeholder."""

    def generate(self, prompt, max_new_tokens=64, stop_when=None):
        tail = " asks for action on [[SLOT]]." if "[[SLOT]]" in prompt else " asks for action."
        return tail, {"wall_ms": 0.0}


def _device(temp=50.0):
    return lambda: {"temp_c": temp, "throttled": "0x0", "power_w": 4.0, "ts": 0.0}


def _energy_stub():
    class Stub:
        def start(self):
            pass

        def stop(self):
            return {
                "energy_j": 0.0,
                "mean_w": 0.0,
                "samples": 0,
                "duration_s": 0.0,
                "method": "stub",
            }

    return Stub()


def _scores(p=0.9):
    return {MEDIUM_TEXT: p, HIGH_TEXT: p}


def _decision(result: dict) -> dict:
    """The fields the verdict is made of (not timings, not explainer text)."""
    d = result["decision"]
    return {
        "risk": d["risk"],
        "action": d["action"],
        "score": d["score"],
        "scam_state": d["scam_state"],
        "detector_only": d["detector_only"],
        "evidence": [(e["id"], e["type"], e["weight"]) for e in result["evidence"]],
        "remediation": result["remediation"],
    }


def _analyzer(llm=None, detector=None, table=None, mode="quiet"):
    return EdgeAnalyzer(
        detector or FakeDetector(_scores()),
        llm or FakeLLM(),
        THRESHOLD,
        detector_only_mode=mode,
        explanation_table=table,
    )


def _covered_table_and_message():
    """A table built with the stub explainer, and a fixture message whose every point it covers."""
    doc = bt.build_table(
        TableRunner(), bt.load_samples(REPO / "tests" / "fixtures" / "rule_cases.json")
    )
    table = ExplanationTable(doc)
    for case in load_fixture("rule_cases.json"):
        canon = canonicalize(case["text"])
        evidence = run_rules(canon)
        if not evidence or len({e.type for e in evidence}) != len(evidence):
            continue
        if all(
            table.sentence(e.type, "en", "describe_kw", fact_slot(e), Risk.MEDIUM) for e in evidence
        ):
            return table, case["text"]
    raise AssertionError("no fixture message is fully covered by the stub table")


# --- verdict parity -----------------------------------------------------------------------


@pytest.mark.parametrize("detector_p", [0.1, 0.9])
def test_verdict_first_decision_equals_the_sync_path(detector_p):
    text = MEDIUM_TEXT
    sync = _analyzer(detector=FakeDetector(_scores(detector_p))).analyse(text)
    result, pending = _analyzer(detector=FakeDetector(_scores(detector_p))).analyse_verdict(text)
    assert _decision(result) == _decision(sync)
    if pending is None:
        assert result["advisory"]["mode"] in {"template", "table", "llm"}
    else:
        assert result["advisory"]["path"] == "pending"


def test_pending_explanation_equals_the_sync_result():
    llm_sync = FakeLLM()
    sync = _analyzer(llm=llm_sync).analyse(MEDIUM_TEXT)
    analyzer = _analyzer(llm=FakeLLM())
    result, pending = analyzer.analyse_verdict(MEDIUM_TEXT)
    assert pending is not None
    assert result["advisory"]["mode"] == "template"
    assert all(p["source"] == "template" for p in result["advisory"]["points"])
    assert analyzer.llm.calls == 0, "the verdict-first answer must not call the explainer"
    full = analyzer.explain_pending(pending)
    assert _decision(full) == _decision(sync)
    assert full["advisory"]["points"] == sync["advisory"]["points"]
    assert full["advisory"]["mode"] == sync["advisory"]["mode"]
    assert full["verifier"] == sync["verifier"]


def test_detector_only_is_final_in_verdict_first():
    text = "Hello, the meeting moved to 4pm"
    analyzer = _analyzer(detector=FakeDetector({text: 0.9}))
    result, pending = analyzer.analyse_verdict(text)
    assert pending is None
    assert result["decision"]["detector_only"] is True
    assert analyzer.llm.calls == 0


def test_low_risk_is_final_in_verdict_first_and_others_are_pending():
    # LOW risk never reaches the explainer (G1): the answer is final at once.
    low = _analyzer(detector=FakeDetector({LOW_TEXT: 0.1})).analyse_verdict(LOW_TEXT)
    if low[0]["decision"]["risk"] == "LOW":
        assert low[1] is None
    else:
        assert low[1] is not None
    medium = _analyzer(detector=FakeDetector({MEDIUM_TEXT: 0.9})).analyse_verdict(MEDIUM_TEXT)
    assert medium[0]["decision"]["risk"] != "LOW" and medium[1] is not None


def test_table_hit_answers_without_the_llm_and_matches_the_slot():
    table, text = _covered_table_and_message()
    llm = FakeLLM()
    analyzer = _analyzer(llm=llm, detector=FakeDetector({text: 0.9}), table=table)
    result, pending = analyzer.analyse_verdict(text)
    assert pending is None
    assert llm.calls == 0
    assert result["advisory"]["path"] == "explainer_table"
    assert result["advisory"]["mode"] == "table"
    assert all(p["source"] == "table" for p in result["advisory"]["points"])
    for p in result["advisory"]["points"]:
        assert "[[SLOT]]" not in p["text"]
    # Same verdict as the sync path with the same detector score.
    sync = _analyzer(llm=FakeLLM(), detector=FakeDetector({text: 0.9})).analyse(text)
    assert _decision(result) == _decision(sync)


def test_batch_results_equal_the_single_message_results():
    scores = {MEDIUM_TEXT: 0.9, HIGH_TEXT: 0.2}
    single = [_analyzer(detector=FakeDetector(scores)).analyse_verdict(t)[0] for t in MESSAGES]
    analyzer = _analyzer(detector=FakeDetector(scores))
    batch = analyzer.analyse_verdict_batch(MESSAGES)
    assert analyzer.detector.batch_calls == 1
    for s, (b, _pending) in zip(single, batch, strict=False):
        assert _decision(b) == _decision(s)
        assert b["advisory"]["path"] == s["advisory"]["path"]


# --- ONNX stand-in: exact batch scores and the INT8 helper --------------------------------

onnx = pytest.importorskip("onnx")


@pytest.fixture(scope="module")
def standin_path(tmp_path_factory):
    from standin_onnx import build_standin

    return build_standin(tmp_path_factory.mktemp("standin") / "standin.onnx")


def _runner_adapter(path, batch_onnx=False):
    from rbench.runners.onnx import SmsRunner

    runner = SmsRunner()
    runner.load(path, 1, {})
    return svc._DetectorAdapter(runner, batch_onnx=batch_onnx)


def test_exact_batch_scores_are_bit_identical_to_single_scores(standin_path):
    adapter = _runner_adapter(standin_path)
    texts = MESSAGES + ["Share your OTP now", "Your parcel is held, pay the customs fee"]
    assert adapter.score_many(texts) == [adapter.score(t) for t in texts]


def test_batched_onnx_scores_are_close_to_single_scores(standin_path):
    exact = _runner_adapter(standin_path)
    batched = _runner_adapter(standin_path, batch_onnx=True)
    texts = MESSAGES + ["Share your OTP now"]
    for a, b in zip(exact.score_many(texts), batched.score_many(texts), strict=False):
        assert abs(a - b) < 1e-4


def test_analyser_on_the_stand_in_gives_the_same_verdicts_batched_and_single(standin_path):
    adapter = _runner_adapter(standin_path)
    analyzer = EdgeAnalyzer(adapter, FakeLLM(), THRESHOLD)
    single = [analyzer.analyse_verdict(t)[0] for t in MESSAGES]
    batch = [r for r, _ in analyzer.analyse_verdict_batch(MESSAGES)]
    assert [_decision(r) for r in single] == [_decision(r) for r in batch]


def test_int8_helper_quantises_and_reports_flips(standin_path, tmp_path):
    from onnx_int8 import quantize_int8_copy, verdict_flips

    int8_path = quantize_int8_copy(standin_path, tmp_path / "standin_int8.onnx")
    assert int8_path.exists()
    fp32 = _runner_adapter(standin_path)
    same = verdict_flips(fp32, fp32, MESSAGES, THRESHOLD)
    assert same["flips"] == 0 and same["n"] == len(MESSAGES)
    int8 = verdict_flips(fp32, _runner_adapter(int8_path), MESSAGES, THRESHOLD)
    assert set(int8) >= {"n", "flips", "max_abs_score_diff", "threshold"}


# --- service: verdict path without the lock, explanation jobs, endpoints ------------------


def _service(analyzer, *, temp=50.0, mode="verdict_first"):
    return svc.Service(
        analyzer,
        threads=2,
        temp_limit=75.0,
        log_path=None,
        models={"sms_student": {"state": "ready"}},
        device_fn=_device(temp),
        energy_factory=_energy_stub,
        response_mode=mode,
    )


def test_unknown_response_mode_is_refused():
    with pytest.raises(ValueError):
        _service(_analyzer(), mode="fast")


def test_verdict_path_does_not_wait_for_a_running_explainer():
    release = threading.Event()
    started = threading.Event()

    class BlockingLLM(FakeLLM):
        def generate(self, prompt, max_new_tokens=64, stop_when=None):
            started.set()
            release.wait(timeout=10)  # event-driven: the test controls when the explainer ends
            return super().generate(prompt, max_new_tokens, stop_when)

    service = _service(_analyzer(llm=BlockingLLM()))
    try:
        status, first = service.analyse_verdict(MEDIUM_TEXT)
        assert status == 200 and first["explanation"]["status"] == "pending"
        assert started.wait(timeout=10)  # the explainer now holds the LLM lock
        out = {}

        def second():
            out["resp"] = service.analyse_verdict(HIGH_TEXT)

        worker = threading.Thread(target=second)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive(), "the verdict path waited for the explainer"
        assert out["resp"][0] == 200
    finally:
        release.set()
        service.wait_explanations()
    status, done = service.explanation(first["explanation"]["id"])
    assert status == 200 and done["status"] == "ready"
    assert done["advisory"]["path"] == "explainer"


def test_explanation_is_ready_from_the_table_without_a_job():
    table, text = _covered_table_and_message()
    llm = FakeLLM()
    service = _service(_analyzer(llm=llm, detector=FakeDetector({text: 0.9}), table=table))
    status, body = service.analyse_verdict(text)
    assert status == 200
    assert body["explanation"] == {"status": "ready", "id": None}
    assert llm.calls == 0


def test_cooling_refuses_the_explainer_job_but_not_the_verdict():
    llm = FakeLLM()
    service = _service(_analyzer(llm=llm), temp=80.0)
    status, body = service.analyse_verdict(MEDIUM_TEXT)
    assert status == 200
    assert body["explanation"]["status"] == "refused_cooling"
    assert body["decision"]["risk"]
    assert llm.calls == 0


def test_unknown_explanation_id_is_404():
    service = _service(_analyzer())
    status, body = service.explanation("does-not-exist")
    assert status == 404


def test_service_batch_items_match_single_verdicts():
    scores = {MEDIUM_TEXT: 0.9, HIGH_TEXT: 0.2}
    service = _service(_analyzer(detector=FakeDetector(scores)))
    status, body = service.analyse_batch(MESSAGES)
    assert status == 200 and body["count"] == len(MESSAGES)
    singles = [
        _service(_analyzer(detector=FakeDetector(scores))).analyse_verdict(t)[1] for t in MESSAGES
    ]
    for item, single in zip(body["items"], singles, strict=False):
        assert _decision(item) == _decision(single)
    service.wait_explanations()


def test_http_round_trip_for_analyze_batch_and_explanation():
    service = _service(_analyzer(detector=FakeDetector(_scores())))
    server = svc.serve(service, "127.0.0.1", 0, b"<html></html>")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(route, doc):
        req = urllib.request.Request(
            base + route,
            data=json.dumps(doc).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())

    try:
        status, batch = post("/analyze_batch", {"sms": [MEDIUM_TEXT, LOW_TEXT]})
        assert status == 200 and batch["count"] == 2
        status, single = post("/analyze", {"sms": MEDIUM_TEXT})
        assert status == 200
        service.wait_explanations()
        if single["explanation"]["status"] == "pending":
            with urllib.request.urlopen(base + single["explanation"]["url"], timeout=10) as resp:
                done = json.loads(resp.read())
            assert done["status"] == "ready"
        status = None
        try:
            post("/analyze_batch", {"sms": []})
        except urllib.error.HTTPError as exc:
            status = exc.code
        assert status == 400
    finally:
        server.shutdown()
        server.server_close()
        service.wait_explanations()


def test_default_response_mode_is_sync():
    assert svc.DEFAULT_RESPONSE_MODE == "sync"
