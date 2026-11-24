"""Edge service on the laptop: pipeline, decision mapping, guards, cooling gate, HTTP surface and
energy window. The detector, the LLM runner and device telemetry are mocks; no model is loaded."""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parents[1] / "deploy" / "pi" / "service"
REPO = Path(__file__).resolve().parents[1]
for p in (SERVICE, REPO / "src", REPO / "deploy" / "pi" / "bench", REPO / "deploy" / "pi"):
    sys.path.insert(0, str(p))

import rakshak_edge_service as svc  # noqa: E402
from edge_device import EnergyWindow, read_snapshot  # noqa: E402
from edge_pipeline import (  # noqa: E402
    DETECTOR_ONLY_QUIET_SUMMARY,
    DETECTOR_ONLY_SUMMARY,
    EdgeAnalyzer,
    scam_state_for,
)

from rakshak.canonicalize import canonicalize  # noqa: E402
from rakshak.rules import run_rules  # noqa: E402

LOW_TEXT = "Hey it's Sam from twitter Could I ask an urgent favour please?"
MEDIUM_TEXT = "Congrats! You have won the 1st Prize. For Registration please Contact PAYTM Helpline"
HIGH_TEXT = "Dear customer share your OTP to avoid account suspension"
THRESHOLD = 0.35


class FakeDetector:
    def __init__(self, p: float):
        self.p = p
        self.calls = 0

    def score(self, text: str) -> float:
        self.calls += 1
        return self.p


class FakeLLM:
    """Returns one scripted sentence per call and records the prompts."""

    def __init__(self, sentence: str = " may be a scam because it claims a prize."):
        self.sentence = sentence
        self.calls = 0

    def generate(self, prompt, max_new_tokens=64, stop_when=None):
        self.calls += 1
        text = self.sentence
        if stop_when is not None:
            stop_when(text)
        return text, {
            "wall_ms": 120.0,
            "new_tokens": 6,
            "prompt_tokens": 90,
            "ttft_ms": 40.0,
            "stop_hit": True,
        }


def _device(temp=50.0):
    return lambda: {"temp_c": temp, "throttled": "0x0", "power_w": 4.0, "ts": 0.0}


def _energy_stub():
    class Stub:
        def start(self):
            pass

        def stop(self):
            return {
                "energy_j": 1.5,
                "mean_w": 3.0,
                "samples": 2,
                "duration_s": 0.5,
                "method": "trapezoid",
            }

    return Stub()


# --- decision mapping ---------------------------------------------------------


def test_scam_state_mapping():
    ev = [object()]
    assert scam_state_for(0.9, THRESHOLD, []) == "scam"
    assert scam_state_for(0.9, THRESHOLD, ev) == "scam"
    assert scam_state_for(0.2, THRESHOLD, ev) == "uncertain"
    assert scam_state_for(0.2, THRESHOLD, []) == "benign"
    assert scam_state_for(THRESHOLD, THRESHOLD, []) == "scam"  # >= threshold


def test_rules_run_on_the_canonical_text():
    assert run_rules(canonicalize(HIGH_TEXT))


# --- pipeline -----------------------------------------------------------------


def test_low_risk_makes_no_llm_call_and_is_template_only():
    llm = FakeLLM()
    out = EdgeAnalyzer(FakeDetector(0.05), llm, THRESHOLD).analyse(LOW_TEXT)
    assert out["decision"]["risk"] == "LOW"
    assert llm.calls == 0
    assert out["advisory"]["llm_skipped_low_risk"] is True
    assert out["advisory"]["mode"] == "template"
    assert out["llm_calls"] == []
    assert out["timings_ms"]["llm_total"] == 0.0
    assert all(p["source"] == "template" for p in out["advisory"]["points"])
    assert all(p["check"]["ok"] is None for p in out["advisory"]["points"])


def test_medium_hedged_sentence_is_kept_and_unhedged_is_rejected():
    out = EdgeAnalyzer(FakeDetector(0.9), FakeLLM(), THRESHOLD).analyse(MEDIUM_TEXT)
    assert out["decision"]["risk"] == "MEDIUM"
    assert out["advisory"]["llm_skipped_low_risk"] is False
    # FakeLLM returns the same sentence for every call. Fix D makes one call per evidence type, so
    # the advisory has one point per type; with a single type that is one LLM point.
    points = out["advisory"]["points"]
    assert points[0]["source"] == "llm"
    assert all(p["source"] == "template" for p in points[1:])
    assert all(p["cleanup"]["duplicate_of"] == 0 for p in points[1:])
    assert out["advisory"]["mode"] == ("llm" if len(points) == 1 else "partial")
    bad = EdgeAnalyzer(
        FakeDetector(0.9),
        FakeLLM(" is a scam because it claims a prize."),
        THRESHOLD,
    ).analyse(MEDIUM_TEXT)
    assert bad["advisory"]["mode"] == "template"
    # The repeat rule (Fix C) blanks the second item's identical sentence before the verifier, so
    # only the first sentence is counted as a G2 rejection. Repeats are not counted twice.
    assert len(bad["verifier"]["g2_rejected"]) == 1
    assert all(p["source"] == "template" for p in bad["advisory"]["points"])
    assert "attribution" in " ".join(bad["advisory"]["points"][0]["check"]["problems"])


def test_high_risk_is_not_guarded_and_action_is_allowed():
    out = EdgeAnalyzer(FakeDetector(0.99), FakeLLM(" is a scam."), THRESHOLD).analyse(HIGH_TEXT)
    assert out["decision"]["risk"] == "HIGH"
    assert out["verifier"]["g2_rejected"] == []
    assert out["decision"]["action"] in {
        "DO_NOT_SHARE_OTP",
        "CONTACT_BANK_OFFICIAL",
        "DO_NOT_CLICK",
        "OPEN_OFFICIAL_APP",
    }


def test_result_schema_and_timings():
    out = EdgeAnalyzer(FakeDetector(0.9), FakeLLM(), THRESHOLD).analyse(MEDIUM_TEXT)
    assert set(out) >= {"decision", "evidence", "advisory", "verifier", "llm_calls", "timings_ms"}
    assert set(out["decision"]) >= {"risk", "action", "score", "threshold", "scam_state"}
    t = out["timings_ms"]
    assert set(t) >= {
        "canonicalise",
        "detect",
        "evidence",
        "llm_calls",
        "llm_total",
        "verify",
        "total",
    }
    # One explainer call per evidence type (Fix D), not per evidence item.
    types = {e["type"] for e in out["evidence"]}
    assert len(t["llm_calls"]) == len(out["llm_calls"]) == len(types)
    # llm_total is the runner's own call time; total is the wall clock of the whole analysis.
    assert t["total"] >= t["canonicalise"] + t["detect"] + t["evidence"] - 1e-6
    assert out["advisory"]["summary"]["source"] == "template"
    json.dumps(out)  # serialisable


def test_detector_score_drives_scam_state():
    # Urgency-only evidence with a scam state is UNCERTAIN in evidence.py, not LOW, so the
    # explainer runs (G1 only skips LOW).
    out = EdgeAnalyzer(FakeDetector(0.9), FakeLLM(), THRESHOLD).analyse(LOW_TEXT)
    assert out["decision"]["scam_state"] == "scam"
    assert out["decision"]["risk"] == "UNCERTAIN"
    assert out["advisory"]["llm_skipped_low_risk"] is False


# --- Fix B: detector flags the message, no rule evidence ----------------------------------------

# Constructed: no rule fires on it (asserted below). The Pi scan (prompt_trial.py
# scan-detector-only) found real test-split messages in the same state; none is copied here.
DETECTOR_ONLY_TEXT = "Your weekly draw entry is confirmed. Reply YES to receive your entry details."


def test_detector_only_text_has_no_rule_evidence_and_evidence_risk_is_unchanged():
    from rakshak.evidence import build_package

    canon = canonicalize(DETECTOR_ONLY_TEXT)
    assert run_rules(canon) == []
    # evidence._risk is not changed (Kaggle evals depend on it): scam without evidence is LOW.
    assert build_package(canon, [], "scam").risk.value == "LOW"


def test_default_detector_only_mode_is_quiet():
    out = EdgeAnalyzer(FakeDetector(0.9), FakeLLM(), THRESHOLD).analyse(DETECTOR_ONLY_TEXT)
    assert out["advisory"]["path"] == "detector_only_quiet"


def test_quiet_mode_is_benign_style_no_warning_and_never_calls_the_llm():
    llm = FakeLLM()
    analyzer = EdgeAnalyzer(FakeDetector(0.9), llm, THRESHOLD, detector_only_mode="quiet")
    out = analyzer.analyse(DETECTOR_ONLY_TEXT)
    assert out["decision"]["scam_state"] == "scam"
    assert out["decision"]["detector_only"] is True
    assert out["decision"]["score"] == 0.9 and out["decision"]["threshold"] == THRESHOLD
    assert out["decision"]["risk"] == "LOW"
    assert out["decision"]["action"] == "IGNORE"
    assert out["decision"]["evidence_count"] == 0
    assert out["advisory"]["path"] == "detector_only_quiet"
    assert out["advisory"]["mode"] == "template"
    assert out["advisory"]["points"] == []
    assert out["advisory"]["summary"]["text"] == DETECTOR_ONLY_QUIET_SUMMARY
    assert out["verifier"]["final_contract_ok"] is True
    assert out["llm_calls"] == [] and llm.calls == 0
    assert out["timings_ms"]["llm_total"] == 0.0


def test_quiet_summary_is_neutral_and_never_says_safe_or_no_known_pattern():
    import re

    from rakshak.verifier import _SAFE_RE

    assert DETECTOR_ONLY_QUIET_SUMMARY == "No specific scam pattern was found in this message."
    assert not _SAFE_RE.search(DETECTOR_ONLY_QUIET_SUMMARY)
    assert not re.search(r"(?i)no known scam pattern|safe", DETECTOR_ONLY_QUIET_SUMMARY)
    assert len(DETECTOR_ONLY_QUIET_SUMMARY) <= 280


def test_quiet_and_verify_differ_only_in_the_user_visible_fields():
    quiet = EdgeAnalyzer(FakeDetector(0.9), FakeLLM(), THRESHOLD, detector_only_mode="quiet")
    verify = EdgeAnalyzer(FakeDetector(0.9), FakeLLM(), THRESHOLD, detector_only_mode="verify")
    q = quiet.analyse(DETECTOR_ONLY_TEXT)
    v = verify.analyse(DETECTOR_ONLY_TEXT)
    for key in ("score", "threshold", "scam_state", "evidence_count", "detector_only"):
        assert q["decision"][key] == v["decision"][key]
    assert (q["decision"]["risk"], q["decision"]["action"]) == ("LOW", "IGNORE")
    assert (v["decision"]["risk"], v["decision"]["action"]) == ("UNCERTAIN", "VERIFY_INDEPENDENTLY")
    assert q["advisory"]["path"] == "detector_only_quiet"
    assert v["advisory"]["path"] == "detector_only"


def test_verify_mode_keeps_the_uncertain_verify_notice():
    llm = FakeLLM()
    analyzer = EdgeAnalyzer(FakeDetector(0.9), llm, THRESHOLD, detector_only_mode="verify")
    out = analyzer.analyse(DETECTOR_ONLY_TEXT)
    assert out["decision"]["scam_state"] == "scam"
    assert out["decision"]["risk"] == "UNCERTAIN"
    assert out["decision"]["evidence_count"] == 0
    assert out["decision"]["action"] == "VERIFY_INDEPENDENTLY"
    assert out["advisory"]["path"] == "detector_only"
    assert out["advisory"]["mode"] == "template"
    assert out["advisory"]["points"] == []
    assert out["advisory"]["summary"]["text"] == DETECTOR_ONLY_SUMMARY
    assert "flagged" in out["advisory"]["summary"]["text"]
    assert "no known scam pattern" not in json.dumps(out["advisory"]).lower()
    assert out["verifier"]["final_contract_ok"] is True
    assert out["llm_calls"] == [] and llm.calls == 0
    assert out["timings_ms"]["llm_total"] == 0.0
    assert set(out["timings_ms"]) >= {"canonicalise", "detect", "evidence", "verify", "total"}


@pytest.mark.parametrize("mode", ["quiet", "verify"])
def test_detector_fires_without_evidence_takes_the_flagged_path_at_the_threshold(mode):
    out = EdgeAnalyzer(
        FakeDetector(THRESHOLD), FakeLLM(), THRESHOLD, detector_only_mode=mode
    ).analyse(DETECTOR_ONLY_TEXT)
    assert out["advisory"]["path"] in {"detector_only", "detector_only_quiet"}
    assert out["decision"]["detector_only"] is True


@pytest.mark.parametrize("mode", ["quiet", "verify"])
def test_below_threshold_without_evidence_keeps_the_low_template(mode):
    llm = FakeLLM()
    out = EdgeAnalyzer(FakeDetector(0.05), llm, THRESHOLD, detector_only_mode=mode).analyse(
        DETECTOR_ONLY_TEXT
    )
    assert out["decision"]["risk"] == "LOW"
    assert out["decision"]["detector_only"] is False
    assert out["advisory"]["path"] == "low_template"
    assert "no known scam pattern" in out["advisory"]["summary"]["text"]
    assert llm.calls == 0


def test_unknown_detector_only_mode_is_refused():
    with pytest.raises(ValueError):
        EdgeAnalyzer(FakeDetector(0.9), FakeLLM(), THRESHOLD, detector_only_mode="loud")


def test_detector_only_flag_is_false_on_the_explainer_and_low_paths():
    low = EdgeAnalyzer(FakeDetector(0.05), FakeLLM(), THRESHOLD).analyse(LOW_TEXT)
    med = EdgeAnalyzer(FakeDetector(0.9), FakeLLM(), THRESHOLD).analyse(MEDIUM_TEXT)
    assert low["decision"]["detector_only"] is False
    assert med["decision"]["detector_only"] is False


def test_detector_fires_with_evidence_still_uses_the_explainer():
    out = EdgeAnalyzer(FakeDetector(0.9), FakeLLM(), THRESHOLD).analyse(MEDIUM_TEXT)
    assert out["advisory"]["path"] == "explainer"
    assert out["decision"]["evidence_count"] > 0


def test_detector_only_path_is_not_taken_when_evidence_exists_for_a_low_state():
    out = EdgeAnalyzer(FakeDetector(0.05), FakeLLM(), THRESHOLD).analyse(LOW_TEXT)
    assert out["advisory"]["path"] != "detector_only"


def test_detector_only_summary_is_within_the_summary_limit_and_says_what_to_do():
    from rakshak.verifier import MAX_SUMMARY

    assert len(DETECTOR_ONLY_SUMMARY) <= MAX_SUMMARY
    assert "Do not click links" in DETECTOR_ONLY_SUMMARY


# --- service: cooling gate, lock, logging ---------------------------------------


class FakeAnalyzer:
    def __init__(self):
        self.calls = 0
        self.inside = 0
        self.max_inside = 0
        self.lock = threading.Lock()

    def analyse(self, text, lang="auto"):
        with self.lock:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
        time.sleep(0.02)
        with self.lock:
            self.inside -= 1
            self.calls += 1
        return {
            "decision": {"risk": "LOW", "action": "IGNORE"},
            "advisory": {"mode": "template", "llm_skipped_low_risk": True},
            "timings_ms": {"total": 1.0, "llm_total": 0.0},
            "llm_calls": [],
            "evidence": [],
            "verifier": {},
        }


def _service(temp=50.0, analyzer=None, log=None):
    return svc.Service(
        analyzer or FakeAnalyzer(),
        threads=2,
        temp_limit=75.0,
        log_path=log,
        models={"sms_student": {"state": "ready"}},
        device_fn=_device(temp),
        energy_factory=_energy_stub,
    )


def test_cooling_gate_returns_503_and_does_not_run_the_analyser():
    analyzer = FakeAnalyzer()
    service = _service(temp=76.0, analyzer=analyzer)
    status, body = service.analyse("hello")
    assert status == 503
    assert body["error"] == "cooling"
    assert body["temp_c"] == 76.0
    assert analyzer.calls == 0
    assert service.refused == 1


def test_at_the_limit_is_refused_and_below_it_runs():
    analyzer = FakeAnalyzer()
    assert _service(temp=75.0, analyzer=analyzer).analyse("x")[0] == 503
    assert _service(temp=74.9, analyzer=analyzer).analyse("x")[0] == 200


def test_requests_are_serialised():
    analyzer = FakeAnalyzer()
    service = _service(analyzer=analyzer)
    threads = [threading.Thread(target=service.analyse, args=("x",)) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert analyzer.calls == 5
    assert analyzer.max_inside == 1


def test_analyse_response_has_device_and_energy_and_is_logged(tmp_path):
    log = tmp_path / "edge.log"
    service = _service(log=log)
    status, body = service.analyse("hello there")
    assert status == 200
    assert body["device"]["start"]["temp_c"] == 50.0
    assert body["device"]["energy"]["energy_j"] == 1.5
    assert body["request_id"] and body["input_chars"] == len("hello there")
    line = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert line["event"] == "analyse" and line["temp_start_c"] == 50.0


def test_analyser_exception_is_not_swallowed_by_the_lock():
    class Boom(FakeAnalyzer):
        def analyse(self, text, lang="auto"):
            raise RuntimeError("boom")

    service = _service(analyzer=Boom())
    with pytest.raises(RuntimeError):
        service.analyse("x")
    # the lock was released, so the next request is not blocked
    assert service.lock.acquire(timeout=1)
    service.lock.release()


# --- HTTP surface ---------------------------------------------------------------


@pytest.fixture()
def server():
    service = _service()
    httpd = svc.serve(service, "127.0.0.1", 0, (svc.HERE / "index.html").read_bytes())
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01},
                              daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    yield f"http://{host}:{port}", service
    httpd.shutdown()
    httpd.server_close()


def _req(url, data=None, method=None):
    req = urllib.request.Request(
        url,
        data=data,
        method=method or ("POST" if data else "GET"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type"), exc.read()


def test_health_and_page(server):
    base, _ = server
    status, ctype, body = _req(base + "/health")
    doc = json.loads(body)
    assert status == 200 and doc["status"] == "ok" and doc["threads"] == 2
    assert "device" in doc and "rss_mb" in doc
    status, ctype, body = _req(base + "/")
    assert status == 200 and ctype.startswith("text/html")
    assert b"Rakshak edge pipeline" in body


def test_page_has_no_external_assets_and_fits_a_phone():
    html = (svc.HERE / "index.html").read_text(encoding="utf-8")
    assert "<script src" not in html and "<link" not in html
    assert "https://" not in html and "http://" not in html
    assert 'name="viewport"' in html


def test_analyze_input_validation(server):
    base, _ = server
    assert _req(base + "/analyze", b"not json")[0] == 400
    assert _req(base + "/analyze", b'{"sms": ""}')[0] == 400
    assert _req(base + "/analyze", b'{"sms": 5}')[0] == 400
    assert _req(base + "/analyze", json.dumps({"sms": "x" * 2001}).encode())[0] == 400
    assert _req(base + "/nope")[0] == 404


def test_analyze_ok_over_http(server):
    base, _ = server
    status, _, body = _req(base + "/analyze", json.dumps({"sms": "hello there"}).encode())
    assert status == 200
    doc = json.loads(body)
    assert doc["decision"]["risk"] == "LOW"


def test_refuses_non_loopback_bind():
    with pytest.raises(SystemExit):
        svc.serve(_service(), "0.0.0.0", 0, b"")


# --- device telemetry -----------------------------------------------------------


def test_read_snapshot_uses_the_injected_readers():
    snap = read_snapshot(lambda: 61.5, lambda: "0x50005", lambda: 3.25)
    assert snap["temp_c"] == 61.5 and snap["throttled"] == "0x50005" and snap["power_w"] == 3.25


def test_energy_window_integrates_constant_power():
    window = EnergyWindow(power_fn=lambda: 4.0, interval_s=0.02)
    window.start()
    time.sleep(0.25)
    doc = window.stop()
    assert doc["method"] == "trapezoid"
    assert doc["samples"] >= 3
    assert 4.0 * doc["duration_s"] * 0.8 <= doc["energy_j"] <= 4.0 * doc["duration_s"] * 1.05
    assert doc["mean_w"] == pytest.approx(4.0, abs=0.01)


def test_energy_window_without_power_reads_gives_zero():
    window = EnergyWindow(power_fn=lambda: None, interval_s=0.02)
    window.start()
    doc = window.stop()
    assert doc["energy_j"] == 0.0 and doc["method"] == "none"


def test_rss_is_a_number_on_linux():
    assert svc.rss_mb() is None or svc.rss_mb() > 0


# --- Unix socket, bridge and eval client ------------------------------------------


def _free_port():
    import socket as socket_mod

    with socket_mod.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _short_socket_path(tmp_path: Path, name: str = "edge.sock"):
    """macOS limits AF_UNIX socket paths to 104 bytes; fall back to /tmp when tmp_path is longer."""
    candidate = tmp_path / name
    if len(str(candidate)) <= 95:
        yield candidate
        return
    with tempfile.TemporaryDirectory(prefix="rakshak-sock-") as td:
        yield Path(td) / name


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/self/net is Linux-only")
def test_netns_report_without_probe_lists_interfaces_and_routes():
    report = svc.netns_report(probe=False)
    assert "error" not in report
    assert "lo" in report["interfaces"]
    assert isinstance(report["routes"], int)
    assert "egress_probe" not in report


def test_unix_socket_serves_the_same_api_and_the_bridge_relays_it(tmp_path):
    import socket_bridge

    with _short_socket_path(tmp_path) as sock_path:
        sock = str(sock_path)
        service = _service()
        index = (svc.HERE / "index.html").read_bytes()
        unix_server = svc.serve_unix(service, sock, index)
        threading.Thread(
            target=unix_server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()
        port = _free_port()
        threading.Thread(
            target=socket_bridge.serve, args=("127.0.0.1", port, sock), daemon=True
        ).start()
        try:
            # the bridge is a plain TCP listener on loopback: this is what the browser uses
            deadline = time.time() + 5
            status = body = None
            while time.time() < deadline:
                try:
                    status, _, body = _req(f"http://127.0.0.1:{port}/health")
                    break
                except OSError:
                    time.sleep(0.05)
            assert status == 200
            assert json.loads(body)["status"] == "ok"
            status, _, body = _req(f"http://127.0.0.1:{port}/")
            assert status == 200 and b"Rakshak edge pipeline" in body
        finally:
            unix_server.shutdown()
            unix_server.server_close()


def test_eval_client_talks_to_the_unix_socket(tmp_path, monkeypatch):
    import eval_service

    with _short_socket_path(tmp_path) as sock:
        service = _service()
        unix_server = svc.serve_unix(service, str(sock), b"<html></html>")
        threading.Thread(
            target=unix_server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()
        monkeypatch.setattr(eval_service, "SOCKET", sock)
        try:
            assert eval_service._get("/health")["status"] == "ok"
            status, body = eval_service._post("hello there")
            assert status == 200 and body["decision"]["risk"] == "LOW"
        finally:
            unix_server.shutdown()
            unix_server.server_close()


def test_eval_percentiles_and_counts():
    import eval_service

    assert eval_service._percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert eval_service._percentile([], 0.5) is None
    assert eval_service._count(["LOW", "HIGH", "LOW"]) == {"HIGH": 1, "LOW": 2}


def test_health_reports_the_detector_only_mode():
    assert _service().health()["detector_only_mode"] == "quiet"
    service = svc.Service(
        FakeAnalyzer(),
        threads=1,
        temp_limit=75.0,
        log_path=None,
        models={},
        detector_only_mode="verify",
        device_fn=_device(50.0),
        energy_factory=_energy_stub,
    )
    assert service.health()["detector_only_mode"] == "verify"
