"""Dashboard Benchmarks API: runs, details, options, the one-at-a-time launch (409), input
validation and the localhost-only rule. The CLI is the real wrapper, run with this interpreter
against a temporary bench root."""

from __future__ import annotations

import fcntl
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pi_dashboard as dash
import pytest
from rbench import store

REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "deploy" / "pi" / "bench" / "bin" / "rbench"
RUN_ID = "20261106T120102Z-aaaaaa"


@pytest.fixture()
def home(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("PYTHONPATH", "")
    results = tmp_path / "results"
    results.mkdir()
    doc = {
        "schema": "rbench/1",
        "run_id": RUN_ID,
        "ts": "2026-11-06T12:01:02Z",
        "name": "",
        "status": "ok",
        "error": None,
        "model": {"id": "sms_student", "kind": "onnx_sms"},
        "dataset": {
            "id": "sms_campaign_test",
            "sha256": "a" * 64,
            "n_used": 300,
            "denominator_notes": "",
        },
        "params": {"threads": 1},
        "env": {},
        "llm": None,
        "perf": {
            "load_s": 0.1,
            "latency_ms": {"p50": 0.6, "p95": 1.2, "p99": 2.0},
            "throughput_items_s": 1500.0,
            "peak_rss_mb": 200.0,
        },
        "quality": {"key_metric": {"name": "f1", "value": 0.97}},
        "thermal": {
            "temp_max_c": 58.0,
            "throttled_before": "0x0",
            "throttled_after": "0x0",
            "pauses": 0,
        },
    }
    (results / RUN_ID).mkdir()
    (results / RUN_ID / "result.json").write_text(json.dumps(doc))
    store.append_index(results, {"run_id": RUN_ID, "status": "ok", "model": "sms_student"})
    return tmp_path


def test_is_local_client_only_loopback():
    assert dash.is_local_client("127.0.0.1")
    assert dash.is_local_client("::1")
    assert not dash.is_local_client("192.168.29.18")
    assert not dash.is_local_client("10.0.0.5")


def test_list_and_get_runs(home):
    bench = dash.BenchControl(home)
    rows = bench.list_runs(10)
    assert rows[0]["run_id"] == RUN_ID
    doc = bench.get_run(RUN_ID)
    assert doc["run_id"] == RUN_ID
    summary = dash._bench_summary(doc)
    assert summary["p50_ms"] == 0.6 and summary["temp_max_c"] == 58.0
    with pytest.raises(dash.BenchError) as bad:
        bench.get_run("../../etc/passwd")
    assert bad.value.status == 400
    with pytest.raises(dash.BenchError) as missing:
        bench.get_run("20261106T120199Z-bbbbbb")
    assert missing.value.status == 404


def test_options_come_from_the_real_cli(home):
    bench = dash.BenchControl(home, python=sys.executable, wrapper=WRAPPER)
    opts = bench.options()
    ids = {m["id"]: m for m in opts["models"]}
    assert "sms_student" in ids
    assert "groundlm_fpr" in ids["sms_student"]["compatible_datasets"]
    assert "url_domain_test" not in ids["sms_student"]["compatible_datasets"]
    assert {d["id"] for d in opts["datasets"]} >= {"sms_campaign_test", "groundlm_fpr"}


def test_start_validation_and_busy(home):
    bench = dash.BenchControl(home, python=sys.executable, wrapper=WRAPPER)
    with pytest.raises(dash.BenchError) as e:
        bench.start({"model": "sms_student", "dataset": "sms_campaign_test", "limit": 0})
    assert e.value.status == 400
    with pytest.raises(dash.BenchError) as e:
        bench.start({"model": "sms_student", "dataset": "sms_campaign_test", "threads": 9})
    assert e.value.status == 400
    with pytest.raises(dash.BenchError) as e:
        bench.start({"model": "nope", "dataset": "sms_campaign_test"})
    assert e.value.status == 400
    with pytest.raises(dash.BenchError) as e:
        bench.start({"model": "sms_student", "dataset": "url_domain_test"})
    assert e.value.status == 400 and "not compatible" in e.value.message
    with pytest.raises(dash.BenchError) as e:
        bench.start({"model": "sms_student", "dataset": "sms_campaign_test", "limit": True})
    assert e.value.status == 400
    # one at a time: a held lock means a benchmark is running
    results = home / "results"
    with (results / ".rbench.lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert bench.busy() is True
        with pytest.raises(dash.BenchError) as e:
            bench.start({"model": "sms_student", "dataset": "sms_campaign_test", "limit": 10})
        assert e.value.status == 409


@pytest.fixture()
def server(home):
    class Sampler:
        stopped = False

        def snapshot(self):
            return {"latest": None, "history": []}

        def wait_next(self, seq, timeout):
            return None, seq

    bench = dash.BenchControl(home, python=sys.executable, wrapper=WRAPPER)
    srv = dash.DashboardServer(("127.0.0.1", 0), Sampler(), bench)
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01},
                              daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def _get(url: str):
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def _post(url: str, body: bytes, ctype: str = "application/json"):
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def test_http_bench_endpoints(server):
    status, runs = _get(server + "/api/bench/runs")
    assert status == 200 and runs["runs"][0]["run_id"] == RUN_ID
    status, detail = _get(server + f"/api/bench/run/{RUN_ID}")
    assert status == 200 and detail["summary"]["model"] == "sms_student"
    assert detail["result"]["schema"] == "rbench/1"
    assert _get(server + "/api/bench/run/nope")[0] == 400
    assert _get(server + "/api/bench/run/20261106T120199Z-bbbbbb")[0] == 404
    assert _get(server + "/api/bench/unknown")[0] == 404
    status, opts = _get(server + "/api/bench/options")
    assert status == 200 and any(m["id"] == "sms_student" for m in opts["models"])
    status, state = _get(server + "/api/bench/status")
    assert status == 200 and state["busy"] is False


def test_http_post_validation_and_conflict(server, home):
    assert _post(server + "/api/bench/run", b"not json")[0] == 400
    assert _post(server + "/api/bench/run", b"[1,2]")[0] == 400
    assert _post(server + "/api/bench/run", b"")[0] == 400
    assert _post(server + "/api/bench/other", b"{}")[0] == 404
    body = json.dumps({"model": "sms_student", "dataset": "sms_campaign_test", "limit": 5})
    with (home / "results" / ".rbench.lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status, doc = _post(server + "/api/bench/run", body.encode())
    assert status == 409 and "already running" in doc["error"]


def test_http_rejects_non_local_clients(server, monkeypatch):
    monkeypatch.setattr(dash, "is_local_client", lambda addr: False)
    assert _get(server + "/api/bench/runs")[0] == 403
    assert _post(server + "/api/bench/run", b"{}")[0] == 403


def test_page_has_benchmarks_tab_and_no_external_assets():
    assert 'id="tab-bench"' in dash.PAGE and 'id="view-bench"' in dash.PAGE
    assert "/api/bench/run" in dash.PAGE
    assert "http://" not in dash.PAGE and "https://" not in dash.PAGE
