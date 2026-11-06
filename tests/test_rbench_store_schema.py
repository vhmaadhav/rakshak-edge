"""rbench result schema, index, comparison and thermal telemetry summary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rbench import SCHEMA, store
from rbench import schema as S
from rbench import telemetry as T


def _result(
    run_id: str,
    p50: float,
    n: int = 300,
    status: str = "ok",
    model="sms_student",
    dataset="sms_campaign_test",
    sha="a" * 64,
) -> dict:
    return {
        "schema": SCHEMA,
        "run_id": run_id,
        "ts": "2026-11-06T00:00:00Z",
        "name": "",
        "status": status,
        "error": None,
        "model": {"id": model, "kind": "onnx_sms"},
        "dataset": {"id": dataset, "sha256": sha, "n_used": n, "denominator_notes": "x"},
        "params": {"threads": 1, "limit": n, "warmup": 10, "seed": 0},
        "env": {"python": "3.13"},
        "perf": {
            "load_s": 0.1,
            "latency_ms": {"p50": p50, "p95": p50 * 2, "p99": p50 * 3},
            "throughput_items_s": 1000.0 / p50,
            "peak_rss_mb": 120.0,
        },
        "llm": None,
        "quality": {
            "at_threshold": {"f1": 0.97, "precision": 0.96, "recall": 0.98, "fpr": 0.02},
            "key_metric": {"name": "f1", "value": 0.97},
        },
        "thermal": {"temp_max_c": 61.0, "throttled_after": "0x0", "paused_s": 0.0},
    }


def test_new_run_id_is_sortable_and_well_formed():
    import datetime as dt

    a = S.new_run_id(dt.datetime(2026, 11, 6, 12, 1, 2, tzinfo=dt.UTC))
    b = S.new_run_id(dt.datetime(2026, 11, 6, 12, 1, 3, tzinfo=dt.UTC))
    assert a.startswith("20261106T120102Z-") and a < b
    assert len(a.split("-")[1]) == 6


def test_validate_result_accepts_valid_and_flags_problems():
    doc = _result("20261106T120102Z-aaaaaa", 1.0)
    doc["llm"] = None
    assert S.validate_result(doc) == []
    broken = dict(doc)
    del broken["perf"]
    broken["status"] = "weird"
    problems = S.validate_result(broken)
    assert any("perf" in p for p in problems) and any("status" in p for p in problems)
    assert S.validate_result({"schema": "other"})[0].startswith("schema must be")


def test_index_roundtrip_and_bad_lines(tmp_path: Path):
    results = store.results_root(tmp_path)
    store.append_index(results, S.index_line(_result("20261106T120102Z-aaaaaa", 1.5), "d1"))
    with (results / "index.jsonl").open("a") as handle:
        handle.write("not json\n\n")
    store.append_index(results, S.index_line(_result("20261106T120103Z-bbbbbb", 2.5), "d2"))
    rows = store.read_index(results)
    assert [r["run_id"] for r in rows] == ["20261106T120102Z-aaaaaa", "20261106T120103Z-bbbbbb"]
    assert rows[0]["p50_ms"] == 1.5 and rows[0]["key_value"] == 0.97
    assert rows[0]["max_temp_c"] == 61.0


def test_load_result_errors(tmp_path: Path):
    results = tmp_path / "results"
    with pytest.raises(FileNotFoundError):
        store.load_result(results, "20261106T120102Z-aaaaaa")
    with pytest.raises(ValueError):
        store.load_result(results, "../etc")
    run = results / "20261106T120102Z-aaaaaa"
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps({"schema": "other"}))
    with pytest.raises(ValueError, match="not an rbench/1"):
        store.load_result(results, "20261106T120102Z-aaaaaa")


def test_compare_flags_different_datasets_and_reports_deltas():
    a = _result("20261106T120102Z-aaaaaa", 2.0)
    b = _result("20261106T120103Z-bbbbbb", 3.0)
    report = store.compare(a, b)
    assert report["comparable"] is True and report["reasons"] == []
    p50 = report["metrics"]["perf.latency_ms.p50"]
    assert p50["delta"] == pytest.approx(1.0) and p50["delta_pct"] == pytest.approx(50.0)

    c = _result("20261106T120104Z-cccccc", 2.0, n=100, dataset="groundlm_fpr", sha="b" * 64)
    report = store.compare(a, c)
    assert report["comparable"] is False
    assert any("different datasets" in r for r in report["reasons"])
    assert any("row counts" in r for r in report["reasons"])


def test_compare_marks_error_runs_and_thread_mismatch():
    a = _result("20261106T120102Z-aaaaaa", 2.0)
    b = _result("20261106T120103Z-bbbbbb", 2.0, status="error")
    b["params"]["threads"] = 4
    report = store.compare(a, b)
    assert any("not status ok" in r for r in report["reasons"])
    assert any("thread counts" in r for r in report["reasons"])


def test_summarise_thermal_rows_and_decoded_flags():
    rows = [
        {
            "ts": "1",
            "temp": "50.0",
            "cap": "2400",
            "freq": "2400",
            "throttled": "0x0",
            "power_w": "2.0",
        },
        {
            "ts": "2",
            "temp": "76.5",
            "cap": "1500",
            "freq": "1500",
            "throttled": "0x80008",
            "power_w": "3.0",
        },
        {"ts": "3", "temp": "", "cap": "", "freq": "", "throttled": "", "power_w": ""},
    ]
    block = T.summarise(
        rows, throttled_before="0x0", throttled_after="0x80008", pauses=1, paused_s=4.2
    )
    assert block["temp_start_c"] == 50.0 and block["temp_max_c"] == 76.5
    assert block["temp_end_c"] == 76.5 and block["cap_mhz_min"] == 1500
    assert block["power_w_avg"] == 2.5 and block["samples"] == 3
    assert block["throttled_after_decoded"]["now"] == ["soft temp limit"]
    assert "soft temp limit" in block["throttled_after_decoded"]["occurred"]
    assert block["throttled_before_decoded"] == {"now": [], "occurred": []}
    assert block["pauses"] == 1 and block["paused_s"] == 4.2


def test_summarise_empty_run_and_missing_csv(tmp_path: Path):
    block = T.summarise([], throttled_before=None, throttled_after=None, pauses=0, paused_s=0)
    assert block["temp_max_c"] is None and block["samples"] == 0
    assert T.read_rows(tmp_path / "missing.csv") == []
    assert T.read_rows(None) == []
