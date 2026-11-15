"""Real-pcap job (kaggle/jobs/nids_real_pcaps.py): schedule matching, TP/FN and benign host-hour
counting, model guard, discovery, the wall-clock budget and the partial write.

Synthetic pcaps only (make_holdout_pcaps.write_holdout). Nothing here reads /kaggle/input.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
JOB = REPO / "kaggle" / "jobs" / "nids_real_pcaps.py"
SERVICE = REPO / "deploy" / "pi" / "service"
MODEL = REPO / "kaggle" / "out" / "nids_baselines" / "output" / "student_cse_cic_ids2018_int8.onnx"
for p in (SERVICE, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

spec = importlib.util.spec_from_file_location("nids_real_pcaps_under_test", JOB)
J = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = J
spec.loader.exec_module(J)

KALI = "205.174.165.73"


def _cs(key, kind, records, observed, alerts=(), flagged=(), model_only=(), path="x", top=None):
    return J.CaptureScore(
        key=key,
        kind=kind,
        path=path,
        flows=len(observed),
        observed_hours=set(observed),
        records=list(records),
        flagged=list(flagged),
        model_only_srcs=set(model_only),
        alert_srcs={r["src"] for r in records},
        alerts=list(alerts),
        top_talkers=top or {},
    )


def _rec(src, typ, first, last, weight=0.9):
    return {"src": src, "type": typ, "first_ms": first, "last_ms": last, "weight": weight}


def test_schedule_is_local_time_at_the_fixed_offset():
    t = J.epoch_ms("2017-07-04", "14:00", -3.0)
    assert t == datetime(2017, 7, 4, 17, 0, tzinfo=UTC).timestamp() * 1000.0
    units = J.cic_units(-3.0)
    assert [u.id for u in units] == ["F1", "F2", "F3", "T1", "T2"]
    t2 = next(u for u in units if u.id == "T2")
    assert t2.family == "brute_force_ssh" and t2.end_ms - t2.start_ms == 3_600_000
    assert t2.srcs == frozenset(J.KALI_IPS)


def test_matching_counts_detections_misses_and_benign_host_hours():
    units = J.cic_units(-3.0)
    t2 = next(u for u in units if u.id == "T2")
    inside = t2.start_ms + 30_000
    h_t2 = J.hour_of(inside)
    h_benign = J.hour_of(J.epoch_ms("2017-07-04", "12:00", -3.0))
    h_night = J.hour_of(J.epoch_ms("2017-07-04", "03:00", -3.0))
    records = [
        _rec(KALI, "brute_force_ssh", inside, inside + 10_000),  # TP, family right
        _rec(KALI, "port_scan", inside + 20_000, inside + 25_000),  # same host window, wrong family
        _rec(
            "10.0.0.9",
            "port_scan",
            J.epoch_ms("2017-07-04", "12:00", -3.0),
            J.epoch_ms("2017-07-04", "12:00", -3.0) + 1000,
        ),  # false alert
        _rec(
            KALI,
            "port_scan",
            J.epoch_ms("2017-07-04", "03:00", -3.0),
            J.epoch_ms("2017-07-04", "03:00", -3.0) + 1000,
        ),  # false alert, no attack hour
    ]
    observed = {("10.0.0.9", h_benign), ("10.0.0.1", h_benign), (KALI, h_t2), (KALI, h_night)}
    cs = _cs("tuesday", "cic", records, observed)
    scores = [cs]
    captures = [
        J.Capture("tuesday", "cic", Path("x.pcap")),
        J.Capture("friday", "cic", None, "missing"),
    ]
    s = J.summarise(scores, captures, -3.0)

    rows = {r["unit"]: r for r in s["units_detail"]}
    assert rows["T2"]["detected"] and rows["T2"]["family_correct"]
    assert rows["T2"]["ttfa_s"] == 30.0
    assert rows["T1"]["detected"] is False and rows["T1"]["family_correct"] is False
    assert s["families"]["brute_force_ssh"]["detected"] == 1
    assert s["families"]["brute_force_other"]["misses"] == ["T1"]
    # friday is not scored: its units are listed, not counted as misses
    assert set(s["not_measured_units"]) == {"F1", "F2", "F3"}
    assert s["units"] == 2
    # benign host-hours: (10.0.0.9,h), (10.0.0.1,h), (KALI,h_night); (KALI,h_t2) is labelled
    assert s["benign"]["host_hours"] == 3
    assert s["benign"]["false_alerts"] == 2
    assert s["benign"]["per_rule"]["port_scan"]["false_alerts"] == 2


def test_dns_unit_source_is_the_heaviest_udp53_host_and_span_is_its_flows():
    flows = [
        {
            "src_ip": "172.16.5.60",
            "dst_port": 53,
            "protocol": 17,
            "src2dst_packets": 40,
            "bidirectional_first_seen_ms": 100.0,
            "bidirectional_duration_ms": 50.0,
        },
        {
            "src_ip": "172.16.5.60",
            "dst_port": 53,
            "protocol": 17,
            "src2dst_packets": 40,
            "bidirectional_first_seen_ms": 900.0,
            "bidirectional_duration_ms": 100.0,
        },
        {
            "src_ip": "172.16.5.20",
            "dst_port": 53,
            "protocol": 17,
            "src2dst_packets": 3,
            "bidirectional_first_seen_ms": 10.0,
            "bidirectional_duration_ms": 1.0,
        },
        {
            "src_ip": "172.16.5.20",
            "dst_port": 443,
            "protocol": 6,
            "src2dst_packets": 999,
            "bidirectional_first_seen_ms": 5.0,
            "bidirectional_duration_ms": 1.0,
        },
    ]
    src, span = J.dns_source(flows)
    assert src == "172.16.5.60"
    assert span == (100.0, 1000.0)
    assert J.dns_source([]) == (None, None)


def test_model_guard_refuses_a_different_file(tmp_path):
    (tmp_path / J.MODEL_NAME).write_bytes(b"not the pinned model")
    with pytest.raises(RuntimeError, match="not the pinned"):
        J.find_model(tmp_path)
    with pytest.raises(RuntimeError, match="expected one"):
        J.find_model(tmp_path / "nowhere")


def test_discover_order_and_missing_days(tmp_path):
    (tmp_path / "Friday-WorkingHours").mkdir()
    (tmp_path / "Friday-WorkingHours" / "Friday-WorkingHours.pcap").write_bytes(b"")
    base = tmp_path / "dns" / "dns-exfiltration-dataset" / "01_original_pcap_files"
    for part, name in (
        ("malicious", "ozymandns/a.pcap"),
        ("malicious", "dnscat2/b.pcap"),
        ("benign", "univ/c.pcap"),
    ):
        f = base / part / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"")
    caps = J.discover(tmp_path)
    keys = [c.key for c in caps]
    assert keys[:2] == ["friday", "tuesday"]
    assert caps[0].path is not None and caps[1].path is None
    assert "missing" in caps[1].note
    assert keys[2:] == ["dns-mal/dnscat2/b.pcap", "dns-mal/ozymandns/a.pcap", "dns-ben/univ/c.pcap"]


def test_budget_stop_before_start_writes_partial_results(tmp_path):
    state = {
        "status": "complete",
        "max_hours": 0.0,
        "flow_source": "stdlib",
        "captures": [],
        "_t0": 0.0,
        "elapsed_s": 0.0,
    }
    caps = [J.Capture("dns-ben/x.pcap", "dns_ben", Path("/nonexistent/x.pcap"))]
    scores = J.run_captures(
        caps, None, "stdlib", tmp_path, state, deadline=0.0, clock=lambda: 1.0, log=lambda m: None
    )
    assert scores == []
    assert state["status"] == "budget_exhausted"
    assert state["captures"][0]["status"] == "budget_not_started"
    doc = json.loads((tmp_path / "results.json").read_text())
    assert doc["status"] == "budget_exhausted"
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["status"] == "budget_exhausted"
    assert (tmp_path / "report.md").read_text().startswith("# Real pcaps")
    assert (tmp_path / "alerts.parquet").exists()


def test_budget_stop_inside_a_capture_is_not_scored(tmp_path):
    holdout = pytest.importorskip("make_holdout_pcaps")
    pcap_dir = tmp_path / "pcaps"
    holdout.write_holdout(pcap_dir)
    pcap = pcap_dir / "ho_ssh_brute_fast.pcap"
    state = {
        "status": "complete",
        "max_hours": 1.0,
        "flow_source": "stdlib",
        "captures": [],
        "_t0": 0.0,
        "elapsed_s": 0.0,
    }
    caps = [J.Capture("dns-ben/fast.pcap", "dns_ben", pcap)]
    seq = iter([0.0])  # first clock read (start check) passes; the in-capture check then expires
    clock = lambda: next(seq, 1e9)  # noqa: E731
    scores = J.run_captures(
        caps, None, "stdlib", tmp_path / "out", state, deadline=1e6, clock=clock, log=lambda m: None
    )
    assert scores == []
    assert state["captures"][0]["status"] == "budget_exhausted"
    assert state["captures"][0].get("note") == "not scored"
    assert (
        json.loads((tmp_path / "out" / "results.json").read_text())["status"] == "budget_exhausted"
    )


@pytest.fixture(scope="module")
def scorer():
    if not MODEL.exists():
        pytest.skip("INT8 model not in kaggle/out (kernel output not fetched)")
    onnxruntime = pytest.importorskip("onnxruntime")
    assert onnxruntime  # noqa: S101
    from rakshak.nids.pi_monitor import OnnxScorer

    return OnnxScorer(str(MODEL))


def test_job_pipeline_matches_the_service_analyser(tmp_path, scorer):
    """The job's per-flow loop gives the same host alerts as NidsAnalyzer.analyse_flows."""
    holdout = pytest.importorskip("make_holdout_pcaps")
    from nids_pipeline import NidsAnalyzer

    from rakshak.nids.pcap_flows import flows_from_pcap_bytes

    pcap_dir = tmp_path / "pcaps"
    holdout.write_holdout(pcap_dir)
    for name in (
        "ho_ssh_brute_fast.pcap",
        "ho_port_scan_20.pcap"
        if (pcap_dir / "ho_port_scan_20.pcap").exists()
        else "ho_ssh_brute_9.pcap",
    ):
        raw = (pcap_dir / name).read_bytes()
        flows = flows_from_pcap_bytes(raw)
        ordered = sorted(flows, key=lambda f: float(f.get("bidirectional_first_seen_ms") or 0.0))
        decisions, hits, alerts = J.score_flows(ordered, scorer, lambda: None)
        mine = sorted((a.src, a.family, a.state, a.is_model_only) for a in alerts)
        service = NidsAnalyzer(
            scorer, scratch=tmp_path, flow_source="stdlib", model_mode="quiet"
        ).analyse_flows(flows)["alerts"]
        theirs = sorted(
            (
                a["src"],
                a["decision"]["family"],
                a["decision"]["state"],
                a["advisory"]["path"] == "model_only_quiet",
            )
            for a in service
        )
        mine_norm = sorted((s, f, st, mo) for s, f, st, mo in mine)
        assert [(s, f, mo) for s, f, _, mo in mine_norm] == [(s, f, mo) for s, f, _, mo in theirs]
        recs = J.rule_records(ordered, decisions, hits)
        rule_hosts = {r["src"] for r in recs}
        assert rule_hosts == {a.src for a in alerts if not a.is_model_only}
