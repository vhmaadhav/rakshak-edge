"""pi_monitor live CLI: optional nfstream, flow-JSON and pcap input, summary file, error exits.

Runs without nfstream, without root and without onnxruntime (the scorer is replaced by a fake).
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from rakshak.nids import pi_monitor
from rakshak.nids.schema import FAMILIES

T0 = 1_700_000_000_000.0
ATT = "10.0.0.66"
HOST = "192.168.1.10"


def benign(x):
    p = np.zeros((1, len(FAMILIES)))
    p[0, 0] = 1.0
    return p


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(pi_monitor, "OnnxScorer", lambda path: benign)


def ssh_flow(t_ms):
    return {
        "src_ip": ATT, "dst_ip": HOST, "src_port": 50000, "dst_port": 22, "protocol": 6,
        "bidirectional_first_seen_ms": T0 + t_ms, "bidirectional_duration_ms": 100.0,
        "bidirectional_bytes": 600, "bidirectional_syn_packets": 1,
        "src2dst_packets": 3, "src2dst_bytes": 600, "dst2src_packets": 3, "dst2src_bytes": 0,
    }


def test_iface_without_nfstream_is_a_clear_error(monkeypatch, capsys, fake_model):
    monkeypatch.setattr(pi_monitor, "nfstream_available", lambda: False)
    rc = pi_monitor.main(["--iface", "eth0", "--model", "m.onnx"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "nfstream" in err and "--flows-json" in err


def test_flows_json_works_without_nfstream(tmp_path, monkeypatch, capsys, fake_model):
    monkeypatch.setattr(pi_monitor, "nfstream_available", lambda: False)
    flows = tmp_path / "flows.json"
    flows.write_text(json.dumps({"flows": [ssh_flow(i * 100.0) for i in range(12)]}),
                     encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    rc = pi_monitor.main(["--flows-json", str(flows), "--model", "m.onnx",
                          "--summary-out", str(summary_path), "--alias-salt", "t"])
    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    alerts = [json.loads(line) for line in out]
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "high"
    assert alerts[0]["host"].startswith("h-")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["source"] == "flows_json"
    assert summary["distinct_hosts"] == 1
    assert summary["alerts"]["total"] == 1
    assert ATT not in out[0] and ATT not in summary_path.read_text(encoding="utf-8")


def test_per_flow_keeps_the_old_one_line_per_flow_output(tmp_path, monkeypatch, capsys,
                                                         fake_model):
    flows = tmp_path / "flows.json"
    flows.write_text(json.dumps([ssh_flow(0.0), ssh_flow(10.0)]), encoding="utf-8")
    rc = pi_monitor.main(["--flows-json", str(flows), "--model", "m.onnx", "--per-flow"])
    assert rc == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert all("state" in json.loads(line) for line in lines)


def test_bad_bounds_are_refused_before_scoring(tmp_path, capsys, fake_model):
    flows = tmp_path / "flows.json"
    flows.write_text("[]", encoding="utf-8")
    rc = pi_monitor.main(["--flows-json", str(flows), "--model", "m.onnx",
                          "--max-history", "10"])
    assert rc == 2
    assert "max_history_per_host" in capsys.readouterr().err


def test_flow_file_of_the_wrong_shape_is_refused(tmp_path, capsys, fake_model):
    flows = tmp_path / "flows.json"
    flows.write_text(json.dumps({"x": 1}), encoding="utf-8")
    assert pi_monitor.main(["--flows-json", str(flows), "--model", "m.onnx"]) == 2


def _eth_ip_tcp(src, dst, sport, dport, flags):
    tcp = struct.pack(">HHIIBBHHH", sport, dport, 0, 0, 0x50, flags, 1024, 0, 0)
    a = bytes(map(int, src.split(".")))
    b = bytes(map(int, dst.split(".")))
    ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(tcp), 0, 0, 64, 6, 0, a, b) + tcp
    return b"\x00" * 12 + b"\x08\x00" + ip


def _pcap(records):
    out = bytearray(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
    for ts_ms, frame in records:
        sec, ms = divmod(ts_ms, 1000)
        out += struct.pack("<IIII", int(sec), int(ms * 1000), len(frame), len(frame))
        out += frame
    return bytes(out)


def test_pcap_without_nfstream_uses_the_stdlib_reader(tmp_path, monkeypatch, capsys,
                                                      fake_model):
    monkeypatch.setattr(pi_monitor, "nfstream_available", lambda: False)
    records = [(int(T0) + 100 * i, _eth_ip_tcp(ATT, HOST, 40000 + i, 22, 0x02))
               for i in range(12)]  # one SYN per new source port: 12 short SSH flows
    cap = tmp_path / "ssh.pcap"
    cap.write_bytes(_pcap(records))
    summary_path = tmp_path / "s.json"
    rc = pi_monitor.main(["--pcap", str(cap), "--model", "m.onnx",
                          "--summary-out", str(summary_path)])
    assert rc == 0
    alerts = [json.loads(x) for x in capsys.readouterr().out.strip().splitlines()]
    assert any(a["family"] == "brute_force_ssh" for a in alerts)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["source"] == "pcap_stdlib"
    assert summary["flows"]["seen"] == 12


def test_help_documents_the_summary_and_alert_only_options(capsys):
    text = pi_monitor.build_parser().format_help()
    for flag in ("--summary-out", "--mode", "--flows-json", "--window-s", "--max-hosts"):
        assert flag in text
