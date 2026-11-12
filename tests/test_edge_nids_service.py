"""Network path of the edge service: /analyze_flows (JSON and pcap), per-host alerts, remediation,
the 503 cooling refusal, the body limits and the model-contract check. No Pi, no model download:
the student is a fake probability function, except the contract test which writes a small ONNX
copy with broken metadata."""

from __future__ import annotations

import json
import shutil
import struct
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "deploy" / "pi" / "service"
for p in (SERVICE, REPO / "src", REPO / "deploy" / "pi" / "bench", REPO / "deploy" / "pi"):
    sys.path.insert(0, str(p))

import rakshak_edge_service as svc  # noqa: E402
from edge_pipeline import EdgeAnalyzer  # noqa: E402
from nids_pipeline import MAX_BODY_BYTES, NidsAnalyzer, flows_from_body  # noqa: E402

from rakshak.evidence import Action, EvidencePackage, Risk  # noqa: E402
from rakshak.nids.evidence import NetAction  # noqa: E402
from rakshak.nids.live import LIVE_FEATURES  # noqa: E402
from rakshak.nids.schema import FAMILIES  # noqa: E402
from rakshak.remediation import (  # noqa: E402
    NET_CODES,
    SMS_CODES,
    SMS_ONLY,
    remediation_for,
    remediation_problems,
)

SCANNER = "10.0.0.66"
TARGET = "192.168.1.10"
BENIGN = "192.168.1.20"
IN_PKTS = LIVE_FEATURES.index("IN_PKTS")
OUT_PKTS = LIVE_FEATURES.index("OUT_PKTS")


def fake_probs(x: np.ndarray) -> np.ndarray:
    """One-packet flows get a port_scan-style score; everything else is benign."""
    row = np.zeros((1, len(FAMILIES)), dtype=np.float32)
    if x[0, IN_PKTS] <= 1 and x[0, OUT_PKTS] <= 1:
        row[0, 0] = 0.05
        row[0, FAMILIES.index("port_scan")] = 0.95
    else:
        row[0, 0] = 0.9999  # benign: attack score 1e-4, below both calibrated thresholds
    return row


def _flow(src, dst, dport, t, in_pkts=1, out_pkts=1):
    return {
        "src_ip": src,
        "dst_ip": dst,
        "src_port": 40000,
        "dst_port": dport,
        "protocol": 6,
        "bidirectional_first_seen_ms": 1_700_000_000_000.0 + t,
        "bidirectional_duration_ms": 1.0,
        "bidirectional_bytes": 80,
        "bidirectional_min_ps": 40,
        "bidirectional_max_ps": 40,
        "src2dst_packets": in_pkts,
        "src2dst_bytes": 40 * in_pkts,
        "src2dst_duration_ms": 0.0,
        "dst2src_packets": out_pkts,
        "dst2src_bytes": 40 * out_pkts,
        "dst2src_duration_ms": 0.0,
        "bidirectional_syn_packets": 1,
        "src2dst_syn_packets": 1,
        "bidirectional_rst_packets": 1,
        "dst2src_rst_packets": 1,
    }


def scan_flows(n=25):
    return [_flow(SCANNER, TARGET, 1 + i, i * 10.0) for i in range(n)]


def benign_flows():
    return [_flow(BENIGN, TARGET, 443, 1000.0 + i, 5, 5) for i in range(3)]


def _pcap_bytes(ports, src=SCANNER, dst=TARGET):
    """A classic little-endian pcap of one SYN per destination port (Ethernet and IPv4 and TCP)."""
    out = bytearray(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
    for i, port in enumerate(ports):
        eth = b"\x02" * 6 + b"\x02" * 6 + b"\x08\x00"
        ip = struct.pack(
            ">BBHHHBBH4s4s",
            0x45,
            0,
            40,
            i,
            0,
            64,
            6,
            0,
            bytes(map(int, src.split("."))),
            bytes(map(int, dst.split("."))),
        )
        tcp = struct.pack(">HHIIBBHHH", 45000, port, 0, 0, 0x50, 0x02, 1024, 0, 0)
        frame = eth + ip + tcp
        out += struct.pack(
            "<IIII", 1_700_000_000 + i // 10, (i % 10) * 1000, len(frame), len(frame)
        )
        out += frame
    return bytes(out)


def _energy_stub():
    class Stub:
        def start(self):
            pass

        def stop(self):
            return {
                "energy_j": 0.5,
                "mean_w": 3.0,
                "samples": 2,
                "duration_s": 0.2,
                "method": "trapezoid",
            }

    return Stub()


def _device(temp=50.0):
    return lambda: {"temp_c": temp, "throttled": "0x0", "power_w": 4.0, "ts": 0.0}


def _service(tmp_path, temp=50.0):
    nids = NidsAnalyzer(fake_probs, scratch=tmp_path / "uploads", flow_source="stdlib")
    service = svc.Service(
        analyzer=None,
        threads=1,
        temp_limit=75.0,
        log_path=tmp_path / "log.jsonl",
        models={},
        device_fn=_device(temp),
        energy_factory=_energy_stub,
        nids=nids,
    )
    return service


@pytest.fixture()
def http_server(tmp_path):
    servers = []

    def start(service):
        server = svc.serve(service, "127.0.0.1", 0, b"<html></html>")
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01},
                         daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def _post(base, path, body: bytes, ctype: str):
    req = urllib.request.Request(
        base + path, data=body, method="POST", headers={"Content-Type": ctype}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# --- aggregation and endpoint -------------------------------------------------


def test_json_flows_give_one_alert_for_the_scanner_only(tmp_path):
    service = _service(tmp_path)
    status, body = service.analyse_flows(
        json.dumps({"flows": scan_flows() + benign_flows()}).encode()
    )
    assert status == 200
    assert body["flows_processed"] == 28
    assert body["flow_source"] == "json"
    assert body["decision"]["alerts"] == 1
    alert = body["alerts"][0]
    assert alert["src"] == SCANNER
    types = [e["type"] for e in alert["evidence"]]
    assert types == ["port_scan"]  # one item per type, from model and window together
    assert alert["decision"]["distinct_dst_ports"] == 25
    assert alert["verifier"]["ok"] is True
    assert alert["remediation_ok"] is True
    assert {r["code"] for r in alert["remediation"]} <= NET_CODES
    assert "BLOCK_SOURCE_IP" in {r["code"] for r in alert["remediation"]}
    assert body["timings_ms"]["total"] >= 0
    assert body["flows_per_s"] > 0
    assert "device" in body and "wall_ms" in body


def test_benign_only_has_no_alert(tmp_path):
    status, body = _service(tmp_path).analyse_flows(json.dumps({"flows": benign_flows()}).encode())
    assert status == 200
    assert body["alerts"] == []
    assert body["decision"]["state"] == "SAFE"
    assert body["watchlist"] == []
    assert body["decision"]["risk"] == "LOW"


def test_pcap_body_is_parsed_and_scored(tmp_path):
    status, body = _service(tmp_path).analyse_flows(_pcap_bytes(range(1, 31)))
    assert status == 200
    assert body["flow_source"] == "pcap_stdlib"
    assert body["flows_processed"] == 30
    assert body["decision"]["alerts"] == 1
    assert body["alerts"][0]["src"] == SCANNER


def test_http_routes_and_limits(tmp_path, http_server):
    base = http_server(_service(tmp_path))
    status, body = _post(
        base, "/analyze_flows", _pcap_bytes(range(1, 26)), "application/vnd.tcpdump.pcap"
    )
    assert status == 200 and body["decision"]["alerts"] == 1
    status, body = _post(base, "/analyze_flows", b'{"flows": []}', "application/json")
    assert status == 400
    status, body = _post(base, "/analyze_flows", b"not a pcap or json", "application/json")
    assert status == 400
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", int(base.rsplit(":", 1)[1]), timeout=10)
    conn.putrequest("POST", "/analyze_flows")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
    conn.endheaders()  # the server refuses on the header alone, so no body is sent
    assert conn.getresponse().status == 413
    conn.close()
    req = urllib.request.Request(base + "/analyze_flows", data=b"", method="POST")
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req, timeout=10)
    assert err.value.code == 411


def test_cooling_refusal_is_503_and_nothing_is_scored(tmp_path, http_server):
    base = http_server(_service(tmp_path, temp=80.0))
    status, body = _post(
        base, "/analyze_flows", json.dumps({"flows": scan_flows()}).encode(), "application/json"
    )
    assert status == 503
    assert body["error"] == "cooling"
    assert body["temp_c"] == 80.0
    assert "flows_processed" not in body


def test_missing_network_model_is_503(tmp_path):
    service = _service(tmp_path)
    service.nids = None
    status, body = service.analyse_flows(b'{"flows": []}')
    assert status == 503 and "network model" in body["error"]


def test_flow_body_errors(tmp_path):
    with pytest.raises(ValueError):
        flows_from_body(b'{"flows": [{"src_ip": 5}]}', "auto", tmp_path)
    with pytest.raises(ValueError):
        flows_from_body(b"\xa1\xb2\xc3\xd4" + b"\x00" * 20, "auto", tmp_path)  # bad link type
    with pytest.raises(ValueError):
        flows_from_body(b"\x0a\x0d\x0d\x0a" + b"\x00" * 40, "auto", tmp_path)  # pcapng


def test_model_contract_mismatch_is_refused(tmp_path):
    onnx = pytest.importorskip("onnx")
    from rakshak.nids.pi_monitor import OnnxScorer

    src = (
        REPO / "kaggle" / "out" / "nids_baselines" / "output" / "student_cse_cic_ids2018_int8.onnx"
    )
    if not src.exists():
        pytest.skip("student ONNX not present")
    bad = tmp_path / "bad.onnx"
    shutil.copy(src, bad)
    model = onnx.load(str(bad))
    for prop in model.metadata_props:
        if prop.key == "rakshak.schema":
            prop.value = "netflow-v1-other"
    onnx.save(model, str(bad))
    with pytest.raises(ValueError, match="model schema"):
        OnnxScorer(str(bad))


# --- remediation ----------------------------------------------------------------


def _sms_package(actions):
    return EvidencePackage(
        risk=Risk.HIGH,
        confidence_state="scam",
        evidence=[],
        allowed_actions=list(actions),
        domain="sms",
    )


def _net_package(actions):
    return EvidencePackage(
        risk=Risk.HIGH,
        confidence_state="uncertain",
        evidence=[],
        allowed_actions=list(actions),
        domain="network",
    )


def test_sms_and_network_remediation_do_not_mix():
    sms = remediation_for(
        _sms_package([Action.DO_NOT_SHARE_OTP, Action.REPORT_1930_OR_CHAKSHU]),
        Action.DO_NOT_SHARE_OTP.value,
    )
    codes = [r["code"] for r in sms]
    assert codes == ["DO_NOT_SHARE_OTP", "REPORT_1930_OR_CHAKSHU"]
    assert any("sancharsaathi.gov.in/sfc" in s for s in sms[1]["steps"])

    net = remediation_for(
        _net_package([NetAction.BLOCK_SOURCE_IP, NetAction.CHANGE_PASSWORD]),
        NetAction.BLOCK_SOURCE_IP.value,
    )
    assert {r["code"] for r in net} <= NET_CODES
    assert not {r["code"] for r in net} & SMS_ONLY
    assert not any("1930" in s for r in net for s in r["steps"])


def test_remediation_rejects_cross_domain_codes():
    mixed = [{"code": "DO_NOT_CLICK", "title": "x", "steps": ["x"]}]
    assert remediation_problems(mixed, _net_package([NetAction.BLOCK_SOURCE_IP]))
    net_item = [{"code": "BLOCK_SOURCE_IP", "title": "x", "steps": ["x"]}]
    assert remediation_problems(net_item, _sms_package([Action.DO_NOT_CLICK]))
    with pytest.raises(ValueError):
        remediation_for(_net_package([NetAction.BLOCK_SOURCE_IP]), "DO_NOT_CLICK")


def test_hinglish_variant_differs_and_keeps_codes():
    pkg = _sms_package([Action.DO_NOT_SHARE_OTP, Action.CONTACT_BANK_OFFICIAL])
    en = remediation_for(pkg, Action.DO_NOT_SHARE_OTP.value, "en")
    hi = remediation_for(pkg, Action.DO_NOT_SHARE_OTP.value, "hinglish")
    assert [r["code"] for r in en] == [r["code"] for r in hi]
    assert en[0]["title"] != hi[0]["title"]
    assert en[0]["steps"] != hi[0]["steps"]
    assert SMS_CODES >= {r["code"] for r in hi}


def test_sms_pipeline_result_carries_remediation(tmp_path):
    class Det:
        def score(self, text):
            return 0.9

    class Llm:
        def generate(self, prompt, max_new_tokens=64, stop_when=None):
            raise AssertionError("no LLM call expected for this check")

    out = EdgeAnalyzer(Det(), Llm(), 0.35, detector_only_mode="quiet").analyse(
        "Hello friend, how are you doing today?"
    )
    assert isinstance(out["remediation"], list) and out["remediation"]
    assert out["verifier"]["remediation_ok"] is True
    assert {r["code"] for r in out["remediation"]} <= SMS_CODES


def test_uncertain_only_host_is_watchlisted_not_alerted(tmp_path):
    def mid_probs(x):
        row = np.zeros((1, len(FAMILIES)), dtype=np.float32)
        row[0, 0] = 0.999  # attack score 1e-3: above the uncertain threshold, below the alert one
        row[0, FAMILIES.index("dos_ddos")] = 1.0
        return row

    # Alert mode keeps the 2026-11-06 watchlist. Quiet mode is tested in test_nids_rules.py.
    nids = NidsAnalyzer(mid_probs, scratch=tmp_path / "u", flow_source="stdlib", model_mode="alert")
    out = nids.analyse_flows(benign_flows())
    assert out["alerts"] == []
    assert out["watchlist"] and out["watchlist"][0]["src"] == BENIGN
    assert out["watchlist"][0]["uncertain_flows"] == 3
