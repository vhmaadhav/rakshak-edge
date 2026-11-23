"""Rules-first network policy (docs/NIDS_EDGE.md, 2026-11-07): rule boundaries, quiet vs alert mode,
model-only advisories, rule family over model family, DNS name parity (stdlib and nfstream), and the
SMS / network remediation separation. Every flow here is a dict; no pcap is sent or captured."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "deploy" / "pi" / "service"
for p in (SERVICE, REPO / "src", REPO / "deploy" / "pi" / "bench", REPO / "deploy" / "pi"):
    sys.path.insert(0, str(p))

from nids_pipeline import NidsAnalyzer  # noqa: E402

from rakshak.evidence import Risk  # noqa: E402
from rakshak.nids import host_rules as R  # noqa: E402
from rakshak.nids.evidence import (  # noqa: E402
    MODEL_ONLY_SUMMARY,
    NetAction,
    NetEvidence,
    build_model_only_package,
    build_network_package,
    net_template,
)
from rakshak.nids.host_alerts import host_alerts  # noqa: E402
from rakshak.nids.live import LIVE_FEATURES  # noqa: E402
from rakshak.nids.pcap_flows import dns_question_name, flows_from_pcap_bytes  # noqa: E402
from rakshak.nids.pi_monitor import FlowDecision  # noqa: E402
from rakshak.nids.schema import FAMILIES  # noqa: E402
from rakshak.remediation import remediation_problems  # noqa: E402

T0 = 1_700_000_000_000.0
ATT = "10.0.0.66"
HOST = "192.168.1.10"
IN_PKTS = LIVE_FEATURES.index("IN_PKTS")
OUT_PKTS = LIVE_FEATURES.index("OUT_PKTS")


def auth_flow(t_ms, dport=22, dur=100.0, size=600, dst=HOST, src=ATT):
    return {
        "src_ip": src,
        "dst_ip": dst,
        "src_port": 50000,
        "dst_port": dport,
        "protocol": 6,
        "bidirectional_first_seen_ms": T0 + t_ms,
        "bidirectional_duration_ms": dur,
        "bidirectional_bytes": size,
        "bidirectional_syn_packets": 1,
        "dst2src_packets": 3,
    }


def dns_flow(t_ms, name, src=ATT):
    return {
        "src_ip": src,
        "dst_ip": "192.168.1.1",
        "src_port": 51000,
        "dst_port": 53,
        "protocol": 17,
        "bidirectional_first_seen_ms": T0 + t_ms,
        "bidirectional_duration_ms": 1.0,
        "bidirectional_bytes": 120,
        "bidirectional_syn_packets": 0,
        "dst2src_packets": 0,
        "requested_server_name": name,
    }


def syn_flow(t_ms, dst=HOST, replied=False):
    return {
        "src_ip": ATT,
        "dst_ip": dst,
        "src_port": 40000,
        "dst_port": 80,
        "protocol": 6,
        "bidirectional_first_seen_ms": T0 + t_ms,
        "bidirectional_duration_ms": 0.1,
        "bidirectional_bytes": 60,
        "bidirectional_syn_packets": 1,
        "dst2src_packets": 1 if replied else 0,
    }


def kinds(hits):
    return sorted(h.type for h in hits)


# --- brute force: count, short flow, window ---------------------------------------------------


def test_ssh_nine_short_flows_do_not_fire_and_ten_do():
    assert R.evaluate([auth_flow(i * 1000) for i in range(9)]) == []
    hits = R.evaluate([auth_flow(i * 1000) for i in range(10)])
    assert kinds(hits) == ["brute_force_ssh"]
    assert hits[0].weight == 0.9 and hits[0].source == "window" and hits[0].dst == HOST
    assert len(hits[0].flow_index) == 10


def test_duration_and_bytes_limits_are_inclusive_at_the_boundary():
    at_limit = [auth_flow(i * 1000, dur=5000.0, size=10_240) for i in range(10)]
    assert kinds(R.evaluate(at_limit)) == ["brute_force_ssh"]
    over_duration = [auth_flow(i * 1000, dur=5001.0) for i in range(10)]
    assert R.evaluate(over_duration) == []
    over_bytes = [auth_flow(i * 1000, size=10_241) for i in range(10)]
    assert R.evaluate(over_bytes) == []


def test_brute_force_window_is_sixty_seconds_span():
    fits = [auth_flow(i * 6666) for i in range(10)]  # span 59 994 ms
    assert kinds(R.evaluate(fits)) == ["brute_force_ssh"]
    too_long = [auth_flow(i * 6667) for i in range(10)]  # span 60 003 ms
    assert R.evaluate(too_long) == []


def test_slow_attack_at_two_seconds_spacing_fires_in_sixty_seconds():
    slow = [auth_flow(i * 2000) for i in range(12)]
    hits = R.evaluate(slow)
    assert kinds(hits) == ["brute_force_ssh"] and len(hits[0].flow_index) == 12


def test_other_auth_ports_and_other_protocols_and_destinations():
    assert kinds(R.evaluate([auth_flow(i, dport=3389) for i in range(10)])) == ["brute_force_other"]
    assert R.evaluate([auth_flow(i, dport=80) for i in range(10)]) == []
    udp = [dict(auth_flow(i), protocol=17) for i in range(10)]
    assert R.evaluate(udp) == []
    split = [auth_flow(i, dst=HOST) for i in range(5)] + [
        auth_flow(i, dst="192.168.1.11") for i in range(5)
    ]
    assert R.evaluate(split) == []


def test_flow_missing_statistics_is_not_counted():
    no_bytes = [auth_flow(i * 1000) for i in range(10)]
    for f in no_bytes[:1]:
        f["bidirectional_bytes"] = None
    assert R.evaluate(no_bytes) == []


# --- dns tunnel: name tests and count ---------------------------------------------------------


def test_dns_name_tests_at_their_thresholds():
    assert not R.dns_name_is_suspicious("www.example.com")
    assert not R.dns_name_is_suspicious("a" * 39 + ".example.com")
    assert R.dns_name_is_suspicious("a" * 40 + ".example.com")
    assert not R.dns_name_is_suspicious(".".join(["x" * 33] * 3)[:99])  # 99 chars, labels 33
    assert R.dns_name_is_suspicious(".".join(["x" * 33] * 3))  # 101 chars
    assert R.dns_name_is_suspicious("abcdefghijklmnop.example.com")  # 16 distinct: 4.0 bits
    assert not R.dns_name_is_suspicious("aaaaaaaaaaaaaaaa.example.com")  # 0 bits
    assert not R.dns_name_is_suspicious(None)
    assert not R.dns_name_is_suspicious("")


def test_shannon_bits_values():
    assert R.shannon_bits("ab") == pytest.approx(1.0)
    assert R.shannon_bits("abcd") == pytest.approx(2.0)
    assert R.shannon_bits("") == 0.0


def test_dns_fires_at_five_suspicious_queries_only():
    long_name = "a" * 60 + ".tunnel.example.internal"
    four = [dns_flow(i * 1000, long_name) for i in range(4)]
    assert R.evaluate(four) == []
    five = [dns_flow(i * 1000, long_name) for i in range(5)]
    hits = R.evaluate(five)
    assert kinds(hits) == ["dns_tunnel"]
    assert hits[0].weight == 0.9 and hits[0].source == "rule" and hits[0].dst is None
    assert hits[0].dst_port == 53


def test_dns_ignores_normal_names_and_missing_names():
    normal = [dns_flow(i * 1000, f"host{i}.example.internal") for i in range(8)]
    assert R.evaluate(normal) == []
    missing = [dns_flow(i * 1000, None) for i in range(8)]
    assert R.evaluate(missing) == []


def test_dns_min_distinct_suppresses_repeated_cdn_lookalike_but_keeps_tunnel():
    cdn = "assets-prod-eu-west-1-7f3a9c2e1b0d4a6f8e5c.cdn.example.net"
    repeated = [dns_flow(i * 1000, cdn) for i in range(6)]
    # Pre-registered default (dns_min_distinct=1) still fires
    assert kinds(R.evaluate(repeated)) == ["dns_tunnel"]
    # Hardened distinct-qname guard suppresses repeated single-hostname CDN lookups
    assert R.evaluate(repeated, dns_min_distinct=3) == []
    # Real DNS tunneling with distinct encoded subdomains still fires
    tunnel = [
        dns_flow(i * 1000, f"chunk{i:02d}-7f3a9c2e1b0d4a6f8e5c9a0b1c2d3e4f.tunnel.example.net")
        for i in range(5)
    ]
    assert kinds(R.evaluate(tunnel, dns_min_distinct=3)) == ["dns_tunnel"]


# --- dos flood: syn-only, count ----------------------------------------------------------------


def test_syn_flood_boundary_is_two_hundred_unanswered_flows():
    assert R.evaluate([syn_flow(i * 10) for i in range(199)]) == []
    hits = R.evaluate([syn_flow(i * 10) for i in range(200)])
    assert kinds(hits) == ["dos_flood"]
    assert hits[0].weight == 0.85 and hits[0].source == "window"


def test_answered_syns_do_not_count_as_flood():
    mixed = [syn_flow(i * 10) for i in range(150)] + [
        syn_flow(i * 10, replied=True) for i in range(60)
    ]
    assert R.evaluate(mixed) == []


# --- DNS name parity: stdlib reader and the nfstream-shaped field -----------------------------


def _dns_query_payload(name: str) -> bytes:
    header = struct.pack(">HHHHHH", 0x1234, 0x0100, 1, 0, 0, 0)
    labels = b"".join(bytes([len(x)]) + x.encode() for x in name.split("."))
    return header + labels + b"\x00" + struct.pack(">HH", 16, 1)


def _udp_frame(src, dst, sport, dport, payload):
    eth = b"\x02" * 6 + b"\x02" * 6 + b"\x08\x00"
    a = bytes(map(int, src.split(".")))
    b = bytes(map(int, dst.split(".")))
    udp = struct.pack(">HHHH", sport, dport, 8 + len(payload), 0) + payload
    ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(udp), 0, 0, 64, 17, 0, a, b)
    return eth + ip + udp


def _pcap(records):
    out = bytearray(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
    for ts_ms, frame in records:
        sec, ms = divmod(ts_ms, 1000)
        out += struct.pack("<IIII", int(sec), int(ms * 1000), len(frame), len(frame)) + frame
    return bytes(out)


def test_dns_question_name_parses_plain_labels_and_rejects_pointers():
    assert dns_question_name(_dns_query_payload("A.Example.COM")) == "a.example.com"
    assert dns_question_name(b"\x00" * 12) is None  # qdcount 0
    bad = bytearray(_dns_query_payload("ab.cd"))
    bad[12] = 0xC0  # a compression pointer in the question
    assert dns_question_name(bytes(bad)) is None
    assert dns_question_name(b"\x00" * 5) is None


def test_stdlib_pcap_gives_the_name_that_the_rule_reads():
    long_name = "b" * 45 + ".tunnel.example.internal"
    recs = [
        (
            1_700_000_000_000 + i * 1000,
            _udp_frame(ATT, "192.168.1.1", 52000 + i, 53, _dns_query_payload(long_name)),
        )
        for i in range(5)
    ]
    flows = flows_from_pcap_bytes(_pcap(recs))
    assert all(f["requested_server_name"] == long_name for f in flows)
    assert kinds(R.evaluate(flows)) == ["dns_tunnel"]


def test_nfstream_path_parity_when_nfstream_is_installed(tmp_path):
    nfstream = pytest.importorskip("nfstream")  # noqa: F841 - skipped on the laptop (not installed)
    from rakshak.nids.live import iter_nfstream

    long_name = "c" * 45 + ".tunnel.example.internal"
    recs = [
        (
            1_700_000_000_000 + i * 1000,
            _udp_frame(ATT, "192.168.1.1", 52100 + i, 53, _dns_query_payload(long_name)),
        )
        for i in range(5)
    ]
    path = tmp_path / "dns.pcap"
    path.write_bytes(_pcap(recs))
    std = {f["requested_server_name"] for f in flows_from_pcap_bytes(path.read_bytes())}
    nf = {f.get("requested_server_name") for f in iter_nfstream(str(path))}
    assert std == {long_name} and long_name in nf


# --- quiet vs alert mode, model-only advisories, rule family over model family ----------------


def _decision(src, dport, family, p, state, evidence, t=0):
    return FlowDecision(
        state=state, p_attack=p, family=family, src=src, dst=HOST, dst_port=dport, evidence=evidence
    )


def model_ev(etype, weight, family):
    return NetEvidence(id="N1", type=etype, weight=weight, family=family, source="model")


def test_model_only_host_is_uncertain_and_verify_only_in_quiet_mode():
    d = _decision(
        "10.9.9.9", 443, "dos_ddos", 0.9, "scam", [model_ev("dos_flood", 0.9, "dos_ddos")]
    )
    alerts = host_alerts([d], [], mode="quiet")
    assert len(alerts) == 1
    a = alerts[0]
    assert a.path == "model_only_quiet" and a.is_model_only
    assert a.state == "uncertain" and a.family == "model_only"
    assert a.package.risk is Risk.UNCERTAIN
    assert [x.value for x in a.package.allowed_actions] == ["VERIFY_INDEPENDENTLY"]
    assert a.template["action"] == "VERIFY_INDEPENDENTLY"
    assert a.template["summary"] == MODEL_ONLY_SUMMARY
    assert a.verified
    assert a.model_family == "dos_ddos" and a.model_flagged_flows == 1


def test_same_host_in_alert_mode_keeps_the_model_alert():
    d = _decision(
        "10.9.9.9", 443, "dos_ddos", 0.9, "scam", [model_ev("dos_flood", 0.9, "dos_ddos")]
    )
    a = host_alerts([d], [], mode="alert")[0]
    assert a.path == "model" and not a.is_model_only
    assert a.state == "scam" and a.family == "dos_ddos"
    assert a.template["action"] == NetAction.RATE_LIMIT.value


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        host_alerts([], [], mode="loud")


def test_rule_family_overrides_model_family_and_its_action():
    # Ten short SSH flows; the model calls every one of them a dos flood.
    flows = [auth_flow(i * 1000) for i in range(10)]
    decisions = [
        _decision(ATT, 22, "dos_ddos", 0.9, "scam", [model_ev("dos_flood", 0.9, "dos_ddos")])
        for _ in flows
    ]
    hits = R.evaluate(flows)
    alerts = host_alerts(decisions, hits, mode="quiet")
    a = alerts[0]
    assert a.path == "rule" and a.family == "brute_force_ssh"
    assert a.template["action"] == "CHANGE_PASSWORD"
    allowed = {x.value for x in a.package.allowed_actions}
    assert {"CHANGE_PASSWORD", "DISABLE_PASSWORD_LOGIN"} <= allowed
    assert "RATE_LIMIT" not in allowed
    assert a.model_family == "dos_ddos" and a.model_flagged_flows == 10
    assert a.p_attack == pytest.approx(0.9)


def test_rule_host_state_and_risk_match_the_port_scan_rule():
    flows = [auth_flow(i * 1000) for i in range(10)]
    decisions = [_decision(ATT, 22, "benign", 0.0, "benign", []) for _ in flows]
    a = host_alerts(decisions, R.evaluate(flows), mode="quiet")[0]
    assert a.state == "uncertain" and a.package.risk is Risk.HIGH
    assert a.evidence[0].source == "window"


def test_quiet_mode_watchlist_is_not_used_and_uncertain_only_host_is_advisory():
    d = _decision(
        "10.9.9.8", 80, "port_scan", 0.01, "uncertain", [model_ev("port_scan", 0.01, "port_scan")]
    )
    alerts = host_alerts([d], [], mode="quiet")
    assert [a.path for a in alerts] == ["model_only_quiet"]


def test_remediation_is_verify_only_for_model_only_and_stays_in_network_codes():
    d = _decision(
        "10.9.9.9", 443, "dos_ddos", 0.9, "scam", [model_ev("dos_flood", 0.9, "dos_ddos")]
    )
    a = host_alerts([d], [], mode="quiet")[0]
    from rakshak.remediation import remediation_for

    rem = remediation_for(a.package, a.template["action"], "en")
    assert [r["code"] for r in rem] == ["VERIFY_INDEPENDENTLY"]
    assert remediation_problems(rem, a.package) == []


def test_net_template_uses_rule_map_order_for_rule_types():
    pkg = build_network_package(
        [
            NetEvidence(
                id="N1",
                type="brute_force_ssh",
                weight=0.9,
                family="brute_force_ssh",
                source="window",
            )
        ],
        "uncertain",
    )
    assert net_template(pkg)["action"] == "CHANGE_PASSWORD"
    mo = build_model_only_package([model_ev("dos_flood", 0.9, "dos_ddos")])
    assert mo.risk is Risk.UNCERTAIN


# --- end to end through NidsAnalyzer (fake student, no model file) ----------------------------


def _fake_probs(p_attack, family):
    def fn(x):
        row = np.zeros((1, len(FAMILIES)), dtype=np.float32)
        row[0, 0] = 1.0 - p_attack
        row[0, FAMILIES.index(family)] = 1.0
        return row

    return fn


def _analyse(tmp_path, flows, probs, mode):
    nids = NidsAnalyzer(
        probs,
        scratch=tmp_path / "u",
        flow_source="stdlib",
        model_mode=mode,
        alert=0.0015,
        uncertain=0.00055,
    )
    return nids.analyse_flows(flows)


def test_end_to_end_ssh_is_password_advice_in_quiet_mode(tmp_path):
    flows = [auth_flow(i * 2000) for i in range(12)]
    out = _analyse(tmp_path, flows, _fake_probs(0.9, "dos_ddos"), "quiet")
    assert out["model"]["mode"] == "quiet"
    assert out["decision"]["alerts"] == 1 and out["decision"]["model_only_hosts"] == 0
    alert = out["alerts"][0]
    assert alert["decision"]["family"] == "brute_force_ssh"
    assert alert["advisory"]["action"] == "CHANGE_PASSWORD"
    assert "RATE_LIMIT" not in alert["advisory"]["allowed_actions"]
    assert out["decision"]["model_flagged_flows"] == 12
    assert out["decision"]["model_p_attack_max"] == pytest.approx(0.9, abs=1e-3)
    assert out["decision"]["rule_hits"] == 1
    assert alert["verifier"]["ok"] is True
    assert out["alerts"][0]["remediation_ok"] is True


def test_end_to_end_model_only_is_uncertain_not_an_alert(tmp_path):
    flows = [
        dict(auth_flow(i * 500, dport=8080 + i, dst=f"192.168.1.{20 + i}"), src_ip=ATT)
        for i in range(4)
    ]
    out = _analyse(tmp_path, flows, _fake_probs(0.9, "dos_ddos"), "quiet")
    assert out["decision"]["alerts"] == 0
    assert out["decision"]["model_only_hosts"] == 1
    assert out["decision"]["state"] == "UNCERTAIN" and out["decision"]["risk"] == "UNCERTAIN"
    assert out["alerts"][0]["advisory"]["path"] == "model_only_quiet"
    assert [r["code"] for r in out["alerts"][0]["remediation"]] == ["VERIFY_INDEPENDENTLY"]


def test_end_to_end_alert_mode_still_raises_the_model_alert(tmp_path):
    flows = [
        dict(auth_flow(i * 500, dport=8080 + i, dst=f"192.168.1.{20 + i}"), src_ip=ATT)
        for i in range(4)
    ]
    out = _analyse(tmp_path, flows, _fake_probs(0.9, "dos_ddos"), "alert")
    assert out["model"]["mode"] == "alert"
    assert out["decision"]["alerts"] == 1
    assert out["decision"]["state"] == "ALERT"


def test_end_to_end_long_dns_names_are_dns_tunnel(tmp_path):
    long_name = "d" * 60 + ".tunnel.example.internal"
    recs = [
        (
            1_700_000_000_000 + i * 1000,
            _udp_frame(ATT, "192.168.1.1", 52200 + i, 53, _dns_query_payload(long_name)),
        )
        for i in range(5)
    ]
    nids = NidsAnalyzer(_fake_probs(0.0, "benign"), scratch=tmp_path / "u", flow_source="stdlib")
    out = nids.analyse_body(_pcap(recs))
    assert out["alerts"][0]["decision"]["family"] == "dns_tunnel"
    assert out["alerts"][0]["advisory"]["action"] == "ISOLATE_DEVICE"


def test_feature_list_unchanged_by_rules():
    # The rules read flow fields the model never sees: the model input width is unchanged.
    assert len(LIVE_FEATURES) == 19
