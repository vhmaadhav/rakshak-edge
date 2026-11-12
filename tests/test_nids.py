"""Network-intrusion module: schema, taxonomy, live featurizer, splits, evidence, verifier, monitor.

Everything runs on tiny synthetic fixtures. No real flows, no captures, no attack traffic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rakshak.nids import evidence as E
from rakshak.nids import live as L
from rakshak.nids import pi_monitor as M
from rakshak.nids import schema as S
from rakshak.nids import splits as SP
from rakshak.verifier import verify

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import nids_baselines as job  # noqa: E402

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_schema_is_frozen_and_identifier_free():
    assert len(S.SHARED_FEATURES) == 33
    assert len(set(S.SHARED_FEATURES)) == 33
    banned = {"IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
              "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "MIN_TTL", "MAX_TTL",
              "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "Label", "Attack"}
    assert not banned & set(S.SHARED_FEATURES)
    assert set(S.CAPS) == set(S.SHARED_FEATURES)
    assert set(S.LOG_FEATURES) == set(S.SHARED_FEATURES) - S.CATEGORICAL


def test_coerce_clips_logs_and_handles_nan():
    names = ["PROTOCOL", "IN_BYTES", "IN_PKTS"]
    raw = np.array([[6.0, np.nan, -5.0], [17.0, np.inf, 3.0]])
    out = S.coerce_matrix(raw, names)
    assert out.dtype == np.float32
    assert out[0, 0] == 6.0  # categorical stays raw
    assert out[0, 1] == 0.0  # NaN -> 0
    assert out[0, 2] == 0.0  # negative -> clipped to 0
    assert out[1, 1] == pytest.approx(np.log1p(S.CAPS["IN_BYTES"]), rel=1e-6)  # inf -> cap, log1p
    assert out[1, 2] == pytest.approx(np.log1p(3.0))


def test_coerce_frame_reports_missing_columns():
    df = pd.DataFrame({"PROTOCOL": [6.0], "IN_BYTES": [10.0]})
    with pytest.raises(KeyError, match="missing feature columns"):
        S.coerce_frame(df, ["PROTOCOL", "IN_BYTES", "OUT_BYTES"])
    out = S.coerce_frame(df, ["PROTOCOL", "IN_BYTES"])
    assert out.shape == (1, 2)


def test_coerce_matrix_rejects_wrong_width():
    with pytest.raises(ValueError, match="expected"):
        S.coerce_matrix(np.zeros((2, 3)), ["PROTOCOL", "IN_BYTES"])


# ---------------------------------------------------------------------------
# Label taxonomy
# ---------------------------------------------------------------------------

KNOWN_STRINGS = {
    # NF-UNSW-NB15-V2
    "Benign": "benign", "Normal": "benign", "Reconnaissance": "port_scan",
    "DoS": "dos_ddos", "Exploits": "injection_exploit", "Shellcode": "injection_exploit",
    "Backdoor": "botnet_c2", "Worms": "other", "Fuzzers": "other", "Generic": "other",
    "Analysis": "other",
    # NF-ToN-IoT-V2
    "normal": "benign", "scanning": "port_scan", "password": "brute_force", "ddos": "dos_ddos",
    "dos": "dos_ddos", "injection": "injection_exploit", "xss": "injection_exploit",
    "mitm": "other", "ransomware": "other", "backdoor": "botnet_c2",
    # NF-BoT-IoT-V2
    "DDoS": "dos_ddos", "Theft": "exfil_theft",
    # NF-CSE-CIC-IDS2018-V2 (CICFlowMeter label strings)
    "Bot": "botnet_c2", "Brute Force -Web": "brute_force", "Brute Force -XSS": "injection_exploit",
    "DDOS attack-HOIC": "dos_ddos", "DDOS attack-LOIC-UDP": "dos_ddos",
    "DDoS attacks-LOIC-HTTP": "dos_ddos", "DoS attacks-GoldenEye": "dos_ddos",
    "DoS attacks-Hulk": "dos_ddos", "DoS attacks-SlowHTTPTest": "dos_ddos",
    "DoS attacks-Slowloris": "dos_ddos", "FTP-BruteForce": "brute_force",
    "SSH-Bruteforce": "brute_force", "Infilteration": "other", "SQL Injection": "injection_exploit",
}


@pytest.mark.parametrize("name,family", sorted(KNOWN_STRINGS.items()))
def test_taxonomy_maps_known_label_strings(name, family):
    got, how = S.map_attack(name, 0 if family == "benign" else 1)
    assert got == family, (name, how)
    if family != "benign":
        assert how in {"exact", "rule"}, (name, how)


def test_taxonomy_families_are_all_reachable():
    reached = {S.map_attack(n, 1)[0] for n in KNOWN_STRINGS if n not in ("Benign", "Normal")}
    assert set(S.ATTACK_FAMILIES) <= reached | {"other"}
    assert set(S.ATTACK_FAMILIES) == set(S.FAMILIES[1:])
    assert S.FAMILIES[0] == "benign"


def test_taxonomy_label_and_unknown_rules():
    assert S.map_attack("DoS", 0) == ("benign", "label")  # label wins for benign rows
    assert S.map_attack("", 1) == ("other", "unknown")
    assert S.map_attack(None, 1) == ("other", "unknown")
    assert S.map_attack("Quantum Zap", 1) == ("other", "unknown")
    assert S.map_attack("Benign", 1) == ("other", "conflict")  # benign name on attack row


def test_plan_quotas_water_fill_keeps_rare_attacks_whole():
    keep, benign_keep = SP.plan_quotas({"rare": 10, "mid": 500, "big": 5000}, 4000, 2000)
    assert keep["rare"] == 1.0 and keep["mid"] == 1.0  # small strings are kept in full
    assert 0.0 < keep["big"] < 1.0  # only the largest string is cut
    kept = {k: keep[k] * c for k, c in {"rare": 10, "mid": 500, "big": 5000}.items()}
    assert sum(kept.values()) + benign_keep * 4000 == pytest.approx(2000, rel=1e-6)
    assert SP.plan_quotas({"a": 3}, 2, 100) == ({"a": 1.0}, 1.0)  # nothing to cut


# ---------------------------------------------------------------------------
# Live featurizer (nfstream-like dicts; nfstream itself is never imported)
# ---------------------------------------------------------------------------


def _flow(**overrides):
    base = {k: 0 for k in L.NFSTREAM_KEYS}
    base.update({
        "src_ip": "10.0.0.5", "dst_ip": "10.0.0.9", "src_port": 40000, "dst_port": 443,
        "protocol": 6, "bidirectional_duration_ms": 2000, "bidirectional_bytes": 4000,
        "bidirectional_min_ps": 40, "bidirectional_max_ps": 1500,
        "src2dst_duration_ms": 1500, "src2dst_packets": 10, "src2dst_bytes": 1000,
        "dst2src_duration_ms": 1800, "dst2src_packets": 20, "dst2src_bytes": 3000,
        "bidirectional_syn_packets": 1, "bidirectional_ack_packets": 30,
        "bidirectional_fin_packets": 1,
        "src2dst_syn_packets": 1, "src2dst_ack_packets": 10,
        "dst2src_ack_packets": 20,
    })
    base.update(overrides)
    return base


def test_featurize_flow_maps_nfstream_attributes():
    vec = L.featurize_flow(_flow())
    idx = S.SHARED_FEATURES.index
    assert vec.shape == (33,)
    assert vec[idx("PROTOCOL")] == 6
    assert vec[idx("IN_BYTES")] == 1000 and vec[idx("OUT_BYTES")] == 3000
    assert vec[idx("IN_PKTS")] == 10 and vec[idx("OUT_PKTS")] == 20
    assert vec[idx("FLOW_DURATION_MILLISECONDS")] == 2000
    assert vec[idx("DURATION_IN")] == 1500 and vec[idx("DURATION_OUT")] == 1800
    assert vec[idx("SRC_TO_DST_SECOND_BYTES")] == pytest.approx(500.0)
    assert vec[idx("DST_TO_SRC_SECOND_BYTES")] == pytest.approx(1500.0)
    assert vec[idx("SRC_TO_DST_AVG_THROUGHPUT")] == pytest.approx(4000.0)
    assert vec[idx("DST_TO_SRC_AVG_THROUGHPUT")] == pytest.approx(12000.0)
    assert vec[idx("MIN_IP_PKT_LEN")] == 40 and vec[idx("SHORTEST_FLOW_PKT")] == 40
    assert vec[idx("MAX_IP_PKT_LEN")] == 1500 and vec[idx("LONGEST_FLOW_PKT")] == 1500
    # Flags: bidirectional fin|syn|ack = 0x01|0x02|0x10; client syn|ack = 0x12; server ack = 0x10.
    assert vec[idx("TCP_FLAGS")] == 0x13
    assert vec[idx("CLIENT_TCP_FLAGS")] == 0x12
    assert vec[idx("SERVER_TCP_FLAGS")] == 0x10


def test_featurize_flow_unmappable_are_nan_and_listed():
    vec = L.featurize_flow(_flow())
    for name in L.UNMAPPABLE_LIVE:
        assert np.isnan(vec[S.SHARED_FEATURES.index(name)]), name
    assert len(L.LIVE_FEATURES) == 19
    assert set(L.UNMAPPABLE_LIVE) | set(L.LIVE_FEATURES) == set(S.SHARED_FEATURES)
    assert set(L.LIVE_COLUMNS.tolist()) == {S.SHARED_FEATURES.index(f) for f in L.LIVE_FEATURES}


def test_featurize_flow_missing_keys_and_zero_duration_do_not_raise():
    vec = L.featurize_flow({"protocol": 17})
    assert vec[S.SHARED_FEATURES.index("PROTOCOL")] == 17
    assert np.isnan(vec[S.SHARED_FEATURES.index("IN_BYTES")])  # absent key -> NaN, not zero
    assert np.isnan(vec[S.SHARED_FEATURES.index("TCP_FLAGS")])  # no flag counters -> NaN
    zero = L.featurize_flow(_flow(bidirectional_duration_ms=0, src2dst_duration_ms=0))
    assert zero[S.SHARED_FEATURES.index("SRC_TO_DST_SECOND_BYTES")] == 0.0


def test_flow_from_nfstream_reads_attributes_and_defaults_to_none():
    obj = SimpleNamespace(src_ip="10.0.0.1", protocol=6, bidirectional_bytes=12)
    flow = L.flow_from_nfstream(obj)
    assert flow["src_ip"] == "10.0.0.1" and flow["protocol"] == 6
    assert flow["dst_port"] is None
    assert set(flow) == set(L.NFSTREAM_KEYS)


def test_host_window_counts_distinct_ports_and_expires_old_entries():
    win = L.HostWindow(window_ms=1000)
    for i, port in enumerate(range(20, 30)):
        stats = win.observe("10.0.0.7", port, ts_ms=i * 10)
    assert stats.distinct_dst_ports == 10 and stats.flows == 10
    later = win.observe("10.0.0.7", 80, ts_ms=5000)  # all earlier entries are now outside
    assert later.distinct_dst_ports == 1 and later.flows == 1


# ---------------------------------------------------------------------------
# Splits: host disjointness for T4, stratification, grouping
# ---------------------------------------------------------------------------


def test_t4_group_split_has_no_host_in_two_parts():
    rng = np.random.default_rng(0)
    src = [f"10.0.0.{i}" for i in rng.integers(1, 60, 800)]
    dst = [f"10.1.0.{i}" for i in rng.integers(1, 60, 800)]
    groups, method = SP.host_groups(src, dst)
    assert method in {"component", "source"}
    parts = SP.group_split(groups, (0.7, 0.15, 0.15), seed=3)
    assert sum(len(p) for p in parts) == 800
    assert len(np.unique(np.concatenate(parts))) == 800
    group_sets = [set(groups[p].tolist()) for p in parts]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not group_sets[i] & group_sets[j], "a host group is in two parts"


def test_union_find_links_shared_endpoints_only():
    # Flows (a,b) and (b,c) share host b, so they are one group. (d,e) is separate.
    comp = SP.union_find_groups(["a", "b", "d"], ["b", "c", "e"])
    assert comp[0] == comp[1]
    assert comp[2] != comp[0]


def test_host_groups_falls_back_to_source_when_one_component_dominates():
    src = ["10.0.0.1"] * 10 + [f"10.0.1.{i}" for i in range(10)]
    dst = ["10.9.9.9"] * 20  # one shared server links every flow into one component
    groups, method = SP.host_groups(src, dst, max_share=0.4)
    assert method == "source"
    assert groups[0] == groups[9]  # same source address, same group


def test_random_split_is_stratified_and_partitions_rows():
    strata = np.repeat([0, 1, 2], [60, 30, 10])
    parts = SP.random_split(100, (0.7, 0.15, 0.15), seed=1, strata=strata)
    assert sum(len(p) for p in parts) == 100
    assert len(np.unique(np.concatenate(parts))) == 100
    for p in parts:
        assert set(np.unique(strata[p]).tolist()) == {0, 1, 2}
    with pytest.raises(ValueError):
        SP.random_split(10, (0.5, 0.2), seed=0)


# ---------------------------------------------------------------------------
# Evidence and actions
# ---------------------------------------------------------------------------


def test_family_to_evidence_mapping_and_brute_force_port():
    ev = E.evidence_for_flow("brute_force", 0.8, dst_port=22)
    assert [e.type for e in ev] == ["brute_force_ssh"]
    ev = E.evidence_for_flow("brute_force", 0.8, dst_port=3389)
    assert [e.type for e in ev] == ["brute_force_other"]
    assert E.evidence_for_flow("dos_ddos", 0.95)[0].type == "dos_flood"
    assert E.evidence_for_flow("botnet_c2", 0.7)[0].type == "botnet_beaconing"
    assert E.evidence_for_flow("exfil_theft", 0.7)[0].type == "data_exfil"
    assert E.evidence_for_flow("injection_exploit", 0.7)[0].type == "exploit_suspect"
    assert E.evidence_for_flow("other", 0.7)[0].type == "anomalous_flow"
    assert E.evidence_for_flow("benign", 0.05) == []
    with pytest.raises(ValueError):
        E.evidence_for_flow("nonsense", 0.5)


def test_window_port_scan_evidence_from_benign_flow():
    win = E.WindowStats(src_ip="10.0.0.7", distinct_dst_ports=25, flows=40)
    ev = E.evidence_for_flow("benign", 0.1, window=win)
    assert [(e.type, e.source) for e in ev] == [("port_scan", "window")]
    assert ev[0].weight == pytest.approx(0.9)


def test_net_action_map_has_no_sms_only_codes():
    sms_only = {"DO_NOT_CLICK", "DO_NOT_SHARE_OTP", "REPORT_1930_OR_CHAKSHU", "BLOCK_SENDER",
                "CONTACT_BANK_OFFICIAL", "OPEN_OFFICIAL_APP"}
    used = {a.value for acts in E.NET_ACTION_MAP.values() for a in acts}
    assert not used & sms_only
    assert set(E.NET_ACTION_MAP) == set(E.NET_EVIDENCE_TYPES)
    assert {"IGNORE", "VERIFY_INDEPENDENTLY"} <= {a.value for a in E.NetAction}


def test_build_package_risk_states_and_actions():
    high = E.evidence_for_flow("port_scan", 0.95)
    pkg = E.build_network_package(high, "scam")
    assert pkg.risk.value == "HIGH" and pkg.domain == "network"
    assert {a.value for a in pkg.allowed_actions} >= {"BLOCK_SOURCE_IP", "CHECK_ROUTER_FIREWALL"}
    low = E.build_network_package([], "benign")
    assert low.risk.value == "LOW" and [a.value for a in low.allowed_actions] == ["IGNORE"]
    unc = E.build_network_package([], "uncertain")
    assert unc.risk.value == "UNCERTAIN"
    assert [a.value for a in unc.allowed_actions] == ["VERIFY_INDEPENDENTLY"]
    with pytest.raises(ValueError):
        E.build_network_package([], "maybe")


def test_net_template_is_verifier_clean_and_compact():
    pkg = E.build_network_package(E.evidence_for_flow("dos_ddos", 0.9), "scam")
    tmpl = E.net_template(pkg)
    assert len(tmpl["summary"]) <= E.MAX_SUMMARY
    assert verify(json.dumps(tmpl), pkg).ok


# ---------------------------------------------------------------------------
# Verifier on network packages
# ---------------------------------------------------------------------------


def _network_pkg():
    return E.build_network_package(E.evidence_for_flow("dos_ddos", 0.9), "scam")


def test_verifier_accepts_a_valid_network_explanation():
    pkg = _network_pkg()
    ok = {"summary": "Heavy traffic is aimed at one target.", "reasons": ["N1"],
          "action": "RATE_LIMIT"}
    assert verify(json.dumps(ok), pkg).ok


def test_verifier_rejects_invented_evidence_action_and_domain():
    pkg = _network_pkg()
    bad_evidence = {"summary": "Traffic flood.", "reasons": ["N9"], "action": "RATE_LIMIT"}
    assert not verify(json.dumps(bad_evidence), pkg).ok
    bad_action = {"summary": "Traffic flood.", "reasons": ["N1"], "action": "BLOCK_SOURCE_IP"}
    assert not verify(json.dumps(bad_action), pkg).ok  # not in this package's allowed set
    bad_domain = {"summary": "Traffic flood from evil.example.", "reasons": ["N1"],
                  "action": "RATE_LIMIT"}
    assert not verify(json.dumps(bad_domain), pkg).ok


def test_verifier_rejects_sms_only_action_on_network_package_even_if_allowed():
    pkg = _network_pkg()
    pkg.allowed_actions = list(pkg.allowed_actions)
    from rakshak.evidence import Action

    pkg.allowed_actions.append(Action.DO_NOT_CLICK)
    out = {"summary": "Traffic flood.", "reasons": ["N1"], "action": "DO_NOT_CLICK"}
    result = verify(json.dumps(out), pkg)
    assert not result.ok
    assert any("SMS-only" in r for r in result.reasons)


def test_verifier_still_accepts_sms_packages_unchanged():
    from rakshak.evidence import Action, EvidencePackage, Risk

    sms = EvidencePackage(risk=Risk.LOW, confidence_state="benign", evidence=[],
                          allowed_actions=[Action.IGNORE])
    assert sms.domain == "sms"
    assert verify(json.dumps({"summary": "No scam signals.", "reasons": [],
                              "action": "IGNORE"}), sms).ok


# ---------------------------------------------------------------------------
# Pi monitor (pure per-flow path with a stub scorer)
# ---------------------------------------------------------------------------


def _probs(p_attack: float, top_family: int = 1):
    out = np.full(len(S.FAMILIES), 0.0)
    out[0] = 1.0 - p_attack
    out[top_family] = p_attack
    return out[None, :]


def test_assess_flow_alert_safe_and_uncertain():
    win = L.HostWindow(window_ms=10_000)
    flow = _flow(src_ip="10.0.0.5", dst_port=80)
    alert = M.assess_flow(flow, lambda x: _probs(0.95, top_family=3), win, alert=0.5,
                          uncertain=0.2)
    assert alert["state"] == "ALERT" and alert["family"] == "dos_ddos"
    assert alert["verified"] is True
    assert any(e["type"] == "dos_flood" for e in alert["evidence"])
    safe = M.assess_flow(flow, lambda x: _probs(0.01), win, alert=0.5, uncertain=0.2)
    assert safe["state"] == "SAFE" and safe["evidence"] == [] and safe["action"] == "IGNORE"
    unc = M.assess_flow(flow, lambda x: _probs(0.3, top_family=2), win, alert=0.5,
                        uncertain=0.2)
    assert unc["state"] == "UNCERTAIN" and unc["family"] == "brute_force"


def test_assess_flow_port_scan_window_raises_safe_to_uncertain():
    win = L.HostWindow(window_ms=10_000)
    state = None
    for port in range(1000, 1025):  # 25 distinct ports from one source, all scored benign
        record = M.assess_flow(_flow(src_ip="10.0.0.66", dst_port=port,
                                     bidirectional_first_seen_ms=port),
                               lambda x: _probs(0.0), win, alert=0.5, uncertain=0.2)
        state = record["state"]
    assert state == "UNCERTAIN"
    assert any(e["type"] == "port_scan" for e in record["evidence"])


def test_assess_flow_rejects_bad_thresholds_and_probability_width():
    win = L.HostWindow()
    with pytest.raises(ValueError):
        M.assess_flow(_flow(), lambda x: _probs(0.5), win, alert=0.2, uncertain=0.5)
    with pytest.raises(ValueError):
        M.assess_flow(_flow(), lambda x: np.zeros((1, 3)), win, alert=0.5, uncertain=0.2)


# ---------------------------------------------------------------------------
# Job helpers and an end-to-end run on synthetic parquet
# ---------------------------------------------------------------------------


def test_fold_first_layer_matches_scaled_forward():
    rng = np.random.default_rng(0)
    mean, scale = rng.normal(size=3), rng.uniform(0.5, 2.0, size=3)
    W, b = rng.normal(size=(3, 4)), rng.normal(size=4)
    x = rng.normal(size=(7, 3))
    W2, b2 = job.fold_first_layer(mean, scale, W, b)
    np.testing.assert_allclose(x @ W2 + b2, ((x - mean) / scale) @ W + b, rtol=1e-9, atol=1e-9)


def test_student_forward_returns_probabilities():
    rng = np.random.default_rng(1)
    layers = [(rng.normal(size=(5, 4)), rng.normal(size=4)),
              (rng.normal(size=(4, 8)), rng.normal(size=8))]
    probs = job.student_forward_np(rng.normal(size=(6, 5)), layers)
    assert probs.shape == (6, 8)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-9)


def _synthetic_flows(n: int, families: list[str], rng: np.random.Generator) -> pd.DataFrame:
    data = {f: rng.gamma(2.0, 50.0, n) for f in S.SHARED_FEATURES}
    data["PROTOCOL"] = rng.choice([6.0, 17.0], n)
    data["TCP_FLAGS"] = rng.integers(0, 256, n).astype(float)
    data["CLIENT_TCP_FLAGS"] = rng.integers(0, 256, n).astype(float)
    data["SERVER_TCP_FLAGS"] = rng.integers(0, 256, n).astype(float)
    attack = np.asarray(families, dtype=object)
    for i, fam in enumerate(sorted(set(families))):
        mask = attack == fam
        data["OUT_PKTS"] = np.where(mask, data["OUT_PKTS"] * (i + 3) * 4.0, data["OUT_PKTS"])
    data["IPV4_SRC_ADDR"] = [f"10.0.0.{i}" for i in rng.integers(1, 30, n)]
    data["IPV4_DST_ADDR"] = [f"10.1.0.{i}" for i in rng.integers(1, 30, n)]
    data["Attack"] = attack
    data["Label"] = (attack != "Benign").astype(np.int64)
    return pd.DataFrame(data)


@pytest.fixture()
def synthetic_root(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    unsw = ["Benign"] * 500 + ["Reconnaissance"] * 150 + ["DoS"] * 150 + ["Exploits"] * 100
    bot = ["Benign"] * 500 + ["DDoS"] * 120 + ["Reconnaissance"] * 120 + ["Theft"] * 60
    for slug, fname, families in (("nfunswnb15v2", "NF-UNSW-NB15-V2.parquet", unsw),
                                  ("nfbotiotv2", "NF-BoT-IoT-V2.parquet", bot)):
        folder = tmp_path / slug
        folder.mkdir()
        df = _synthetic_flows(len(families), families, rng)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), folder / fname)
        (folder / "NetFlow v2 Features.csv").write_text(
            "Feature\n" + "\n".join(S.SHARED_FEATURES[:10]) + "\n", encoding="utf-8")
    return tmp_path


class FakeCtx:
    def __init__(self, work: Path, argv: list[str]):
        self.results: dict = {}
        self.lines: list[str] = []
        self.argv = argv
        self.work = work
        self.saves = 0

    def log(self, msg):
        self.lines.append(str(msg))

    def save(self):
        self.saves += 1


def test_discover_matches_kaggle_mount_folders(synthetic_root):
    found = job.discover(synthetic_root, ["unsw_nb15", "bot_iot", "ton_iot"])
    assert set(found) == {"unsw_nb15", "bot_iot"}
    assert found["bot_iot"].name == "NF-BoT-IoT-V2.parquet"


def test_sampling_is_label_stratified_and_t4_is_host_disjoint(synthetic_root):
    path = job.discover(synthetic_root, ["unsw_nb15"])["unsw_nb15"]
    ds = job.sample_dataset(path, "unsw_nb15", max_rows=300, seed=0)
    assert ds.X.shape[1] == len(S.SHARED_FEATURES)
    assert ds.X.dtype == np.float32
    assert 0 < ds.y.sum() < len(ds.y)
    assert ds.info["duplicates_removed"] == 0
    parts = SP.group_split(ds.groups, job.FRACTIONS, seed=0)
    sets = [set(ds.groups[p].tolist()) for p in parts]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]


def test_end_to_end_job_on_tiny_synthetic_data(synthetic_root, monkeypatch):
    # The tests check result structure, not accuracy, so small ensembles are enough.
    monkeypatch.setattr(job, "LGBM_N_ESTIMATORS", 30)
    monkeypatch.setattr(job, "IFOREST_N_ESTIMATORS", 20)
    argv = ["--data-root", str(synthetic_root), "--datasets", "unsw_nb15", "bot_iot",
            "--max-rows", "400", "--models", "lightgbm", "logreg",
            "--protocols", "T0", "T1", "T2", "T4", "--t2-cross-models", "lightgbm",
            "--n-boot", "5", "--no-student"]
    ctx = FakeCtx(synthetic_root / "out", argv)
    status = job.run(ctx)
    assert status in (True, "ok", "partial")
    results = ctx.results
    assert set(results["datasets"]) == {"unsw_nb15", "bot_iot"}
    assert results["datasets"]["unsw_nb15"]["feature_csv"]["file"] == "NetFlow v2 Features.csv"
    assert "T0" in results["protocols"] and "T1" in results["protocols"]
    assert "unsw_nb15->bot_iot" in results["protocols"]["T1"]
    assert "lightgbm" in results["protocols"]["T1"]["unsw_nb15->bot_iot"]
    t0 = results["protocols"]["T0"]["unsw_nb15"]["lightgbm"]
    assert "bootstrap" in t0 and "family" in t0
    report = (synthetic_root / "out" / "report.md").read_text(encoding="utf-8")
    assert "T1 cross-dataset" in report and "Feature schema" in report
    assert "student" not in results or results.get("student") is None


def test_student_skips_cleanly_when_onnx_is_missing(synthetic_root, monkeypatch):
    monkeypatch.setattr(job.importlib.util, "find_spec",
                        lambda name: None if name in {"onnx", "onnxruntime"} else object())
    assert "skipped" in job.student_available()


def test_render_report_handles_empty_results():
    text = job.render_report({"meta": {"schema": S.SCHEMA_VERSION}, "schema": {}})
    assert text.startswith("# Network-IDS baselines")


# ---------------------------------------------------------------------------
# Column resolution: optional host columns, explicit aliases, loud failures
# ---------------------------------------------------------------------------


def test_resolve_columns_exact_alias_missing_and_ambiguous():
    matched, missing = S.resolve_columns(["in_bytes", "Label", "x"], ["IN_BYTES", "Label", "OUT"])
    assert matched == {"IN_BYTES": "in_bytes", "Label": "Label"}
    assert missing == ["OUT"]
    with pytest.raises(ValueError, match="ambiguous"):
        S.resolve_columns(["in_bytes", "IN-BYTES"], ["IN_BYTES"])


def _write_nids_parquet(folder: Path, fname: str, families: list[str], rng, *,
                        drop=(), rename=None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    df = _synthetic_flows(len(families), families, rng)
    df = df.drop(columns=[c for c in drop if c in df.columns])
    if rename:
        df = df.rename(columns=rename)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), folder / fname)


def test_job_without_host_columns_skips_t4_and_invents_no_hosts(tmp_path):
    rng = np.random.default_rng(3)
    fams = ["Benign"] * 300 + ["Reconnaissance"] * 80 + ["DoS"] * 80
    _write_nids_parquet(tmp_path / "nfunswnb15v2", "NF-UNSW-NB15-V2.parquet", fams, rng,
                        drop=("IPV4_SRC_ADDR", "IPV4_DST_ADDR"))
    argv = ["--data-root", str(tmp_path), "--datasets", "unsw_nb15", "--max-rows", "300",
            "--models", "lightgbm", "--protocols", "T0", "T4", "--n-boot", "5", "--no-student"]
    ctx = FakeCtx(tmp_path / "out", argv)
    job.run(ctx)
    info = ctx.results["datasets"]["unsw_nb15"]
    assert info["host_columns"] is False
    assert info["host_groups"]["reason"] == "skipped: no host columns"
    assert "IPV4_SRC_ADDR" not in info["columns"]
    assert ctx.results["skipped"]["T4 unsw_nb15"] == "skipped: no host columns"
    assert not ctx.results["protocols"].get("T4")  # no T4 result without host columns
    assert "lightgbm" in ctx.results["protocols"]["T0"]["unsw_nb15"]
    assert "unsw_nb15" in ctx.results["parquet_columns"]
    bootstrap = ctx.results["protocols"]["T0"]["unsw_nb15"]["lightgbm"]["bootstrap"]
    assert bootstrap["method"] == "iid_stratified"  # no clusters without hosts


def test_missing_feature_column_fails_loudly_with_full_list(tmp_path):
    rng = np.random.default_rng(4)
    fams = ["Benign"] * 50 + ["DoS"] * 50
    _write_nids_parquet(tmp_path / "d", "a.parquet", fams, rng, drop=("OUT_PKTS", "DNS_TTL_ANSWER"))
    with pytest.raises(KeyError) as err:
        job.sample_dataset(tmp_path / "d" / "a.parquet", "unsw_nb15", 100, 0)
    text = str(err.value)
    assert "OUT_PKTS" in text and "DNS_TTL_ANSWER" in text
    assert "parquet columns are" in text


def test_lowercase_feature_names_are_mapped_and_recorded(tmp_path):
    rng = np.random.default_rng(5)
    fams = ["Benign"] * 60 + ["DoS"] * 60
    _write_nids_parquet(tmp_path / "d", "a.parquet", fams, rng,
                        rename={"IN_BYTES": "in_bytes", "label": "label"})
    ds = job.sample_dataset(tmp_path / "d" / "a.parquet", "unsw_nb15", 500, 0)
    assert ds.info["column_aliases"]["IN_BYTES"] == "in_bytes"
    assert ds.info["host_columns"] is True
    assert ds.X.shape[1] == len(S.SHARED_FEATURES)


# ---------------------------------------------------------------------------
# Home-baseline calibration, LODO (T5), benign-only detectors, ONNX round-trip
# ---------------------------------------------------------------------------

from rakshak.nids import baseline as B  # noqa: E402

CAT = np.asarray([f in S.CATEGORICAL for f in S.SHARED_FEATURES], dtype=bool)
TAG_STEP = 10_000


def _fake_dataset(key: str, n: int, tag: int, rng: np.random.Generator) -> job.Dataset:
    """Synthetic dataset whose column 0 is a row identity (tag + position), so tests can tell which
    dataset and which row a split holds. Labels and families are random."""
    y = (rng.random(n) < 0.4).astype(np.int64)
    fam_idx = np.where(y == 0, 0, rng.integers(1, len(S.FAMILIES), n)).astype(np.int64)
    X = rng.gamma(2.0, 5.0, size=(n, len(S.SHARED_FEATURES))).astype(np.float32)
    X[:, 0] = tag + np.arange(n, dtype=np.float32)
    fam = np.asarray([S.FAMILIES[i] for i in fam_idx], dtype=object)
    return job.Dataset(key, X, y, fam_idx, fam, None, {})


def _partitions(datasets: dict, seed: int = 0) -> dict:
    return {key: {"T0": dict(zip(("tr", "ca", "te"),
                                 SP.random_split(len(ds.y), job.FRACTIONS, seed,
                                                 strata=ds.fam_idx), strict=True))}
            for key, ds in datasets.items()}


def test_calibration_slice_is_disjoint_and_removed_from_test():
    n, k = 500, 80
    head, rest = B.calibration_slice(n, k, seed=3)
    assert len(head) == k and len(rest) == n - k
    assert set(head.tolist()) | set(rest.tolist()) == set(range(n))
    assert not set(head.tolist()) & set(rest.tolist())
    assert np.array_equal(head, B.unlabelled_head(n, k, 3))
    with pytest.raises(ValueError, match="leaves no rows"):
        B.calibration_slice(10, 10, 0)
    # The test split built from ``rest`` holds no slice row (row identity in column 0).
    ds = _fake_dataset("t", n, 0, np.random.default_rng(0))
    test = job.take(ds, rest)
    assert not set(test.X[:, 0].tolist()) & set(ds.X[head, 0].tolist())


def test_calibration_reads_no_labels():
    import inspect

    for fn in (B.fit_site_baseline, B.calibration_slice, B.unlabelled_head):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"y", "label", "labels", "fam", "fam_idx"}, fn.__name__
    with pytest.raises(TypeError):  # labels cannot be passed in at all
        B.fit_site_baseline(np.zeros((5, len(CAT))), CAT, y=np.zeros(5))
    rng = np.random.default_rng(1)
    ds = _fake_dataset("t", 300, 0, rng)
    flipped = job.Dataset(ds.key, ds.X, 1 - ds.y, ds.fam_idx, ds.fam, None, {})
    head_a, _ = B.calibration_slice(len(ds.y), 40, 5)
    head_b, _ = B.calibration_slice(len(flipped.y), 40, 5)
    assert np.array_equal(head_a, head_b)
    t_a = B.fit_site_baseline(ds.X[head_a], CAT).transform(ds.X)
    t_b = B.fit_site_baseline(flipped.X[head_b], CAT).transform(flipped.X)
    np.testing.assert_array_equal(t_a, t_b)


def test_site_transform_maps_numeric_uniform_and_passes_categorical_through():
    rng = np.random.default_rng(2)
    base = rng.gamma(2.0, 5.0, size=(400, len(S.SHARED_FEATURES))).astype(np.float64)
    base[:, S.SHARED_FEATURES.index("PROTOCOL")] = rng.choice([6.0, 17.0], 400)
    site = B.fit_site_baseline(base, CAT)
    out = site.transform(base)
    j = S.SHARED_FEATURES.index("PROTOCOL")
    np.testing.assert_array_equal(out[:, j], base[:, j].astype(np.float32))  # categorical kept
    num = ~CAT
    assert out[:, num].min() >= 0.0 and out[:, num].max() <= 1.0
    far = base[:1].copy()
    far[:, num] *= 1000.0  # beyond the baseline maximum: saturates at 1
    assert np.allclose(site.transform(far)[:, num], 1.0, atol=1e-6)
    with pytest.raises(ValueError):
        B.fit_site_baseline(np.zeros((5, 3)), CAT)


def test_site_rank_and_fpr_at_recall():
    base = np.arange(10.0)
    ranks = B.site_rank(base, np.array([-1.0, 4.0, 9.0, 99.0]))
    np.testing.assert_allclose(ranks, [0.0, 0.5, 1.0, 1.0])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.95, 0.4, 0.7])
    assert B.fpr_at_recall(y, s, 1.0) == pytest.approx(0.25)  # the lowest-scored attack needs 1/4
    assert np.isnan(B.fpr_at_recall(np.zeros(4), s[:4], 0.9))


def test_benign_only_detectors_rank_outliers_higher():
    rng = np.random.default_rng(4)
    benign = rng.normal(size=(400, 5))
    outliers = benign[:20] + 8.0
    for det in (B.IsolationForestScorer(0).fit(benign), B.AutoencoderScorer(0).fit(benign)):
        assert det.score(outliers).mean() > det.score(benign[20:]).mean(), det.name
    with pytest.raises(ValueError, match="at least"):
        B.IsolationForestScorer(0).fit(benign[:10])


def test_lodo_balanced_rows_exclude_held_out_and_balance_sources():
    rows = {"a": np.arange(0, 100), "b": np.arange(100, 250), "c": np.arange(250, 400),
            "d": np.arange(400, 430)}
    out = SP.lodo_balanced_rows(rows, "d", seed=0)
    assert set(out) == {"a", "b", "c"}
    assert {len(v) for v in out.values()} == {100}  # min over the three sources
    for key, picked in out.items():
        assert set(picked.tolist()) <= set(rows[key].tolist())
        assert not set(picked.tolist()) & set(rows["d"].tolist())
    with pytest.raises(KeyError):
        SP.lodo_sources(["a", "b"], "z")
    with pytest.raises(ValueError, match="two source"):
        SP.lodo_balanced_rows({"a": np.arange(5), "d": np.arange(5)}, "d", 0)


def test_lodo_sites_never_put_held_out_rows_on_the_training_side():
    rng = np.random.default_rng(6)
    datasets = {k: _fake_dataset(k, n, tag, rng) for k, n, tag in
                (("a", 260, 0), ("b", 300, TAG_STEP), ("c", 220, 2 * TAG_STEP),
                 ("d", 240, 3 * TAG_STEP))}
    parts = _partitions(datasets, seed=1)
    for held in datasets:
        sites = job.lodo_sites(datasets, parts, held, k=30, seed=2)
        names = [s.name for s in sites]
        assert held not in names and len(names) == 3
        lo = datasets[held].X[0, 0]
        hi = lo + len(datasets[held].y)
        for site in sites:
            for part in (site.train.X[:, 0], site.cal.X[:, 0], site.base[:, 0]):
                assert not ((part >= lo) & (part < hi)).any(), (held, site.name)
        assert len({len(s.train.y) for s in sites}) == 1  # equal rows per source


def test_t1_slice_and_site_base_come_from_the_right_rows():
    rng = np.random.default_rng(8)
    ds = _fake_dataset("tgt", 200, 5 * TAG_STEP, rng)
    slice_rows, rest = B.calibration_slice(len(ds.y), 25, seed=0)
    test = job.take(ds, rest)
    base = ds.X[slice_rows]
    assert len(test.y) == 175 and len(base) == 25
    assert not set(test.X[:, 0].tolist()) & set(base[:, 0].tolist())


def test_student_onnx_round_trip_and_ir_pin(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    rng = np.random.default_rng(2)
    n_in = 19
    layers = [(rng.normal(size=(n_in, 64)) * 0.1, rng.normal(size=64) * 0.1),
              (rng.normal(size=(64, 32)) * 0.1, np.zeros(32)),
              (rng.normal(size=(32, 8)) * 0.1, np.zeros(8))]
    path = tmp_path / "student_fp32.onnx"
    job.build_student_onnx(layers, n_in, path, {"rakshak.schema": "test"})
    model = onnx.load(str(path))
    assert model.ir_version == job.ONNX_IR_VERSION <= 9
    assert model.opset_import[0].version == job.ONNX_OPSET
    X = rng.normal(size=(40, n_in)).astype(np.float32)
    ref = job.student_forward_np(X.astype(np.float64), layers)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(sess.run(None, {"features": X})[0], ref, atol=1e-5)
    assert sess.run(None, {"features": X[:1]})[0].shape == (1, 8)  # dynamic batch axis
    int8 = tmp_path / "student_int8.onnx"
    quantize_dynamic(str(path), str(int8), weight_type=QuantType.QInt8)
    job._ensure_metadata(int8, {"rakshak.schema": "test"})
    assert onnx.load(str(int8)).ir_version <= 9
    sess8 = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    got8 = sess8.run(None, {"features": X})[0]
    assert np.mean(got8.argmax(1) == ref.argmax(1)) > 0.9


def _three_dataset_root(tmp_path: Path) -> Path:
    rng = np.random.default_rng(11)
    specs = (("nfunswnb15v2", "NF-UNSW-NB15-V2.parquet",
              ["Benign"] * 300 + ["Reconnaissance"] * 90 + ["DoS"] * 90 + ["Exploits"] * 60),
             ("nfbotiotv2", "NF-BoT-IoT-V2.parquet",
              ["Benign"] * 300 + ["DDoS"] * 90 + ["Reconnaissance"] * 90 + ["Theft"] * 50),
             ("nftoniotv2", "NF-ToN-IoT-V2.parquet",
              ["Benign"] * 300 + ["scanning"] * 80 + ["password"] * 80 + ["xss"] * 60))
    for slug, fname, families in specs:
        _write_nids_parquet(tmp_path / slug, fname, families, rng)
    return tmp_path


def test_end_to_end_calibrated_lodo_and_hybrid_on_synthetic_data(tmp_path, monkeypatch):
    # The tests check result structure, not accuracy, so small ensembles are enough.
    monkeypatch.setattr(job, "LGBM_N_ESTIMATORS", 30)
    monkeypatch.setattr(job, "IFOREST_N_ESTIMATORS", 20)
    root = _three_dataset_root(tmp_path)
    argv = ["--data-root", str(root), "--max-rows", "400", "--models", "lightgbm",
            "--protocols", "T1", "T2", "T5", "--t2-cross-models", "lightgbm",
            "--calib-k", "40", "--calib-sensitivity", "70", "--n-boot", "3", "--no-student"]
    ctx = FakeCtx(root / "out", argv)
    job.run(ctx)
    prot = ctx.results["protocols"]
    assert ctx.results["meta"]["calibration"]["headline_K"] == 40
    pair = "unsw_nb15->bot_iot"
    entry = prot["T1_calib"][pair]["K40"]
    assert {"raw", "calib", "slice"} <= set(entry)
    calib = entry["calib"]["variants"]
    assert {"lightgbm", "isolation_forest", "autoencoder", "hybrid_isolation_forest",
            "hybrid_autoencoder"} <= set(calib)
    assert "fpr_at_recall" in calib["lightgbm"] and "family" in calib["isolation_forest"]
    assert entry["slice"]["n"] == 40
    assert calib["lightgbm"]["sizes"]["test"] == entry["calib"]["meta"]["test_rows"]
    assert set(prot["T5"]) == {"unsw_nb15", "bot_iot", "ton_iot"}
    for held, ks in prot["T5"].items():
        assert held not in ks["K40"]["sources"]
    assert "T2_anomaly" in prot and "T2_transfer" in prot
    report = (root / "out" / "report.md").read_text(encoding="utf-8")
    assert "T1 home-baseline calibration" in report and "T5 leave-one-dataset-out" in report
    assert "K sensitivity" in report and "Calibration slices" in report
