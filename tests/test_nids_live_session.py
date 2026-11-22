"""Live NIDS session: windows, per-host state across windows, bounds, alert-only retention, metrics.

Every flow here is a synthetic dict in the nfstream attribute shape (``live.NFSTREAM_KEYS``). No
nfstream, no root, no interface. The model is a fake callable, so the tests run without onnxruntime.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from rakshak.nids import host_rules as R
from rakshak.nids.live import NFSTREAM_KEYS
from rakshak.nids.live_session import (
    DNS_SUSPICIOUS_SENTINEL,
    RETAINED_KEYS,
    HostAliaser,
    LiveConfig,
    LiveSession,
    drive,
    load_flows_json,
    run_session,
    sanitize_flow,
    write_summary,
)
from rakshak.nids.schema import FAMILIES

T0 = 1_700_000_000_000.0
ATT = "10.0.0.66"
HOST = "192.168.1.10"
SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "rakshak" / "nids"


def benign(x):
    p = np.zeros((1, len(FAMILIES)))
    p[0, 0] = 1.0
    return p


def attacker(x):
    p = np.zeros((1, len(FAMILIES)))
    p[0, 0] = 0.05
    p[0, FAMILIES.index("brute_force")] = 0.95
    return p


def flow(t_ms, *, src=ATT, dst=HOST, dport=22, proto=6, dur=100.0, size=600, syn=1, back=3,
         name=None):
    return {
        "src_ip": src,
        "dst_ip": dst,
        "src_port": 50000,
        "dst_port": dport,
        "protocol": proto,
        "bidirectional_first_seen_ms": T0 + t_ms,
        "bidirectional_duration_ms": dur,
        "bidirectional_bytes": size,
        "bidirectional_min_ps": 60,
        "bidirectional_max_ps": 1500,
        "bidirectional_syn_packets": syn,
        "src2dst_packets": 3,
        "src2dst_bytes": size,
        "src2dst_duration_ms": dur,
        "dst2src_packets": back,
        "dst2src_bytes": 0,
        "dst2src_duration_ms": dur,
        "requested_server_name": name,
    }


def ssh_burst(start_ms, count, step_ms=1.0, src=ATT):
    return [flow(start_ms + i * step_ms, src=src) for i in range(count)]


class FakeClock:
    """A wall clock the test moves by hand."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def make(probs=benign, clock=None, **cfg):
    alerts: list[dict] = []
    summaries: list[dict] = []
    session = LiveSession(
        probs,
        alert=0.5,
        uncertain=0.2,
        config=LiveConfig(**cfg),
        aliaser=HostAliaser("test-salt"),
        on_alert=alerts.append,
        summary_sink=summaries.append,
        source="test",
        clock=clock or (lambda: 0.0),
    )
    return session, alerts, summaries


def types(record):
    return [e["type"] for e in record["evidence"]]


# --- windows and state across windows -----------------------------------------------------


def test_brute_force_spanning_two_windows_fires_only_when_both_are_seen():
    # 6 SSH flows in window 0 (0-5 s) and 4 in window 1 (5-10 s): 10 in 60 s, but 6 per window.
    session, alerts, _ = make(window_s=5, alert_cooldown_s=300)
    first = ssh_burst(0, 5, step_ms=1000.0) + [flow(4500.0)]
    for f in first:
        session.feed(f)
    session.feed(flow(5000.0))  # first flow of window 1: flushes window 0
    assert alerts == []  # 6 flows so far, below the 10 the rule needs
    for f in [flow(6000.0), flow(7000.0), flow(8000.0)]:
        session.feed(f)
    summary = session.close()
    assert len(alerts) == 1
    assert "brute_force_ssh" in types(alerts[0])
    assert alerts[0]["time_ms"] == T0 + 8000.0  # fired when window 1 was processed
    assert summary["windows_processed"] == 2


def test_repeat_of_same_alert_is_suppressed_within_cooldown():
    session, alerts, _ = make(window_s=5, alert_cooldown_s=300)
    for f in ssh_burst(0, 10, step_ms=500.0):
        session.feed(f)
    session.feed(flow(5000.0))
    session.feed(flow(6000.0))
    summary = session.close()
    assert len(alerts) == 1
    assert summary["alerts"]["suppressed_repeats"] >= 1


def test_late_flow_is_counted_and_scored_in_the_open_window():
    session, _, _ = make(window_s=1)
    session.feed(flow(1500.0, dport=443, back=3))
    session.feed(flow(200.0, dport=443, back=3))  # belongs to an earlier window
    summary = session.close()
    assert summary["flows"]["late"] == 1
    assert summary["flows"]["scored"] == 2


def test_dos_flood_in_one_window_is_detected():
    session, alerts, _ = make(max_history_per_host=200)
    flood = [flow(float(i), dport=80, syn=1, back=0) for i in range(250)]
    for f in flood:
        session.feed(f)
    session.close()
    assert any("dos_flood" in types(a) for a in alerts)


# --- bounds -----------------------------------------------------------------------------


def test_tracked_hosts_are_capped_and_the_oldest_are_evicted():
    session, _, _ = make(max_hosts=3)
    for i in range(10):
        session.feed(flow(float(i), src=f"10.1.0.{i}", dport=443, back=3))
    summary = session.close()
    assert session.tracked_hosts == 3
    assert summary["distinct_hosts"] == 10
    assert summary["bounds"]["hosts_evicted_cap"] == 7


def test_per_host_history_is_capped_and_the_loss_is_counted():
    session, _, _ = make(max_history_per_host=200)
    for f in [flow(float(i), dport=80, syn=1, back=0) for i in range(250)]:
        session.feed(f)
    summary = session.close()
    assert summary["flows"]["history_evicted"] == 50
    assert summary["flows"]["dropped_bounds"] == 50


def test_flows_beyond_the_window_cap_are_dropped_and_counted():
    session, _, _ = make(max_flows_per_window=5)
    for i in range(8):
        session.feed(flow(float(i), dport=443, back=3))
    summary = session.close()
    assert summary["flows"]["scored"] == 5
    assert summary["flows"]["dropped_window_cap"] == 3
    assert summary["flows"]["dropped_bounds"] == 3


def test_idle_hosts_are_forgotten_after_the_idle_limit():
    session, _, _ = make(window_s=1, idle_evict_s=10)
    session.feed(flow(0.0, src="10.2.0.1", dport=443, back=3))
    session.feed(flow(100_000.0, src="10.2.0.2", dport=443, back=3))
    summary = session.close()
    assert session.tracked_hosts == 1
    assert summary["bounds"]["hosts_evicted_idle"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_history_per_host": 50},  # below the largest rule threshold (200)
        {"window_s": 0},
        {"max_hosts": 0},
        {"mode": "block"},
        {"alert_cooldown_s": -1},
    ],
)
def test_invalid_config_is_refused(kwargs):
    with pytest.raises(ValueError):
        LiveConfig(**kwargs)


def test_invalid_flows_are_counted_not_scored():
    session, _, _ = make()
    session.feed({"bidirectional_first_seen_ms": T0})  # no src_ip
    session.feed({"src_ip": ATT})  # no start time
    summary = session.close()
    assert summary["flows"]["invalid"] == 2
    assert summary["flows"]["scored"] == 0


# --- alert-only, retention -------------------------------------------------------------


def test_alert_only_is_the_only_mode_and_nothing_blocks():
    assert LiveConfig().mode == "alert-only"
    source = (SRC_DIR / "live_session.py").read_text(encoding="utf-8")
    for forbidden in ("import socket", "subprocess", "os.system", "iptables", "nftables",
                      "import signal", "shutil"):
        assert forbidden not in source, forbidden


def test_sanitize_keeps_only_retained_keys_and_aliases_addresses():
    raw = flow(0.0, name="x" * 60 + ".tunnel.example")
    raw["src_port"] = 51515
    kept = sanitize_flow(raw, HostAliaser("s"))
    assert set(kept) <= RETAINED_KEYS
    assert "src_port" not in kept
    assert ATT not in json.dumps(kept)
    assert HOST not in json.dumps(kept)
    assert kept["src_ip"].startswith("h-")
    assert kept["requested_server_name"] == DNS_SUSPICIOUS_SENTINEL
    assert "tunnel.example" not in json.dumps(kept)


def test_benign_dns_name_is_not_kept_at_all():
    kept = sanitize_flow(flow(0.0, dport=53, proto=17, name="www.example.org"), HostAliaser("s"))
    assert kept["requested_server_name"] is None


def test_all_retained_keys_are_real_nfstream_keys_or_aliases():
    for key in RETAINED_KEYS:
        assert key in NFSTREAM_KEYS


def test_no_raw_address_or_dns_name_survives_in_state_alerts_or_summary():
    session, alerts, _ = make(window_s=5)
    for f in ssh_burst(0, 12, step_ms=300.0):
        session.feed(f)
    session.feed(flow(0.0, src="10.9.9.9", dport=53, proto=17,
                      name="y" * 70 + ".suspicious.example"))
    summary = session.close()
    blob = repr(session._hosts) + json.dumps(alerts) + json.dumps(summary)
    assert ATT not in blob
    assert "10.9.9.9" not in blob
    assert "suspicious.example" not in blob
    assert alerts and all(a["host"].startswith("h-") for a in alerts)


def test_dns_tunnel_is_detected_without_keeping_the_name():
    session, alerts, _ = make(window_s=60)
    for i in range(5):
        session.feed(flow(float(i), dport=53, proto=17, back=0, dur=1.0, size=120,
                          name="z" * 60 + f".t{i}.example"))
    session.close()
    assert any("dns_tunnel" in types(a) for a in alerts)
    assert "example" not in repr(session._hosts)


def test_aliases_are_stable_for_a_salt_and_differ_between_salts():
    assert HostAliaser("a").alias(ATT) == HostAliaser("a").alias(ATT)
    assert HostAliaser("a").alias(ATT) != HostAliaser("b").alias(ATT)
    assert HostAliaser("a").alias(None) is None


def test_rule_relevance_matches_the_rule_inputs():
    assert R.rule_relevant(flow(0.0, dport=22, dur=100.0, size=600))
    assert not R.rule_relevant(flow(0.0, dport=22, dur=60_000.0, size=600))  # not short
    assert R.rule_relevant(flow(0.0, dport=80, syn=1, back=0))  # SYN only
    assert not R.rule_relevant(flow(0.0, dport=443, syn=1, back=3))
    assert not R.rule_relevant({"src_ip": ATT})  # no time, no protocol


# --- model-only and benign -------------------------------------------------------------


def test_model_only_flags_are_advisories_not_alerts():
    session, alerts, _ = make(probs=attacker, window_s=60)
    for i in range(3):
        session.feed(flow(float(i), dport=443, back=3))
    summary = session.close()
    assert alerts == []
    assert summary["alerts"]["total"] == 0
    assert summary["alerts"]["model_only_advisories"] >= 1


def test_model_only_can_be_printed_on_request_but_is_not_counted():
    session, alerts, _ = make(probs=attacker, window_s=60, include_model_only=True)
    session.feed(flow(0.0, dport=443, back=3))
    summary = session.close()
    assert [a["kind"] for a in alerts] == ["model_only_advisory"]
    assert summary["alerts"]["total"] == 0


def test_benign_traffic_gives_no_alerts_and_zero_rate():
    session, alerts, _ = make(window_s=60)
    for i in range(50):
        session.feed(flow(i * 1000.0, src=f"10.3.0.{i % 5}", dport=443, back=3))
    summary = session.close()
    assert alerts == []
    assert summary["alerts"]["per_device_hour"] == 0.0
    assert summary["time_to_first_alert_s"] is None


# --- metrics ----------------------------------------------------------------------------


def test_summary_fields_are_consistent_and_serialisable(tmp_path):
    session, alerts, _ = make(window_s=5)
    for f in ssh_burst(0, 12, step_ms=400.0):
        session.feed(f)
    session.feed(flow(20_000.0, src="10.4.0.1", dport=443, back=3))
    summary = session.close()
    s = json.loads(json.dumps(summary))
    assert s["schema"].startswith("rakshak.nids.live_summary/")
    assert s["distinct_hosts"] == 2
    assert s["observed"]["duration_s"] == pytest.approx(20.0, abs=1e-3)
    # Host-active exposure: window 0 (ATT) and window 4 (the other host), 5 s each.
    assert s["device_hours"] == pytest.approx(2 * 5.0 / 3600.0, abs=1e-4)
    assert s["device_hours_upper_bound"] == pytest.approx(2 * 20.0 / 3600.0, abs=1e-4)
    assert "host-active" in s["device_hours_basis"]
    assert "overstates" in s["device_hours_upper_bound_basis"]
    total = s["alerts"]["total"]
    assert total == len(alerts) == 1
    assert s["alerts"]["by_severity"]["high"] == 1  # SSH brute force: rule weight 0.9
    # The summary rounds device_hours for display; the rate uses the unrounded exposure.
    assert s["alerts"]["per_device_hour"] == pytest.approx(total / (2 * 5.0 / 3600.0), rel=1e-6)
    assert s["time_to_first_alert_s"] == pytest.approx((alerts[0]["time_ms"] - T0) / 1000.0)
    assert s["windows_processed"] >= 2
    assert s["process"]["cpu_time_s"] >= 0.0
    assert s["process"]["wall_s"] >= 0.0
    assert ATT not in json.dumps(s)
    out = tmp_path / "summary.json"
    write_summary(out, s)
    assert json.loads(out.read_text(encoding="utf-8"))["alerts"]["total"] == total


def test_periodic_summary_is_written_during_the_run():
    ticks = iter(range(0, 10_000))
    session = LiveSession(
        benign,
        alert=0.5,
        uncertain=0.2,
        config=LiveConfig(window_s=1, summary_every_s=1.0),
        aliaser=HostAliaser("p"),
        summary_sink=lambda s: sinks.append(s),
        clock=lambda: float(next(ticks)),
    )
    sinks: list[dict] = []
    for i in range(6):
        session.feed(flow(i * 1000.0, dport=443, back=3))
    session.close()
    assert len(sinks) >= 2  # periodic writes plus the final one


def test_run_session_helper_and_max_flows():
    flows = [flow(i * 10.0, dport=443, back=3) for i in range(30)]
    summary = run_session(flows, benign, alert=0.5, uncertain=0.2, max_flows=10,
                          config=LiveConfig(window_s=60), aliaser=HostAliaser("m"))
    assert summary["flows"]["seen"] == 10


def test_flows_json_loader_accepts_both_shapes(tmp_path):
    a = tmp_path / "list.json"
    b = tmp_path / "obj.json"
    a.write_text(json.dumps([flow(0.0)]), encoding="utf-8")
    b.write_text(json.dumps({"flows": [flow(0.0)]}), encoding="utf-8")
    assert len(load_flows_json(a)) == 1
    assert len(load_flows_json(b)) == 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"nope": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_flows_json(bad)


# --- wall-clock window timer ------------------------------------------------------------


def test_open_window_closes_on_the_wall_clock_deadline_on_a_quiet_link():
    clock = FakeClock()
    session, alerts, _ = make(clock=clock, window_s=5, grace_s=1.0)
    for f in ssh_burst(0, 12, step_ms=100.0):
        session.feed(f)
    assert alerts == []  # window still open: no later flow has arrived
    clock.t = 5.9
    session.poll()
    assert alerts == []  # deadline is opening time + window_s + grace_s = 6.0
    clock.t = 6.0
    session.poll()
    assert len(alerts) == 1
    assert session.summary()["windows_processed"] == 1


def test_drive_closes_windows_while_the_source_is_silent():
    clock = FakeClock()
    session, alerts, _ = make(clock=clock, window_s=5, grace_s=1.0)
    seen: dict[str, int] = {}

    def stream():
        yield from ssh_burst(0, 12, step_ms=100.0)
        # Wait until the main thread has fed all 12 flows, then move the wall clock past the
        # deadline and go quiet. The main thread must close the window on its poll.
        for _ in range(500):
            if session.summary()["flows"]["seen"] >= 12:
                break
            time.sleep(0.01)
        clock.t = 10.0
        for _ in range(500):
            if alerts:
                break
            time.sleep(0.01)
        seen["alerts_while_silent"] = len(alerts)

    drive(session, stream(), poll_every_s=0.01)
    session.close()
    assert seen["alerts_while_silent"] == 1


def test_drive_reraises_a_source_error_in_the_caller():
    session, _, _ = make()

    def broken():
        yield flow(0.0, dport=443, back=3)
        raise OSError("interface went away")

    with pytest.raises(OSError, match="interface went away"):
        drive(session, broken(), poll_every_s=0.01)


def test_grace_must_not_be_negative():
    with pytest.raises(ValueError):
        LiveConfig(grace_s=-1)
