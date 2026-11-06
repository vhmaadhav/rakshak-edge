"""Tests for deploy/pi/pi_dashboard.py and the PMIC/throttle helpers in pi_common."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pi_common as pc
import pi_dashboard as dash
import pytest

# Real `vcgencmd pmic_read_adc` output from the Pi 5 (12 rail pairs plus EXT5V/BATT).
PMIC_SAMPLE = """\
   3V7_WL_SW_A current(0)=0.05757987A
   3V3_SYS_A current(1)=0.05270022A
   1V8_SYS_A current(2)=0.15712470A
  DDR_VDD2_A current(3)=0.00097593A
  DDR_VDDQ_A current(4)=0.00000000A
   1V1_SYS_A current(5)=0.18933040A
    0V8_SW_A current(6)=0.26935670A
  VDD_CORE_A current(7)=0.58938000A
   3V3_DAC_A current(17)=0.00042735A
   3V3_ADC_A current(18)=0.00091575A
   0V8_AON_A current(16)=0.00317460A
      HDMI_A current(22)=0.02454210A
 3V7_WL_SW_V volt(8)=3.69606400V
   3V3_SYS_V volt(9)=3.31096100V
   1V8_SYS_V volt(10)=1.79828900V
  DDR_VDD2_V volt(11)=1.10146400V
  DDR_VDDQ_V volt(12)=0.60695910V
   1V1_SYS_V volt(13)=1.09999900V
    0V8_SW_V volt(14)=0.79926660V
  VDD_CORE_V volt(15)=0.74971840V
   3V3_DAC_V volt(20)=3.30402600V
   3V3_ADC_V volt(21)=3.30036300V
   0V8_AON_V volt(19)=0.79853400V
      HDMI_V volt(23)=5.18714000V
     EXT5V_V volt(24)=5.16972000V
      BATT_V volt(25)=0.00940170V
"""

# The shorter sample from the task brief.
SPEC_SAMPLE = """\
   VDD_CORE_A current(7)=0.93765000A
   VDD_CORE_V volt(15)=0.85421160V
   EXT5V_V volt(24)=5.17642000V
"""

PID_STAT = "100 (python3) S 1 1 1 0 -1 4194560 100 0 0 0 {utime} {stime} 0 0 20 0 1 0 100 1 300\n"
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
CLK_TCK = os.sysconf("SC_CLK_TCK")


# -- pmic / throttle ----------------------------------------------------------


def test_parse_pmic_and_power_from_real_sample():
    rails = pc.parse_pmic_adc(PMIC_SAMPLE)
    assert rails["VDD_CORE_A"] == pytest.approx(0.58938)
    assert rails["EXT5V_V"] == pytest.approx(5.16972)
    assert pc.pmic_power_w(rails) == pytest.approx(1.671, abs=1e-3)


def test_power_from_brief_sample_uses_only_paired_rails():
    rails = pc.parse_pmic_adc(SPEC_SAMPLE)
    # EXT5V_V has no _A partner, so it must not be counted.
    assert pc.pmic_power_w(rails) == pytest.approx(0.801, abs=1e-3)
    assert rails["EXT5V_V"] == pytest.approx(5.17642)


def test_power_none_without_paired_rails():
    assert pc.pmic_power_w(pc.parse_pmic_adc("EXT5V_V volt(24)=5.1V\n")) is None
    assert pc.pmic_power_w({}) is None


def test_decode_throttled_0x50005():
    assert pc.decode_throttled("0x50005") == {
        "now": ["under-voltage", "throttled"],
        "occurred": ["under-voltage", "throttled"],
    }


def test_decode_throttled_edge_cases():
    assert pc.decode_throttled("0x0") == {"now": [], "occurred": []}
    assert pc.decode_throttled(None) == {"now": [], "occurred": []}
    all_now = pc.decode_throttled(0xF)
    assert all_now["now"] == ["under-voltage", "arm freq capped", "throttled", "soft temp limit"]
    assert all_now["occurred"] == []
    soft = pc.decode_throttled(1 << 19)
    assert soft == {"now": [], "occurred": ["soft temp limit"]}


def test_parse_throttled_text():
    assert pc.parse_throttled("throttled=0x50005\n") == "0x50005"
    assert pc.parse_throttled("garbage") is None
    assert pc.parse_throttled(None) is None


def test_run_vcgencmd_failure_is_graceful(monkeypatch):
    def boom(*_args, **_kwargs):
        raise FileNotFoundError("vcgencmd")

    monkeypatch.setattr(pc.subprocess, "run", boom)
    assert pc.run_vcgencmd(["get_throttled"]) is None
    assert pc.read_throttled_hex() is None


# -- /proc parsing -------------------------------------------------------------


def test_proc_stat_delta_math():
    prev = dash.parse_proc_stat("cpu  100 0 100 800 0 0 0 0\ncpu0 50 0 50 400 0 0 0 0\n")
    cur = dash.parse_proc_stat("cpu  150 0 150 850 0 0 0 0\ncpu0 75 0 75 425 0 0 0 0\n")
    pct = dash.cpu_percentages(prev, cur)
    # total: 1000 -> 1150 (+150); idle: 800 -> 850 (+50); busy 100/150
    assert pct["cpu"] == pytest.approx(66.7, abs=0.05)
    assert pct["cpu0"] == pytest.approx(66.7, abs=0.05)
    summary = dash.cpu_summary(pct)
    assert summary["total"] == pct["cpu"]
    assert summary["cores"] == [pct["cpu0"]]


def test_proc_stat_iowait_counts_as_idle_and_first_sample_is_none():
    prev = dash.parse_proc_stat("cpu  0 0 0 100 100 0 0 0\n")
    cur = dash.parse_proc_stat("cpu  10 0 0 100 190 0 0 0\n")
    # idle+iowait goes 200 -> 290 (+90); total 200 -> 300 (+100): busy 10%.
    assert dash.cpu_percentages(prev, cur)["cpu"] == pytest.approx(10.0)
    assert dash.cpu_percentages({}, cur) is None


def test_parse_pid_stat_handles_spaces_in_name():
    text = "1234 (my proc) S 1 1 1 0 -1 4194560 100 0 0 0 50 25 0 0 20 0 1 0 100 1000000 300"
    assert dash.parse_pid_stat(text) == ("my proc", 75, 300)
    assert dash.parse_pid_stat("junk") is None


def test_process_top_uses_tick_delta_over_wall_time():
    prev = {10: 100}
    cur = {10: ("py", 150, 5), 11: ("idle", 0, 1)}
    rows = dash.process_top(prev, cur, dt=1.0, clk_tck=100, page_size=4096)
    assert rows == [{"pid": 10, "name": "py", "cpu_pct": 50.0, "rss_mb": 0.0}]
    assert dash.process_top(prev, cur, dt=None, clk_tck=100, page_size=4096) == []


def test_net_rates_kbps():
    rates = dash.net_rates((1000, 2000, 10.0), 5096, 10192, 12.0)
    assert rates == {"rx_kBps": 2.0, "tx_kBps": 4.0}
    assert dash.net_rates(None, 1, 1, 1.0) is None


def test_parse_meminfo_and_mem_totals():
    text = "MemTotal: 4000000 kB\nMemAvailable: 3000000 kB\nSwapTotal: 2000 kB\nSwapFree: 1000 kB\n"
    mem = dash.build_mem(dash.parse_meminfo(text))
    assert mem == {"total_mb": 3906, "avail_mb": 2930, "used_mb": 977, "swap_used_mb": 1}


# -- collector and snapshot shape ---------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_fake_pi(root: Path, *, with_thermald: bool) -> dict[str, Path]:
    sysfs = root / "sys"
    proc = root / "proc"
    thermal = root / "rakshak"
    run = root / "run"
    cpufreq = sysfs / "devices/system/cpu/cpufreq/policy0"
    _write(cpufreq / "scaling_max_freq", "2400000\n")
    _write(cpufreq / "scaling_governor", "ondemand\n")
    _write(sysfs / "devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "1800000\n")
    _write(sysfs / "devices/system/cpu/cpu1/cpufreq/scaling_cur_freq", "1500000\n")
    _write(sysfs / "class/thermal/thermal_zone0/temp", "61500\n")
    _write(sysfs / "class/net/wlan0/statistics/rx_bytes", "1000\n")
    _write(sysfs / "class/net/wlan0/statistics/tx_bytes", "2000\n")
    _write(proc / "stat", "cpu  100 0 100 800 0 0 0 0\ncpu0 50 0 50 400 0 0 0 0\n")
    _write(proc / "meminfo", "MemTotal: 4000000 kB\nMemAvailable: 3000000 kB\n")
    _write(proc / "loadavg", "0.50 0.40 0.30 1/200 999\n")
    _write(proc / "uptime", "12345.67 999.0\n")
    _write(proc / "100/stat", PID_STAT.format(utime=50, stime=25))
    if with_thermald:
        _write(
            thermal / "thermal.json",
            json.dumps({"ts": 999.0, "cap_mhz": 2400, "reason": "steady", "running": True}),
        )
        _write(thermal / "thermal_events.log", "2026-11-06T10:00:00 start cap=2400\n")
    _write(
        run / "current_job.json",
        json.dumps({"state": "running", "name": "smoke", "log": str(run / "smoke-x.log")}),
    )
    _write(run / "smoke-x.log", "line one\nline two\n")
    return {"sys": sysfs, "proc": proc, "thermal": thermal, "run": run}


def test_collector_snapshot_shape_with_fake_inputs(tmp_path, monkeypatch):
    def fake_vcgencmd(args, timeout=2.0):
        if list(args) == ["get_throttled"]:
            return "throttled=0x50005\n"
        if list(args) == ["pmic_read_adc"]:
            return PMIC_SAMPLE
        return None

    monkeypatch.setattr(pc, "run_vcgencmd", fake_vcgencmd)
    paths = build_fake_pi(tmp_path, with_thermald=True)
    collector = dash.Collector(
        sysfs=paths["sys"],
        proc=paths["proc"],
        thermal_dir=paths["thermal"],
        run_dir=paths["run"],
    )
    first = collector.collect(mono=10.0, wall=1000.0)
    assert first["cpu_pct"] is None or first["cpu_pct"]["total"] == 0.0
    assert first["net"] == {"wlan0": None}

    # Second reading two seconds later: counters moved.
    _write(paths["sys"] / "class/net/wlan0/statistics/rx_bytes", "5096\n")
    _write(paths["sys"] / "class/net/wlan0/statistics/tx_bytes", "10192\n")
    _write(paths["proc"] / "stat", "cpu  150 0 150 850 0 0 0 0\ncpu0 75 0 75 425 0 0 0 0\n")
    _write(paths["proc"] / "100/stat", PID_STAT.format(utime=100, stime=25))
    sample = collector.collect(mono=12.0, wall=1002.0)

    expected_keys = {
        "ts",
        "temp_c",
        "freq_mhz",
        "cap_mhz",
        "governor",
        "throttled",
        "throttled_flags",
        "ext5v_v",
        "power_w",
        "vdd_core",
        "cpu_pct",
        "loadavg",
        "uptime_s",
        "mem",
        "swap_used_mb",
        "disk",
        "net",
        "top",
        "thermald",
        "thermald_events",
        "job",
        "job_log",
    }
    assert expected_keys <= set(sample)
    assert sample["temp_c"] == pytest.approx(61.5)
    assert sample["freq_mhz"] == [1800, 1500]
    assert sample["cap_mhz"] == 2400
    assert sample["governor"] == "ondemand"
    assert sample["throttled"] == "0x50005"
    assert sample["throttled_flags"]["now"] == ["under-voltage", "throttled"]
    assert sample["power_w"] == pytest.approx(1.671, abs=1e-3)
    assert sample["ext5v_v"] == pytest.approx(5.16972)
    assert sample["vdd_core"] == {"v": pytest.approx(0.74971840), "a": pytest.approx(0.58938)}
    assert sample["cpu_pct"]["total"] == pytest.approx(66.7, abs=0.05)
    assert sample["net"] == {"wlan0": {"rx_kBps": 2.0, "tx_kBps": 4.0}}
    assert sample["loadavg"] == [0.5, 0.4, 0.3]
    assert sample["uptime_s"] == 12346
    assert sample["mem"]["used_mb"] == 977
    assert sample["swap_used_mb"] == 0  # no swap fields in the fake meminfo
    top = sample["top"]
    assert top[0]["pid"] == 100 and top[0]["name"] == "python3"
    assert top[0]["cpu_pct"] == pytest.approx(round(100 * 50 / (CLK_TCK * 2), 1), abs=0.05)
    assert top[0]["rss_mb"] == pytest.approx(round(300 * PAGE_SIZE / (1024 * 1024), 1), abs=0.05)
    assert sample["thermald"]["cap_mhz"] == 2400
    assert sample["thermald"]["age_s"] == pytest.approx(3.0)
    assert sample["thermald_events"] == ["2026-11-06T10:00:00 start cap=2400"]
    assert sample["job"]["state"] == "running"
    assert sample["job_log"] == ["line one", "line two"]
    json.dumps(sample)  # must serialise cleanly


def test_thermald_reports_not_installed_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "run_vcgencmd", lambda *a, **k: None)
    paths = build_fake_pi(tmp_path, with_thermald=False)
    collector = dash.Collector(
        sysfs=paths["sys"], proc=paths["proc"], thermal_dir=paths["thermal"], run_dir=paths["run"]
    )
    sample = collector.collect(mono=1.0, wall=1.0)
    assert sample["thermald"] == "not installed"
    assert sample["thermald_events"] == []
    assert sample["power_w"] is None
    assert sample["throttled"] is None


def test_sampler_interval_follows_job_state_and_history_slims(tmp_path):
    state = {"job": None}

    def fake_collect():
        return {
            "ts": time.time(),
            "temp_c": 50.0,
            "cap_mhz": 2400,
            "freq_mhz": [1800, 1600],
            "power_w": 1.0,
            "ext5v_v": 5.1,
            "throttled": "0x0",
            "cpu_pct": {"total": 12.0, "cores": [12.0]},
            "mem": {"used_mb": 500},
            "job": state["job"],
        }

    sampler = dash.Sampler(None, collect=fake_collect, idle_s=5.0, job_s=1.0)
    sampler.tick()
    assert sampler.interval() == 5.0
    state["job"] = {"state": "paused"}
    sampler.tick()
    assert sampler.interval() == 1.0
    snap = sampler.snapshot()
    assert len(snap["history"]) == 2
    assert snap["history"][-1]["freq_avg_mhz"] == 1700
    assert snap["history"][-1]["job_state"] == "paused"
    assert snap["latest"]["temp_c"] == 50.0


def test_http_endpoints_and_sse_frame():
    def fake_collect():
        return {
            "ts": time.time(),
            "temp_c": 55.0,
            "cap_mhz": 2400,
            "freq_mhz": [1800],
            "power_w": 1.2,
            "ext5v_v": 5.1,
            "throttled": "0x0",
            "job": None,
            "cpu_pct": {"total": 10.0, "cores": [10.0]},
        }

    sampler = dash.Sampler(None, collect=fake_collect)
    sampler.tick()
    server = dash.DashboardServer(("127.0.0.1", 0), sampler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        html = urllib.request.urlopen(base + "/", timeout=5).read().decode("utf-8")
        assert "<title>Rakshak Pi 5</title>" in html
        assert "http://" not in html and "https://" not in html  # no external assets
        assert "<script src" not in html

        snap = json.loads(urllib.request.urlopen(base + "/api/snapshot", timeout=5).read())
        assert snap["latest"]["temp_c"] == 55.0
        assert snap["history"][0]["temp_c"] == 55.0

        resp = urllib.request.urlopen(base + "/api/stream", timeout=5)
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        line = resp.readline().decode("utf-8")
        assert line.startswith("data: ")
        assert json.loads(line[len("data: ") :])["temp_c"] == 55.0
        resp.close()

        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(base + "/nope", timeout=5)
        assert err.value.code == 404
    finally:
        sampler.stop()
        server.shutdown()
        server.server_close()
