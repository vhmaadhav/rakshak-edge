"""Tests for deploy/pi/rakshak_thermald.py (governor decisions on a fake sysfs tree)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pi_common as pc
import pytest
import rakshak_thermald as rt

MAX_KHZ = 2_400_000
FLOOR_KHZ = 1_500_000


def make_sysfs(root: Path, policies: tuple[str, ...] = ("policy0", "policy4")) -> Path:
    cpufreq = root / "devices" / "system" / "cpu" / "cpufreq"
    for name in policies:
        policy = cpufreq / name
        policy.mkdir(parents=True)
        (policy / "scaling_max_freq").write_text(f"{MAX_KHZ}\n")
        (policy / "cpuinfo_max_freq").write_text(f"{MAX_KHZ}\n")
        (policy / "scaling_governor").write_text("ondemand\n")
    zone = root / "class" / "thermal" / "thermal_zone0"
    zone.mkdir(parents=True)
    (zone / "temp").write_text("50000\n")
    return root


def read_max(root: Path, policy: str = "policy0") -> int:
    return int((root / "devices/system/cpu/cpufreq" / policy / "scaling_max_freq").read_text())


def make_daemon(root: Path, state: Path, *, dry_run: bool = False, ceiling: int = 2400):
    return rt.ThermalDaemon(
        sysfs_root=root,
        state_dir=state,
        ceiling_mhz=ceiling,
        dry_run=dry_run,
        read_throttled=lambda: "0x0",
    )


# -- pure governor -----------------------------------------------------------


def make_gov(start: int = MAX_KHZ, ceiling: int = MAX_KHZ) -> rt.Governor:
    return rt.Governor(start_khz=start, ceiling_khz=ceiling)


def test_steps_down_100mhz_at_68c():
    gov = make_gov()
    decision = gov.decide(0.0, 68.5)
    assert decision.action == "down"
    assert decision.cap_khz == 2_300_000


def test_holds_below_68c_and_above_60c():
    gov = make_gov()
    assert gov.decide(0.0, 67.9).action == "hold"
    assert gov.decide(10.0, 61.0).action == "hold"
    assert gov.cap_khz == MAX_KHZ


def test_hard_drop_to_floor_at_73c_ignores_step_gap():
    gov = make_gov()
    assert gov.decide(0.0, 70.0).action == "down"
    decision = gov.decide(1.0, 73.0)  # 1 s later, inside the 6 s gap
    assert decision.action == "hard_drop"
    assert decision.cap_khz == FLOOR_KHZ


def test_min_gap_between_ordinary_steps():
    gov = make_gov()
    assert gov.decide(0.0, 70.0).action == "down"
    assert gov.decide(2.0, 70.0).action == "hold"
    assert gov.decide(4.0, 70.0).action == "hold"
    decision = gov.decide(6.0, 70.0)
    assert decision.action == "down"
    assert decision.cap_khz == 2_200_000


def test_sustained_cool_steps_up_after_10s_then_again_after_another_10s():
    gov = make_gov(start=FLOOR_KHZ)
    assert gov.decide(0.0, 55.0).action == "hold"
    assert gov.decide(5.0, 55.0).action == "hold"
    up = gov.decide(10.0, 55.0)
    assert up.action == "up"
    assert up.cap_khz == 1_600_000
    assert gov.decide(15.0, 55.0).action == "hold"  # needs a fresh 10 s hold
    assert gov.decide(20.0, 55.0).action == "up"


def test_hold_band_resets_cool_timer():
    gov = make_gov(start=2_000_000)
    gov.decide(0.0, 55.0)
    gov.decide(5.0, 65.0)  # hysteresis band: resets the cool timer
    assert gov.decide(6.0, 55.0).action == "hold"
    assert gov.decide(12.0, 55.0).action == "hold"  # only 6 s since the reset
    up = gov.decide(16.0, 55.0)
    assert up.action == "up"
    assert up.cap_khz == 2_100_000


def test_never_steps_above_ceiling():
    gov = make_gov(start=2_300_000, ceiling=2_400_000)
    gov.decide(0.0, 50.0)
    assert gov.decide(10.0, 50.0).cap_khz == 2_400_000
    later = gov.decide(30.0, 50.0)
    assert later.action == "hold"
    assert gov.cap_khz == 2_400_000


def test_never_steps_below_floor():
    gov = make_gov()
    for t in range(0, 60, 6):
        gov.decide(float(t), 85.0)
    assert gov.cap_khz == FLOOR_KHZ
    assert gov.decide(100.0, 85.0).action == "hold"


def test_start_cap_clamped_into_range():
    gov = rt.Governor(start_khz=3_000_000, ceiling_khz=MAX_KHZ)
    assert gov.cap_khz == MAX_KHZ


def test_fail_safe_drops_to_floor():
    gov = make_gov()
    decision = gov.fail_safe(0.0, "sensor unreadable")
    assert decision.action == "fail_safe"
    assert decision.cap_khz == FLOOR_KHZ


# -- daemon on a fake sysfs tree ----------------------------------------------


def test_daemon_steps_down_and_writes_every_policy(tmp_path):
    root = make_sysfs(tmp_path / "sys")
    daemon = make_daemon(root, tmp_path / "state")
    daemon.start(wall=0.0)
    daemon.tick(0.0, 70.0, wall=0.0)
    assert read_max(root, "policy0") == 2_300_000
    assert read_max(root, "policy4") == 2_300_000


def test_hard_drop_writes_floor(tmp_path):
    root = make_sysfs(tmp_path / "sys")
    daemon = make_daemon(root, tmp_path / "state")
    daemon.start(wall=0.0)
    daemon.tick(0.0, 80.0, wall=0.0)
    assert read_max(root, "policy0") == FLOOR_KHZ
    assert read_max(root, "policy4") == FLOOR_KHZ


def test_moving_average_ignores_single_spike(tmp_path):
    root = make_sysfs(tmp_path / "sys")
    (root / "devices/system/cpu/cpufreq/policy0/scaling_max_freq").write_text("2000000\n")
    daemon = make_daemon(root, tmp_path / "state")
    daemon.start(wall=0.0)
    actions = [daemon.tick(t, temp, wall=t).action for t, temp in [(0, 50), (2, 50), (4, 80)]]
    assert "hard_drop" not in actions
    assert read_max(root, "policy0") == 2_000_000


def test_dry_run_writes_no_sysfs_but_logs_and_state(tmp_path):
    root = make_sysfs(tmp_path / "sys")
    state = tmp_path / "state"
    daemon = make_daemon(root, state, dry_run=True)
    daemon.start(wall=100.0)
    decision = daemon.tick(0.0, 80.0, wall=100.0)
    assert decision.action == "hard_drop"
    assert read_max(root, "policy0") == MAX_KHZ
    assert read_max(root, "policy4") == MAX_KHZ
    daemon.restore(wall=101.0)
    assert read_max(root, "policy0") == MAX_KHZ
    doc = json.loads((state / "thermal.json").read_text())
    assert doc["dry_run"] is True
    assert doc["cap_mhz"] == 1500
    events = (state / "thermal_events.log").read_text()
    assert "(dry-run)" in events


def test_state_json_shape_and_mode(tmp_path):
    root = make_sysfs(tmp_path / "sys")
    state = tmp_path / "state"
    daemon = make_daemon(root, state)
    daemon.start(wall=5.0)
    daemon.tick(0.0, 62.0, wall=7.0)
    path = state / "thermal.json"
    doc = json.loads(path.read_text())
    for key in (
        "ts",
        "temp_c",
        "cap_mhz",
        "ceiling_mhz",
        "last_action",
        "reason",
        "throttled",
    ):
        assert key in doc
    assert doc["ts"] == 7.0
    assert doc["temp_c"] == 62.0
    assert doc["cap_mhz"] == 2400
    assert doc["ceiling_mhz"] == 2400
    assert doc["throttled"] == "0x0"
    assert path.stat().st_mode & 0o777 == 0o644
    assert not list(state.glob(".thermal.json.*"))  # no temp files left behind


def test_restore_on_exit_puts_cpuinfo_max_back(tmp_path):
    root = make_sysfs(tmp_path / "sys")
    daemon = make_daemon(root, tmp_path / "state")
    daemon.start(wall=0.0)
    daemon.tick(0.0, 80.0, wall=0.0)
    assert read_max(root, "policy0") == FLOOR_KHZ
    daemon.restore(wall=1.0)
    assert read_max(root, "policy0") == MAX_KHZ
    assert read_max(root, "policy4") == MAX_KHZ
    doc = json.loads((tmp_path / "state" / "thermal.json").read_text())
    assert doc["running"] is False


def test_run_restores_even_when_stopped_before_loop(tmp_path):
    root = make_sysfs(tmp_path / "sys")
    (root / "devices/system/cpu/cpufreq/policy0/scaling_max_freq").write_text("1500000\n")
    daemon = make_daemon(root, tmp_path / "state")
    stop = threading.Event()
    stop.set()
    daemon.run(interval=0.01, stop=stop)
    assert read_max(root, "policy0") == MAX_KHZ


def test_unreadable_sensor_fails_safe_on_third_miss(tmp_path):
    root = make_sysfs(tmp_path / "sys")
    daemon = make_daemon(root, tmp_path / "state")
    daemon.start(wall=0.0)
    assert daemon.tick(0.0, None, wall=0.0).action == "hold"
    assert daemon.tick(2.0, None, wall=2.0).action == "hold"
    assert daemon.tick(4.0, None, wall=4.0).action == "fail_safe"
    assert read_max(root, "policy0") == FLOOR_KHZ


def test_events_log_keeps_newest_500_lines(tmp_path):
    log = tmp_path / "events.log"
    for i in range(600):
        pc.append_event(log, f"line {i}", keep=500)
    lines = log.read_text().splitlines()
    assert len(lines) == 500
    assert lines[0] == "line 100"
    assert lines[-1] == "line 599"


def test_missing_cpufreq_raises(tmp_path):
    (tmp_path / "sys").mkdir()
    with pytest.raises(RuntimeError):
        make_daemon(tmp_path / "sys", tmp_path / "state")


def test_policy_order_is_numeric(tmp_path):
    root = make_sysfs(tmp_path / "sys", policies=("policy10", "policy4", "policy0"))
    names = [p.name for p in pc.cpufreq_policies(root)]
    assert names == ["policy0", "policy4", "policy10"]


def test_restore_keeps_last_temperature_reading(tmp_path):
    root = make_sysfs(tmp_path / "sys")
    state = tmp_path / "state"
    daemon = make_daemon(root, state)
    daemon.start(wall=0.0)
    daemon.tick(0.0, 62.0, wall=1.0)
    daemon.restore(wall=2.0)
    doc = json.loads((state / "thermal.json").read_text())
    assert doc["temp_c"] == 62.0
    assert doc["avg_temp_c"] == 62.0
    assert doc["running"] is False
