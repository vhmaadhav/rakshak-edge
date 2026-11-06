"""Tests for deploy/pi/thermal_run.py: guard decisions plus real child-process control."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
import thermal_run as tr


def _proc_state(pid: int) -> str:
    """Single-letter process state from /proc (T = stopped, S/R = running)."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return "X"
    return text[text.rfind(")") + 2]


def _probe() -> dict[str, object]:
    return {"cap_mhz": 2400, "freq_mhz": 1800, "throttled": "0x0", "power_w": 1.5}


def _job(tmp_path: Path, cmd: list[str], temps, **kwargs) -> tr.Job:
    seq = iter(temps)
    defaults = dict(
        name="t",
        cmd=cmd,
        guard=tr.Guard(75.0, 65.0, 80.0),
        run_dir=tmp_path,
        read_temp=lambda: next(seq, 60.0),
        probe=_probe,
        sample_s=0.05,
        kill_grace_s=2.0,
        echo=None,
    )
    defaults.update(kwargs)
    return tr.Job(**defaults)


# -- guard decisions ---------------------------------------------------------


def test_guard_pause_resume_kill_decisions():
    guard = tr.Guard(pause_c=75.0, resume_c=65.0, kill_c=80.0)
    assert guard.decide(70.0, paused=False) == "none"
    assert guard.decide(75.0, paused=False) == "pause"
    assert guard.decide(76.0, paused=True) == "none"  # already paused, not cool yet
    assert guard.decide(65.0, paused=True) == "resume"
    assert guard.decide(64.9, paused=True) == "resume"
    assert guard.decide(80.0, paused=False) == "kill"
    assert guard.decide(81.0, paused=True) == "kill"  # kill wins even when paused
    assert guard.decide(None, paused=False) == "none"  # unreadable sensor: no action


def test_guard_rejects_bad_ordering():
    with pytest.raises(ValueError):
        tr.Guard(pause_c=70.0, resume_c=72.0, kill_c=80.0)
    with pytest.raises(ValueError):
        tr.Guard(pause_c=75.0, resume_c=65.0, kill_c=74.0)


def test_wrap_command_keeps_user_command_last():
    wrapped = tr.wrap_command(["sleep", "1"])
    assert wrapped[-2:] == ["sleep", "1"]
    assert wrapped[0] in ("nice", "sleep")


def test_main_requires_separator(capsys):
    with pytest.raises(SystemExit):
        tr.main(["--name", "x"])


# -- real child processes -------------------------------------------------------


def test_pause_then_resume_stops_and_continues_real_child(tmp_path):
    seen: list[str] = []
    job_ref: dict[str, tr.Job] = {}
    temps = itertools.chain([70.0, 76.0, 76.0, 70.0, 64.0], itertools.repeat(64.0))

    def read_temp() -> float:
        job = job_ref["job"]
        if job.proc is not None:
            seen.append(_proc_state(job.proc.pid))
        return next(temps)

    job = _job(tmp_path, ["sh", "-c", "echo hello; sleep 1; echo bye"], [], read_temp=read_temp)
    job_ref["job"] = job
    rc = job.run()

    assert rc == 0
    assert job.pauses == 1
    assert job.state == "done"
    assert "T" in seen, f"child was never observed stopped: {seen}"
    assert job.paused_s > 0
    log = job.log_path.read_text()
    assert "hello" in log and "bye" in log
    doc = json.loads((tmp_path / "current_job.json").read_text())
    assert doc["state"] == "done"
    assert doc["exit_code"] == 0
    assert doc["pauses"] == 1
    rows = job.csv_path.read_text().splitlines()
    assert rows[0] == "ts,temp,cap,freq,throttled,power_w"
    assert len(rows) >= 3
    assert rows[1].split(",")[2] == "2400"


def test_kill_threshold_terminates_real_child(tmp_path):
    job = _job(tmp_path, ["sleep", "30"], [81.0])
    rc = job.run()
    assert rc == tr.EXIT_KILLED
    assert job.state == "killed"
    assert job.exit_code == tr.EXIT_KILLED
    doc = json.loads((tmp_path / "current_job.json").read_text())
    assert doc["state"] == "killed"


def test_kill_while_paused_still_terminates_stopped_child(tmp_path):
    # SIGTERM stays pending on a stopped process, so the guard must SIGCONT first.
    job = _job(tmp_path, ["sleep", "30"], [76.0, 81.0])
    rc = job.run()
    assert rc == tr.EXIT_KILLED
    assert job.pauses == 1
    assert job.state == "killed"


def test_stop_request_kills_the_job(tmp_path):
    job_ref: dict[str, tr.Job] = {}

    def read_temp() -> float:
        job_ref["job"].request_stop()
        return 60.0

    job = _job(tmp_path, ["sleep", "30"], [], read_temp=read_temp)
    job_ref["job"] = job
    assert job.run() == tr.EXIT_KILLED
    assert job.state == "killed"


def test_child_exit_code_passes_through(tmp_path):
    job = _job(tmp_path, ["sh", "-c", "exit 7"], [60.0])
    assert job.run() == 7
    assert job.state == "done"
    assert job.exit_code == 7


def test_unreadable_sensor_never_pauses(tmp_path):
    job = _job(tmp_path, ["sh", "-c", "sleep 0.2"], [None, None, None])
    assert job.run() == 0
    assert job.pauses == 0
