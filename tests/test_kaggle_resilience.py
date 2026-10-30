"""Resilience helpers shared by the Kaggle baseline and teacher jobs (kaggle/jobs/_common.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import _common  # noqa: E402


class FakeCtx:
    def __init__(self):
        self.results: dict = {}
        self.lines: list[str] = []
        self.saves = 0

    def log(self, msg):
        self.lines.append(str(msg))

    def save(self):
        self.saves += 1


@pytest.mark.parametrize(("ok", "failed", "expected"), [
    (3, 0, "ok"),
    (2, 1, "partial"),
    (0, 4, "fail"),
    (0, 0, "fail"),
])
def test_job_status(ok, failed, expected):
    assert _common.job_status(ok, failed) == expected


def test_single_class_reason_only_for_one_class():
    assert _common.single_class_reason([0, 1, 0], "split") is None
    reason = _common.single_class_reason([1, 1, 1], "url_temporal")
    assert reason.startswith("skipped: url_temporal test split has only label 1")
    assert _common.single_class_reason([], "empty").startswith("skipped: empty")


def test_record_skip_keeps_reason_in_results():
    ctx = FakeCtx()
    _common.record_skip(ctx, "url_temporal", "skipped: one class")
    assert ctx.results["skipped"] == {"url_temporal": "skipped: one class"}
    assert ctx.lines == ["skipped: one class"]


def test_units_record_failure_and_continue():
    ctx = FakeCtx()
    units = _common.Units(ctx)

    def good(x):
        return x * 2

    def bad():
        raise ValueError("boom")

    assert units.run({"split": "a", "model": "m"}, good, 2) == 4
    assert units.run({"split": "a", "model": "n"}, bad) is None
    assert units.run({"split": "b", "model": "m"}, good, 5) == 10

    assert units.status() == "partial"
    (failure,) = ctx.results["failures"]
    assert failure["split"] == "a" and failure["model"] == "n"
    assert failure["error"] == "ValueError: boom"
    assert "Traceback" in failure["traceback"] and "boom" in failure["traceback"]
    assert ctx.saves == 3  # results.json saved after every unit


def test_units_all_failing_is_fail():
    ctx = FakeCtx()
    units = _common.Units(ctx)
    units.run({"split": "a"}, lambda: 1 / 0)
    assert units.status() == "fail"
    assert len(ctx.results["failures"]) == 1
