"""The generated Kaggle runner must load a job whose dataclasses use postponed annotations.

Regression for sms_teacher: @dataclass with `from __future__ import annotations` looks the module up
in sys.modules while the class body runs, so an unregistered job module crashed at import.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

JOB_SOURCE = """\
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    fp16: bool
    max_len: int


def run(ctx):
    return Config(fp16=True, max_len=128).max_len == 128
"""


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "build_kernel", REPO / "kaggle" / "build_kernel.py"
    )
    bk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bk)

    repo = tmp_path / "repo"
    (repo / "src" / "rakshak").mkdir(parents=True)
    (repo / "src" / "rakshak" / "__init__.py").write_text("")
    (repo / "kaggle" / "jobs").mkdir(parents=True)
    (repo / "kaggle" / "jobs" / "_common.py").write_text("")
    (repo / "kaggle" / "jobs" / "dc_job.py").write_text(JOB_SOURCE)
    pins = tmp_path / "requirements-kaggle.txt"
    pins.write_text("numpy==2.0.0\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)

    monkeypatch.setattr(bk, "REPO", repo)
    monkeypatch.setattr(bk, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(bk, "PINS_FILE", pins)
    return bk, tmp_path


def test_generated_loader_imports_job_with_dataclass(fake_repo, monkeypatch):
    bk, tmp_path = fake_repo
    monkeypatch.setattr(sys, "path", list(sys.path))  # load_job prepends to sys.path
    bk.main(["--job", "dc_job"])

    run_py = tmp_path / "out" / "dc_job" / "kernel" / "run.py"
    ns = {"__name__": "rakshak_kernel_test"}
    exec(compile(run_py.read_text(), str(run_py), "exec"), ns)
    try:
        ns["unpack"]()
        module = ns["load_job"]()
        assert module.__name__ == "rakshak_job_dc_job"
        assert sys.modules["rakshak_job_dc_job"] is module
        assert module.run(None) is True
        assert module.Config(fp16=True, max_len=128).max_len == 128
    finally:
        sys.modules.pop("rakshak_job_dc_job", None)
        shutil.rmtree(ns["SCRATCH"], ignore_errors=True)
