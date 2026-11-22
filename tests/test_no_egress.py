"""Regression layer for offline inference: with Python-level socket connects and name lookups
patched to raise, the shipped models still load and run one inference through the normal code
paths.

This is a regression layer only. It cannot see native code: ONNX Runtime, BLAS and any C library
open sockets without going through Python's ``socket`` module. The stronger, OS-level check is
``scripts/offline_check.sh``, which runs the selection inside a network namespace that has no
interfaces at all. Run both.

Model files are not in the repository. Tests that need one skip when it is absent.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "deploy" / "pi" / "bench", REPO / "deploy" / "pi"):
    sys.path.insert(0, str(p))

SMS_NAME = "best_student.onnx"
SAMPLE_SMS = "Dear customer share your OTP to avoid account suspension"
SAMPLE_URL_TEXT = "Your KYC is blocked. Update now at https://sbi-kyc.in/upd/verify?acct=123"


def _find_model(name: str) -> Path | None:
    kaggle = REPO / "kaggle"
    if not kaggle.is_dir():
        return None
    return next(iter(sorted(kaggle.rglob(name))), None)


@pytest.fixture
def no_network(monkeypatch):
    """Make every Python-level route to the network fail loudly."""

    def refuse(*args, **kwargs):
        raise OSError("network access is disabled for this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket, "gethostbyname", refuse)
    return refuse


def test_patch_blocks_python_sockets(no_network):
    with pytest.raises(OSError):
        socket.getaddrinfo("example.invalid", 443)
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", 9), timeout=0.1)


def test_sms_student_runs_one_inference_without_network(no_network):
    path = _find_model(SMS_NAME)
    if path is None:
        pytest.skip(f"{SMS_NAME} not present; the shipped SMS model is not in this checkout")
    from rbench.runners.onnx import SmsRunner

    runner = SmsRunner()
    runner.load(path, 1, {})
    outputs = runner.run(runner.make_feed({"text": SAMPLE_SMS}))
    prob, logit = runner.score(outputs)
    assert 0.0 <= prob <= 1.0 and logit == logit  # finite, in range


def test_url_expert_runs_one_inference_without_network(no_network):
    from rakshak.fusion.url_expert import FrozenURLExpert, frozen_url_folder

    try:
        folder = frozen_url_folder(REPO / "kaggle")
    except ValueError as exc:
        pytest.skip(f"hash-pinned URL model not usable in this checkout: {exc}")
    expert = FrozenURLExpert(folder)
    scores, present = expert.score([SAMPLE_URL_TEXT])
    assert bool(present[0]) is True
    assert 0.0 <= float(scores[0]) <= 1.0
