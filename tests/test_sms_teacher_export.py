"""Tests for kaggle/jobs/sms_teacher_export.py: the pure helpers (parity statistics, latency
summary, export.json builder, failed-export payload, argument parsing, the fixed string list).
Synthetic arrays only; no torch, no onnxruntime, no model."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import sms_teacher_export as job  # noqa: E402

TAG = re.compile(r"<[A-Z][A-Z0-9_]*>")


# ---- argument parsing ----


def test_parse_args_defaults():
    args = job.parse_args([])
    assert args.teacher_dir == "/kaggle/input"
    assert args.teacher_sha == job.TEACHER_SHA
    assert job.TEACHER_SHA == "673b970e9f4746dbbbb5a00aa84eb056d6750cbc2897ba712ab1aa1b234abc7e"


def test_parse_args_values():
    other = "a" * 64
    args = job.parse_args(["--teacher-dir", "/data", "--teacher-sha", other])
    assert args.teacher_dir == "/data"
    assert args.teacher_sha == other


@pytest.mark.parametrize("sha", ["ABC", "g" * 64, "a" * 63, "A" * 64])
def test_parse_args_rejects_bad_sha(sha):
    with pytest.raises(SystemExit):
        job.parse_args(["--teacher-sha", sha])


# ---- the fixed string list ----


def test_sms_strings_are_64_unique_non_empty_without_tags():
    strings = job.sms_strings()
    assert len(strings) == 64
    assert len(set(strings)) == 64
    assert all(s.strip() for s in strings)
    assert not any(TAG.search(s) for s in strings)


def test_sms_strings_cover_scripts_urls_and_amounts():
    strings = job.sms_strings()
    assert any(re.search(r"[ऀ-ॿ]", s) for s in strings)
    assert any("http" in s for s in strings)
    assert any(re.search(r"Rs \d", s) for s in strings)


# ---- parity statistics ----


def test_scam_probability_is_softmax_of_class_one():
    p = job.scam_probability(np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0]]))
    assert p[0] == pytest.approx(0.5)
    assert p[1] > 0.85
    assert p[2] < 0.15


def test_logit_parity_perfect_agreement():
    ref = np.array([[1.0, -1.0], [-2.0, 3.0], [0.5, 0.25]])
    out = job.logit_parity(ref, ref.copy())
    assert out == {"n": 3, "max_abs_logit_diff": 0.0, "argmax_agreement": 1.0}


def test_logit_parity_detects_differences_and_flips():
    ref = np.array([[1.0, -1.0], [-2.0, 3.0]])
    got = np.array([[1.5, -1.0], [3.0, -3.0]])  # second row flips class
    out = job.logit_parity(ref, got)
    assert out["max_abs_logit_diff"] == pytest.approx(6.0)
    assert out["argmax_agreement"] == pytest.approx(0.5)


def test_probability_parity_perfect_agreement():
    ref = np.array([[0.2, -0.4], [1.0, 2.0], [-3.0, 0.5], [0.0, 0.1]])
    out = job.probability_parity(ref, ref.copy())
    assert out["max_abs_prob_diff"] == pytest.approx(0.0)
    assert out["mean_abs_prob_diff"] == pytest.approx(0.0)
    assert out["argmax_agreement"] == 1.0
    assert out["pearson_r"] == pytest.approx(1.0)
    assert out["n"] == 4


def test_probability_parity_constant_probabilities_have_no_pearson():
    # both arrays hold the same constant probability for every row: correlation is undefined
    ref = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    out = job.probability_parity(ref, ref.copy())
    assert out["pearson_r"] is None
    assert out["max_abs_prob_diff"] == 0.0
    assert out["argmax_agreement"] == 1.0


def test_probability_parity_constant_on_one_side_only():
    ref = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    got = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    out = job.probability_parity(ref, got)
    assert out["pearson_r"] is None
    assert out["max_abs_prob_diff"] > 0


def test_probability_parity_negative_correlation_and_shifts():
    ref = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    got = -ref
    out = job.probability_parity(ref, got)
    assert out["pearson_r"] < 0
    assert out["argmax_agreement"] < 1.0


@pytest.mark.parametrize("fn", [job.logit_parity, job.probability_parity])
def test_parity_rejects_mismatched_or_empty_inputs(fn):
    with pytest.raises(ValueError):
        fn(np.zeros((2, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        fn(np.zeros((0, 2)), np.zeros((0, 2)))


# ---- latency ----


def test_latency_summary_fields():
    s = job.latency_summary([10.0, 20.0, 30.0, 40.0, 50.0])
    assert s["n"] == 5
    assert s["p50"] == 30.0
    assert s["p95"] == pytest.approx(48.0)
    assert s["mean"] == 30.0
    assert s["min"] == 10.0


def test_latency_summary_rejects_empty():
    with pytest.raises(ValueError):
        job.latency_summary([])


def test_pad_to_truncates_and_pads_with_mask():
    assert job.pad_to([5, 6, 7], 5, 0) == ([5, 6, 7, 0, 0], [1, 1, 1, 0, 0])
    assert job.pad_to(list(range(9)), 4, 0) == ([0, 1, 2, 3], [1, 1, 1, 1])
    with pytest.raises(ValueError):
        job.pad_to([1], 0, 0)


# ---- export.json and the failed-export payload ----


def _success_doc():
    return job.build_export_doc(
        status="ok",
        teacher_sha=job.TEACHER_SHA,
        exporter="torch.onnx.export",
        attempts=[job.attempt_record("torch.onnx.export", False, None)],
        opset=17,
        ir_version=8,
        max_len=128,
        files={"teacher_fp32.onnx": {"bytes": 10, "sha256": "0" * 64}},
        quantisation={"quantised_op_types": ["MatMul"], "error_type": None},
        parity={"fp32_onnx_vs_pytorch": {"max_abs_logit_diff": 1e-5}},
        latency_ms={"fp32": {"32": {"p50": 1.0, "p95": 2.0}}},
        versions={"onnx": "1.23.2"},
        n_strings=64,
        notes=["n"],
    )


def test_export_doc_has_every_required_key():
    doc = _success_doc()
    for key in job.REQUIRED_KEYS:
        assert key in doc
    assert doc["job"] == "sms_teacher_export"
    assert doc["teacher_sha256"] == job.TEACHER_SHA
    json.dumps(doc)  # serialisable


def test_export_doc_refuses_missing_keys(monkeypatch):
    monkeypatch.setattr(job, "REQUIRED_KEYS", (*job.REQUIRED_KEYS, "never_set"))
    with pytest.raises(ValueError):
        _success_doc()


def test_failed_export_payload_is_well_formed():
    attempts = [
        job.attempt_record("torch.onnx.export", False, RuntimeError("unsupported op")),
        job.attempt_record("torch.onnx.export(dynamo=True)", True, ValueError("guard failed")),
    ]
    doc = job.failed_export_doc(
        teacher_sha=job.TEACHER_SHA, attempts=attempts, versions={"onnx": None}, max_len=128
    )
    for key in job.REQUIRED_KEYS:
        assert key in doc
    assert doc["status"] == "export_failed"
    assert doc["exporter"] is None
    assert doc["files"] == {}
    assert doc["quantisation"] is None and doc["parity"] is None
    assert [a["ok"] for a in doc["export_attempts"]] == [False, False]
    assert doc["export_attempts"][0]["error_type"] == "RuntimeError"
    assert doc["export_attempts"][0]["error_message"] == "unsupported op"
    json.dumps(doc)


def test_attempt_record_success_and_failure():
    ok = job.attempt_record("x", False, None)
    assert ok == {
        "exporter": "x",
        "dynamo": False,
        "ok": True,
        "error_type": None,
        "error_message": None,
    }
    bad = job.attempt_record("y", True, KeyError("k"))
    assert bad["ok"] is False and bad["error_type"] == "KeyError" and bad["dynamo"] is True


def test_render_report_for_success_and_failure():
    text = job.render_report(_success_doc())
    assert "torch.onnx.export" in text and "teacher_fp32.onnx" in text
    failed = job.failed_export_doc(
        teacher_sha=job.TEACHER_SHA,
        attempts=[job.attempt_record("torch.onnx.export", False, RuntimeError("boom"))],
        versions={},
        max_len=128,
    )
    text = job.render_report(failed)
    assert "export_failed" in text and "boom" in text


def test_package_versions_has_every_name():
    versions = job.package_versions()
    assert set(versions) == set(job.PACKAGES)
    assert job.package_versions(("definitely-not-a-package-xyz",)) == {
        "definitely-not-a-package-xyz": None
    }


def test_job_imports_without_torch():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import sms_teacher_export; "
        "print('torch' in sys.modules, 'onnxruntime' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(JOBS)], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["False", "False"]


def test_launch_line_names_cpu_kernel_and_teacher_kernel():
    lines = [ln.strip() for ln in job.__doc__.splitlines() if "kaggle/launch.sh" in ln]
    assert lines == [
        "kaggle/launch.sh sms_teacher_export --internet --kernels maadhavvh/rakshak-sms-teacher"
    ]
    assert "--gpu" not in lines[0]
