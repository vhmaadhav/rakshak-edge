"""Pure functions of the frozen baseline v0 job (kaggle/jobs/baseline_v0.py). No Kaggle needed.

The job's pins are checked against the frozen source files in the working tree when they exist, so a
change to rules.py, canonicalize.py, evidence.py or verifier.py since the pinned commit fails here.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import baseline_v0 as job  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def test_check_pins_reports_mismatch_and_missing():
    pins = {"a": "x", "b": "y", "c": "z"}
    assert job.check_pins({"a": "x", "b": "y", "c": "z"}, pins) == []
    problems = job.check_pins({"a": "x", "b": "NOPE"}, pins)
    assert any(p.startswith("b: sha256 NOPE") for p in problems)
    assert any(p == "c: not found" for p in problems)
    assert not any(p.startswith("a:") for p in problems)


def test_locate_artifact_picks_the_pinned_copy(tmp_path):
    good = tmp_path / "kernel_a"
    other = tmp_path / "kernel_b"
    good.mkdir()
    other.mkdir()
    (good / "best.onnx").write_bytes(b"pinned bytes")
    (other / "best.onnx").write_bytes(b"other bytes")
    sha = hashlib.sha256(b"pinned bytes").hexdigest()
    found = job.locate_artifact([tmp_path], "best.onnx", sha)
    assert found == good / "best.onnx"


def test_locate_artifact_refuses_when_no_copy_matches(tmp_path):
    (tmp_path / "k").mkdir()
    (tmp_path / "k" / "best.onnx").write_bytes(b"drifted")
    with pytest.raises(FileNotFoundError, match="no copy with sha256"):
        job.locate_artifact([tmp_path], "best.onnx", "0" * 64)


def test_wilson_interval_edges_and_symmetry():
    lo, hi = job.wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775, abs=5e-4)
    lo5, hi5 = job.wilson_interval(5, 10)
    assert lo5 + hi5 == pytest.approx(1.0, abs=1e-9)
    assert job.wilson_interval(0, 0)[0] != job.wilson_interval(0, 0)[0]  # NaN


def test_url_side_matrix_standardises_clips_and_fills():
    df = pd.DataFrame({"a": [1.0, np.nan, 1000.0], "b": [2.0, 2.0, 2.0]})
    out = job.url_side_matrix(df, ["a", "b"], mean=[0.0, 2.0], std=[1.0, 1.0])
    assert out.dtype == np.float32
    assert out.shape == (3, 2)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(0.0)  # NaN filled with 0 before standardising
    assert out[2, 0] == pytest.approx(10.0)  # clipped at +10
    assert np.allclose(out[:, 1], 0.0)


def test_url_side_matrix_refuses_missing_columns():
    with pytest.raises(ValueError, match="missing"):
        job.url_side_matrix(pd.DataFrame({"a": [1.0]}), ["a", "zzz"], [0, 0], [1, 1])


def _split(n_pos: int = 40, n_neg: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.ones(n_pos, np.int64), np.zeros(n_neg, np.int64)])
    s = np.clip(np.concatenate([rng.beta(5, 2, n_pos), rng.beta(2, 5, n_neg)]), 0, 1)
    clusters = [f"c{i % 15}" for i in range(len(y))]
    return y, s, clusters


def test_sms_run_table_shipped_point_matches_binary_metrics():
    y, s, clusters = _split()
    table = job.sms_run_table(y, s, y, s, clusters, n_boot=50)
    assert table["n_test"] == 100 and table["n_test_pos"] == 40
    shipped = table["shipped_threshold"]
    assert shipped["threshold"] == job.SMS_THRESHOLD
    assert shipped["f1"] == pytest.approx(job.M.binary_metrics(y, s, job.SMS_THRESHOLD)["f1"])
    assert "bootstrap_shipped" in table
    assert {"pr_auc", "roc_auc"} <= set(table["threshold_free"])
    assert "cal_fpr_1pct" in table


def test_groundlm_denominator_is_policy_negative_scoreable_only():
    df = pd.DataFrame(
        {
            "policy_negative": [True, True, True, False, True],
            "text_available": [True, True, False, True, True],
            "eval_ok": [True, True, False, True, True],
        }
    )
    scores = np.array([0.9, 0.1, 0.99, 0.99, 0.5])
    rules = np.array([True, False, True, True, False])
    out = job.groundlm_table(df, scores, rules, threshold=0.35)
    assert out["policy_negatives_total"] == 4
    assert out["policy_negatives_hash_only"] == 1
    # rows 0, 1 and 4 are policy negatives with text and eval_ok (row 3 is not a policy negative)
    assert out["scoreable"] == 3
    assert out["student_at_threshold"]["k"] == 2  # scores 0.9 and 0.5 clear 0.35
    assert out["rule_evidence_any"]["k"] == 1  # row 0 only
    assert out["rule_evidence_and_student"]["k"] == 1


def test_smishx_uses_smishing_and_legitimate_with_eval_ok():
    df = pd.DataFrame(
        {
            "eval_ok": [True, True, True, False, True],
            "label_fine": ["smishing", "smishing", "legitimate", "smishing", "spam"],
        }
    )
    scores = np.array([0.9, 0.1, 0.8, 0.9, 0.9])
    rules = np.array([True, False, False, False, False])
    out = job.smishx_table(df, scores, rules, threshold=0.35)
    assert out["smishing_recall"]["student_at_threshold"]["n"] == 2
    assert out["smishing_recall"]["student_at_threshold"]["k"] == 1
    assert out["legitimate_fpr"]["student_at_threshold"]["k"] == 1
    assert out["legitimate_fpr"]["student_at_threshold"]["n"] == 1


def test_run_onnx_batches_and_applies_sigmoid():
    class FakeSession:
        def __init__(self):
            self.calls = []

        def run(self, names, feeds):
            assert names == ["logit"]
            n = len(feeds["raw"])
            self.calls.append(n)
            return [np.zeros((n, 1), dtype=np.float32)]

    session = FakeSession()
    feeds = {"raw": np.zeros((5, 4), dtype=np.int64)}
    probs = job.run_onnx(session, feeds, 5, batch=2)
    assert session.calls == [2, 2, 1]
    assert np.allclose(probs, 0.5)


def test_url_domain_table_uses_calibration_threshold_and_pr_auc():
    y, s, clusters = _split(seed=3)
    out = job.url_domain_table(y, s, y, s, clusters, n_boot=50)
    assert out["n_test_pos"] == 40
    assert 0.0 <= out["pr_auc"] <= 1.0
    thr = out["cal_fpr_1pct"]["threshold"]
    assert thr == thr
    assert out["at_cal_fpr_1pct"]["threshold"] == pytest.approx(thr)
    assert "bootstrap_at_cal_fpr_1pct" in out


def test_protocol_scores_reuses_phreshphish_protocol():
    class Data:
        test = np.arange(20)
        bench_full = np.arange(20)
        subsets = [("0.05%", np.arange(20))]

    labels = np.array([1, 0] * 10)
    scores = np.linspace(0, 1, 20)
    section = job.protocol_scores(labels, scores, Data)
    assert "per_rate" in section and "0.05%" in section["per_rate"]
    assert section["per_rate"]["0.05%"]["n_subsets"] == 1


def test_frozen_source_files_match_their_pins_when_present():
    package = REPO / "src" / "rakshak"
    for name, want in job.SOURCE_PINS.items():
        path = package / name
        if not path.exists():
            pytest.skip(f"{name} not in this checkout")
        assert job.sha256_file(path) == want, f"{name} changed since ac45aea"


def test_artifact_pins_are_well_formed():
    for name, (sha, size, kernel) in job.ARTIFACT_PINS.items():
        assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), name
        assert size > 0
        assert kernel.startswith("maadhavvh/rakshak-")
    assert job.SMS_THRESHOLD == 0.35127517580986023
