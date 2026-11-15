"""Pure functions of the url_threshold job (kaggle/jobs/url_threshold.py). No Kaggle, no data,
no ONNX.

Covers: the tau_url rule (smallest observed cal score with benign FPR <= 1 %), the order gate,
the max-over-URLs aggregation with absent messages, the carve size check, and pin mismatch
failures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import url_threshold as job  # noqa: E402


def test_threshold_rule_is_smallest_observed_score_within_one_percent():
    # 1000 benign scores 0.000 .. 0.999, so 1 % allows 10 benign at or above tau.
    neg = np.arange(1000) / 1000.0
    pos = np.full(200, 0.995)
    y = np.concatenate([np.zeros(1000, dtype=int), np.ones(200, dtype=int)])
    s = np.concatenate([neg, pos])
    rec = job.threshold_rule(y, s)
    assert rec["tau"] == pytest.approx(0.990)
    assert rec["cal_fp"] == 10
    assert rec["cal_fpr"] == pytest.approx(0.01)
    assert rec["cal_tp"] == 200
    assert rec["cal_recall"] == pytest.approx(1.0)
    assert rec["cal_n_benign"] == 1000 and rec["cal_n_phish"] == 200
    # one observed step lower breaks the bound, so tau is the smallest that holds
    lower = job.threshold_rule(y, s, target=0.01)
    assert lower["tau"] == rec["tau"]
    assert job.at_threshold_counts(y, s, 0.989)["fp"] == 11


def test_threshold_rule_refuses_a_tie_block_larger_than_the_allowance():
    y = np.zeros(1000, dtype=int)
    y[:5] = 1
    s = np.full(1000, 0.5)  # every benign score ties; 1000 false positives at any tau
    with pytest.raises(ValueError, match="no cal score meets FPR"):
        job.threshold_rule(y, s)


def test_threshold_rule_needs_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        job.threshold_rule(np.zeros(10, dtype=int), np.linspace(0, 1, 10))


def test_carve_size_mismatch_reports_the_numbers():
    job.check_carve_sizes(job.CAL_N, job.FIT_N)  # no raise
    with pytest.raises(RuntimeError, match="cal 18000 .* fit 480000"):
        job.check_carve_sizes(18000, 480000)


def test_order_gate_refuses_post_sets_before_the_threshold_and_pre_sets_after():
    logged: list[str] = []
    written: list[dict] = []
    gate = job.ThresholdGate(logged.append, written.append)
    with pytest.raises(RuntimeError, match="before the threshold was written"):
        gate.score_post("phreshphish_test", lambda: 1)
    assert gate.score_pre("phreshphish_cal", lambda: 2) == 2
    gate.write({"tau": 0.25})
    assert written == [{"tau": 0.25}]
    assert gate.tau == 0.25
    assert logged and "tau_url=0.25" in logged[0]
    with pytest.raises(RuntimeError, match="calibration set"):
        gate.score_pre("late_cal", lambda: 3)
    with pytest.raises(RuntimeError, match="already written"):
        gate.write({"tau": 0.3})
    assert gate.score_post("url_domain_test", lambda: 4) == 4
    assert gate.events == ["score:phreshphish_cal", "threshold", "score:url_domain_test"]


def test_order_gate_scores_each_set_once():
    gate = job.ThresholdGate(lambda _m: None, lambda _r: None)
    gate.score_pre("phreshphish_cal", lambda: 0)
    gate.write({"tau": 0.5})
    gate.score_post("groundlm", lambda: 0)
    with pytest.raises(ValueError, match="scored twice"):
        gate.score_post("groundlm", lambda: 0)


def test_gate_rejects_a_non_finite_threshold():
    gate = job.ThresholdGate(lambda _m: None, lambda _r: None)
    with pytest.raises(ValueError, match="not finite"):
        gate.write({"tau": float("nan")})
    assert gate.tau is None


def test_max_over_urls_and_absent_messages():
    url_lists = [["a", "b"], [], ["c"]]
    scores, has = job.aggregate_message_scores(url_lists, {"a": 0.2, "b": 0.7, "c": 0.1})
    assert scores[0] == pytest.approx(0.7)
    assert np.isnan(scores[1]) and not has[1]
    assert scores[2] == pytest.approx(0.1) and has[2]
    flags = job.flags_at(scores, has, 0.05)
    assert flags.tolist() == [True, False, True]
    # a message with no URL is never flagged, even at a threshold of zero
    assert not job.flags_at(scores, has, 0.0)[1]


def test_message_scores_come_from_canonicalized_urls():
    def fake_scores(urls):
        return np.asarray([0.9 if "evil" in u else 0.1 for u in urls], dtype=np.float64)

    texts = [
        "Your parcel is held, visit http://evil.example.com/pay to update",
        "Your parcel is held, no link here",
    ]
    scores, has = job.message_url_scores(texts, fake_scores)
    assert has.tolist() == [True, False]
    assert scores[0] == pytest.approx(0.9)
    assert np.isnan(scores[1])


def test_data_pin_mismatch_and_missing_files_fail(tmp_path):
    (tmp_path / "urls.parquet").write_bytes(b"not the pinned bytes")
    with pytest.raises(RuntimeError, match="data pin mismatch") as info:
        job.check_data_pins(tmp_path)
    message = str(info.value)
    assert "urls.parquet: sha256" in message
    assert "urls_phreshphish.parquet: not found" in message


def test_artifact_pin_mismatch_fails(tmp_path):
    kernel = tmp_path / "rakshak-url-student"
    kernel.mkdir()
    (kernel / job.SCALER_NAME).write_text('{"drifted": true}')
    with pytest.raises(FileNotFoundError, match="no copy with sha256"):
        job.base.locate_artifact([tmp_path], job.SCALER_NAME, job.SCALER_SHA)
    (kernel / job.ONNX_NAME).write_bytes(b"drifted onnx")
    with pytest.raises(FileNotFoundError, match="no copy with sha256"):
        job.base.locate_artifact([tmp_path], job.ONNX_NAME, job.ONNX_SHA)


def test_parity_scores_are_float32_hex_little_endian():
    frame = job.parity_frame(["7"], ["https://a.example/x"], [1], [0.5])
    assert frame["score_f32_hex"].iloc[0] == "0000003f"  # 0.5 as float32, little-endian
    assert frame["score"].dtype == np.float32
