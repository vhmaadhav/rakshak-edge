"""rbench metrics against hand-worked values."""

from __future__ import annotations

import numpy as np
import pytest
from rbench import metrics as M


def test_confusion_and_threshold_metrics():
    y = [1, 0, 1, 0]
    s = [0.9, 0.8, 0.7, 0.1]
    c = M.confusion(y, s, 0.75)
    assert c == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    t = M.threshold_metrics(y, s, 0.5)
    assert t["tp"] == 2 and t["fp"] == 1 and t["tn"] == 1 and t["fn"] == 0
    assert t["precision"] == pytest.approx(2 / 3)
    assert t["recall"] == pytest.approx(1.0)
    assert t["f1"] == pytest.approx(0.8)
    assert t["fpr"] == pytest.approx(0.5)
    assert t["n"] == 4


def test_decision_rule_is_greater_or_equal():
    assert M.confusion([1, 0], [0.5, 0.5], 0.5) == {"tp": 1, "fp": 1, "tn": 0, "fn": 0}


def test_zero_division_is_none_not_nan():
    t = M.threshold_metrics([0, 0], [0.1, 0.2], 0.9)
    assert t["recall"] is None and t["precision"] is None and t["f1"] is None
    assert t["fpr"] == 0.0


def test_roc_auc_known_value_and_ties():
    assert M.roc_auc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(0.75)
    assert M.roc_auc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert M.roc_auc([1, 1], [0.3, 0.9]) is None  # one class


def test_average_precision_known_value():
    # ranked: P(0.9), N(0.8), P(0.7), N(0.1) -> (1/1 + 2/3) / 2
    assert M.average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(5 / 6)
    assert M.average_precision([0, 0], [0.1, 0.2]) is None


def test_average_precision_agrees_with_sklearn_on_random_data():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    s = rng.random(500) + 0.3 * y
    assert M.average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s), abs=1e-9)
    assert M.roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s), abs=1e-9)


def test_wilson_interval_known_values():
    lo, hi = M.wilson_interval(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(0.27753, abs=1e-4)
    lo, hi = M.wilson_interval(5, 10)
    assert lo == pytest.approx(0.23659, abs=1e-4) and hi == pytest.approx(0.76341, abs=1e-4)
    assert M.wilson_interval(0, 0) == (None, None)


def test_fpr_block_reports_denominator():
    block = M.fpr_block(134, 300)
    assert block["denominator"] == 300 and block["fp"] == 134
    assert block["fpr"] == pytest.approx(134 / 300)
    lo, hi = block["fpr_ci95"]
    assert lo < block["fpr"] < hi


def test_percentiles():
    p = M.percentiles([1.0, 2.0, 3.0, 4.0, 100.0])
    assert p["p50"] == 3.0
    assert p["max"] == 100.0 and p["min"] == 1.0 and p["n"] == 5
    assert p["p95"] == pytest.approx(np.percentile([1, 2, 3, 4, 100], 95), abs=1e-3)
    empty = M.percentiles([])
    assert empty["p50"] is None and empty["n"] == 0


def test_label_and_length_validation():
    with pytest.raises(ValueError):
        M.roc_auc([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError):
        M.confusion([0, 1], [0.1], 0.5)
