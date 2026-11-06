"""Metrics for the benchmark: confusion metrics at a threshold, ROC-AUC, average precision,
Wilson intervals and latency percentiles. Numpy only, so the Pi venv stays small.

Decision rule everywhere: ``score >= threshold`` means scam (rakshak.eval.metrics).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Z95 = 1.959963984540054


def _as_arrays(y_true: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError("labels and scores differ in length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    return y, s


def confusion(y_true: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, int]:
    y, s = _as_arrays(y_true, scores)
    pred = s >= threshold
    return {
        "tp": int(np.sum(pred & (y == 1))),
        "fp": int(np.sum(pred & (y == 0))),
        "tn": int(np.sum(~pred & (y == 0))),
        "fn": int(np.sum(~pred & (y == 1))),
    }


def _ratio(num: float, den: float) -> float | None:
    return None if den == 0 else float(num) / float(den)


def threshold_metrics(
    y_true: Sequence[int], scores: Sequence[float], threshold: float
) -> dict[str, float | int | None]:
    """Precision, recall, F1, FPR, accuracy at a fixed threshold, plus the raw counts."""
    c = confusion(y_true, scores, threshold)
    precision = _ratio(c["tp"], c["tp"] + c["fp"])
    recall = _ratio(c["tp"], c["tp"] + c["fn"])
    fpr = _ratio(c["fp"], c["fp"] + c["tn"])
    if precision is None or recall is None or precision + recall == 0:
        f1 = None if precision is None or recall is None else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    n = sum(c.values())
    accuracy = _ratio(c["tp"] + c["tn"], n)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "accuracy": accuracy,
        **c,
        "n": n,
    }


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float | None:
    """Area under the ROC curve (Mann-Whitney with average ranks for ties). None if one class."""
    y, s = _as_arrays(y_true, scores)
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum = ranks[y == 1].sum()
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision(y_true: Sequence[int], scores: Sequence[float]) -> float | None:
    """Step-wise average precision (PR-AUC as in sklearn's average_precision_score)."""
    y, s = _as_arrays(y_true, scores)
    pos = int(y.sum())
    if pos == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    # keep the last index of each distinct score (tied scores form one threshold)
    last = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s_sorted) - 1]
    tp = tps[last].astype(np.float64)
    fp = fps[last].astype(np.float64)
    precision = tp / (tp + fp)
    recall = tp / pos
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float | None, float | None]:
    """Wilson score interval for k successes out of n (95% by default)."""
    if n <= 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def percentiles(values_ms: Sequence[float]) -> dict[str, float | int | None]:
    """p50, p95, p99, mean, min, max of latency samples (linear interpolation)."""
    arr = np.asarray(values_ms, dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "p50": None,
            "p95": None,
            "p99": None,
            "mean": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(arr.size),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
        "mean": round(float(arr.mean()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def fpr_block(fp: int, negatives: int) -> dict[str, float | int | None]:
    """FPR over policy-negatives with its denominator and Wilson 95% interval."""
    lo, hi = wilson_interval(fp, negatives)
    return {
        "fp": int(fp),
        "denominator": int(negatives),
        "fpr": _ratio(fp, negatives),
        "fpr_ci95": [lo, hi],
    }
