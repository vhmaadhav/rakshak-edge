"""Binary detection metrics for the rakshak evaluation protocol.

Conventions:
- Labels are 0 (benign) or 1 (scam). Higher scores mean more scam-like.
- A decision is ``score >= threshold``.
- Undefined quantities are NaN rather than a silent 0: recall without positives, FPR without
  negatives, precision without predicted positives.
- Thresholds are chosen on the calibration split only (``threshold_max_f1``,
  ``threshold_for_fpr``). The test split is never used to pick one.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

FPR_TARGETS: tuple[float, ...] = (0.001, 0.01, 0.05)
FPR_OPERATING_POINT = 0.01
ECE_BINS = 15
# A recall@FPR figure is only resolvable when the negatives can show this many false positives.
MIN_FALSE_POSITIVES = 10
NAN = float("nan")
_FIXED_THRESHOLD_METRICS = ("precision", "recall", "f1", "mcc", "fpr", "accuracy")


def _labels(y_true: Sequence[int] | np.ndarray) -> np.ndarray:
    y = np.asarray(y_true)
    if y.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    y = y.astype(np.int64)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    return y


def _pair(y_true, score) -> tuple[np.ndarray, np.ndarray]:
    y = _labels(y_true)
    s = np.asarray(score, dtype=np.float64)
    if s.shape != y.shape:
        raise ValueError("labels and scores differ in length")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    return y, s


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else NAN


def confusion_counts(y_true, pred) -> tuple[int, int, int, int]:
    """Return (tp, fp, tn, fn) for boolean predictions."""
    y = _labels(y_true)
    p = np.asarray(pred, dtype=bool)
    tp = int(np.sum(p & (y == 1)))
    fp = int(np.sum(p & (y == 0)))
    fn = int(np.sum(~p & (y == 1)))
    tn = int(np.sum(~p & (y == 0)))
    return tp, fp, tn, fn


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Precision, recall, F1, MCC, accuracy and FPR from a confusion matrix.

    MCC is 0 when its denominator is 0 (the sklearn convention); every other undefined ratio is
    NaN.
    """
    n_pos = tp + fn
    n_neg = fp + tn
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return {
        "n": n_pos + n_neg,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, n_pos),
        "fpr": _ratio(fp, n_neg),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "mcc": float(mcc),
        "accuracy": _ratio(tp + tn, n_pos + n_neg),
    }


def binary_metrics(y_true, score, threshold: float) -> dict:
    """Confusion-matrix metrics at a fixed threshold (``score >= threshold`` is scam)."""
    y, s = _pair(y_true, score)
    out = metrics_from_counts(*confusion_counts(y, s >= threshold))
    out["threshold"] = float(threshold)
    return out


def roc_auc(y_true, score) -> float:
    y, s = _pair(y_true, score)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return NAN
    return float(roc_auc_score(y, s))


def pr_auc(y_true, score) -> float:
    """Average precision (the usual PR-AUC estimator)."""
    y, s = _pair(y_true, score)
    if (y == 1).sum() == 0:
        return NAN
    return float(average_precision_score(y, s))


def threshold_max_f1(y_true, score) -> tuple[float, float]:
    """Threshold maximising F1 on the given (calibration) split. Returns (threshold, best_f1).

    Ties (common when cal is nearly separable) go to the lowest threshold that reaches the best F1,
    which keeps the most recall. A threshold at the top of the score range would sit right on the
    boundary of the calibration data and generalise poorly.
    """
    y, s = _pair(y_true, score)
    if (y == 1).sum() == 0:
        return NAN, NAN
    precision, recall, thresholds = precision_recall_curve(y, s)
    p, r = precision[:-1], recall[:-1]
    f1 = np.where(p + r > 0, 2 * p * r / np.where(p + r > 0, p + r, 1.0), 0.0)
    best_f1 = float(f1.max())
    tied = np.flatnonzero(f1 >= best_f1 - 1e-12)
    # thresholds ascend, so the last tied index is the lowest threshold
    pick = int(tied[-1])
    return float(thresholds[pick]), best_f1


def _roc_points(y: np.ndarray, s: np.ndarray):
    return roc_curve(y, s, drop_intermediate=False)


def _best_point_within(fpr: np.ndarray, tpr: np.ndarray, target: float) -> int:
    """Index of the highest-TPR point with FPR <= target. Ties go to the first (most conservative)
    point. The first ROC point (FPR 0, threshold +inf) always qualifies."""
    idx = np.flatnonzero(fpr <= target + 1e-12)
    return int(idx[np.argmax(tpr[idx])])


def threshold_for_fpr(y_true, score, target: float) -> dict:
    """Threshold on this split whose FPR is <= target with the highest recall.

    Returns threshold, the FPR and TPR it achieves on this split. NaN values when a class is
    missing.
    """
    y, s = _pair(y_true, score)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return {"threshold": NAN, "fpr": NAN, "tpr": NAN}
    fpr, tpr, thr = _roc_points(y, s)
    i = _best_point_within(fpr, tpr, target)
    return {"threshold": float(thr[i]), "fpr": float(fpr[i]), "tpr": float(tpr[i])}


def recall_at_fpr(y_true, score, target: float) -> dict:
    """Oracle recall at FPR <= target on this split, with resolvability.

    ``resolvable`` is False when the negatives are too few for ``MIN_FALSE_POSITIVES`` false
    positives to exist at this FPR target, so the figure is a step size, not a measurement.
    """
    y, s = _pair(y_true, score)
    n_neg = int((y == 0).sum())
    n_required = math.ceil(MIN_FALSE_POSITIVES / target)
    out = threshold_for_fpr(y, s, target)
    out.update(
        {
            "target": float(target),
            "recall": out["tpr"],
            "n_neg": n_neg,
            "n_neg_required": n_required,
            "resolvable": bool(n_neg >= n_required),
        }
    )
    return out


def expected_calibration_error(y_true, prob, n_bins: int = ECE_BINS) -> float:
    """Equal-width-bin ECE: sum over bins of |bin fraction| * |mean label - mean probability|."""
    y, p = _pair(y_true, prob)
    if len(y) == 0:
        return NAN
    if p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("ECE needs probabilities in [0, 1]")
    idx = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    sum_p = np.bincount(idx, weights=p, minlength=n_bins)
    sum_y = np.bincount(idx, weights=y.astype(np.float64), minlength=n_bins)
    nz = counts > 0
    gap = np.abs(sum_y[nz] / counts[nz] - sum_p[nz] / counts[nz])
    return float(np.sum(counts[nz] / len(y) * gap))


def brier_score(y_true, prob) -> float:
    y, p = _pair(y_true, prob)
    if len(y) == 0:
        return NAN
    return float(np.mean((p - y) ** 2))


def coverage_error_curve(y_true, pred, confidence, *, points: int = 101) -> dict:
    """Selective risk when the least confident predictions are abstained on.

    Predictions are sorted by ``confidence`` (descending). ``risk[k]`` is the error rate among the
    top ``coverage[k]`` fraction. ``aurc`` is the mean risk over all coverage levels (the area
    under the risk-coverage curve, Geifman and El-Yaniv 2017). ``curve`` is downsampled to
    ``points`` values for storage.
    """
    y = _labels(y_true)
    p = np.asarray(pred, dtype=bool)
    conf = np.asarray(confidence, dtype=np.float64)
    n = len(y)
    if n == 0:
        return {"aurc": NAN, "curve": {"coverage": [], "risk": []}}
    order = np.argsort(-conf, kind="stable")
    wrong = (p[order] != (y[order] == 1)).astype(np.float64)
    k = np.arange(1, n + 1)
    risk = np.cumsum(wrong) / k
    coverage = k / n
    keep = np.unique(np.linspace(0, n - 1, min(points, n)).round().astype(np.int64))
    return {
        "aurc": float(np.mean(risk)),
        "curve": {
            "coverage": [float(c) for c in coverage[keep]],
            "risk": [float(r) for r in risk[keep]],
        },
    }


BOOTSTRAP_CLUSTER = "campaign_cluster"
BOOTSTRAP_IID = "iid_stratified"
BOOTSTRAP_METHODS: tuple[str, ...] = (BOOTSTRAP_CLUSTER, BOOTSTRAP_IID)
_METHOD_ALIASES = {"cluster": BOOTSTRAP_CLUSTER, "iid": BOOTSTRAP_IID}


def resolve_bootstrap_method(method: str | None, cluster_ids) -> str:
    """Campaign-clustered when cluster ids exist (the default), otherwise iid stratified.

    Passing ``method="iid"`` on purpose still works with cluster ids present; asking for the
    cluster bootstrap without cluster ids is an error, never a silent fallback.
    """
    if method is None:
        return BOOTSTRAP_CLUSTER if cluster_ids is not None else BOOTSTRAP_IID
    method = _METHOD_ALIASES.get(method, method)
    if method not in BOOTSTRAP_METHODS:
        raise ValueError(f"unknown bootstrap method {method!r}; use one of {BOOTSTRAP_METHODS}")
    if method == BOOTSTRAP_CLUSTER and cluster_ids is None:
        raise ValueError("the cluster bootstrap needs cluster ids (for example campaign_id)")
    return method


def _cluster_index(y: np.ndarray, cluster_ids) -> dict:
    """Rows grouped by cluster. A cluster is a positive stratum if it holds any scam row."""
    ids = np.asarray([str(v) for v in cluster_ids], dtype=object)
    if len(ids) != len(y):
        raise ValueError("cluster ids and labels differ in length")
    uniq, codes = np.unique(ids, return_inverse=True)
    n = len(uniq)
    sizes = np.bincount(codes, minlength=n).astype(np.int64)
    pos_rows = np.bincount(codes, weights=(y == 1).astype(np.float64), minlength=n)
    has_pos = pos_rows > 0
    return {
        "order": np.argsort(codes, kind="stable"),
        "starts": np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64),
        "sizes": sizes,
        "pos_clusters": np.flatnonzero(has_pos),
        "neg_clusters": np.flatnonzero(~has_pos),
        "n_clusters": int(n),
    }


def _cluster_draw(index: dict, rng: np.random.Generator) -> np.ndarray:
    """Row positions of one resample: clusters drawn with replacement within each stratum
    (positive-containing and all-benign), each contributing all of its rows."""
    pos, neg = index["pos_clusters"], index["neg_clusters"]
    drawn = np.concatenate((
        rng.choice(pos, size=len(pos), replace=True) if len(pos) else np.zeros(0, np.int64),
        rng.choice(neg, size=len(neg), replace=True) if len(neg) else np.zeros(0, np.int64),
    )).astype(np.int64)
    sizes = index["sizes"][drawn]
    cum = np.cumsum(sizes)
    total = int(cum[-1]) if len(cum) else 0
    base = np.repeat(index["starts"][drawn] - (cum - sizes), sizes)
    return index["order"][base + np.arange(total, dtype=np.int64)]


def _iid_draw(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row positions of one resample: positives and negatives drawn separately, with replacement."""
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return np.concatenate((
        rng.choice(pos, size=len(pos), replace=True),
        rng.choice(neg, size=len(neg), replace=True),
    ))


def _resample_metrics(yb, sb, threshold: float, fpr_targets: Sequence[float]) -> dict:
    """Every per-resample metric. Fixed-threshold ones use ``threshold``; the rest do not."""
    vals = metrics_from_counts(*confusion_counts(yb, sb >= threshold))
    out = {k: vals[k] for k in _FIXED_THRESHOLD_METRICS}
    fpr_b, tpr_b, _ = _roc_points(yb, sb)
    out["roc_auc"] = float(np.trapezoid(tpr_b, fpr_b))
    out["pr_auc"] = float(average_precision_score(yb, sb))
    for target in fpr_targets:
        best = _best_point_within(fpr_b, tpr_b, target)
        out[f"recall_at_fpr_{target:g}"] = float(tpr_b[best])
    return out


def bootstrap_intervals(
    y_true,
    score,
    threshold: float,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    fpr_targets: Sequence[float] = FPR_TARGETS,
    alpha: float = 0.05,
    cluster_ids: Sequence | None = None,
    method: str | None = None,
) -> dict:
    """Percentile intervals, campaign-clustered by default.

    With ``cluster_ids`` (campaign ids) the default is the cluster bootstrap: clusters are
    resampled with replacement, stratified by whether the cluster holds any scam row, and each
    drawn cluster contributes all of its rows. Campaign-level dependence therefore widens the
    intervals, which the iid version understates. Without cluster ids the default is the iid
    stratified bootstrap: positives and negatives resampled separately, so class counts are kept.
    ``method`` forces either one; the chosen method is returned as ``method``.

    The threshold is held fixed at ``threshold`` (chosen on calibration). Threshold-free metrics
    (PR-AUC, ROC-AUC, recall at FPR) are recomputed on each resample.
    Returns ``{metric: [low, high]}`` plus ``method``, ``n_boot``, ``seed`` and, for the cluster
    method, ``n_clusters``.
    """
    y, s = _pair(y_true, score)
    method = resolve_bootstrap_method(method, cluster_ids)
    clustered = method == BOOTSTRAP_CLUSTER
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return {"method": method, "n_boot": 0, "seed": seed}
    index = _cluster_index(y, cluster_ids) if clustered else None
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = _cluster_draw(index, rng) if clustered else _iid_draw(y, rng)
        values = _resample_metrics(y[idx], s[idx], threshold, fpr_targets)
        for key, value in values.items():
            draws.setdefault(key, []).append(value)
    out: dict = {"method": method, "n_boot": n_boot, "seed": seed, "alpha": alpha}
    if clustered:
        out["n_clusters"] = index["n_clusters"]
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    for key, arr in draws.items():
        values = np.asarray(arr, dtype=np.float64)
        with warnings.catch_warnings():
            # An all-NaN metric (for example PR-AUC with no positives) gives NaN bounds, quietly.
            warnings.simplefilter("ignore", RuntimeWarning)
            out[key] = [float(np.nanpercentile(values, lo_q)),
                        float(np.nanpercentile(values, hi_q))]
    return out


PAIRED_METRICS: tuple[str, ...] = ("pr_auc", "roc_auc", "f1", "recall_at_fpr_0.01")
_FIXED_THRESHOLD_NAMES = frozenset(_FIXED_THRESHOLD_METRICS)


def paired_bootstrap_test(
    y_true,
    score_a,
    score_b,
    *,
    metric: str,
    threshold_a: float | None = None,
    threshold_b: float | None = None,
    cluster_ids: Sequence | None = None,
    method: str | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Is model A better than model B on ``metric``? Paired, on the same test rows.

    ``metric`` is ``pr_auc``, ``roc_auc``, ``recall_at_fpr_<target>`` (oracle threshold per
    resample, as in ``bootstrap_intervals``), or a fixed-threshold metric such as ``f1`` or
    ``recall``, which needs ``threshold_a`` and ``threshold_b`` (take them from calibration).

    Each resample draws one set of rows (campaigns, by default) and scores both models on it, so
    the difference keeps the pairing. ``delta = A - B`` on the full test set, ``delta_ci`` is its
    percentile interval, and ``p_value`` is the two-sided null-centred bootstrap p-value:
    ``(1 + #{|delta* - delta| >= |delta|}) / (1 + #valid resamples)``.
    """
    y, a = _pair(y_true, score_a)
    _, b = _pair(y_true, score_b)
    method = resolve_bootstrap_method(method, cluster_ids)
    clustered = method == BOOTSTRAP_CLUSTER
    if metric in _FIXED_THRESHOLD_NAMES:
        if threshold_a is None or threshold_b is None:
            raise ValueError(f"{metric} needs a threshold for each model (choose it on cal)")
        thr_a, thr_b = float(threshold_a), float(threshold_b)
    elif metric.startswith("recall_at_fpr_") or metric in ("pr_auc", "roc_auc"):
        thr_a = thr_b = float("nan")
    else:
        raise ValueError(f"unknown paired metric {metric!r}")
    target = None
    if metric.startswith("recall_at_fpr_"):
        target = float(metric[len("recall_at_fpr_"):])
    targets = (target,) if target is not None else ()
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return {"metric": metric, "method": method, "n_boot": 0, "seed": seed}

    point_a = _resample_metrics(y, a, thr_a, targets)[metric]
    point_b = _resample_metrics(y, b, thr_b, targets)[metric]
    delta_hat = point_a - point_b

    index = _cluster_index(y, cluster_ids) if clustered else None
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = _cluster_draw(index, rng) if clustered else _iid_draw(y, rng)
        yb = y[idx]
        va = _resample_metrics(yb, a[idx], thr_a, targets)[metric]
        vb = _resample_metrics(yb, b[idx], thr_b, targets)[metric]
        deltas.append(va - vb)
    d = np.asarray(deltas, dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) == 0 or not np.isfinite(delta_hat):
        p_value, ci = float("nan"), [float("nan"), float("nan")]
    else:
        p_value = float((1 + np.sum(np.abs(d - delta_hat) >= abs(delta_hat))) / (1 + len(d)))
        ci = [float(np.percentile(d, 100 * alpha / 2)),
              float(np.percentile(d, 100 * (1 - alpha / 2)))]
    out = {
        "metric": metric,
        "method": method,
        "n_boot": n_boot,
        "seed": seed,
        "alpha": alpha,
        "a": float(point_a),
        "b": float(point_b),
        "delta": float(delta_hat),
        "delta_ci": ci,
        "p_value": p_value,
        "n_valid": int(len(d)),
    }
    if clustered:
        out["n_clusters"] = index["n_clusters"]
    if target is not None or metric in _FIXED_THRESHOLD_NAMES:
        out["thresholds"] = {"a": thr_a, "b": thr_b}
    return out
