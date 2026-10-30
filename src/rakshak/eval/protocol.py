"""The evaluation protocol shared by every SMS and URL job.

Thresholds come from the calibration split only:
- ``max_f1``: the threshold maximising F1 on cal (the primary operating point).
- ``fpr_1pct``: the threshold giving the most recall on cal with FPR <= 1%.

Test metrics are then computed at those fixed thresholds. Recall at a given FPR is also reported
on test with its own threshold (the oracle view) and flagged when not resolvable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from rakshak.eval.metrics import (
    FPR_OPERATING_POINT,
    FPR_TARGETS,
    binary_metrics,
    bootstrap_intervals,
    brier_score,
    coverage_error_curve,
    expected_calibration_error,
    pr_auc,
    recall_at_fpr,
    roc_auc,
    threshold_for_fpr,
    threshold_max_f1,
)


def group_metrics(y_true, score, threshold: float, groups: Mapping[str, Sequence]) -> dict:
    """Per-group metrics at one fixed threshold, one list of rows per grouping column.

    Each row has the group value, its size and class counts, the confusion metrics, and PR-AUC
    (NaN when the group has one class). Rows are sorted by size, largest first.
    """
    y = np.asarray(y_true).astype(np.int64)
    s = np.asarray(score, dtype=np.float64)
    out: dict[str, list[dict]] = {}
    for name, values in groups.items():
        labels = ["(missing)" if v is None or v == "" else str(v) for v in values]
        arr = np.asarray(labels, dtype=object)
        rows = []
        for value in sorted(set(arr.tolist())):
            m = arr == value
            row = {"value": value}
            row.update(binary_metrics(y[m], s[m], threshold))
            row["pr_auc"] = pr_auc(y[m], s[m])
            rows.append(row)
        rows.sort(key=lambda r: (-r["n"], r["value"]))
        out[name] = rows
    return out


def evaluate_scores(
    y_cal,
    s_cal,
    y_test,
    s_test,
    *,
    groups: Mapping[str, Sequence] | None = None,
    fpr_targets: Sequence[float] = FPR_TARGETS,
    n_boot: int = 1000,
    seed: int = 0,
    bootstrap: bool = True,
) -> dict:
    """Full protocol result for one model on one split.

    ``s_*`` are probabilities or any monotone score where higher means scam. Calibration arrays
    pick thresholds. Test arrays give every reported test number.
    """
    y_cal = np.asarray(y_cal).astype(np.int64)
    y_test = np.asarray(y_test).astype(np.int64)
    s_cal = np.asarray(s_cal, dtype=np.float64)
    s_test = np.asarray(s_test, dtype=np.float64)
    if (y_cal == 1).sum() == 0:
        raise ValueError("calibration split has no positives; thresholds are undefined")
    if (y_test == 1).sum() == 0 or (y_test == 0).sum() == 0:
        raise ValueError("test split needs both classes")

    thr_f1, cal_f1 = threshold_max_f1(y_cal, s_cal)
    op_fpr = threshold_for_fpr(y_cal, s_cal, FPR_OPERATING_POINT)
    thresholds = {
        "max_f1": {"threshold": thr_f1, "cal_f1": cal_f1},
        "fpr_1pct": {
            "threshold": op_fpr["threshold"],
            "cal_fpr": op_fpr["fpr"],
            "cal_recall": op_fpr["tpr"],
        },
    }
    operating = {
        "max_f1": binary_metrics(y_test, s_test, thr_f1),
        "fpr_1pct": binary_metrics(y_test, s_test, op_fpr["threshold"]),
    }

    recall_cal_thr: dict[str, dict] = {}
    recall_oracle: dict[str, dict] = {}
    for target in fpr_targets:
        cal_thr = threshold_for_fpr(y_cal, s_cal, target)
        on_test = binary_metrics(y_test, s_test, cal_thr["threshold"])
        recall_cal_thr[f"{target:g}"] = {
            "threshold": cal_thr["threshold"],
            "cal_fpr": cal_thr["fpr"],
            "test_recall": on_test["recall"],
            "test_fpr": on_test["fpr"],
        }
        recall_oracle[f"{target:g}"] = recall_at_fpr(y_test, s_test, target)

    threshold_free = {
        "roc_auc": roc_auc(y_test, s_test),
        "pr_auc": pr_auc(y_test, s_test),
        "ece": expected_calibration_error(y_test, np.clip(s_test, 0.0, 1.0)),
        "brier": brier_score(y_test, np.clip(s_test, 0.0, 1.0)),
        "recall_at_cal_fpr": recall_cal_thr,
        "recall_at_fpr_oracle": recall_oracle,
    }
    cal_free = {"pr_auc": pr_auc(y_cal, s_cal), "roc_auc": roc_auc(y_cal, s_cal)}

    confidence = np.abs(s_test - thr_f1)
    coverage = coverage_error_curve(y_test, s_test >= thr_f1, confidence)

    result = {
        "sizes": {
            "cal": int(len(y_cal)),
            "cal_pos": int((y_cal == 1).sum()),
            "test": int(len(y_test)),
            "test_pos": int((y_test == 1).sum()),
            "test_neg": int((y_test == 0).sum()),
        },
        "thresholds": thresholds,
        "operating_points": operating,
        "threshold_free": threshold_free,
        "cal_threshold_free": cal_free,
        "coverage": {
            "confidence": "abs(score - max_f1 threshold)",
            "aurc": coverage["aurc"],
            "curve": coverage["curve"],
        },
        "groups": group_metrics(y_test, s_test, thr_f1, groups or {}),
    }
    if bootstrap:
        result["bootstrap"] = bootstrap_intervals(
            y_test, s_test, thr_f1, n_boot=n_boot, seed=seed, fpr_targets=fpr_targets
        )
    return result


def headline(result: Mapping) -> dict:
    """The few numbers a summary row needs, from one protocol result."""
    op = result["operating_points"]["max_f1"]
    tf = result["threshold_free"]
    return {
        "f1": op["f1"],
        "pr_auc": tf["pr_auc"],
        "recall_at_1pct_fpr_cal": tf["recall_at_cal_fpr"].get("0.01", {}).get("test_recall"),
        "fpr_at_max_f1": op["fpr"],
        "ece": tf["ece"],
    }
