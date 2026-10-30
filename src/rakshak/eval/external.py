"""Policy-negative false-positive rate for the external SMS sets (GroundLM 2026 primary endpoint).

The GroundLM paper's primary metric is the global FPR over the policy-negative rows (label 0 of
the 1,393-message test set), plus the FPR within each template category (Bank/account,
UPI/Payment, ...). ``flagged`` is ``score >= threshold`` with the threshold fixed on internal
calibration data, never on the external set.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

GROUNDLM_FPR_TARGET = 0.01


def policy_fpr(
    score: Sequence[float] | np.ndarray,
    threshold: float,
    is_negative: Sequence[bool] | np.ndarray,
    categories: Sequence[str | None] | None = None,
    target: float = GROUNDLM_FPR_TARGET,
) -> dict:
    """Global FPR over policy-negatives and FPR per category.

    Returns the flagged count, denominator and rate for the whole set, and a ``by_category``
    table (sorted by denominator, largest first). ``meets_target`` compares the global rate with
    ``target`` (1%, the GroundLM go/no-go line). A NaN rate means an empty denominator.
    """
    s = np.asarray(score, dtype=np.float64)
    neg = np.asarray(is_negative, dtype=bool)
    flagged = s >= threshold
    n_neg = int(neg.sum())
    fp = int((flagged & neg).sum())
    fpr = fp / n_neg if n_neg else float("nan")
    by_category: dict[str, dict] = {}
    if categories is not None:
        cats = np.asarray(
            ["(missing)" if c in (None, "") else str(c) for c in categories], dtype=object
        )
        for cat in sorted(set(cats[neg].tolist())):
            m = neg & (cats == cat)
            n = int(m.sum())
            k = int((flagged & m).sum())
            by_category[cat] = {"fp": k, "n": n, "fpr": k / n if n else float("nan")}
        by_category = dict(sorted(by_category.items(), key=lambda kv: (-kv[1]["n"], kv[0])))
    return {
        "threshold": float(threshold),
        "n_negatives": n_neg,
        "false_positives": fp,
        "global_fpr": fpr,
        "target": float(target),
        "meets_target": bool(fpr <= target) if n_neg else None,
        "by_category": by_category,
    }
