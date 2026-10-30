"""Calibration-only selective decisions with explicit denominators and finite-sample CRC."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from rakshak.eval.metrics import coverage_error_curve
from rakshak.fusion.calibrate import labelled, scores_array

LOW = float(np.nextafter(0.0, -np.inf))
HIGH = float(np.nextafter(1.0, np.inf))


@dataclass(frozen=True)
class SelectiveThresholds:
    """SAFE at p <= t_safe; SCAM at p >= t_scam; otherwise UNCERTAIN."""

    t_safe: float
    t_scam: float

    def __post_init__(self) -> None:
        if not LOW <= self.t_safe < self.t_scam <= HIGH:
            raise ValueError("finite ordered thresholds required")

    def transform(self, prob) -> np.ndarray:
        p = scores_array(prob, probability=True)
        return np.where(p <= self.t_safe, "SAFE",
                        np.where(p >= self.t_scam, "SCAM", "UNCERTAIN"))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, state: dict) -> SelectiveThresholds:
        return cls(**state)


def selective_metrics(y, prob, thresholds: SelectiveThresholds) -> dict:
    """FPR = FP / decided benign; miss = FN / decided scams. Undefined = NaN."""
    p, y = labelled(prob, y, probability=True)
    pred = thresholds.transform(p)
    decided = pred != "UNCERTAIN"
    neg = int(np.sum(decided & (y == 0)))
    pos = int(np.sum(decided & (y == 1)))
    fp = int(np.sum((pred == "SCAM") & (y == 0)))
    fn = int(np.sum((pred == "SAFE") & (y == 1)))
    return {"n": len(y), "decided": int(decided.sum()), "decided_benign": neg,
            "decided_scam": pos, "fp": fp, "fn": fn, "coverage": float(decided.mean()),
            "fpr": fp / neg if neg else float("nan"),
            "miss_rate": fn / pos if pos else float("nan"),
            "risk": (fp + fn) / (neg + pos) if neg + pos else float("nan")}


def search_thresholds(y, prob, *, alpha: float = 0.01, beta: float = 0.05) -> dict:
    """Exact maximum empirical coverage over tied-score boundaries, O(n log n).

    Both decided-class denominators must be positive; abstaining on an entire
    class does not satisfy its constraint. Ties prefer fewer errors, then the
    earlier ascending scam boundary. This is empirical selection, not a guarantee.
    """
    p, y = labelled(prob, y, probability=True)
    if not 0 <= alpha < 1 or not 0 <= beta < 1:
        raise ValueError("alpha and beta must lie in [0, 1)")
    values, inv = np.unique(p, return_inverse=True)
    neg = np.r_[0, np.cumsum(np.bincount(inv, weights=y == 0, minlength=len(values)))]
    pos = np.r_[0, np.cumsum(np.bincount(inv, weights=y == 1, minlength=len(values)))]
    best = None
    for j in range(len(values) + 1):
        fp, tp = neg[-1] - neg[j], pos[-1] - pos[j]
        # Largest safe prefix allowed by the miss constraint, without overlapping SCAM.
        cap = beta * tp / (1 - beta)
        k = min(j, int(np.searchsorted(pos, cap, side="right") - 1))
        tn, fn = neg[k], pos[k]
        if tn + fp == 0 or tp + fn == 0:
            continue
        if fp > alpha * (tn + fp) or fn > beta * (tp + fn):
            continue
        rank = (int(tn + fn + fp + tp), -int(fp + fn))
        if best is None or rank > best[0]:
            threshold = SelectiveThresholds(float(values[k - 1]) if k else LOW,
                                           float(values[j]) if j < len(values) else HIGH)
            best = rank, threshold
    if best is None:
        return {"status": "infeasible", "alpha": alpha, "beta": beta, "thresholds": None,
                "reason": "no pair has defined class risks satisfying both constraints"}
    return {"status": "ok", "alpha": alpha, "beta": beta,
            "thresholds": best[1].to_dict(), "cal": selective_metrics(y, p, best[1])}


def conformal_fpr(y, prob, *, t_safe: float, alpha: float = 0.01) -> dict:
    """CRC (Angelopoulos et al., ICLR 2024) for selective benign false alarms.

    Fix the scorer AND t_safe independently of these calibration observations.
    Benign calibration observations and the next benign observation must be
    exchangeable. Use decreasing bounded loss L_t = 1{SCAM} + alpha*1{UNCERTAIN}
    on benign rows, with B=1. Select the smallest representable threshold with
    (sum L_t + 1)/(n_benign + 1) <= alpha. If none exists return infeasible.

    CRC controls expected surrogate risk, hence E[FP] <= alpha*E[decided] for
    a new benign observation (expectations include calibration randomness).
    This is a marginal selective FPR bound, NOT an expected ratio of dataset
    counts, a conditional-on-fitted-threshold guarantee, or a high-probability CI.
    It does not control scam misses. No guarantee under campaign shift, and
    dependent messages within campaigns need not be exchangeable. Campaign
    disjointness alone does not establish the assumption. Infeasible selection
    must not be deployed as a certified rule; conditioning on success is not
    covered by the unconditional CRC theorem.
    """
    p, y = labelled(prob, y, probability=True)
    if not 0 < alpha < 1 or not LOW <= t_safe < HIGH:
        raise ValueError("invalid alpha or fixed safe threshold")
    benign = np.sort(p[y == 0])
    n = len(benign)
    candidates = np.unique(np.r_[np.nextafter(t_safe, np.inf),
                                 np.nextafter(benign[benign > t_safe], np.inf), HIGH])
    safe = int(np.sum(benign <= t_safe))
    for threshold in candidates:
        fp = n - int(np.searchsorted(benign, threshold, side="left"))
        uncertain = n - safe - fp
        corrected = (fp + alpha * uncertain + 1) / (n + 1)
        if n and corrected <= alpha:
            return {"status": "ok", "alpha": alpha, "n_benign": n, "bound": 1,
                    "corrected_risk": float(corrected), "loss_sum": fp + alpha * uncertain,
                    "thresholds": SelectiveThresholds(t_safe, float(threshold)).to_dict(),
                    "guarantee": "marginal selective FPR under stated exchangeability assumptions"}
    return {"status": "infeasible", "alpha": alpha, "n_benign": n, "thresholds": None,
            "reason": "no threshold satisfies the finite-sample corrected surrogate risk"}


def risk_coverage(y, prob, *, points: int = 101) -> dict:
    """Confidence ranking, reusing eval AURC. Coverage is monotone; risk need not be.

    Confidence ties retain stable input order as in the existing helper; points
    within ties describe a ranking, not deployable scalar-threshold policies.
    """
    p, y = labelled(prob, y, probability=True)
    return coverage_error_curve(y, p >= 0.5, np.abs(p - 0.5), points=points)


def group_coverage(prob, thresholds: SelectiveThresholds, groups: dict) -> dict:
    decided = thresholds.transform(prob) != "UNCERTAIN"
    out = {}
    for name, values in groups.items():
        values = np.asarray([str(v) for v in values])
        if values.shape != decided.shape:
            raise ValueError("group labels are not aligned")
        out[name] = {value: {"n": int(np.sum(values == value)),
                             "coverage": float(decided[values == value].mean())}
                     for value in np.unique(values)}
    return out


def campaign_size_buckets(campaign_ids) -> list[str]:
    """Sizes within the supplied evaluation population, not the full corpus."""
    _, inverse, counts = np.unique(campaign_ids, return_inverse=True, return_counts=True)
    return ["1" if n == 1 else "2-5" if n <= 5 else "6-20" if n <= 20 else "21+"
            for n in counts[inverse]]
