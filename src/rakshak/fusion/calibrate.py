"""Portable scalar calibration and reliability data; fits require held-out scores."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from rakshak.eval.metrics import brier_score, expected_calibration_error


def scores_array(scores, *, probability: bool = False) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or not np.isfinite(s).all():
        raise ValueError("scores must be a finite vector")
    if probability and ((s < 0) | (s > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    return s


def labelled(scores, y, *, probability: bool = False) -> tuple[np.ndarray, np.ndarray]:
    s = scores_array(scores, probability=probability)
    labels = np.asarray(y)
    if labels.shape != s.shape or not np.isin(labels, [0, 1]).all() or not len(s):
        raise ValueError("nonempty aligned binary labels required")
    return s, labels.astype(int)


def logits(prob) -> np.ndarray:
    p = np.clip(scores_array(prob, probability=True), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


class TemperatureScaling:
    """Fit T on binary logits using bounded NLL minimisation; transform expects logits."""

    def __init__(self, bounds: tuple[float, float] = (0.01, 100.0)):
        if not 0 < bounds[0] < bounds[1] < float("inf"):
            raise ValueError("temperature bounds must be finite and positive")
        self.bounds = tuple(bounds)
        self.temperature: float | None = None

    def fit(self, scores, y) -> TemperatureScaling:
        s, y = labelled(scores, y)
        result = minimize_scalar(
            lambda t: float(np.mean(np.logaddexp(0, s / t) - y * s / t)),
            bounds=self.bounds, method="bounded",
        )
        if not result.success:
            raise RuntimeError("temperature optimisation failed")
        self.temperature = float(result.x)
        return self

    def transform(self, scores) -> np.ndarray:
        if self.temperature is None:
            raise ValueError("calibrator is not fitted")
        return expit(scores_array(scores) / self.temperature)

    def to_dict(self) -> dict:
        if self.temperature is None:
            raise ValueError("calibrator is not fitted")
        return {"kind": "temperature", "bounds": list(self.bounds),
                "temperature": self.temperature}

    @classmethod
    def from_dict(cls, state: dict) -> TemperatureScaling:
        obj = cls(tuple(state["bounds"]))
        obj.temperature = float(state["temperature"])
        if not np.isfinite(obj.temperature) or obj.temperature <= 0:
            raise ValueError("invalid temperature")
        return obj


class PlattScaling:
    """Unpenalised sigmoid fit on scalar scores (typically logits)."""

    def fit(self, scores, y) -> PlattScaling:
        s, y = labelled(scores, y)
        model = LogisticRegression(C=np.inf, max_iter=1000).fit(s[:, None], y)
        self.weight = float(model.coef_[0, 0])
        self.intercept = float(model.intercept_[0])
        return self

    def transform(self, scores) -> np.ndarray:
        return expit(self.weight * scores_array(scores) + self.intercept)

    def to_dict(self) -> dict:
        return {"kind": "platt", "weight": self.weight, "intercept": self.intercept}

    @classmethod
    def from_dict(cls, state: dict) -> PlattScaling:
        obj = cls()
        obj.weight, obj.intercept = float(state["weight"]), float(state["intercept"])
        return obj


class IsotonicCalibration:
    """Increasing piecewise-linear map with clipped out-of-range predictions."""

    def fit(self, scores, y) -> IsotonicCalibration:
        s, y = labelled(scores, y)
        model = IsotonicRegression(out_of_bounds="clip").fit(s, y)
        self.x, self.p = model.X_thresholds_, model.y_thresholds_
        return self

    def transform(self, scores) -> np.ndarray:
        return np.interp(scores_array(scores), self.x, self.p)

    def to_dict(self) -> dict:
        return {"kind": "isotonic", "x": self.x.tolist(), "p": self.p.tolist()}

    @classmethod
    def from_dict(cls, state: dict) -> IsotonicCalibration:
        obj = cls()
        obj.x, obj.p = np.asarray(state["x"], float), np.asarray(state["p"], float)
        return obj


def reliability_diagram(y, prob, *, n_bins: int = 15, strategy: str = "width") -> list[dict]:
    """Bin counts, mean probability and event rate. Empty bins have null means.

    Equal-mass bins use quantile boundaries; duplicate boundaries collapse rather
    than splitting identical predictions arbitrarily. Width bins always number n_bins.
    """
    p, y = labelled(prob, y, probability=True)
    if n_bins < 1 or strategy not in ("width", "mass"):
        raise ValueError("positive n_bins and width/mass strategy required")
    edges = np.linspace(0, 1, n_bins + 1)
    if strategy == "mass":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        if len(edges) == 1:
            edges = np.array([edges[0], edges[0]])
    bins = np.searchsorted(edges[1:-1], p, side="right")
    out = []
    for i in range(len(edges) - 1):
        mask = bins == i
        out.append({"lower": float(edges[i]), "upper": float(edges[i + 1]),
                    "count": int(mask.sum()),
                    "mean_probability": float(p[mask].mean()) if mask.any() else None,
                    "event_rate": float(y[mask].mean()) if mask.any() else None})
    return out


def calibration_metrics(y, prob) -> dict:
    p, y = labelled(prob, y, probability=True)
    mass = reliability_diagram(y, p, strategy="mass")
    return {"ece_width": expected_calibration_error(y, p, n_bins=15),
            "ece_mass": sum(b["count"] / len(y) * abs(b["mean_probability"] - b["event_rate"])
                            for b in mass if b["count"]),
            "brier": brier_score(y, p), "reliability_mass": mass,
            "reliability_width": reliability_diagram(y, p)}
