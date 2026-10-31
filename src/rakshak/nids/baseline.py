"""Home-baseline calibration and benign-only anomaly scores for the network-IDS protocols.

Label-free by construction. The calibration slice is chosen from row positions only
(``calibration_slice``), the site transform takes a feature matrix and nothing else
(``fit_site_baseline``), and the anomaly detectors are fitted on the benign rows the caller
passes in. A site's rank compares a flow with that site's own unlabelled baseline.

Deployment reading: a Raspberry Pi sees its own home or SME traffic for its first hours. That
traffic is unlabelled and mostly benign. The "home baseline" is the transform and rank reference
a sensor could fit from those hours. The slice's attack share is whatever the dataset's base
rate is. Labels are read only to report that share, never to fit anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

QUANTILE_CAP = 1000
MIN_BENIGN_FOR_DETECTOR = 50


def unlabelled_head(n: int, k: int, seed: int) -> np.ndarray:
    """Sorted positions of the first ``min(k, n)`` rows of a seeded random permutation.

    Depends on ``n``, ``k`` and ``seed`` only. This is the "random unlabelled subset, first K"
    rule: a random order, then the first K flows, as a sensor would collect them.
    """
    if n <= 0 or k <= 0:
        raise ValueError(f"need n > 0 and k > 0, got n={n}, k={k}")
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[: min(k, n)])


def calibration_slice(n: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(slice, rest) row positions of a target with ``n`` rows.

    ``slice`` is ``unlabelled_head``. ``rest`` is every other position and is the only set a
    test may use. The two are disjoint and cover ``range(n)``. A slice that would leave nothing to
    test raises, so the caller records a skip.
    """
    if k >= n:
        raise ValueError(f"calibration slice of {k} rows leaves no rows of {n} to test")
    head = unlabelled_head(n, k, seed)
    rest = np.setdiff1d(np.arange(n), head, assume_unique=True)
    return head, rest


@dataclass
class SiteBaseline:
    """Per-feature quantile map fitted on one site's unlabelled baseline. Categorical columns are
    passed through unchanged, because a quantile map would change what a protocol or a flag
    value means."""

    numeric: np.ndarray  # bool mask over feature columns
    quantile: object  # fitted sklearn QuantileTransformer on the numeric columns

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = np.asarray(X, dtype=np.float64).copy()
        if self.numeric.any():
            out[:, self.numeric] = self.quantile.transform(out[:, self.numeric])
        return out.astype(np.float32)


def fit_site_baseline(X: np.ndarray, categorical: np.ndarray) -> SiteBaseline:
    """Fit the site transform on feature rows only. There is no label argument by design.

    ``categorical`` marks the columns that pass through. The map is uniform on [0, 1], so a value
    above the baseline maximum saturates at 1.
    """
    from sklearn.preprocessing import QuantileTransformer

    X = np.asarray(X, dtype=np.float64)
    cat = np.asarray(categorical, dtype=bool)
    if X.ndim != 2 or X.shape[1] != cat.shape[0]:
        raise ValueError(f"expected (rows, {cat.shape[0]}) baseline, got shape {X.shape}")
    if len(X) < 2:
        raise ValueError("a site baseline needs at least two rows")
    numeric = ~cat
    quantile = None
    if numeric.any():
        quantile = QuantileTransformer(
            n_quantiles=min(QUANTILE_CAP, len(X)), output_distribution="uniform",
            subsample=len(X), copy=True, random_state=0,
        ).fit(X[:, numeric])
    return SiteBaseline(numeric=numeric, quantile=quantile)


def site_rank(baseline_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Fraction of the site's baseline scores at or below each score, in [0, 1]. Higher means more
    unusual for this site, whatever the detector's native scale."""
    base = np.sort(np.asarray(baseline_scores, dtype=np.float64))
    if base.size == 0:
        raise ValueError("site rank needs a non-empty baseline")
    return np.searchsorted(base, np.asarray(scores, dtype=np.float64), side="right") / base.size


def fpr_at_recall(y_true: np.ndarray, score: np.ndarray, recall: float) -> float:
    """Smallest FPR on this split at which recall reaches ``recall`` (threshold-free, an oracle on
    the split it is computed on). NaN when a class is missing or the recall is never reached."""
    from sklearn.metrics import roc_curve

    y = np.asarray(y_true).astype(np.int64)
    s = np.asarray(score, dtype=np.float64)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    hit = np.flatnonzero(tpr >= recall)
    return float(fpr[hit[0]]) if hit.size else float("nan")


def _check_benign(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if len(X) < MIN_BENIGN_FOR_DETECTOR:
        raise ValueError(
            f"benign-only detector needs at least {MIN_BENIGN_FOR_DETECTOR} benign rows, "
            f"got {len(X)}")
    return X


class IsolationForestScorer:
    """Benign-only isolation forest. ``score`` is higher for more anomalous rows."""

    name = "isolation_forest"

    def __init__(self, seed: int, n_estimators: int = 200, max_samples: int = 256):
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.model = None

    def fit(self, X_benign: np.ndarray) -> IsolationForestScorer:
        from sklearn.ensemble import IsolationForest

        X = _check_benign(X_benign)
        self.model = IsolationForest(
            n_estimators=self.n_estimators, max_samples=min(self.max_samples, len(X)),
            random_state=self.seed).fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(np.asarray(X, dtype=np.float64))


class AutoencoderScorer:
    """Small benign-only MLP autoencoder on standardised features. ``score`` is the mean squared
    reconstruction error per row, higher for more anomalous rows."""

    name = "autoencoder"

    def __init__(self, seed: int, hidden: tuple[int, ...] = (16, 6, 16)):
        self.seed = seed
        self.hidden = hidden
        self.scaler = None
        self.net = None

    def fit(self, X_benign: np.ndarray) -> AutoencoderScorer:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        X = _check_benign(X_benign)
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.net = MLPRegressor(
            hidden_layer_sizes=self.hidden, activation="relu", max_iter=40,
            early_stopping=True, n_iter_no_change=5, batch_size=512,
            random_state=self.seed).fit(Z, Z)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(np.asarray(X, dtype=np.float64))
        return np.mean((self.net.predict(Z) - Z) ** 2, axis=1)
