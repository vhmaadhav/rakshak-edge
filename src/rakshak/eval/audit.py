"""Clustered intervals for audit rates, including single-class benign cohorts."""

from __future__ import annotations

import numpy as np


def clustered_rate(values, cluster_ids, *, n_boot=1000, seed=13):
    """Resample whole campaigns with replacement; never substitute an IID interval.

    The percentile CI is descriptive for rare events: all-zero outcomes produce [0, 0],
    which does not establish zero population risk. Report Wilson bounds separately.
    Signed values support paired differences using exactly the same cluster draws.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(cluster_ids, dtype=str)
    if len(values) != len(groups):
        raise ValueError("one campaign ID per outcome is required")
    if not np.isfinite(values).all():
        raise ValueError("outcomes must be finite")
    if not len(values):
        return {"n": 0, "rate": None, "ci95": [None, None], "n_clusters": 0}
    _, codes = np.unique(groups, return_inverse=True)
    sizes = np.bincount(codes)
    totals = np.bincount(codes, weights=values)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.integers(len(sizes), size=len(sizes))
        draws[i] = totals[chosen].sum() / sizes[chosen].sum()
    return {
        "n": len(values), "rate": float(values.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "n_clusters": len(sizes), "n_boot": n_boot, "seed": seed,
        "method": "campaign_cluster_percentile",
        "degenerate": bool(np.all(values == values[0])),
    }


def paired_rate_test(a, b, cluster_ids, *, n_boot=1000, seed=13):
    """Paired A-B rate difference and a two-sided null-centred cluster bootstrap test."""
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    out = clustered_rate(delta, cluster_ids, n_boot=n_boot, seed=seed)
    if not len(delta):
        return {**out, "p_value": None}
    # Reuse identical seeded draws to test the centred bootstrap null distribution.
    _, codes = np.unique(np.asarray(cluster_ids, dtype=str), return_inverse=True)
    sizes, totals = np.bincount(codes), np.bincount(codes, weights=delta)
    rng = np.random.default_rng(seed)
    observed = float(delta.mean())
    extreme = 0
    for _ in range(n_boot):
        chosen = rng.integers(len(sizes), size=len(sizes))
        resampled = totals[chosen].sum() / sizes[chosen].sum()
        extreme += abs(resampled - observed) >= abs(observed) - 1e-12
    return {**out, "p_value": float((extreme + 1) / (n_boot + 1)), "direction": "A minus B"}


def clustered_ratio(numerators, denominators, cluster_ids, *, n_boot=1000, seed=13):
    """Pooled point pass fraction, resampling campaigns with all their points together."""
    num, den = np.asarray(numerators, dtype=float), np.asarray(denominators, dtype=float)
    groups = np.asarray(cluster_ids, dtype=str)
    if len(num) != len(den) or len(num) != len(groups):
        raise ValueError("point counts and campaign IDs must align")
    if not (np.isfinite(num).all() and np.isfinite(den).all()) or np.any((num < 0) | (num > den)):
        raise ValueError("point pass counts must be finite and between zero and denominator")
    if not len(num) or den.sum() == 0:
        return {"rate": None, "ci95": [None, None], "events": int(num.sum()),
                "total": int(den.sum()), "n_boot": 0}
    _, codes = np.unique(groups, return_inverse=True)
    num_total, den_total = np.bincount(codes, weights=num), np.bincount(codes, weights=den)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        chosen = rng.integers(len(num_total), size=len(num_total))
        d = den_total[chosen].sum()
        if d > 0:
            draws.append(num_total[chosen].sum() / d)
    return {
        "rate": float(num.sum() / den.sum()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist() if draws else [None, None],
        "events": int(num.sum()), "total": int(den.sum()),
        "n_clusters": len(num_total), "n_boot": n_boot, "valid_draws": len(draws),
        "seed": seed, "method": "campaign_cluster_ratio_percentile",
    }
