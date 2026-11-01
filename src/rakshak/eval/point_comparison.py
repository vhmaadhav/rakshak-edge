"""Paired pooled point-pass difference, conditional on fixed model outputs."""

import numpy as np


def paired_point_pass(base_pass, tuned_pass, totals, campaigns, *, n_boot=1000, seed=13):
    base, tuned, den = (np.asarray(x, dtype=float) for x in (base_pass, tuned_pass, totals))
    if not (base.shape == tuned.shape == den.shape == (len(campaigns),)):
        raise ValueError("one pair of pass counts and a denominator per campaign row required")
    if (not all(np.isfinite(x).all() for x in (base, tuned, den))
            or np.any(base < 0) or np.any(tuned < 0)
            or np.any(base > den) or np.any(tuned > den)):
        raise ValueError("invalid point counts")
    if not den.sum():
        return {"total": 0, "delta": None, "ci95": [None, None], "p_value": None}
    _, groups = np.unique(np.asarray(campaigns, dtype=str), return_inverse=True)
    diff = np.bincount(groups, weights=tuned - base)
    sizes = np.bincount(groups, weights=den)
    observed = float(diff.sum() / sizes.sum())
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        chosen = rng.integers(len(sizes), size=len(sizes))
        denominator = sizes[chosen].sum()
        if denominator > 0:
            draws.append(diff[chosen].sum() / denominator)
    samples = np.asarray(draws)
    return {"base": float(base.sum() / den.sum()), "tuned": float(tuned.sum() / den.sum()),
            "total": int(den.sum()), "delta": observed,
            "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "p_value": float((1 + (np.abs(samples - observed) >= abs(observed) - 1e-12).sum())
                             / (len(samples) + 1)),
            "valid_draws": len(samples), "n_boot": n_boot, "seed": seed,
            "n_campaigns": len(sizes), "direction": "tuned minus base",
            "method": "paired campaign bootstrap; pooled point counts; null-centred p-value"}
