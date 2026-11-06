"""Paired, campaign-clustered mean F1 over fixed training seeds.

Resample campaigns once per draw and apply those multiplicities to every seed and arm.
The interval describes campaign uncertainty conditional on these three training seeds.
"""

import numpy as np


def paired_seed_f1(y, pred_a, pred_b, campaigns, *, n_boot=1000, seed=13):
    y = np.asarray(y, dtype=bool)
    a, b = np.asarray(pred_a, dtype=bool), np.asarray(pred_b, dtype=bool)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != len(y):
        raise ValueError("predictions must align as seed by message")
    if len(campaigns) != len(y):
        raise ValueError("campaigns must align")
    _, groups = np.unique(np.asarray(campaigns, dtype=str), return_inverse=True)
    n = int(groups.max()) + 1

    def totals(predictions):
        return np.asarray([[np.bincount(groups, weights=values, minlength=n)
                            for values in (p & y, p & ~y, ~p & y)] for p in predictions])

    ta, tb = totals(a), totals(b)

    def f1(t, chosen):
        tp, fp, fn = t[:, :, chosen].sum(axis=2).T
        den = 2 * tp + fp + fn
        return float(np.divide(2 * tp, den, out=np.zeros_like(den), where=den > 0).mean())

    all_groups = np.arange(n)
    va, vb = f1(ta, all_groups), f1(tb, all_groups)
    rng = np.random.default_rng(seed)
    draws = np.asarray([(f1(ta, chosen), f1(tb, chosen))
                        for chosen in (rng.integers(n, size=n) for _ in range(n_boot))])
    delta = draws[:, 0] - draws[:, 1]
    return {"a": va, "b": vb, "delta": va - vb,
            "a_ci95": np.quantile(draws[:, 0], [0.025, 0.975]).tolist(),
            "b_ci95": np.quantile(draws[:, 1], [0.025, 0.975]).tolist(),
            "delta_ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
            "p_value": float((1 + (np.abs(delta - (va - vb)) >= abs(va - vb)).sum())
                             / (n_boot + 1)),
            "n_boot": n_boot, "seed": seed, "n_campaigns": n, "training_seeds": len(a),
            "method": "paired campaign bootstrap of mean F1 over fixed training seeds"}
