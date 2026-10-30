"""Attack harness: how far a detector's recall falls under each perturbation family.

For every family in ``rakshak.perturb.FAMILIES`` and every seed, each test scam is perturbed and
rescored at the calibration threshold. Reported per family:
- ``recall_under_attack``: fraction of perturbed scams still flagged, averaged over seeds.
- ``flip_rate``: 1 - recall_under_attack (scam classified as ham).
- ``attack_success_rate``: among scams flagged when clean, the fraction a perturbation makes
  miss. This is the usual evasion ASR.
- ``changed_rate``: fraction of messages the family actually modified. A family that changes
  nothing shows zero, so an unchanged message cannot count as a robust result.
- ``fpr_under_attack``: only for ``benign_sentence_insert``, applied to benign messages too.

The ``predict`` function receives raw message text and returns scam probabilities. Any
canonicalisation belongs to the caller, so the same harness runs with and without the
canonicaliser.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from rakshak.perturb import FAMILIES, perturb

ATTACK_SEEDS: tuple[int, ...] = (0, 1, 2)
HAM_FAMILIES: tuple[str, ...] = ("benign_sentence_insert",)
Predict = Callable[[list[str]], np.ndarray]


def _score(predict: Predict, texts: Sequence[str]) -> np.ndarray:
    if len(texts) == 0:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(predict(list(texts)), dtype=np.float64)


def robustness_report(
    predict: Predict,
    positives: Sequence[str],
    negatives: Sequence[str],
    threshold: float,
    *,
    families: Sequence[str] | None = None,
    seeds: Sequence[int] = ATTACK_SEEDS,
    max_positives: int | None = None,
    subsample_seed: int = 0,
) -> dict:
    """Run every family over the scam texts (and benign texts for the ham families).

    ``max_positives`` caps the scams used, by a seeded random subsample, for a quicker run.
    """
    families = tuple(FAMILIES) if families is None else tuple(families)
    unknown = [f for f in families if f not in FAMILIES]
    if unknown:
        raise KeyError(f"unknown families: {unknown}")
    pos = list(positives)
    neg = list(negatives)
    if max_positives is not None and len(pos) > max_positives:
        keep = np.random.default_rng(subsample_seed).choice(len(pos), max_positives, replace=False)
        pos = [pos[i] for i in sorted(keep)]

    clean_pos = _score(predict, pos)
    clean_neg = _score(predict, neg)
    clean_hit = clean_pos >= threshold
    report: dict = {
        "threshold": float(threshold),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "seeds": [int(s) for s in seeds],
        "clean": {
            "recall": float(clean_hit.mean()) if len(pos) else float("nan"),
            "fpr": float((clean_neg >= threshold).mean()) if len(neg) else float("nan"),
        },
        "families": {},
    }

    for family in families:
        recalls, asrs, changed = [], [], []
        fprs: list[float] = []
        for seed in seeds:
            perturbed = [perturb(t, family, seed) for t in pos]
            diff = [a != b for a, b in zip(perturbed, pos, strict=True)]
            changed.append(float(np.mean(diff)) if pos else float("nan"))
            hit = _score(predict, perturbed) >= threshold
            recalls.append(float(hit.mean()) if len(pos) else float("nan"))
            if clean_hit.any():
                asrs.append(float((~hit[clean_hit]).mean()))
            else:
                asrs.append(float("nan"))
            if family in HAM_FAMILIES and neg:
                ham_perturbed = [perturb(t, family, seed) for t in neg]
                fprs.append(float((_score(predict, ham_perturbed) >= threshold).mean()))
        rec = float(np.nanmean(recalls))
        entry = {
            "recall_under_attack": rec,
            "recall_under_attack_std": float(np.nanstd(recalls)),
            "flip_rate": 1.0 - rec,
            "attack_success_rate": float(np.nanmean(asrs)),
            "changed_rate": float(np.nanmean(changed)),
        }
        if family in HAM_FAMILIES:
            entry["fpr_under_attack"] = float(np.mean(fprs)) if fprs else float("nan")
        report["families"][family] = entry

    if report["families"]:
        recalls = {f: v["recall_under_attack"] for f, v in report["families"].items()}
        worst = min(recalls, key=lambda f: (np.nan_to_num(recalls[f], nan=2.0), f))
        report["summary"] = {
            "mean_recall_under_attack": float(np.nanmean(list(recalls.values()))),
            "worst_family": worst,
            "worst_recall_under_attack": recalls[worst],
            "mean_attack_success_rate": float(
                np.nanmean([v["attack_success_rate"] for v in report["families"].values()])
            ),
            "mean_changed_rate": float(
                np.nanmean([v["changed_rate"] for v in report["families"].values()])
            ),
        }
    return report
