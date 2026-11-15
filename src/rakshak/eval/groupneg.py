"""Group-disjoint folds for cross-fitted negatives (docs/DETECTOR_FIX_PLAN.md, section 3).

Pure standard library, so it runs without torch. Two rows are linked when their normalised template
keys are equal or their char-5-gram Jaccard is at least the threshold (the repo's near-duplicate
threshold, 0.7). Connected components are the clusters, and clusters (never rows) are assigned to
folds, so no near-duplicate template crosses a fold boundary.
"""

from __future__ import annotations

from collections.abc import Sequence

NEAR_DUP_JACCARD = 0.7


def _char5(key: str) -> frozenset[str]:
    padded = f" {key} "
    return frozenset(padded[i : i + 5] for i in range(max(1, len(padded) - 4)))


def negative_clusters(keys: Sequence[str], threshold: float = NEAR_DUP_JACCARD) -> list[int]:
    """Cluster index per row (union-find over equal keys and near-duplicate keys).

    Cluster indices are numbered by first appearance, so the output is deterministic.
    """
    norm = [str(k or "").strip().lower() for k in keys]
    grams = [_char5(k) for k in norm]
    parent = list(range(len(norm)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(norm)):
        for j in range(i + 1, len(norm)):
            if find(i) == find(j):
                continue
            if norm[i] == norm[j]:
                linked = True
            else:
                union = grams[i] | grams[j]
                linked = bool(union) and len(grams[i] & grams[j]) / len(union) >= threshold
            if linked:
                parent[find(j)] = find(i)

    ids: dict[int, int] = {}
    out: list[int] = []
    for i in range(len(norm)):
        root = find(i)
        ids.setdefault(root, len(ids))
        out.append(ids[root])
    return out


def assign_folds(clusters: Sequence[int], n_folds: int = 2) -> list[int]:
    """Fold per row. Whole clusters go to the fold with the fewest rows so far (largest cluster
    first, ties by cluster index), so a cluster never spans two folds."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    sizes: dict[int, int] = {}
    for c in clusters:
        sizes[c] = sizes.get(c, 0) + 1
    load = [0] * n_folds
    fold_of: dict[int, int] = {}
    for c in sorted(sizes, key=lambda c: (-sizes[c], c)):
        k = min(range(n_folds), key=lambda f: (load[f], f))
        fold_of[c] = k
        load[k] += sizes[c]
    return [fold_of[c] for c in clusters]


def cross_fit_folds(keys: Sequence[str], n_folds: int = 2) -> tuple[list[int], list[int]]:
    """The cross-fit fold construction of kaggle/jobs/sms_groupneg.py: template clusters over the
    keys (negative_clusters), then whole clusters assigned to folds (assign_folds).

    Returns (cluster per row, fold per row). Used by sms_groupneg and sms_crossfit_confirm, so both
    jobs build the same folds.
    """
    clusters = negative_clusters(keys)
    return clusters, assign_folds(clusters, n_folds=n_folds)
