"""Row sampling, host groups and the split functions used by the NIDS protocols.

Everything here is pure numpy and deterministic under a seed. The Kaggle job and the tests call
the same functions, so the disjointness guarantees are tested directly.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np


def plan_quotas(
    attack_counts: dict[str, int], benign_count: int, max_rows: int
) -> tuple[dict[str, float], float]:
    """Keep probabilities per stratum for a label-stratified cap of about ``max_rows`` rows.

    Benign rows get their share of the total (``benign_count / total``). The attack budget is
    split across attack strings by water-filling: each string gets ``min(count, level)``, so a
    rare attack keeps all of its rows and a common one is cut to the level. Returns the keep
    probability per attack string and the benign keep probability. Probabilities are 1 when
    nothing needs cutting.
    """
    total = benign_count + sum(attack_counts.values())
    if total <= max_rows:
        return {k: 1.0 for k in attack_counts}, 1.0
    if total <= 0:
        return {}, 1.0
    benign_quota = max_rows * benign_count / total if total else 0.0
    attack_budget = max(max_rows - benign_quota, 0.0)
    keep: dict[str, float] = {}
    counts = sorted(attack_counts.items(), key=lambda kv: kv[1])
    remaining = attack_budget
    for idx, (name, count) in enumerate(counts):
        share = remaining / (len(counts) - idx)
        take = min(count, share)
        keep[name] = (take / count) if count else 1.0
        remaining -= take
    benign_keep = (benign_quota / benign_count) if benign_count else 1.0
    return keep, min(benign_keep, 1.0)


def random_split(
    n: int, fractions: Sequence[float], seed: int, strata: np.ndarray | None = None
) -> list[np.ndarray]:
    """Shuffle row positions into parts with the given fractions (they must sum to 1).

    With ``strata`` the split is stratified: each stratum is shuffled and cut in the same
    proportions, so every part sees each stratum that has enough rows.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    rng = np.random.default_rng(seed)
    parts: list[list[np.ndarray]] = [[] for _ in fractions]
    groups = [np.arange(n)] if strata is None else [
        np.flatnonzero(strata == s) for s in np.unique(strata)
    ]
    for rows in groups:
        rows = rng.permutation(rows)
        edges = np.cumsum([round(f * len(rows)) for f in fractions[:-1]])
        pieces = np.split(rows, edges)
        for part, piece in zip(parts, pieces, strict=True):
            part.append(piece)
    return [np.sort(np.concatenate(p)) if p else np.zeros(0, np.int64) for p in parts]


def group_split(
    groups: np.ndarray, fractions: Sequence[float], seed: int
) -> list[np.ndarray]:
    """Assign whole groups to parts so that no group is in two parts.

    Groups are shuffled, then filled into parts in order until each part holds about its
    fraction of rows. Returns row positions per part. A part can be empty when there are too few
    groups; the caller checks class coverage.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    groups = np.asarray(groups)
    n = len(groups)
    uniq, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    targets = np.cumsum(np.asarray(fractions, dtype=np.float64)) * n
    part_of_group = np.empty(len(uniq), dtype=np.int64)
    filled = 0
    part = 0
    for g in order:
        while part < len(fractions) - 1 and filled >= targets[part]:
            part += 1
        part_of_group[g] = part
        filled += sizes[g]
    row_part = part_of_group[inverse]
    return [np.flatnonzero(row_part == p) for p in range(len(fractions))]


def union_find_groups(src: Sequence[Hashable], dst: Sequence[Hashable]) -> np.ndarray:
    """Connected components of the host graph where each flow links its two endpoints.

    Returns an integer group id per flow. Two flows share a group when a chain of shared
    endpoints links them. This is the strict host grouping: a host can sit in one group only.
    """
    parent: dict[Hashable, Hashable] = {}

    def find(x: Hashable) -> Hashable:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in zip(src, dst, strict=True):
        for node in (a, b):
            parent.setdefault(node, node)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    ids: dict[Hashable, int] = {}
    out = np.empty(len(src), dtype=np.int64)
    for i, a in enumerate(src):
        root = find(a)
        out[i] = ids.setdefault(root, len(ids))
    return out


def host_groups(
    src: Sequence[Hashable], dst: Sequence[Hashable], max_share: float = 0.4
) -> tuple[np.ndarray, str]:
    """Host groups for T4 and the bootstrap clusters.

    The strict connected-component grouping is used when its largest component holds at most
    ``max_share`` of the flows. Otherwise a single attacker or server ties most of the traffic
    into one component, which cannot be split, and the source-address grouping is used instead.
    Returns the group ids and the method name ("component" or "source").
    """
    comp = union_find_groups(src, dst)
    biggest = np.bincount(comp).max() / len(comp) if len(comp) else 0.0
    if biggest <= max_share:
        return comp, "component"
    uniq, inverse = np.unique(np.asarray([str(s) for s in src], dtype=object),
                              return_inverse=True)
    return inverse.astype(np.int64), "source"


def leave_family_out(families: np.ndarray, held_out: str) -> np.ndarray:
    """Boolean mask of the rows that stay in training when ``held_out`` is removed."""
    return np.asarray(families) != held_out


def lodo_sources(keys: Sequence[str], held_out: str) -> list[str]:
    """The source datasets for leave-one-dataset-out: every key except the held-out one."""
    if held_out not in keys:
        raise KeyError(f"held-out dataset {held_out!r} is not among {list(keys)}")
    return [k for k in keys if k != held_out]


def balanced_subsample(rows: np.ndarray, n: int, seed: int) -> np.ndarray:
    """``n`` row positions drawn uniformly without replacement, sorted. Requires n <= len(rows)."""
    rows = np.asarray(rows)
    if n > len(rows):
        raise ValueError(f"cannot draw {n} rows from {len(rows)}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, size=n, replace=False))


def lodo_balanced_rows(
    rows_by_key: dict[str, np.ndarray], held_out: str, seed: int
) -> dict[str, np.ndarray]:
    """Equal row count per source dataset for a leave-one-dataset-out fit.

    Each source keeps ``min`` of the source row counts, drawn at random, so no source dominates.
    The held-out dataset is removed before anything is drawn, and the result is asserted to
    contain no held-out key, so target rows cannot reach the training side.
    """
    sources = lodo_sources(list(rows_by_key), held_out)
    if len(sources) < 2:
        raise ValueError("leave-one-dataset-out needs at least two source datasets")
    n = min(len(np.asarray(rows_by_key[k])) for k in sources)
    out = {k: balanced_subsample(np.asarray(rows_by_key[k]), n, seed + i)
           for i, k in enumerate(sources)}
    assert held_out not in out, "held-out dataset leaked into the LODO training side"
    return out
