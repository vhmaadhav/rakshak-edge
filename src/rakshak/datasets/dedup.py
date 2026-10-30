"""Exact dedup, MinHash-LSH near-duplicate pairs and campaign assignment.

Campaign = connected component of the graph whose edges are
  (a) near-duplicate template keys (estimated Jaccard >= NEAR_DUP_THRESHOLD over char
      5-gram shingles), computed over all kept messages, legitimate ones included; and
  (b) scam messages that share a registered domain which is not a shortener or a shared
      platform, provided no more than DOMAIN_FANOUT_CAP scam messages share it.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd
from datasketch import MinHash, MinHashLSH

from rakshak.datasets.common import (
    DOMAIN_FANOUT_CAP,
    NEAR_DUP_THRESHOLD,
    NON_CAMPAIGN_DOMAINS,
    NUM_PERM,
    SEED,
    SHINGLE_K,
)
from rakshak.datasets.unionfind import UnionFind


def mark_exact_duplicates(df: pd.DataFrame, key_col: str = "dedup_key") -> pd.DataFrame:
    """Add ``dup_of`` (representative id, else missing) and ``label_conflict``.

    ``df`` must already be sorted by source priority then row order: the first row of each
    key group is the representative. ``label_conflict`` is True for every row whose key group
    carries more than one binary label.
    """
    out = df.copy()
    rep_id = out.groupby(key_col, sort=False)["id"].transform("first")
    out["dup_of"] = rep_id.where(rep_id != out["id"])
    conflicts = out.groupby(key_col, sort=False)["label_binary"].transform("nunique") > 1
    out["label_conflict"] = conflicts.astype(bool)
    return out


def _shingles(template: str, k: int = SHINGLE_K) -> list[bytes]:
    s = template if len(template) >= k else template.ljust(k)
    return [s[i : i + k].encode("utf-8") for i in range(len(s) - k + 1)]


def minhash_signatures(templates: list[str]) -> list[MinHash]:
    """128-permutation MinHash over char 5-gram shingles, fixed seed (deterministic)."""
    sigs = []
    for t in templates:
        m = MinHash(num_perm=NUM_PERM, seed=SEED)
        m.update_batch(_shingles(t))
        sigs.append(m)
    return sigs


def near_duplicate_pairs(
    sigs: list[MinHash], threshold: float = NEAR_DUP_THRESHOLD
) -> list[tuple[int, int, float]]:
    """Index pairs (i, j, est_jaccard) with estimated Jaccard >= threshold.

    LSH (banding tuned by datasketch for ``threshold``) proposes candidates; each candidate is
    verified with the MinHash Jaccard estimate, so the output has no pair below threshold.
    LSH recall near the threshold is probabilistic, so pairs just above it can be missed.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    with lsh.insertion_session() as session:
        for i, m in enumerate(sigs):
            session.insert(i, m)
    out = []
    for i, m in enumerate(sigs):
        for j in lsh.query(m):
            if j <= i:
                continue
            est = m.jaccard(sigs[j])
            if est >= threshold:
                out.append((i, j, est))
    return sorted(out)


def cross_near_duplicates(
    pool: list[MinHash], queries: list[MinHash], threshold: float = NEAR_DUP_THRESHOLD
) -> list[list[tuple[int, float]]]:
    """For each query, the pool entries whose estimated Jaccard is at least ``threshold``.

    Returns one list per query of (pool index, estimated Jaccard), best first. Same LSH and
    verification as ``near_duplicate_pairs``, so the two agree on what counts as a near-duplicate.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    with lsh.insertion_session() as session:
        for j, m in enumerate(pool):
            session.insert(j, m)
    out: list[list[tuple[int, float]]] = []
    for q in queries:
        hits = []
        for j in lsh.query(q):
            est = q.jaccard(pool[j])
            if est >= threshold:
                hits.append((j, float(est)))
        out.append(sorted(hits, key=lambda h: (-h[1], h[0])))
    return out


def assign_campaigns(
    df: pd.DataFrame,
    threshold: float = NEAR_DUP_THRESHOLD,
    domain_cap: int = DOMAIN_FANOUT_CAP,
) -> tuple[pd.DataFrame, dict]:
    """Assign ``campaign_id`` to every row of ``df``.

    Needs columns: id, template_key, label_binary, reg_domains, dup_of. Only non-duplicate
    rows take part in clustering; a duplicate inherits its representative's campaign.
    Campaign ids are ``camp_000000`` onward, largest campaign first.
    """
    kept = df[df["dup_of"].isna()].reset_index(drop=True)
    uf = UnionFind(len(kept))
    sigs = minhash_signatures(kept["template_key"].tolist())
    pairs = near_duplicate_pairs(sigs, threshold)
    for i, j, _ in pairs:
        uf.union(i, j)

    domain_rows: dict[str, list[int]] = defaultdict(list)
    labels_kept = kept["label_binary"].tolist()
    for idx, doms in enumerate(kept["reg_domains"]):
        label = labels_kept[idx]
        if label != 1:
            continue
        for dom in set(doms):
            if dom and dom not in NON_CAMPAIGN_DOMAINS:
                domain_rows[dom].append(idx)
    capped: dict[str, int] = {}
    shared_joins = 0
    for dom, rows in domain_rows.items():
        if len(rows) < 2:
            continue
        if len(rows) > domain_cap:
            capped[dom] = len(rows)
            continue
        shared_joins += 1
        for r in rows[1:]:
            uf.union(rows[0], r)

    raw_ids = [f"x{lab}" for lab in uf.labels()]
    sizes = Counter(raw_ids)
    order = sorted(sizes, key=lambda c: (-sizes[c], c))
    rename = {old: f"camp_{i:06d}" for i, old in enumerate(order)}
    kept_campaign = dict(zip(kept["id"], (rename[c] for c in raw_ids), strict=True))

    out = df.copy()
    rep_of = out["dup_of"].where(out["dup_of"].notna(), out["id"])
    out["campaign_id"] = rep_of.map(kept_campaign)
    diagnostics = {
        "near_dup_pairs": len(pairs),
        "shared_domain_joins": shared_joins,
        "domains_over_fanout_cap": capped,
        "n_campaigns": len(sizes),
    }
    return out, diagnostics


def campaign_size_histogram(sizes: list[int]) -> dict[str, int]:
    bins = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 50), (51, 200), (201, 10**9)]
    labels = ["1", "2", "3-5", "6-10", "11-50", "51-200", "201+"]
    hist = dict.fromkeys(labels, 0)
    for s in sizes:
        for (lo, hi), lab in zip(bins, labels, strict=True):
            if lo <= s <= hi:
                hist[lab] += 1
                break
    return hist


def near_dup_sensitivity(sigs: list[MinHash], thresholds: list[float]) -> dict[str, int]:
    """Connected components over near-duplicate edges only, at each threshold."""
    out = {}
    for thr in thresholds:
        uf = UnionFind(len(sigs))
        for i, j, _ in near_duplicate_pairs(sigs, thr):
            uf.union(i, j)
        out[f"{thr:.2f}"] = len(set(uf.labels()))
    return out
