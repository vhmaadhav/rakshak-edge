"""Data pools for the SMS student retrain (docs/SMS_RETRAIN_PLAN.md, sections 2-4).

Two-level clustering (plan amendment, section 2.2):
  * split groups = connected components at char-5-gram Jaccard >= 0.7 (SPLIT_JACCARD). Splits are
    group-disjoint by split group. The FinEE validation hold-out is drawn by whole split group.
  * cap groups = connected components at Jaccard >= 0.9 (CAP_JACCARD), nested inside split groups.
    The 5-per-group cap and the round-robin budget apply to cap groups.

Pure numpy and pandas. MinHash is a numpy implementation (crc32 char-5-gram shingles, 128
permutations, LSH banding) so it scales to ~150k rows. ``rakshak.eval.groupneg`` is an O(n^2)
pure-Python clusterer, so it is not used here. Test-only texts are read only to remove groups.
"""

from __future__ import annotations

import hashlib
import os
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from rakshak.data.minimal_pairs import make_minimal_pairs

NUM_PERM = 128
BANDS = 32
ROWS = 4  # BANDS * ROWS == NUM_PERM
PRIME = 2_147_483_647
SIG_SEED = 20261107
SPLIT_JACCARD = 0.7
CAP_JACCARD = 0.9
TEST_JACCARD = 0.8
CAP_BUCKET = 200  # max rows per LSH bucket in self-joins (extra rows are skipped and recorded)
CAP_CROSS = 500  # max test rows per LSH bucket in cross-joins
PER_GROUP_CAP = 5
TOTAL_CAP = 12_000
TOTAL_CAP_FRACTION = 0.25
HOLDOUT_FRAC = 0.15
FPR_VAL_LIMIT = 0.01
FPR_CAL_LIMIT = 0.02

SOURCE_FINEE = "finee_synth"
SOURCE_CLOVE = "cloveai_india_spam_sms"
SOURCE_SKY = "skyaara"
SOURCE_RIDHAM = "ridham115"
# present in the external directory but never read into any train or validation pool
NEVER_IN_TRAIN = ("smsshield_eval", "ieee_ham", "groundlm_xlmr_probs")

_URL = re.compile(
    r"(?:https?://|www\.)\S+|\b[a-z0-9-]+\.(?:com|in|org|net|co|info|xyz|me|io|ly|ai)(?:/\S*)?"
)


def normalize(text: str) -> str:
    """Lowercase, URLs to ``<url>``, digits to ``0``, whitespace collapsed."""
    s = str(text).lower()
    s = _URL.sub(" <url> ", s)
    s = re.sub(r"\d", "0", s)
    return re.sub(r"\s+", " ", s).strip()


def load_indian_sources(root: Path) -> dict[str, pd.DataFrame]:
    """Every ``*.parquet`` in ``<root>/external/indian_legit`` (or in ``root`` itself if that
    folder is absent), keyed by file stem. Missing sources are absent keys."""
    root = Path(root)
    folder = root / "external" / "indian_legit"
    if not folder.is_dir():
        folder = root
    if not folder.is_dir():
        return {}
    return {p.stem: pd.read_parquet(p) for p in sorted(folder.glob("*.parquet"))}


@dataclass
class PrepResult:
    train_extra: pd.DataFrame  # id,text,label,source,cluster,cap_group (+pair_of,edit_family)
    legit_val: pd.DataFrame  # id,text,label=0,source,cluster,cap_group
    val_scam: pd.DataFrame  # id,text,label=1,source,cluster,cap_group
    report: dict = field(default_factory=dict)


# ---------------------------------------------------------------- MinHash / LSH (numpy)


def _shingles(norm: str) -> np.ndarray:
    s = norm if len(norm) >= 5 else norm.ljust(5)
    vals = np.fromiter(
        (zlib.crc32(s[i : i + 5].encode("utf-8")) for i in range(len(s) - 4)),
        dtype=np.uint64,
        count=len(s) - 4,
    )
    return np.unique(vals % np.uint64(PRIME))


_RNG = np.random.default_rng(SIG_SEED)
_A = _RNG.integers(1, PRIME, size=NUM_PERM, dtype=np.uint64)
_B = _RNG.integers(0, PRIME, size=NUM_PERM, dtype=np.uint64)
_COEF = np.arange(1, ROWS + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)


def signatures(norms: list[str]) -> np.ndarray:
    sig = np.empty((len(norms), NUM_PERM), dtype=np.uint32)
    p = np.uint64(PRIME)
    for r, norm in enumerate(norms):
        hs = _shingles(norm)
        vals = (_A[:, None] * hs[None, :] + _B[:, None]) % p
        sig[r] = vals.min(axis=1).astype(np.uint32)
    return sig


def band_keys(sig: np.ndarray) -> np.ndarray:
    blocks = sig.astype(np.uint64).reshape(sig.shape[0], BANDS, ROWS)
    return (blocks * _COEF).sum(axis=2, dtype=np.uint64)


def _estimates(sig: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    est = np.empty(len(ia), dtype=np.float64)
    for s in range(0, len(ia), 200_000):
        e = s + 200_000
        est[s:e] = (sig[ia[s:e]] == sig[ib[s:e]]).mean(axis=1)
    return est


def _verify(
    sig_a: np.ndarray, ia: np.ndarray, sig_b: np.ndarray, ib: np.ndarray, thr: float
) -> np.ndarray:
    return _estimates_between(sig_a, ia, sig_b, ib) >= thr


def _estimates_between(sig_a: np.ndarray, ia: np.ndarray, sig_b: np.ndarray, ib: np.ndarray):
    est = np.empty(len(ia), dtype=np.float64)
    for s in range(0, len(ia), 200_000):
        e = s + 200_000
        est[s:e] = (sig_a[ia[s:e]] == sig_b[ib[s:e]]).mean(axis=1)
    return est


def _self_candidates(keys: np.ndarray, stats: dict, tag: str) -> tuple[np.ndarray, np.ndarray]:
    n = keys.shape[0]
    out_i, out_j = [], []
    for band in range(BANDS):
        col = keys[:, band]
        order = np.argsort(col, kind="stable")
        sk = col[order]
        bounds = np.flatnonzero(np.diff(sk)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [n]))
        sizes = ends - starts
        for s, e in zip(starts[sizes >= 2], ends[sizes >= 2], strict=False):
            mem = order[s:e]
            if len(mem) > CAP_BUCKET:
                stats[f"{tag}_capped_buckets"] = stats.get(f"{tag}_capped_buckets", 0) + 1
                stats[f"{tag}_rows_beyond_cap"] = (
                    stats.get(f"{tag}_rows_beyond_cap", 0) + len(mem) - CAP_BUCKET
                )
                mem = mem[:CAP_BUCKET]
            ii, jj = np.triu_indices(len(mem), 1)
            out_i.append(mem[ii])
            out_j.append(mem[jj])
    if not out_i:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    a = np.concatenate(out_i)
    b = np.concatenate(out_j)
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    code = np.unique(lo * n + hi)
    return code // n, code % n


def _candidate_edges(norms: list[str], stats: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Candidate pairs with estimated Jaccard. Exact normalised duplicates are added explicitly
    (LSH bucket caps could otherwise drop them)."""
    sig = signatures(norms)
    keys = band_keys(sig)
    ia, ib = _self_candidates(keys, stats, "lsh")
    est = _estimates(sig, ia, ib)
    first: dict[str, int] = {}
    ex_a: list[int] = []
    ex_b: list[int] = []
    for r, t in enumerate(norms):
        if t in first:
            ex_a.append(first[t])
            ex_b.append(r)
        else:
            first[t] = r
    ia = np.concatenate([ia, np.asarray(ex_a, dtype=np.int64)])
    ib = np.concatenate([ib, np.asarray(ex_b, dtype=np.int64)])
    est = np.concatenate([est, np.ones(len(ex_a))])
    stats["verified_candidate_pairs_all"] = int(len(ia))
    return ia, ib, est


def _components(n: int, ia: np.ndarray, ib: np.ndarray, est: np.ndarray, thr: float) -> np.ndarray:
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    keep = est >= thr
    for a, b in zip(ia[keep].tolist(), ib[keep].tolist(), strict=False):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(r) for r in range(n)], dtype=np.int64)
    _, ids = np.unique(roots, return_inverse=True)
    return ids.astype(np.int64)


def two_level_labels(
    norms: list[str],
    stats: dict | None = None,
    split_thr: float = SPLIT_JACCARD,
    cap_thr: float = CAP_JACCARD,
) -> tuple[np.ndarray, np.ndarray]:
    """(split group, cap group) per row. Cap groups are components at ``cap_thr``; each is
    intersected with its split group, so a cap group always sits inside exactly one split group."""
    stats = {} if stats is None else stats
    n = len(norms)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if cap_thr < split_thr:
        raise ValueError("cap_thr must be at least split_thr (cap groups nest inside split groups)")
    ia, ib, est = _candidate_edges(norms, stats)
    split = _components(n, ia, ib, est, split_thr)
    cap_raw = _components(n, ia, ib, est, cap_thr)
    pairs = pd.DataFrame({"s": split, "c": cap_raw}).drop_duplicates()
    spanning = int((pairs.groupby("c")["s"].nunique() > 1).sum())
    stats["cap_groups_spanning_split_groups"] = spanning
    stats["split_groups"] = int(split.max()) + 1
    stats["cap_groups"] = int(cap_raw.max()) + 1
    cap = pd.Series(list(zip(split.tolist(), cap_raw.tolist(), strict=True))).factorize()[0]
    return split, np.asarray(cap, dtype=np.int64)


def test_contaminated(
    norms: list[str], test_norms: list[str], stats: dict | None = None
) -> np.ndarray:
    """True for each row that is an exact or near (Jaccard >= 0.8) duplicate of a test-only text."""
    stats = {} if stats is None else stats
    if not norms:
        return np.zeros(0, dtype=bool)
    exact = set(test_norms)
    flagged = np.array([t in exact for t in norms], dtype=bool)
    if not test_norms:
        return flagged
    sig_c = signatures(norms)
    sig_t = signatures(test_norms)
    keys_c = band_keys(sig_c)
    keys_t = band_keys(sig_t)
    for band in range(BANDS):
        tk = pd.DataFrame({"k": keys_t[:, band], "tid": np.arange(len(test_norms))})
        sizes = tk.groupby("k").size()
        over = set(sizes.index[sizes > CAP_CROSS].tolist())
        if over:
            stats["test_capped_keys"] = stats.get("test_capped_keys", 0) + len(over)
        tk = tk[~tk["k"].isin(over)]
        ck = pd.DataFrame({"k": keys_c[:, band], "row": np.arange(len(norms))})
        m = ck.merge(tk, on="k")
        if m.empty:
            continue
        keep = _verify(sig_c, m["row"].to_numpy(), sig_t, m["tid"].to_numpy(), TEST_JACCARD)
        flagged[m["row"].to_numpy()[keep]] = True
    stats["test_flagged_rows"] = int(flagged.sum())
    return flagged


# ---------------------------------------------------------------- pools


def threshold_rule(
    s_legit_val: np.ndarray, s_cal: np.ndarray, y_cal: np.ndarray
) -> tuple[float, bool]:
    """Smallest threshold with FPR <= 1 % on Indian legit validation and <= 2 % on cal negatives.

    Score >= tau is a positive. If no threshold passes both, fall back to the campaign-cal max-F1
    threshold and return flag True. Takes no test input.
    """
    s_val = np.sort(np.asarray(s_legit_val, dtype=float))
    s_c = np.asarray(s_cal, dtype=float)
    y_c = np.asarray(y_cal).astype(int)
    if s_val.size == 0:
        raise ValueError("legit validation set is empty")
    neg = np.sort(s_c[y_c == 0])
    if neg.size == 0:
        raise ValueError("campaign cal has no negatives")
    candidates = np.unique(np.concatenate([s_val, s_c]))
    val_fp = s_val.size - np.searchsorted(s_val, candidates, side="left")
    cal_fp = neg.size - np.searchsorted(neg, candidates, side="left")
    ok = (val_fp / s_val.size <= FPR_VAL_LIMIT) & (cal_fp / neg.size <= FPR_CAL_LIMIT)
    if ok.any():
        return float(candidates[np.argmax(ok)]), False
    pos = int((y_c == 1).sum())
    best_t, best_f1 = float(np.max(s_c)) if s_c.size else 0.5, -1.0
    for t in np.unique(s_c):
        pred = s_c >= t
        tp = int(np.sum(pred & (y_c == 1)))
        fp = int(np.sum(pred & (y_c == 0)))
        denom = 2 * tp + fp + (pos - tp)
        f1 = 2 * tp / denom if denom else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, True


def _cap_per_group(df: pd.DataFrame, col: str, cap: int) -> pd.DataFrame:
    rank = df.groupby(col).cumcount()
    return df[rank < cap]


def _round_robin_cap(
    df: pd.DataFrame, budget: int, seed: int, col: str = "cap_group"
) -> pd.DataFrame:
    """Keep at most ``budget`` rows, one per cap group per round (groups in seeded order)."""
    if len(df) <= budget:
        return df
    rng = np.random.default_rng(seed)
    groups = df[col].unique()
    group_rank = dict(zip(groups.tolist(), rng.permutation(len(groups)).tolist(), strict=True))
    depth = df.groupby(col).cumcount()
    key = pd.DataFrame(
        {"depth": depth.to_numpy(), "grank": df[col].map(group_rank).to_numpy()},
        index=df.index,
    )
    order = key.sort_values(["depth", "grank"]).index[:budget]
    return df.loc[order]


def build_pools(
    arm: str,
    sources: dict[str, pd.DataFrame],
    test_only_texts: list[str],
    n_campaign_train_neg: int,
    seed: int,
    groundlm_texts: list[str] | None = None,
    split_jaccard: float = SPLIT_JACCARD,
    cap_jaccard: float = CAP_JACCARD,
) -> PrepResult:
    """Train extra and validation pools for one arm ("A", "B" or "C"); plan sections 2 and 3."""
    if arm not in ("A", "B", "C"):
        raise ValueError(f"unknown arm {arm!r}")
    report: dict = {
        "arm": arm,
        "seed": seed,
        "split_jaccard": split_jaccard,
        "cap_jaccard": cap_jaccard,
        "sources_present": sorted(sources),
        "stages": {},
    }
    report["never_in_train_or_val"] = [s for s in NEVER_IN_TRAIN if s in sources]

    parts = []
    if SOURCE_FINEE in sources:
        f = sources[SOURCE_FINEE].copy()
        f["role"] = "train"
        parts.append(f)
    if arm == "C" and SOURCE_CLOVE in sources:
        c = sources[SOURCE_CLOVE].copy()
        c["role"] = "train"
        parts.append(c)
    if SOURCE_SKY in sources:
        s = sources[SOURCE_SKY].copy()
        s["role"] = "legit_val"
        parts.append(s)
    if SOURCE_RIDHAM in sources:
        r = sources[SOURCE_RIDHAM].copy()
        r["role"] = np.where(r["label"].astype(int) == 1, "val_scam", "legit_val")
        parts.append(r)
    cols = ["id", "text", "label", "source", "role"]
    cand = (
        pd.concat([p[cols] for p in parts], ignore_index=True)
        if parts
        else pd.DataFrame(columns=cols)
    )
    cand["label"] = cand["label"].astype(int)
    cand = cand.drop_duplicates(subset=["id"]).reset_index(drop=True)
    report["stages"]["candidates"] = int(len(cand))
    report["stages"]["candidates_by_source"] = cand["source"].value_counts().to_dict()

    norms = [normalize(t) for t in cand["text"]]
    test_norms = [normalize(t) for t in test_only_texts]
    lsh: dict = {}
    split_ids, cap_ids = two_level_labels(norms, lsh, split_jaccard, cap_jaccard)
    cand["cluster"] = split_ids  # split group: group-disjointness unit
    cand["cap_group"] = cap_ids  # cap group: 5-per-group cap and round-robin unit
    report["lsh_group_stats"] = lsh
    report["stages"]["split_groups"] = int(lsh.get("split_groups", 0))
    report["stages"]["cap_groups"] = int(lsh.get("cap_groups", 0))

    # 1. test-only contamination: drop whole split groups that touch a test-only text
    flagged = test_contaminated(norms, test_norms, report.setdefault("lsh_test_stats", {}))
    bad = set(cand.loc[flagged, "cluster"].tolist())
    report["stages"]["rows_dropped_test_contaminated_split_groups"] = int(
        cand["cluster"].isin(bad).sum()
    )
    report["stages"]["split_groups_dropped_test_contaminated"] = len(bad)
    cand = cand[~cand["cluster"].isin(bad)].reset_index(drop=True)

    # 2. 5-per-cap-group cap on train-side rows, applied before the split so the held-out FinEE
    #    rows get the same cap (validation is not flooded). Validation sources stay whole.
    rng = np.random.default_rng(seed)
    shuffled = cand.iloc[rng.permutation(len(cand))]
    is_train = shuffled["role"] == "train"
    capped = _cap_per_group(shuffled[is_train], "cap_group", PER_GROUP_CAP)
    report["stages"]["train_side_rows_before_cap"] = int(is_train.sum())
    report["stages"]["train_side_rows_after_cap"] = int(len(capped))
    cand = pd.concat([capped, shuffled[~is_train]], ignore_index=True)

    # 3. group-disjoint hold-out: whole split groups, seeded
    finee_groups = sorted(cand.loc[cand["source"] == SOURCE_FINEE, "cluster"].unique().tolist())
    n_hold = int(round(HOLDOUT_FRAC * len(finee_groups)))
    held = set(np.random.default_rng(seed + 1).permutation(finee_groups)[:n_hold].tolist())
    val_side = set(cand.loc[cand["role"] != "train", "cluster"].tolist()) | held
    report["stages"]["finee_split_groups"] = len(finee_groups)
    report["stages"]["finee_split_groups_held_out"] = len(held)

    val_mask = cand["cluster"].isin(val_side)
    legit_val_mask = (cand["role"] == "legit_val") | (
        (cand["source"] == SOURCE_FINEE) & cand["cluster"].isin(held)
    )
    legit_val = cand[val_mask & legit_val_mask & (cand["label"] == 0)].copy()
    val_scam = cand[cand["role"] == "val_scam"].copy()
    train_pool = cand[(~val_mask) & (cand["role"] == "train")].copy()
    report["stages"]["train_rows_in_validation_groups_dropped"] = int(
        ((cand["role"] == "train") & val_mask).sum()
    )

    # 4. arm C: drop every CloveAI split group that touches a GroundLM text (near-dup, 0.8)
    if arm == "C":
        gl_src = groundlm_texts if groundlm_texts is not None else test_only_texts
        gl_norms = [normalize(t) for t in gl_src]
        clove_rows = cand["source"].to_numpy() == SOURCE_CLOVE
        touch = test_contaminated([normalize(t) for t in cand["text"]], gl_norms, {})
        gl_groups = set(cand.loc[touch & clove_rows, "cluster"].tolist())
        before = len(train_pool)
        train_pool = train_pool[~train_pool["cluster"].isin(gl_groups)]
        report["stages"]["arm_c_clove_groundlm_split_groups_dropped"] = len(gl_groups)
        report["stages"]["arm_c_rows_dropped"] = before - len(train_pool)

    # 5. total cap on legit rows (round-robin over cap groups); cloveai scam rows get the budget
    budget = min(TOTAL_CAP, int(TOTAL_CAP_FRACTION * n_campaign_train_neg))
    report["stages"]["total_cap_budget"] = budget
    legit_train = _round_robin_cap(train_pool[train_pool["label"] == 0], budget, seed + 2)
    scam_train = _round_robin_cap(train_pool[train_pool["label"] == 1], budget, seed + 3)
    report["stages"]["train_extra_legit"] = int(len(legit_train))
    report["stages"]["train_extra_scam"] = int(len(scam_train))
    train_extra = pd.concat([legit_train, scam_train], ignore_index=True)

    # 6. arm B: minimal scam twins of the capped legit templates (twins stay in train)
    if arm == "B":
        twins = legit_train.assign(cluster=legit_train["cap_group"])
        pairs = make_minimal_pairs(twins, n=2000, seed=seed)
        if len(pairs):
            by_id = legit_train.set_index("id")
            pairs["cluster"] = pairs["pair_of"].map(by_id["cluster"]).to_numpy()
            pairs["cap_group"] = pairs["pair_of"].map(by_id["cap_group"]).to_numpy()
        report["stages"]["minimal_pairs"] = int(len(pairs))
        train_extra = pd.concat([train_extra, pairs], ignore_index=True)

    legit_val = legit_val.reset_index(drop=True)
    report["stages"]["legit_val"] = int(len(legit_val))
    report["stages"]["legit_val_by_source"] = legit_val["source"].value_counts().to_dict()
    report["stages"]["legit_val_meets_1500"] = bool(len(legit_val) >= 1500)
    report["stages"]["val_scam"] = int(len(val_scam))
    report["stages"]["train_extra_total"] = int(len(train_extra))
    report["stages"]["train_extra_split_groups"] = int(train_extra["cluster"].nunique())
    report["stages"]["train_extra_cap_groups"] = int(train_extra["cap_group"].nunique())
    val_split = set(legit_val["cluster"].tolist()) | set(val_scam["cluster"].tolist())
    report["stages"]["group_disjoint_train_vs_val"] = bool(
        set(train_extra["cluster"].tolist()).isdisjoint(val_split)
    )
    report["stages"]["max_rows_per_cap_group"] = (
        int(pd.concat([train_extra, legit_val, val_scam]).groupby("cap_group").size().max())
        if len(train_extra) + len(legit_val) + len(val_scam)
        else 0
    )

    def _out(df: pd.DataFrame, label: int | None = None) -> pd.DataFrame:
        out = df.copy()
        if label is not None:
            out["label"] = label
        return out[["id", "text", "label", "source", "cluster", "cap_group"]].reset_index(drop=True)

    extra_cols = ["id", "text", "label", "source", "cluster", "cap_group"]
    extra_cols += [c for c in ("pair_of", "edit_family") if c in train_extra.columns]
    return PrepResult(
        train_extra=train_extra[extra_cols].reset_index(drop=True),
        legit_val=_out(legit_val, 0),
        val_scam=_out(val_scam, 1),
        report=report,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_pinned(root: Path, name: str, sha256: str) -> Path:
    """The copy of ``name`` under ``root`` whose sha256 is the pin. Raises when none or several
    match. Errors carry names, digest prefixes and counts only, never file contents."""
    if not root.exists():
        raise FileNotFoundError(f"folder {root} does not exist")
    hits = [Path(d) / name for d, _dirs, files in os.walk(root) if name in files]
    if not hits:
        raise FileNotFoundError(f"no {name} under {root}")
    found = {str(p): _sha256(p) for p in hits}
    good = [p for p in hits if found[str(p)] == sha256]
    if not good:
        shas = sorted(v[:12] for v in found.values())
        raise ValueError(f"{name} sha256 mismatch: found {shas}, pinned {sha256[:12]}")
    if len(good) > 1:
        raise FileNotFoundError(f"{len(good)} copies of {name} match the pin")
    return good[0]
