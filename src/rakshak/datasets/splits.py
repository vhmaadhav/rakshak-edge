"""Partitions over the SMS and URL corpora. Every partition stores ids only.

Invariants (asserted by ``check_disjoint`` and covered by tests):
  * no campaign_id (SMS) or reg_domain (URL) appears in more than one split, except for the
    temporal partitions, which are defined by time and are reported with their overlap;
  * rows with eval_ok == False (synthetic, augmentation-only, label-conflicting) never appear
    in a test or cal set, and a campaign containing any such row is kept whole in train;
  * exact duplicates are never in any partition (they are not in the kept frame).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from rakshak.datasets.common import SEED
from rakshak.datasets.loaders import SOURCE_IMC25, SOURCE_MISHRA, SOURCE_SCAMSHIELD, SOURCE_UCI

CAMPAIGN_FRACTIONS = {"train": 0.70, "cal": 0.10, "test": 0.20}
TEMPORAL_FRACTIONS = {"train": 0.70, "cal": 0.10, "test": 0.20}
SOURCE_HOLDOUT_SOURCES = [SOURCE_UCI, SOURCE_MISHRA, SOURCE_SCAMSHIELD]
HINDI_SHIFT_LANGS = ("hinglish", "hi_deva")


def assign_group_splits(
    df: pd.DataFrame,
    group_col: str,
    fractions: dict[str, float],
    strat_cols: list[str],
    forced_col: str | None = None,
    seed: int = SEED,
) -> pd.Series:
    """Assign each row to a split with whole groups kept together.

    Groups containing any row with ``forced_col`` True go to the first split in
    ``fractions`` (train). The remaining groups are placed greedily, largest first (ties in
    a seeded random order), into the split that minimises the squared deviation from the
    target fractions, summed over every value of every stratification column (label, source,
    language). Returns a Series of split names aligned with ``df``.
    """
    names = list(fractions)
    train = names[0]
    totals: Counter = Counter()
    for col in strat_cols:
        for val, cnt in df[col].value_counts().items():
            totals[(col, str(val))] += int(cnt)
    total_rows = len(df)
    counts = {s: Counter() for s in names}
    sizes = dict.fromkeys(names, 0)
    assigned = np.empty(total_rows, dtype=object)

    arrays = {col: df[col].astype(str).to_numpy() for col in strat_cols}

    def group_counter(idx: np.ndarray) -> Counter:
        c: Counter = Counter()
        for col in strat_cols:
            c.update((col, v) for v in arrays[col][idx])
        return c

    groups = df.groupby(group_col, sort=True).indices
    keys = list(groups)
    rng = np.random.default_rng(seed)
    keys = [keys[i] for i in rng.permutation(len(keys))]
    keys.sort(key=lambda k: -len(groups[k]))

    forced_mask = df[forced_col].to_numpy(dtype=bool) if forced_col else np.zeros(total_rows, bool)
    free: list[tuple[object, np.ndarray, Counter]] = []
    for key in keys:
        idx = groups[key]
        if forced_mask[idx].any():
            assigned[idx] = train
            counts[train].update(group_counter(idx))
            sizes[train] += len(idx)
        else:
            free.append((key, idx, group_counter(idx)))

    def cost(split: str, add: Counter) -> float:
        frac = fractions[split]
        total = 0.0
        for k, tot in totals.items():
            dev = (counts[split][k] + add.get(k, 0) - frac * tot) / tot
            total += dev * dev
        return total

    for _key, idx, gcount in free:
        # Choose the split whose deviation shrinks most (or grows least). Comparing absolute
        # post-add deviations would always favour the split that is already full.
        best = min(names, key=lambda s: cost(s, gcount) - cost(s, Counter()))
        assigned[idx] = best
        counts[best].update(gcount)
        sizes[best] += len(idx)
    return pd.Series(assigned, index=df.index, dtype=object)


def _ids(df: pd.DataFrame, mask: pd.Series | np.ndarray) -> list[str]:
    return sorted(df.loc[mask, "id"].tolist())


def _ids_at(df: pd.DataFrame, positions: np.ndarray) -> list[str]:
    return sorted(df["id"].iloc[positions].tolist())


def _sms_forced(df: pd.DataFrame) -> pd.Series:
    return ~df["eval_ok"].astype(bool)


def campaign_partition(kept: pd.DataFrame) -> tuple[dict[str, list[str]], pd.Series]:
    """Campaign-disjoint 70/10/20 split. Returns ids per split and the per-row split."""
    df = kept.assign(_forced=_sms_forced(kept))
    split = assign_group_splits(
        df,
        "campaign_id",
        CAMPAIGN_FRACTIONS,
        ["label_binary", "source", "lang"],
        forced_col="_forced",
    )
    ids = {s: _ids(kept, split == s) for s in CAMPAIGN_FRACTIONS}
    return ids, split


def temporal_partition(kept: pd.DataFrame, campaign_split: pd.Series) -> dict[str, list[str]]:
    """Time-ordered split on IMC25 smishing (earliest 70 / next 10 / latest 20 percent).

    Ham has no dates, so it is paired with the campaign split: ham in campaign-train goes to
    temporal train, ham in campaign-cal to cal and ham in campaign-test to test. Scam rows
    come only from IMC25, so the temporal test's classes differ in source (see DATACARD).
    """
    imc = kept[(kept["source"] == SOURCE_IMC25) & kept["date"].notna()]
    imc = imc.sort_values(["date", "id"])
    n = len(imc)
    cut1 = int(round(n * TEMPORAL_FRACTIONS["train"]))
    cut2 = int(round(n * (TEMPORAL_FRACTIONS["train"] + TEMPORAL_FRACTIONS["cal"])))
    out: dict[str, list[str]] = {}
    ids_sorted = imc["id"].tolist()
    out["train"] = ids_sorted[:cut1]
    out["cal"] = ids_sorted[cut1:cut2]
    out["test"] = ids_sorted[cut2:]
    ham = kept[(kept["label_binary"] == 0) & kept["eval_ok"].astype(bool)]
    for s in ("train", "cal", "test"):
        ham_ids = ham.loc[campaign_split.reindex(ham.index) == s, "id"].tolist()
        out[s] = sorted(set(out[s]) | set(ham_ids))
    return out


def source_holdout_partitions(
    kept: pd.DataFrame, seed: int = SEED
) -> dict[str, dict[str, list[str]]]:
    """Leave-one-source-out folds for the sources that carry both classes.

    Test = eval-eligible rows of the held-out source. Train/cal = every other source, minus any
    row whose campaign also occurs in the held-out test set, split 90/10 by campaign.
    """
    folds: dict[str, dict[str, list[str]]] = {}
    for src in SOURCE_HOLDOUT_SOURCES:
        is_test = (kept["source"] == src) & kept["eval_ok"].astype(bool)
        test_campaigns = set(kept.loc[is_test, "campaign_id"])
        rest = kept[(kept["source"] != src) & ~kept["campaign_id"].isin(test_campaigns)]
        rest = rest.assign(_forced=_sms_forced(rest))
        split = assign_group_splits(
            rest,
            "campaign_id",
            {"train": 0.9, "cal": 0.1},
            ["label_binary", "source", "lang"],
            forced_col="_forced",
            seed=seed,
        )
        folds[f"holdout_{src}"] = {
            "train": _ids(rest, split == "train"),
            "cal": _ids(rest, split == "cal"),
            "test": _ids(kept, is_test),
        }
    return folds


def lang_shift_partition(kept: pd.DataFrame, campaign_split: pd.Series) -> dict[str, list[str]]:
    """Train and cal on English; test on Hinglish and Devanagari Hindi (eval-eligible only)."""
    en = kept["lang"] == "en"
    shift = kept["lang"].isin(HINDI_SHIFT_LANGS) & kept["eval_ok"].astype(bool)
    camp = campaign_split.reindex(kept.index)
    return {
        "train": _ids(kept, en & (camp == "train")),
        "cal": _ids(kept, en & (camp == "cal") & kept["eval_ok"].astype(bool)),
        "test": _ids(kept, shift & (camp == "test")),
    }


def legacy_partitions(kept: pd.DataFrame) -> dict[str, dict[str, dict[str, list[str]]]]:
    """Published-style protocols on the deduplicated pool, for comparability only.

    UCI and Mishra & Soni: stratified 80/20 holdout plus stratified 5-fold CV (seed 13).
    ScamShield: its official train / val / test files, with duplicates and ineligible rows
    removed from cal and test. Anmol and bolewara have no eval-eligible rows and are omitted.
    """
    out: dict[str, dict[str, dict[str, list[str]]]] = {}
    for name, src in (("legacy_uci_sms_spam", SOURCE_UCI), ("legacy_mishra_soni", SOURCE_MISHRA)):
        pool = kept[(kept["source"] == src) & kept["eval_ok"].astype(bool)]
        y = pool["label_binary"].to_numpy()
        idx = np.arange(len(pool))
        tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
        parts: dict[str, dict[str, list[str]]] = {
            "holdout": {"train": _ids_at(pool, tr), "test": _ids_at(pool, te)}
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for k, (tr_k, te_k) in enumerate(skf.split(idx, y)):
            parts[f"fold_{k}"] = {"train": _ids_at(pool, tr_k), "test": _ids_at(pool, te_k)}
        out[name] = parts
    sc = kept[kept["source"] == SOURCE_SCAMSHIELD]
    elig = sc["eval_ok"].astype(bool)
    out["legacy_scamshield"] = {
        "official": {
            "train": _ids(sc, sc["source_split"] == "train"),
            "cal": _ids(sc, (sc["source_split"] == "val") & elig),
            "test": _ids(sc, (sc["source_split"] == "test") & elig),
        }
    }
    return out


def url_domain_partition(urls: pd.DataFrame) -> tuple[dict[str, list[str]], pd.Series]:
    """Registered-domain-grouped 70/10/20 split. No registered domain crosses splits."""
    group = urls["reg_domain"].where(urls["reg_domain"] != "", urls["url_norm"])
    df = urls.assign(_group=group)
    split = assign_group_splits(df, "_group", CAMPAIGN_FRACTIONS, ["label", "source"])
    return {s: _ids(urls, split == s) for s in CAMPAIGN_FRACTIONS}, split


def url_temporal_partition(urls: pd.DataFrame) -> dict[str, list[str]]:
    """Time-ordered 70/10/20 split over the rows that carry a date (PhishTank, PhreshPhish)."""
    dated = urls[urls["date"].notna()].sort_values(["date", "id"])
    n = len(dated)
    c1 = int(round(n * 0.7))
    c2 = int(round(n * 0.8))
    ids = dated["id"].tolist()
    return {"train": ids[:c1], "cal": ids[c1:c2], "test": ids[c2:]}


def check_disjoint(frame: pd.DataFrame, partition: dict[str, list[str]], group_col: str) -> dict:
    """Groups shared between each pair of splits (should be 0 for group-disjoint splits)."""
    by_id = frame.set_index("id")[group_col]
    groups = {s: set(by_id.reindex(ids).dropna()) for s, ids in partition.items()}
    names = list(partition)
    overlap = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap[f"{a}-{b}"] = len(groups[a] & groups[b])
    return overlap
