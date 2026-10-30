"""Summary statistics for the build report (JSON) and the datacard tables."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from rakshak.datasets.dedup import campaign_size_histogram


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def label_balance(df: pd.DataFrame, by: str = "source", label: str = "label_binary") -> dict:
    out: dict[str, dict[str, int]] = {}
    for src, grp in df.groupby(by):
        out[str(src)] = {
            "negative": int((grp[label] == 0).sum()),
            "positive": int((grp[label] == 1).sum()),
        }
    return out


def campaign_stats(kept: pd.DataFrame, top_n: int = 20) -> dict:
    sizes = kept.groupby("campaign_id").size()
    sources_per = kept.groupby("campaign_id")["source"].nunique()
    labels_per = kept.groupby("campaign_id")["label_binary"].nunique()
    top = []
    for cid in sizes.sort_values(ascending=False).index[:top_n]:
        grp = kept[kept["campaign_id"] == cid]
        top.append(
            {
                "campaign_id": cid,
                "size": int(len(grp)),
                "label_mix": _counts(grp["label_fine"]),
                "sources": _counts(grp["source"]),
                "example": str(grp["text"].iloc[0])[:120],
                "template_example": str(grp["template_key"].iloc[0])[:120],
            }
        )
    return {
        "n_campaigns": int(len(sizes)),
        "n_rows": int(len(kept)),
        "singletons": int((sizes == 1).sum()),
        "largest": int(sizes.max()),
        "size_histogram": campaign_size_histogram(sizes.tolist()),
        "campaigns_spanning_multiple_sources": int((sources_per > 1).sum()),
        "campaigns_with_both_labels": int((labels_per > 1).sum()),
        "top": top,
    }


def partition_stats(frame: pd.DataFrame, partition: dict[str, list[str]]) -> dict[str, dict]:
    by_id = frame.set_index("id")
    out = {}
    for split, ids in partition.items():
        sub = by_id.loc[ids] if ids else by_id.iloc[0:0]
        entry: dict = {"n": int(len(sub))}
        if len(sub):
            label_col = "label_binary" if "label_binary" in sub else "label"
            entry["label_binary"] = _counts(sub[label_col])
            entry["source"] = _counts(sub["source"])
            if "lang" in sub:
                entry["lang"] = _counts(sub["lang"])
            if "campaign_id" in sub:
                entry["n_campaigns"] = int(sub["campaign_id"].nunique())
            if "reg_domain" in sub:
                entry["n_reg_domains"] = int(sub["reg_domain"].replace("", None).nunique())
            if "synthetic" in sub:
                entry["n_synthetic_known"] = int(sub["synthetic"].fillna(False).astype(bool).sum())
        out[split] = entry
    return out


def language_agreement(frame: pd.DataFrame) -> dict[str, dict[str, dict[str, int]]]:
    """Cross-tab of the detector's language bucket against the source's own language label."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    sub = frame[frame["src_lang"].notna() & frame["dup_of"].isna()]
    for src, grp in sub.groupby("source"):
        table: dict[str, dict[str, int]] = {}
        for src_l, det in zip(grp["src_lang"], grp["lang"], strict=True):
            table.setdefault(str(src_l), {})
            table[str(src_l)][str(det)] = table[str(src_l)].get(str(det), 0) + 1
        out[str(src)] = table
    return out


def top_counter(values: list[str], n: int = 10) -> dict[str, int]:
    return dict(Counter(values).most_common(n))
