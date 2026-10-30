"""External SMS test sets. Evaluation only: their rows never enter a training partition.

- SmishX (Wang et al., SOUPS 2025; MIT): ``smishx_soups2025/dataset.csv``, 1,200 manually
  relabelled messages with label legitimate / spam / smishing.
- GroundLM 2026 cleaned Indian financial SMS test set (Gupta; Zenodo 20480151, Apache-2.0 for
  the code and redistributable data): member ``data/cleaned_test_1393.csv`` of the v4 archive,
  read in memory. 1,393 rows, label_cleaned 1 = fraud, 0 = safe. The 1,294 rows with
  label_cleaned 0 are the policy-negatives behind the paper's primary endpoint (global FPR).
  Rows from the IIITD collection are hash-only in the release (``sha256:<hex>``), so their text
  is not available and they cannot be scored or deduplicated.

Dedup against the training pool: exact (dedup key of the canonical text) and near-duplicate
(MinHash over template keys, the same threshold as the campaign clustering). Pool rows that
overlap any external message are returned so the build can drop them from TRAIN partitions. The
external messages themselves are never removed for overlap.

Raw files are untrusted: parsed with pandas and zipfile only, never extracted or executed.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from rakshak.canonicalize import canonicalize
from rakshak.datasets.common import NEAR_DUP_THRESHOLD, dedup_key, stable_id
from rakshak.datasets.dedup import cross_near_duplicates, minhash_signatures
from rakshak.datasets.lang import detect_lang
from rakshak.datasets.loaders import SOURCE_GROUNDLM, SOURCE_SMISHX
from rakshak.datasets.taxonomy import policy_for
from rakshak.datasets.template import template_key

SMISHX_PATH = Path("smishx_soups2025") / "dataset.csv"
GROUNDLM_ZIP = Path("zenodo_20480151_scamshield_v4") / "scamshield_emnlp_artifact_v4_2026-05-31.zip"
GROUNDLM_MEMBER = "scamshield_emnlp_artifact_v4/data/cleaned_test_1393.csv"
HASH_PREFIX = "sha256:"
EXTERNAL_COLUMNS = [
    "id",
    "source",
    "row_key",
    "text",
    "text_canon",
    "label_binary",
    "label_fine",
    "label_policy",
    "label_scam",
    "policy_negative",
    "category",
    "lang",
    "template_key",
    "dedup_key",
    "text_available",
    "dup_within_set",
    "label_conflict_in_set",
    "eval_ok",
    "pool_exact",
    "pool_near",
    "pool_matches",
]


def load_smishx(raw_dir: Path) -> pd.DataFrame:
    data = pd.read_csv(raw_dir / SMISHX_PATH, dtype=str, keep_default_na=False, encoding="utf-8")
    rows = []
    for i, (text, label) in enumerate(zip(data["SMS"], data["label"], strict=True)):
        fine = label.strip().lower()
        if fine not in {"legitimate", "spam", "smishing"}:
            raise ValueError(f"SmishX row {i}: unexpected label {label!r}")
        rows.append(
            dict(
                source=SOURCE_SMISHX,
                row_key=f"{i:05d}",
                text=text,
                label_fine=fine,
                # Legacy binary (spam + scam = 1), as for the training sources.
                label_binary=0 if fine == "legitimate" else 1,
                policy_negative=None,
                category=None,
            )
        )
    return pd.DataFrame(rows)


def load_groundlm(raw_dir: Path) -> pd.DataFrame:
    with zipfile.ZipFile(raw_dir / GROUNDLM_ZIP) as zf:
        payload = zf.read(GROUNDLM_MEMBER)
    data = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False, encoding="utf-8")
    rows = []
    for rec in data.itertuples(index=False):
        if rec.label_cleaned not in {"0", "1"}:
            raise ValueError(f"GroundLM row {rec.idx}: unexpected label {rec.label_cleaned!r}")
        text = None if rec.text.startswith(HASH_PREFIX) else rec.text
        is_fraud = rec.label_cleaned == "1"
        rows.append(
            dict(
                source=SOURCE_GROUNDLM,
                row_key=str(rec.idx),
                text=text,
                label_fine="fraud" if is_fraud else "safe",
                label_binary=int(is_fraud),
                policy_negative=not is_fraud,
                category=rec.template_bucket,
            )
        )
    return pd.DataFrame(rows)


def build_external_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Canonical text, language, keys, label taxonomy and within-set flags for one or more
    external sets. Rows without text keep their labels but are not eval-eligible."""
    df = raw.reset_index(drop=True).copy()
    df["text_available"] = df["text"].notna()
    ok = df["text_available"].to_numpy(dtype=bool)
    canon = [canonicalize(t).normalized if k else None for t, k in zip(df["text"], ok, strict=True)]
    df["text_canon"] = canon
    df["lang"] = [detect_lang(c) if c is not None else None for c in canon]
    df["template_key"] = [template_key(c) if c is not None else None for c in canon]
    df["dedup_key"] = [dedup_key(c) if c is not None else None for c in canon]
    df["id"] = [stable_id(s, k) for s, k in zip(df["source"], df["row_key"], strict=True)]
    # Within-set duplicates: the first occurrence is eval-eligible, later copies are not.
    df["dup_within_set"] = False
    df["label_conflict_in_set"] = False
    if ok.any():
        checked = df.loc[ok]
        df.loc[ok, "dup_within_set"] = checked.duplicated("dedup_key", keep="first").to_numpy()
        # Texts that appear with both labels inside one set are ambiguous: not eval-eligible.
        n_labels = checked.groupby("dedup_key")["label_binary"].transform("nunique")
        df.loc[ok, "label_conflict_in_set"] = (n_labels > 1).to_numpy()
    policies = [
        policy_for(s, f, bool(c))
        for s, f, c in zip(df["source"], df["label_fine"], df["label_conflict_in_set"], strict=True)
    ]
    df["label_policy"] = [p.label_policy for p in policies]
    df["label_scam"] = pd.array([p.label_scam for p in policies], dtype="Int8")
    df["eval_ok"] = (
        ok
        & ~df["dup_within_set"].to_numpy(dtype=bool)
        & ~df["label_conflict_in_set"].to_numpy(dtype=bool)
    )
    return df


def pool_overlap(
    external: pd.DataFrame, pool: pd.DataFrame, threshold: float = NEAR_DUP_THRESHOLD
) -> tuple[pd.DataFrame, set[str]]:
    """Mark external rows that also occur in the training pool.

    ``pool`` needs id, dedup_key and template_key (kept SMS rows). Adds pool_exact (same dedup
    key), pool_near (estimated Jaccard >= threshold over template keys, not exact), and
    pool_matches (count of pool rows matched by either). Returns the updated frame and the set
    of pool ids that match any checkable external row (the rows to drop from TRAIN).
    """
    out = external.copy()
    out["pool_exact"] = False
    out["pool_near"] = False
    out["pool_matches"] = 0
    pool_ids: set[str] = set()
    checkable = out.index[out["text_available"]]
    if len(checkable) == 0 or pool.empty:
        return out, pool_ids
    by_key: dict[str, list[str]] = {}
    for pid, key in zip(pool["id"], pool["dedup_key"], strict=True):
        by_key.setdefault(key, []).append(pid)
    pool_templates = pool["template_key"].tolist()
    ext_templates = out.loc[checkable, "template_key"].tolist()
    near = cross_near_duplicates(
        minhash_signatures(pool_templates), minhash_signatures(ext_templates), threshold
    )
    for pos, idx in enumerate(checkable):
        exact_ids = by_key.get(out.at[idx, "dedup_key"], [])
        near_idx = [j for j, _ in near[pos]]
        near_ids = [pool["id"].iat[j] for j in near_idx]
        matched = set(exact_ids) | set(near_ids)
        out.at[idx, "pool_exact"] = bool(exact_ids)
        out.at[idx, "pool_near"] = bool(set(near_ids) - set(exact_ids))
        out.at[idx, "pool_matches"] = len(matched)
        pool_ids |= matched
    return out, pool_ids


def overlap_report(external: pd.DataFrame, pool_size: int) -> dict:
    """Per-set counts for the build report. Counts are over every row of each set."""
    report: dict[str, dict] = {"pool_rows": int(pool_size), "sets": {}}
    for src, grp in external.groupby("source", sort=True):
        checkable = grp["text_available"]
        report["sets"][src] = {
            "rows": int(len(grp)),
            "text_available": int(checkable.sum()),
            "text_hash_only": int((~checkable).sum()),
            "eval_ok": int(grp["eval_ok"].sum()),
            "within_set_duplicates": int(grp["dup_within_set"].sum()),
            "label_conflict_in_set": int(grp["label_conflict_in_set"].sum()),
            "pool_exact": int(grp["pool_exact"].sum()),
            "pool_near_only": int(grp["pool_near"].sum()),
            "pool_any": int(((grp["pool_exact"]) | (grp["pool_near"])).sum()),
        }
    return report


def cross_set_overlap(external: pd.DataFrame) -> dict:
    """Exact and near-duplicate overlap between the external sets themselves (checkable rows)."""
    keys = external[external["text_available"]]
    names = sorted(keys["source"].unique())
    out: dict[str, int] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ka = set(keys.loc[keys["source"] == a, "dedup_key"])
            kb = set(keys.loc[keys["source"] == b, "dedup_key"])
            out[f"{a}|{b}|exact"] = len(ka & kb)
    return out


def policy_negative_mask(external: pd.DataFrame) -> np.ndarray:
    """Rows of the GroundLM policy-negative endpoint that can be scored (text available)."""
    flag = external["policy_negative"].fillna(False).astype(bool).to_numpy()
    return flag & external["text_available"].to_numpy(dtype=bool)
