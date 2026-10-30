"""SMS corpus assembly: raw frame -> analysis-ready frame with dedup and campaigns."""

from __future__ import annotations

import pandas as pd

from rakshak.canonicalize import canonicalize
from rakshak.datasets.common import cached_registered_domain, dedup_key, stable_id
from rakshak.datasets.dedup import assign_campaigns, mark_exact_duplicates
from rakshak.datasets.lang import detect_lang
from rakshak.datasets.loaders import (
    SOURCE_ANMOL,
    SOURCE_BOLEWARA,
    SOURCE_IMC25,
    SOURCE_MISHRA,
    SOURCE_SCAMSHIELD,
    SOURCE_UCI,
)
from rakshak.datasets.taxonomy import policy_for
from rakshak.datasets.template import template_key

# Exact-duplicate representative = first row in this source order (provenance strength).
SOURCE_PRIORITY = {
    SOURCE_UCI: 0,
    SOURCE_MISHRA: 1,
    SOURCE_IMC25: 2,
    SOURCE_SCAMSHIELD: 3,
    SOURCE_ANMOL: 4,
    SOURCE_BOLEWARA: 5,
}

SMS_COLUMNS = [
    "id",
    "text",
    "text_canon",
    "label_binary",
    "label_fine",
    "label_policy",
    "label_scam",
    "source",
    "subsource",
    "source_split",
    "lang",
    "src_lang",
    "date",
    "urls",
    "reg_domains",
    "synthetic",
    "augmentation_only",
    "eval_ok",
    "template_key",
    "campaign_id",
    "dup_of",
    "label_conflict",
]


def _is_true(value: object) -> bool:
    """True only for a known-positive flag (pd.NA / None / NaN count as unknown)."""
    return not pd.isna(value) and bool(value)


def build_sms_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Canonicalise, detect language, dedup and cluster campaigns.

    Returns the full frame (duplicates included, ``dup_of`` set) and campaign diagnostics.

    ``eval_ok`` marks rows that may appear in a test or calibration set: kept (not a
    duplicate), not known-synthetic, not from an augmentation-only source, and not part of a
    text with conflicting labels.
    """
    df = raw.reset_index(drop=True).copy()
    df["row_order"] = range(len(df))
    canon = [canonicalize(t) for t in df["text"].astype(str)]
    df["text_canon"] = [c.normalized for c in canon]
    df["urls"] = [c.urls for c in canon]
    df["reg_domains"] = [
        sorted({d for d in (cached_registered_domain(u) for u in c.urls) if d}) for c in canon
    ]
    df["lang"] = [detect_lang(t) for t in df["text_canon"]]
    df["id"] = [stable_id(s, k) for s, k in zip(df["source"], df["row_key"], strict=True)]
    df["template_key"] = [template_key(t) for t in df["text_canon"]]
    df["dedup_key"] = [dedup_key(t) for t in df["text_canon"]]
    df["source_rank"] = df["source"].map(SOURCE_PRIORITY)
    # Within a source, known-real rows win over synthetic ones, then file order.
    df["synth_rank"] = [1 if _is_true(v) else 0 for v in df["synthetic"]]
    df = df.sort_values(["source_rank", "synth_rank", "row_order"]).reset_index(drop=True)

    df = mark_exact_duplicates(df, "dedup_key")
    policies = [
        policy_for(src, fine, bool(conflict))
        for src, fine, conflict in zip(
            df["source"], df["label_fine"], df["label_conflict"], strict=True
        )
    ]
    df["label_policy"] = [p.label_policy for p in policies]
    df["label_scam"] = pd.array([p.label_scam for p in policies], dtype="Int8")
    df, campaign_diag = assign_campaigns(df)

    df["eval_ok"] = [
        bool(pd.isna(dup))  # kept, not a duplicate
        and not _is_true(s)
        and not aug
        and not conflict
        for dup, s, aug, conflict in zip(
            df["dup_of"],
            df["synthetic"],
            df["augmentation_only"],
            df["label_conflict"],
            strict=True,
        )
    ]
    df["synthetic"] = df["synthetic"].astype("boolean")
    out = df[SMS_COLUMNS + ["dedup_key"]].copy()
    return out.reset_index(drop=True), campaign_diag


def cross_source_overlap(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Pre-dedup overlap: for each ordered pair (A, B), rows of A whose text_canon key also
    appears in B. Computed over the full raw corpus (before dedup removes the copies)."""
    keys = {src: set(grp["dedup_key"]) for src, grp in frame.groupby("source")}
    names = sorted(keys)
    return {a: {b: len(keys[a] & keys[b]) for b in names} for a in names}
