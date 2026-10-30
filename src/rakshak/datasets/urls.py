"""URL corpus assembly: normalisation, dedup, label-conflict removal and url_features."""

from __future__ import annotations

from urllib.parse import urlsplit

import pandas as pd

from rakshak.datasets.common import cached_registered_domain, stable_id
from rakshak.datasets.loaders import (
    SOURCE_LEGITPHISH,
    SOURCE_OPENPHISH,
    SOURCE_PHISHTANK,
    SOURCE_PHRESHPHISH,
)
from rakshak.url_features import url_features

URL_SOURCE_PRIORITY = {
    SOURCE_PHISHTANK: 0,
    SOURCE_PHRESHPHISH: 1,
    SOURCE_LEGITPHISH: 2,
    SOURCE_OPENPHISH: 3,
}


def normalize_url(url: str) -> str:
    """Dedup key for a URL: scheme-free host + path + query.

    Lower-cases the host, drops a leading 'www.', the default ports, the fragment, the scheme
    and a trailing slash on an empty path. http and https copies of one page therefore dedup
    together (the scheme is often absent in feeds and in message text).
    """
    raw = url.strip()
    candidate = raw if "://" in raw else "http://" + raw
    try:
        parts = urlsplit(candidate)
        port = parts.port  # raises ValueError on a malformed port
    except ValueError:
        return raw.lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if port and port not in (80, 443):
        host = f"{host}:{port}"
    path = "" if parts.path in ("", "/") else parts.path
    query = f"?{parts.query}" if parts.query else ""
    return f"{host}{path}{query}"


def build_url_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Dedup on the normalised URL and attach url_features.

    URLs whose sources disagree on the label are dropped entirely (both sides), because their
    label cannot be trusted. Among agreeing duplicates the first row by source priority is
    kept. Returns (kept rows, per-source accounting).
    """
    df = raw.reset_index(drop=True).copy()
    df["row_order"] = range(len(df))
    df["url_norm"] = [normalize_url(u) for u in df["url"]]
    df["reg_domain"] = [cached_registered_domain(u) for u in df["url_norm"]]
    df["id"] = [stable_id(s, k) for s, k in zip(df["source"], df["row_key"], strict=True)]
    df["source_rank"] = df["source"].map(URL_SOURCE_PRIORITY)
    df = df.sort_values(["source_rank", "row_order"]).reset_index(drop=True)

    conflict = df.groupby("url_norm", sort=False)["label"].transform("nunique") > 1
    accounting: dict[str, dict[str, int]] = {}
    for src, grp in df.groupby("source", sort=False):
        accounting[src] = {
            "raw": int(len(grp)),
            "label_conflict_dropped": int(conflict[grp.index].sum()),
        }
    df = df[~conflict].reset_index(drop=True)

    rep = df.groupby("url_norm", sort=False)["id"].transform("first")
    is_dup = rep != df["id"]
    for src, grp in df.groupby("source", sort=False):
        accounting[src]["duplicate_dropped"] = int(is_dup[grp.index].sum())
    kept = df[~is_dup].reset_index(drop=True)
    for src, grp in kept.groupby("source", sort=False):
        accounting[src]["kept"] = int(len(grp))

    feats = pd.DataFrame([url_features(u) for u in kept["url"]])
    kept = pd.concat([kept, feats], axis=1)
    return kept, accounting
