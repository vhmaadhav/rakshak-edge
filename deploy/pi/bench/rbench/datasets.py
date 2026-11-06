"""Dataset loading for the benchmark: registry filters, row sampling and denominators.

Rows come from the parquet files named in registry.toml. The split files are the same JSON
the repo's eval code reads (flat ``partitions`` with train/cal/test id lists). Sampling uses a
seeded permutation so the same (dataset, limit, seed) always picks the same rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rbench.errors import ArtifactError, MissingDependency
from rbench.registry import Dataset

NIDS_FEATURES = 19


@dataclass
class Rows:
    """Sampled rows for one run. ``items`` is a list of dicts (row_id, text, label)."""

    items: list[dict[str, Any]]
    n_file: int
    n_filtered: int
    n_used: int
    denominator_notes: str
    extra: dict[str, Any] = field(default_factory=dict)


def _read_split_ids(path: Path, part: str) -> list[str]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read split file {path}: {exc}") from exc
    container = obj.get("partitions", obj)
    if part not in container or not isinstance(container[part], list):
        raise ArtifactError(f"split file {path.name} has no list {part!r}")
    return [str(i) for i in container[part]]


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise MissingDependency("pandas is not installed (pip install pandas pyarrow)") from exc
    return pd


def _sample(n: int, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.permutation(n)[:limit]


def load_rows(root: Path, dataset: Dataset, limit: int | None, seed: int) -> Rows:
    if dataset.synthetic:
        raise ValueError("synthetic datasets are built by synthetic_features()")
    pd = _pandas()
    data_path = root / dataset.path
    if not data_path.is_file():
        raise ArtifactError(f"dataset file missing: {data_path}")
    df = pd.read_parquet(data_path)
    n_file = len(df)
    notes: list[str] = [f"{dataset.path}: {n_file} rows"]
    if dataset.split_file:
        ids = set(_read_split_ids(root / dataset.split_file, dataset.split_part))
        df = df[df["id"].astype(str).isin(ids)]
        notes.append(
            f"{dataset.split_part} partition of {Path(dataset.split_file).name}: {len(df)} rows"
        )
    if dataset.split_column:
        df = df[df[dataset.split_column] == dataset.split_value]
        notes.append(f"{dataset.split_column} == {dataset.split_value!r}: {len(df)} rows")
    if dataset.row_filter == "eval_mask":
        keep = np.ones(len(df), dtype=bool)
        if "eval_ok" in df:
            keep &= df["eval_ok"].fillna(False).astype(bool).to_numpy()
        if "synthetic" in df:
            keep &= ~df["synthetic"].fillna(False).astype(bool).to_numpy()
        df = df[keep]
        notes.append(f"eval_ok and not synthetic: {len(df)} rows")
    elif dataset.row_filter == "policy_negative_scoreable":
        keep = (
            df["policy_negative"].fillna(False).astype(bool).to_numpy()
            & df["text_available"].fillna(False).astype(bool).to_numpy()
            & df["eval_ok"].fillna(False).astype(bool).to_numpy()
            & (df["label_binary"].to_numpy() == 0)
        )
        df = df[keep]
        notes.append(f"policy-negatives with text available: {len(df)} rows (denominator for FPR)")
    n_filtered = len(df)
    if n_filtered == 0:
        raise ArtifactError(f"dataset {dataset.id} has no rows after its filters")
    order = _sample(n_filtered, limit, seed)
    sub = df.iloc[order]
    label_col = "label_binary" if "label_binary" in sub.columns else "label"
    # ela_prompts rows carry the raw SMS; the prompt is built from it by the worker
    # (src/rakshak/ela_eval.py), so the SMS text is the item text here.
    text_col = {"url_binary": "url"}.get(dataset.task, "text")
    if text_col not in sub.columns:
        raise ArtifactError(f"dataset {dataset.id} has no {text_col!r} column")
    texts = sub[text_col].fillna("").astype(str).tolist()
    labels = sub[label_col].astype(int).tolist() if label_col in sub.columns else [None] * len(sub)
    row_ids = sub["id"].astype(str).tolist()
    items = [
        {"row_id": r, "text": t, "label": y} for r, t, y in zip(row_ids, texts, labels, strict=True)
    ]
    if limit is not None and limit < n_filtered:
        notes.append(f"sampled {len(items)} of {n_filtered} (seed {seed})")
    else:
        notes.append(f"used all {len(items)} rows")
    return Rows(
        items=items,
        n_file=n_file,
        n_filtered=n_filtered,
        n_used=len(items),
        denominator_notes="; ".join(notes),
    )


def synthetic_features(limit: int | None, seed: int, n_default: int = 2000) -> Rows:
    """Random NIDS feature rows for latency only. Not real traffic; no quality is reported."""
    n = int(limit) if limit else n_default
    rng = np.random.default_rng(seed)
    matrix = rng.lognormal(mean=2.0, sigma=1.5, size=(n, NIDS_FEATURES)).astype(np.float32)
    items = [{"row_id": f"synthetic-{i}", "features": matrix[i], "label": None} for i in range(n)]
    note = (
        f"SYNTHETIC: {n} seeded random rows x {NIDS_FEATURES} live features "
        f"(seed {seed}); latency only, no quality metrics"
    )
    return Rows(items=items, n_file=n, n_filtered=n, n_used=n, denominator_notes=note)
