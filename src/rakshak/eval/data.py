"""Data-contract loading: sms.parquet, urls.parquet and splits/*.json (see docs/CONTEXT.md).

The data root is configurable: an explicit path (which alone is tried), then ``$RAKSHAK_DATA``,
the known candidates (``/kaggle/input/rakshak-data``, ``/content/rakshak-data`` on Colab, the
repository's ``data/processed``), and
finally a recursive search of ``/kaggle/input`` (directories up to 5 levels deep). The first
location that holds the marker file (directly or one folder down) wins; among several recursive
hits a path containing ``rakshak-data`` is preferred.

Split files come in two layouts, and both are handled:
- flat: ``{"train": [...], "cal": [...], "test": [...]}``
- partitioned: ``{"family": ..., "seed": ..., "partitions": {name: {"train", "cal"?, "test"}}}``.
  Each partition is one run. A partition without ``cal`` gets a stratified 10% of its train
  rows carved out as calibration, seeded by the file's seed.

Evaluation rows are the test and calibration rows that are ``eval_ok`` and not ``synthetic``.
Synthetic rows can train a model but never make a headline number, as the context requires.
"""

from __future__ import annotations

import fnmatch
import functools
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from rakshak.canonicalize import Canon, canonicalize

logger = logging.getLogger(__name__)

ENV_VAR = "RAKSHAK_DATA"
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_DATA = KAGGLE_INPUT / "rakshak-data"
COLAB_DATA = Path("/content/rakshak-data")
KAGGLE_SEARCH_DEPTH = 5
DATA_FOLDER_NAME = "rakshak-data"
REPO_DATA = Path(__file__).resolve().parents[3] / "data" / "processed"
SMS_FILE = "sms.parquet"
URL_FILE = "urls.parquet"
SPLIT_DIR = "splits"
GROUP_COLUMNS: tuple[str, ...] = ("source", "lang", "label_fine")
URL_META_COLUMNS = frozenset({"id", "url", "url_norm", "label", "source", "row_key", "reg_domain",
                              "date", "brand"})
CAL_FRACTION = 0.1


def _direct_or_child(folder: Path, marker: str) -> Path | None:
    """``folder`` if it holds the marker, else the first child folder that does."""
    if (folder / marker).is_file():
        return folder
    if folder.is_dir():
        hits = sorted(folder.glob(f"*/{marker}"))
        if hits:
            return hits[0].parent
    return None


def _search_tree(base: Path, marker: str, max_depth: int = KAGGLE_SEARCH_DEPTH) -> list[Path]:
    """Every folder under ``base`` (itself at depth 0, at most ``max_depth`` levels down) that
    directly holds the marker file."""
    if not base.is_dir():
        return []
    base_depth = len(base.parts)
    hits: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        if marker in filenames:
            hits.append(Path(dirpath))
        if len(Path(dirpath).parts) - base_depth >= max_depth:
            dirnames[:] = []  # do not descend below the depth limit
    return hits


def _choose_hit(hits: list[Path]) -> Path:
    """Prefer a folder named like the dataset; then the shallowest, then the lexically first."""
    preferred = [h for h in hits if DATA_FOLDER_NAME in str(h)]
    pool = preferred or hits
    return min(pool, key=lambda p: (len(p.parts), str(p)))


def resolve_data_root(explicit: str | os.PathLike | None = None, marker: str = SMS_FILE) -> Path:
    """Find the folder holding the data contract (``marker`` file, directly or one folder down).

    An explicit path is the only location tried. Otherwise the order is ``$RAKSHAK_DATA``, the
    known candidates, then a recursive search of ``KAGGLE_INPUT`` for the marker file.
    """
    if explicit is not None:
        candidates = [Path(explicit)]
    else:
        candidates = [Path(os.environ[ENV_VAR])] if os.environ.get(ENV_VAR) else []
        candidates += [KAGGLE_DATA, COLAB_DATA, REPO_DATA]
    for candidate in candidates:
        found = _direct_or_child(candidate, marker)
        if found is not None:
            return found
    where = [str(c) for c in candidates]
    if explicit is None:
        hits = _search_tree(KAGGLE_INPUT, marker)
        if hits:
            chosen = _choose_hit(hits)
            if len(hits) > 1:
                logger.warning("several %s folders under %s: %s; using %s", marker, KAGGLE_INPUT,
                               sorted(map(str, hits)), chosen)
            logger.info("data root %s (found by searching %s for %s)", chosen, KAGGLE_INPUT, marker)
            return chosen
        where.append(f"{KAGGLE_INPUT} (recursive, depth <= {KAGGLE_SEARCH_DEPTH})")
    raise FileNotFoundError(f"no {marker} under {where}")


def load_sms(root: Path) -> pd.DataFrame:
    df = pd.read_parquet(root / SMS_FILE)
    missing = {"id", "text", "label_binary"} - set(df.columns)
    if missing:
        raise ValueError(f"{SMS_FILE} lacks columns {sorted(missing)}")
    df["id"] = df["id"].astype(str)
    if not df["id"].is_unique:
        raise ValueError(f"{SMS_FILE} has duplicate ids")
    return df.reset_index(drop=True)


def load_urls(root: Path) -> pd.DataFrame:
    df = pd.read_parquet(root / URL_FILE)
    missing = {"id", "url", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{URL_FILE} lacks columns {sorted(missing)}")
    df["id"] = df["id"].astype(str)
    if not df["id"].is_unique:
        raise ValueError(f"{URL_FILE} has duplicate ids")
    return df.reset_index(drop=True)


def url_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric URL feature columns: every numeric column that is not metadata or the label."""
    return [
        c for c in df.columns
        if c not in URL_META_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]


def eval_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows eligible for headline evaluation: ``eval_ok`` and not ``synthetic``."""
    mask = np.ones(len(df), dtype=bool)
    if "eval_ok" in df:
        mask &= df["eval_ok"].fillna(False).astype(bool).to_numpy()
    if "synthetic" in df:
        mask &= ~df["synthetic"].fillna(False).astype(bool).to_numpy()
    return mask


@dataclass(frozen=True)
class Run:
    """One train/cal/test evaluation: row positions into the table."""

    name: str
    train: np.ndarray
    cal: np.ndarray
    test: np.ndarray
    # Every row of the split's test partition, before the eval mask. ``test`` drops masked rows,
    # so transfer sets must exclude ``test_all``, not ``test``.
    test_all: np.ndarray
    notes: dict


def list_split_files(root: Path, patterns: Sequence[str] | None = None) -> list[Path]:
    files = sorted((root / SPLIT_DIR).glob("*.json"))
    if patterns:
        files = [f for f in files if any(fnmatch.fnmatch(f.stem, p) for p in patterns)]
    return files


def _ids_body(body: dict) -> dict[str, list[str]]:
    return {k: [str(i) for i in body[k]] for k in ("train", "cal", "test") if k in body}


def read_partitions(path: Path) -> dict[str, dict[str, list[str]]]:
    """Partition name to its ``train``/``cal``/``test`` id lists.

    A file with train and test lists directly under ``partitions`` (or at top level) is one
    partition named ``default``. Otherwise each dict under ``partitions`` holding train and test
    is one partition.
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    container = obj.get("partitions", obj)
    if isinstance(container.get("train"), list) and isinstance(container.get("test"), list):
        return {"default": _ids_body(container)}
    out = {
        name: _ids_body(body)
        for name, body in container.items()
        if isinstance(body, dict) and "train" in body and "test" in body
    }
    if not out:
        raise ValueError(f"{path.name}: no partition has train and test lists")
    return out


def _positions(index: pd.Index, ids: list[str]) -> tuple[np.ndarray, int]:
    pos = index.get_indexer(ids)
    found = pos[pos >= 0]
    return found.astype(np.int64), int((pos < 0).sum())


def make_run(
    name: str,
    body: dict[str, list[str]],
    df: pd.DataFrame,
    *,
    labels: np.ndarray,
    mask: np.ndarray | None,
    seed: int,
) -> Run:
    """Map one partition's ids to row positions. Applies the eval mask and carves cal if needed."""
    index = pd.Index(df["id"])
    train, miss_tr = _positions(index, body["train"])
    test, miss_te = _positions(index, body["test"])
    test_all = test
    has_cal = "cal" in body
    cal, miss_ca = _positions(index, body.get("cal", []))
    notes: dict = {"missing_ids": {"train": miss_tr, "test": miss_te, "cal": miss_ca}}
    if mask is not None:
        before = {"test": len(test), "cal": len(cal)}
        test = test[mask[test]]
        cal = cal[mask[cal]]
        notes["dropped_not_eval"] = {
            "test": before["test"] - len(test),
            "cal": before["cal"] - len(cal),
        }
    if not has_cal:
        tr_keep, cal_new = train_test_split(
            train, test_size=CAL_FRACTION, stratify=labels[train], random_state=seed
        )
        train, cal = np.sort(tr_keep), np.sort(cal_new)
        notes["cal_carved_from_train"] = int(len(cal))
    overlap = np.intersect1d(train, test).size + np.intersect1d(cal, test).size
    if overlap:
        raise ValueError(f"{name}: {overlap} rows appear in train or cal and in test")
    if np.intersect1d(train, cal).size:
        raise ValueError(f"{name}: rows appear in both train and cal")
    notes["sizes"] = {"train": int(len(train)), "cal": int(len(cal)), "test": int(len(test))}
    return Run(name=name, train=train, cal=cal, test=test, test_all=test_all, notes=notes)


def sample_cap(items: Sequence, cap: int | None, seed: int) -> list:
    """Seeded subsample of ``items`` to at most ``cap`` entries, kept in their original order.
    ``None`` or a cap of 0 or less keeps everything."""
    items = list(items)
    if not cap or cap <= 0 or len(items) <= cap:
        return items
    keep = np.random.default_rng(seed).choice(len(items), cap, replace=False)
    return [items[i] for i in sorted(keep)]


def transfer_positions(n_rows: int, test_all: np.ndarray) -> np.ndarray:
    """Row positions of the transfer set: every row of the table except the split's test
    partition (``Run.test_all``, which includes rows the eval mask drops)."""
    keep = np.ones(n_rows, dtype=bool)
    keep[np.asarray(test_all, dtype=np.int64)] = False
    return np.flatnonzero(keep).astype(np.int64)


def assert_no_test_rows(transfer_ids, test_ids) -> None:
    """Raise when any transfer id is a test id. Test-row logits must never reach a
    distillation file. Raises rather than asserts, so it also holds under ``python -O``."""
    overlap = sorted(set(map(str, transfer_ids)) & set(map(str, test_ids)))
    if overlap:
        raise ValueError(f"transfer set holds {len(overlap)} test ids, e.g. {overlap[:3]}")


def load_runs(
    root: Path,
    df: pd.DataFrame,
    *,
    patterns: Sequence[str] | None,
    labels: np.ndarray,
    mask: np.ndarray | None,
) -> list[Run]:
    """Every partition of every matching split file, as runs named ``<file>:<partition>``."""
    runs: list[Run] = []
    for path in list_split_files(root, patterns):
        obj = json.loads(path.read_text(encoding="utf-8"))
        seed = int(obj.get("seed", 0))
        for pname, body in read_partitions(path).items():
            name = path.stem if pname == "default" else f"{path.stem}:{pname}"
            runs.append(make_run(name, body, df, labels=labels, mask=mask, seed=seed))
    if not runs:
        raise FileNotFoundError(f"no split files match {patterns} under {root / SPLIT_DIR}")
    return runs


def group_values(df: pd.DataFrame, positions: np.ndarray,
                 columns: Sequence[str] = GROUP_COLUMNS) -> dict[str, list]:
    """Group labels for the given rows; columns the table lacks are skipped."""
    return {c: df[c].iloc[positions].tolist() for c in columns if c in df.columns}


CLUSTER_COLUMNS: dict[str, str] = {"sms": "campaign_id", "urls": "reg_domain"}


def cluster_values(df: pd.DataFrame, positions: np.ndarray, column: str) -> list[str]:
    """Cluster id per row for the cluster bootstrap (campaign ids for SMS, registered domains
    for URLs). Missing ids become a unique id per row, so an absent id is its own cluster."""
    if column not in df.columns:
        raise KeyError(f"cluster column {column!r} is not in the table")
    values = df[column].iloc[positions].tolist()
    return [
        f"__row_{int(p)}" if v is None or (isinstance(v, float) and np.isnan(v)) or v == ""
        else str(v)
        for p, v in zip(positions, values, strict=True)
    ]


@functools.lru_cache(maxsize=400_000)
def canon_of(text: str) -> Canon:
    """Cached ``canonicalize``. Use ``canonicalize`` directly for timing."""
    return canonicalize(text)


def canon_text(text: str) -> str:
    return canon_of(text).normalized
