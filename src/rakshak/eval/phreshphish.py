"""PhreshPhish protocol (arXiv 2507.10854 v2, Table 2) on the benchmark subsets.

Shared by kaggle/jobs/url_baselines.py and kaggle/jobs/url_student.py: the data loading (official
split, benchmark subsets, feature columns), the metrics and the report tables. Pure numpy, pandas
and sklearn, so nothing here imports torch.

Per base rate, every benchmark subset is scored and the metrics are averaged over the subsets:
- AP: average precision (the sklearn estimator, the same as ``metrics.pr_auc``).
- P@R=0.90: the highest precision among thresholds whose recall is at least 0.90.

A subset is the shared benign list plus one list of phish samples (benchmark_subsets.json, see
kaggle/jobs/phreshphish_extract.py). Scores come from one model for every benchmark row; a subset
only selects rows. Thresholds are not used, so no calibration split is needed for the protocol.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit

from rakshak.eval import data as D

PHRESH_FAMILY = "url_phreshphish"
PHRESH_FILE = "urls_phreshphish.parquet"
PHRESH_META = ("lang", "split", "bench_benign", "bench_full", "bench_subsets")
RATE_ORDER: tuple[str, ...] = ("0.05%", "0.1%", "0.5%", "1%", "5%")
RATE_FROM_PREFIX = {
    "benchmark-005": "0.05%",
    "benchmark-010": "0.1%",
    "benchmark-050": "0.5%",
    "benchmark-100": "1%",
    "benchmark-500": "5%",
}
RECALL_TARGET = 0.9
FULL_LABEL = "24% (full)"
# Paper v2, Table 2, GTE row (fine-tuned gte-large on URL + HTML): (AP, P@R=0.90) per base rate.
# Quoted for reference only. The paper publishes no URL-only row (checked 2026-10-30 against the
# full v2 text: the baselines Linear, FFN, GTE and the LLM all take URL and HTML).
PAPER_V2_GTE = {
    "0.05%": (0.6050, 0.0891),
    "0.1%": (0.6879, 0.1563),
    "0.5%": (0.8354, 0.3736),
    "1%": (0.8762, 0.5426),
    "5%": (0.9382, 0.8576),
    FULL_LABEL: (0.9779, 0.9746),
}


@dataclass(frozen=True)
class PhreshData:
    """The PhreshPhish release as the protocol uses it. Positions index ``df``."""

    df: pd.DataFrame
    labels: np.ndarray
    train: np.ndarray
    test: np.ndarray
    bench_full: np.ndarray
    subsets: list[tuple[str, np.ndarray]]
    feature_cols: list[str]


def paper_reference() -> dict[str, dict[str, float]]:
    """PAPER_V2_GTE as {rate: {"ap", "p_at_r90"}} for the report sections."""
    return {k: {"ap": a, "p_at_r90": p} for k, (a, p) in PAPER_V2_GTE.items()}


def phreshphish_subsets(df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    """(base-rate label, row positions) for every benchmark subset.

    A subset is the shared benign benchmark list plus the phish rows listed under its id in
    ``bench_subsets`` (benchmark-005/0 and so on). Positions index ``df``.
    """
    benign = np.flatnonzero(df["bench_benign"].to_numpy(dtype=bool))
    phish_by_subset: dict[str, list[int]] = {}
    for pos, ids in enumerate(df["bench_subsets"]):
        for sid in ids:
            phish_by_subset.setdefault(str(sid), []).append(pos)
    subsets = []
    for sid in sorted(phish_by_subset):
        prefix = sid.split("/")[0]
        rate = RATE_FROM_PREFIX.get(prefix, prefix)
        pos = np.concatenate([benign, np.asarray(phish_by_subset[sid], dtype=np.int64)])
        subsets.append((rate, pos))
    return subsets


def load_phreshphish(root: Path) -> PhreshData:
    """Read urls_phreshphish.parquet and the official split (splits/url_phreshphish.json)."""
    df = pd.read_parquet(root / PHRESH_FILE)
    if not df["id"].is_unique:
        raise ValueError(f"{PHRESH_FILE} has duplicate ids")
    part = D.read_partitions(root / D.SPLIT_DIR / f"{PHRESH_FAMILY}.json")["official"]
    index = pd.Index(df["id"])
    tr = index.get_indexer(part["train"])
    te = index.get_indexer(part["test"])
    if (tr < 0).any() or (te < 0).any():
        raise ValueError("PhreshPhish split ids missing from the table")
    return PhreshData(
        df=df,
        labels=df["label"].to_numpy(np.int64),
        train=tr.astype(np.int64),
        test=te.astype(np.int64),
        bench_full=np.flatnonzero(df["bench_full"].to_numpy(dtype=bool)),
        subsets=phreshphish_subsets(df),
        feature_cols=[c for c in D.url_feature_columns(df) if c not in PHRESH_META],
    )


def grouped_cal(
    rows: Sequence[int] | np.ndarray,
    labels: np.ndarray,
    groups: Sequence,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split row positions into (keep, cal) with every group wholly on one side.

    ``groups`` gives one group key per entry of ``rows`` (for PhreshPhish, the registered domain).
    Raises when the calibration side holds a single class, since the cal metrics need both.
    """
    rows = np.asarray(rows, dtype=np.int64)
    g = np.asarray(groups, dtype=object)
    if len(g) != len(rows):
        raise ValueError("one group key per row is required")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    keep_i, cal_i = next(
        GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(
            rows, labels[rows], groups=g
        )
    )
    keep, cal = np.sort(rows[keep_i]), np.sort(rows[cal_i])
    if len(np.unique(labels[cal])) < 2:
        raise ValueError("grouped cal split holds one class; raise the fraction or the rows")
    return keep, cal


def subsample_rows(rows: Sequence[int] | np.ndarray, n: int | None, seed: int) -> np.ndarray:
    """A uniform random subset of ``n`` rows (all rows when n is None or not smaller)."""
    rows = np.asarray(rows, dtype=np.int64)
    if n is None or n >= len(rows):
        return rows
    if n < 2:
        raise ValueError("a subsample needs at least 2 rows")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, size=int(n), replace=False))


def precision_at_recall(
    y_true: Sequence[int] | np.ndarray,
    score: Sequence[float] | np.ndarray,
    target: float = RECALL_TARGET,
) -> float:
    """Highest precision over thresholds whose recall is at least ``target``.

    NaN when the rows hold no positive. Ties in the score are handled by sklearn's curve.
    """
    y = np.asarray(y_true).astype(np.int64)
    s = np.asarray(score, dtype=np.float64)
    if (y == 1).sum() == 0:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y, s)
    ok = recall >= target
    return float(precision[ok].max()) if ok.any() else 0.0


def subset_metrics(y_true, score) -> dict:
    """AP, P@R=0.90, base rate and size for one subset."""
    y = np.asarray(y_true).astype(np.int64)
    s = np.asarray(score, dtype=np.float64)
    return {
        "n": int(len(y)),
        "positives": int((y == 1).sum()),
        "base_rate": float((y == 1).mean()) if len(y) else float("nan"),
        "ap": float(average_precision_score(y, s)) if (y == 1).any() else float("nan"),
        "p_at_r90": precision_at_recall(y, s),
    }


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v == v], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    std = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
    return float(arr.mean()), std


def protocol_over_subsets(
    y_bench, score_bench, subsets: Iterable[tuple[str, np.ndarray]]
) -> dict[str, dict]:
    """Average the subset metrics per base-rate label.

    ``subsets`` yields (rate label, row positions into the benchmark arrays). Returns one entry
    per rate present, in ``RATE_ORDER``, with mean and sample standard deviation (ddof=1) of AP and
    P@R=0.90 over that rate's subsets, plus the subset count and the realised base rates.
    """
    y = np.asarray(y_bench).astype(np.int64)
    s = np.asarray(score_bench, dtype=np.float64)
    by_rate: dict[str, list[dict]] = {}
    for rate, pos in subsets:
        m = subset_metrics(y[pos], s[pos])
        by_rate.setdefault(rate, []).append(m)
    out: dict[str, dict] = {}
    for rate in sorted(by_rate, key=lambda r: RATE_ORDER.index(r) if r in RATE_ORDER else 99):
        rows = by_rate[rate]
        ap_mean, ap_std = _mean_std(r["ap"] for r in rows)
        pr_mean, pr_std = _mean_std(r["p_at_r90"] for r in rows)
        br_mean, br_std = _mean_std(r["base_rate"] for r in rows)
        out[rate] = {
            "n_subsets": len(rows),
            "ap_mean": ap_mean,
            "ap_std": ap_std,
            "p_at_r90_mean": pr_mean,
            "p_at_r90_std": pr_std,
            "base_rate_mean": br_mean,
            "base_rate_std": br_std,
            "subset_size_mean": float(np.mean([r["n"] for r in rows])),
        }
    return out


def protocol_section(
    labels: np.ndarray,
    scores: np.ndarray,
    test: np.ndarray,
    bench_full: np.ndarray,
    subsets: list[tuple[str, np.ndarray]],
) -> dict:
    """Protocol metrics for one model's scores (NaN outside the scored rows).

    ``test`` are the official test positions, ``bench_full`` the 24% benchmark positions and
    ``subsets`` the (rate, positions) list from ``load_phreshphish``. Returns the per-rate table,
    the full-benchmark (24%) figures and the full test-split figures.
    """
    return {
        "per_rate": protocol_over_subsets(labels, scores, subsets),
        "full_benchmark_24pct": subset_metrics(labels[bench_full], scores[bench_full]),
        "full_test_split": subset_metrics(labels[test], scores[test]),
    }


def _pm(mean: float, std: float) -> str:
    if mean != mean:
        return "n/a"
    if std != std:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def _num(value: float | None) -> str:
    return "n/a" if value is None or value != value else f"{value:.4f}"


PROTOCOL_HEADER = [
    "Base rate",
    "Subsets",
    "AP mean ± std",
    "P@R=0.90 mean ± std",
    "Paper GTE AP (URL+HTML)",
    "Paper GTE P@R=0.90 (URL+HTML)",
]
PROTOCOL_NOTE = (
    "Each benchmark subset is scored and AP and P@R=0.90 are averaged over the subsets of a base "
    "rate: mean ± sample std (ddof=1). Paper columns: arXiv 2507.10854 v2, Table 2, GTE "
    "(fine-tuned gte-large, URL and HTML). Quoted for reference; the paper has no URL-only row."
)


def protocol_tables(section: Mapping) -> list[dict]:
    """One report table per model of a protocol section (see ``protocol_section``).

    Each table is {"title", "note", "header", "rows"}; rows hold display strings, so the markdown
    renderer in rakshak.eval.report can print them as they are.
    """
    paper = section.get("paper_v2_gte_reference") or paper_reference()
    tables = []
    for name, res in (section.get("models") or {}).items():
        rows: list[list[str]] = []
        for rate, v in res["per_rate"].items():
            ref = paper.get(rate, {})
            rows.append([
                rate,
                str(v["n_subsets"]),
                _pm(v["ap_mean"], v["ap_std"]),
                _pm(v["p_at_r90_mean"], v["p_at_r90_std"]),
                _num(ref.get("ap")),
                _num(ref.get("p_at_r90")),
            ])
        fb = res["full_benchmark_24pct"]
        ref = paper.get(FULL_LABEL, {})
        rows.append(["24% (full benchmark)", "1", _num(fb["ap"]), _num(fb["p_at_r90"]),
                     _num(ref.get("ap")), _num(ref.get("p_at_r90"))])
        ft = res["full_test_split"]
        rows.append(["full test split", "1", _num(ft["ap"]), _num(ft["p_at_r90"]), "n/a", "n/a"])
        tables.append({
            "title": f"PhreshPhish protocol: {name}",
            "note": PROTOCOL_NOTE,
            "header": PROTOCOL_HEADER,
            "rows": rows,
        })
    return tables
