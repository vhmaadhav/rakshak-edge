"""Cross-dataset generalisation helpers for the NetFlow-v2 network module.

Pure numpy and sklearn; the Kaggle/VM job (kaggle/jobs/nids_xdata.py) calls these functions and
the tests call the same ones. Four pieces:

* feature hygiene: drop identifier, port, TTL, TCP-window, timestamp and flow-id artefacts, and add
  ratio features computed from raw counts (``build_feature_matrix``);
* stability selection: keep features that rank in the top-k by mutual information on every source
  dataset (``stable_features``). It reads source data only, by construction (the caller passes
  source splits);
* benign-quantile normalisation: a per-dataset quantile map fitted on that dataset's benign rows
  (or, for a target, on an unlabelled head) (``fit_benign_quantiles``, via ``baseline``);
* leave-one-dataset-out threshold and hyperparameter choice: a threshold is chosen from
  leave-one-source-out inner folds and never sees the held-out target (``choose_threshold``,
  ``mcc_curve``).

``stage_sample`` is the streaming, label-stratified sampler that writes the per-dataset parquet
used by the experiment, with the 43 NetFlow-v2 feature columns plus ``Label`` and ``Attack``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rakshak.nids import baseline as B
from rakshak.nids import schema as S
from rakshak.nids import splits as SP

# ---------------------------------------------------------------------------
# Feature hygiene
# ---------------------------------------------------------------------------

# Raw NetFlow-v2 columns that identify hosts, services or capture time, or encode TTL and TCP window
# values. They are never read as features. Names that a given parquet copy does not contain are
# ignored.
HYGIENE_DROP_RAW: tuple[str, ...] = (
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "MIN_TTL", "MAX_TTL",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS",
    "FLOW_ID", "DNS_QUERY_ID",
)
# Shared-schema features removed by hygiene: L7_PROTO names the service, DNS_TTL_ANSWER is a TTL.
HYGIENE_DROP_FEATURES: tuple[str, ...] = ("L7_PROTO", "DNS_TTL_ANSWER")
HYGIENE_FEATURES: tuple[str, ...] = tuple(
    f for f in S.SHARED_FEATURES if f not in HYGIENE_DROP_FEATURES)
# Engineered ratios. Each is computed from raw counts and durations, then log1p.
ENGINEERED: tuple[str, ...] = (
    "BYTES_PER_PKT", "IN_OUT_BYTES_RATIO", "IN_OUT_PKTS_RATIO", "DUR_IN_RATIO", "DUR_OUT_RATIO",
)
ALL_NAMES: tuple[str, ...] = tuple(S.SHARED_FEATURES) + ENGINEERED
_IDX = {name: i for i, name in enumerate(ALL_NAMES)}
CATEGORICAL_NAMES: frozenset[str] = frozenset(S.CATEGORICAL)

assert not (set(HYGIENE_DROP_RAW) & set(HYGIENE_FEATURES))
assert set(HYGIENE_DROP_FEATURES) <= set(S.SHARED_FEATURES)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """a / (b + 1), clipped to be non-negative, then log1p. Division by zero is impossible."""
    r = np.clip(np.asarray(num, dtype=np.float64), 0.0, None) / (
        np.clip(np.asarray(den, dtype=np.float64), 0.0, None) + 1.0)
    return np.log1p(np.where(np.isfinite(r), r, 0.0))


def engineered_matrix(raw: Mapping[str, np.ndarray]) -> np.ndarray:
    """Ratio features from raw NetFlow columns. Returns float64 (rows, len(ENGINEERED))."""
    n = len(next(iter(raw.values())))
    get = lambda k: np.nan_to_num(np.asarray(raw[k], dtype=np.float64), nan=0.0)  # noqa: E731
    in_b, out_b = get("IN_BYTES"), get("OUT_BYTES")
    in_p, out_p = get("IN_PKTS"), get("OUT_PKTS")
    dur = get("FLOW_DURATION_MILLISECONDS")
    dur_in, dur_out = get("DURATION_IN"), get("DURATION_OUT")
    cols = [
        np.log1p(np.clip((in_b + out_b) / (in_p + out_p + 1.0), 0.0, None)),
        _ratio(in_b, out_b),
        _ratio(in_p, out_p),
        _ratio(dur_in, dur),
        _ratio(dur_out, dur),
    ]
    out = np.column_stack(cols) if cols else np.zeros((n, 0))
    assert out.shape == (n, len(ENGINEERED))
    return out


def build_feature_matrix(raw: Mapping[str, np.ndarray]) -> np.ndarray:
    """All candidate features for one dataset, in ``ALL_NAMES`` order (float32).

    Shared features go through ``schema.coerce_matrix`` (caps, log1p on non-categorical columns),
    so the base arm is exactly the baseline transform. Engineered ratios are appended. Raw columns
    named in ``HYGIENE_DROP_RAW`` are never read here, so they cannot reach a model.
    """
    base_raw = np.column_stack([np.nan_to_num(np.asarray(raw[f], dtype=np.float64), nan=0.0)
                                for f in S.SHARED_FEATURES])
    base = S.coerce_matrix(base_raw, S.SHARED_FEATURES)
    eng = engineered_matrix(raw).astype(np.float32)
    return np.hstack([base, eng]).astype(np.float32)


def column_index(names: Sequence[str]) -> np.ndarray:
    """Positions of ``names`` inside ``ALL_NAMES``. Unknown names raise KeyError."""
    missing = [n for n in names if n not in _IDX]
    if missing:
        raise KeyError(f"unknown feature names: {missing}")
    return np.asarray([_IDX[n] for n in names], dtype=np.int64)


def categorical_mask(names: Sequence[str]) -> np.ndarray:
    return np.asarray([n in CATEGORICAL_NAMES for n in names], dtype=bool)


# ---------------------------------------------------------------------------
# Stability selection (sources only)
# ---------------------------------------------------------------------------


def stable_features(
    sources: Mapping[str, tuple[np.ndarray, np.ndarray]],
    names: Sequence[str],
    *,
    k: int = 20,
    min_keep: int = 8,
    seed: int = 0,
    max_rows: int = 50_000,
) -> list[str]:
    """Features ranked in the top ``k`` by mutual information on every source dataset.

    ``sources`` maps a source key to ``(X, y)`` with X in the order of ``names``. Each source is
    scored on its own. A feature is kept when its rank is in the top ``k`` on all sources. When
    fewer than ``min_keep`` survive, the ``min_keep`` features with the smallest worst-source rank
    are kept. The held-out target is never an argument, so it cannot be used.
    """
    from sklearn.feature_selection import mutual_info_classif

    names = list(names)
    if not sources:
        raise ValueError("stable_features needs at least one source")
    if not 1 <= min_keep <= len(names):
        raise ValueError(f"min_keep={min_keep} out of range for {len(names)} features")
    cat = categorical_mask(names)
    ranks = []
    for key in sorted(sources):
        X, y = sources[key]
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if X.shape[1] != len(names):
            raise ValueError(f"source {key!r} has {X.shape[1]} columns, expected {len(names)}")
        if len(y) > max_rows:
            pick = np.random.default_rng(seed).choice(len(y), size=max_rows, replace=False)
            X, y = X[pick], y[pick]
        mi = mutual_info_classif(X, y, discrete_features=cat, random_state=seed)
        order = np.argsort(-mi, kind="stable")
        rank = np.empty(len(names), dtype=np.int64)
        rank[order] = np.arange(len(names))
        ranks.append(rank)
    rank_mat = np.vstack(ranks)
    keep = np.flatnonzero((rank_mat < min(k, len(names))).all(axis=0))
    if len(keep) < min_keep:
        worst = rank_mat.max(axis=0)
        keep = np.argsort(worst, kind="stable")[:min_keep]
    return [names[i] for i in sorted(keep.tolist())]


# ---------------------------------------------------------------------------
# Benign-quantile normalisation (per dataset)
# ---------------------------------------------------------------------------


def fit_benign_quantiles(X: np.ndarray, names: Sequence[str]) -> B.SiteBaseline:
    """Per-feature quantile map fitted on ``X`` only: the benign rows of a source, or the unlabelled
    calibration head of a target. Labels are not an argument, so the map cannot see attacks
    through a label. Categorical columns pass through."""
    return B.fit_site_baseline(np.asarray(X, dtype=np.float64), categorical_mask(names))


# ---------------------------------------------------------------------------
# Threshold and hyperparameter choice
# ---------------------------------------------------------------------------


def mcc_curve(y: np.ndarray, score: np.ndarray, thresholds: np.ndarray) -> dict[str, np.ndarray]:
    """MCC, FPR and TPR at each threshold (``score >= threshold`` is an attack). Vectorised with a
    sort, so it costs O(n log n) for any number of thresholds. Degenerate MCC (a zero marginal) is
    0, as in ``rakshak.eval.metrics``."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(score, dtype=np.float64)
    thr = np.asarray(thresholds, dtype=np.float64)
    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    pos_cum = np.concatenate([[0], np.cumsum(y[order] == 1)])
    n = len(s)
    n_pos = int((y == 1).sum())
    n_neg = n - n_pos
    idx = np.searchsorted(s_sorted, thr, side="left")  # first index with score >= thr
    tp = n_pos - pos_cum[idx]
    fp = (n - idx) - tp
    fn = n_pos - tp
    tn = n_neg - fp
    num = tp.astype(np.float64) * tn - fp.astype(np.float64) * fn
    den = np.sqrt((tp + fp).astype(np.float64) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    fpr = np.where(n_neg > 0, fp / max(n_neg, 1), 0.0)
    tpr = np.where(n_pos > 0, tp / max(n_pos, 1), 0.0)
    return {"mcc": mcc, "fpr": fpr, "tpr": tpr}


def candidate_thresholds(scores: Sequence[np.ndarray], points: int = 200) -> np.ndarray:
    """Unique quantiles of the pooled scores (``points`` of them). Fixed before any outer result."""
    pooled = np.concatenate([np.asarray(s, dtype=np.float64) for s in scores])
    return np.unique(np.quantile(pooled, np.linspace(0.0, 1.0, points)))


def choose_threshold(folds: Sequence[tuple[np.ndarray, np.ndarray]], *,
                     points: int = 200) -> dict[str, float]:
    """Threshold maximising the mean MCC across inner folds.

    ``folds`` is a list of ``(y, score)`` pairs, one per inner fold (a source held out of the
    inner training). The mean MCC is taken over the folds at each candidate threshold. Ties go to
    the higher threshold (lower FPR). Returns threshold, mean MCC, mean FPR and the number of
    folds.
    """
    if not folds:
        raise ValueError("choose_threshold needs at least one inner fold")
    thr = candidate_thresholds([s for _, s in folds], points)
    curves = [mcc_curve(y, s, thr) for y, s in folds]
    mean_mcc = np.mean([c["mcc"] for c in curves], axis=0)
    mean_fpr = np.mean([c["fpr"] for c in curves], axis=0)
    best = mean_mcc.max()
    tied = np.flatnonzero(mean_mcc >= best - 1e-12)
    lowest_fpr = mean_fpr[tied].min()
    # thresholds ascend, so the last index is the highest threshold among the lowest-FPR ties
    same = tied[mean_fpr[tied] <= lowest_fpr + 1e-12]
    pick = int(same[-1])
    return {"threshold": float(thr[pick]), "mean_inner_mcc": float(mean_mcc[pick]),
            "mean_inner_fpr": float(mean_fpr[pick]), "folds": len(folds)}


def choose_config(results: Mapping[str, Mapping[str, float]]) -> str:
    """Name of the hyperparameter config with the highest mean inner MCC. Ties go to the lower mean
    inner FPR, then to the name sorted first (stable)."""
    if not results:
        raise ValueError("no configs to choose from")
    return sorted(results, key=lambda c: (-round(results[c]["mean_inner_mcc"], 12),
                                          results[c]["mean_inner_fpr"], c))[0]


# ---------------------------------------------------------------------------
# Metrics on a held-out target
# ---------------------------------------------------------------------------


def _mcc(tp: float, fp: float, tn: float, fn: float) -> float:
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def target_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    """Metrics at a fixed decision (``pred`` is 0/1, chosen before this call). PR-AUC is average
    precision. FPR is on the target's benign flows."""
    from sklearn.metrics import average_precision_score

    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    pr_auc = float(average_precision_score(y, score)) if (y == 1).any() else float("nan")
    return {
        "n": int(len(y)), "prevalence": float((y == 1).mean()),
        "mcc": _mcc(tp, fp, tn, fn), "f1": f1, "precision": prec, "recall": rec,
        "fpr": fp / (fp + tn) if fp + tn > 0 else 0.0, "pr_auc": pr_auc,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def bootstrap_metric_draws(y: np.ndarray, preds: Sequence[np.ndarray], n_boot: int,
                           seed: int) -> np.ndarray:
    """MCC of each prediction vector on the same bootstrap resamples of the rows.

    Returns shape (n_boot, len(preds)). Resampling is shared across the vectors, so differences
    between them are paired."""
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    rng = np.random.default_rng(seed)
    pos = (y == 1).astype(np.float64)
    neg = 1.0 - pos
    out = np.zeros((n_boot, len(preds)))
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(np.float64)
        for j, pr in enumerate(preds):
            p = np.asarray(pr, dtype=np.float64)
            tp = float((counts * pos * p).sum())
            fp = float((counts * neg * p).sum())
            fn = float((counts * pos * (1 - p)).sum())
            tn = float((counts * neg * (1 - p)).sum())
            out[b, j] = _mcc(tp, fp, tn, fn)
    return out


def percentile_interval(draws: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.percentile(
        np.asarray(draws, dtype=np.float64), [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Sampling for the staged per-dataset parquet
# ---------------------------------------------------------------------------


@dataclass
class StageInfo:
    rows_total: int
    benign_total: int
    attack_total: int
    benign_kept: int
    attack_kept: int
    attack_strings: dict[str, int]
    columns: list[str]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def stage_sample(src: Path, out: Path, *, max_benign: int, max_attack: int, seed: int,
                 batch_rows: int = 200_000) -> StageInfo:
    """Label-stratified sample of a NetFlow-v2 parquet, written with its numeric columns.

    Two streaming passes. Pass 1 counts labels and attack strings. Pass 2 keeps benign rows with
    probability ``min(1, max_benign / benign_total)`` and attack rows with the water-filling keep
    probabilities of ``splits.plan_quotas`` (rare attacks keep all their rows, common ones are cut
    to a common level). The kept rows are then cut to the exact caps at random. Only numeric
    columns, ``Label`` and ``Attack`` are written. Host and other string columns are dropped here
    because no model reads them.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(src)
    names = list(pf.schema_arrow.names)
    for need in S.LABEL_COLUMNS:
        if need not in names:
            raise KeyError(f"{src.name}: missing column {need!r}; columns are {names}")
    counts: Counter[str] = Counter()
    benign_total = total = 0
    for batch in pf.iter_batches(batch_size=batch_rows, columns=list(S.LABEL_COLUMNS)):
        df = batch.to_pandas()
        lab = df["Label"].to_numpy().astype(np.int64)
        total += len(lab)
        benign_total += int((lab == 0).sum())
        counts.update(df["Attack"].fillna("").astype(str)[lab == 1].value_counts().to_dict())
    attack_total = total - benign_total
    keep_att, _ = SP.plan_quotas(dict(counts), 0, max_attack)
    keep_ben = min(1.0, max_benign / benign_total) if benign_total else 1.0

    numeric = [n for n in names if n not in S.LABEL_COLUMNS
               and not _is_text(pf.schema_arrow.field(n).type)]
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for batch in pf.iter_batches(batch_size=batch_rows, columns=numeric + list(S.LABEL_COLUMNS)):
        df = batch.to_pandas()
        lab = df["Label"].to_numpy().astype(np.int64)
        att = df["Attack"].fillna("").astype(str).to_numpy()
        prob = np.where(lab == 0, keep_ben, 1.0)
        attack_rows = np.flatnonzero(lab == 1)
        prob[attack_rows] = [keep_att.get(a, 1.0) for a in att[attack_rows]]
        keep = rng.random(len(df)) < prob
        if keep.any():
            parts.append(df.loc[keep, numeric + list(S.LABEL_COLUMNS)])
    sample = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=numeric)
    # exact caps: random drop when the Bernoulli draw overshoots
    lab_all = sample["Label"].to_numpy().astype(np.int64)
    for cls, cap in ((0, max_benign), (1, max_attack)):
        rows = np.flatnonzero(lab_all == cls)
        if len(rows) > cap:
            drop = rng.choice(rows, size=len(rows) - cap, replace=False)
            sample = sample.drop(index=sample.index[drop])
            lab_all = sample["Label"].to_numpy().astype(np.int64)
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if not set(np.unique(sample["Label"].to_numpy()).tolist()) == {0, 1}:
        raise ValueError(f"{src.name}: staged sample must hold both classes")
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out, index=False)
    kept = sample["Label"].to_numpy()
    att_kept = Counter(sample["Attack"].fillna("").astype(str).to_numpy()[kept == 1])
    return StageInfo(
        rows_total=int(total), benign_total=int(benign_total), attack_total=int(attack_total),
        benign_kept=int((kept == 0).sum()), attack_kept=int((kept == 1).sum()),
        attack_strings=dict(att_kept.most_common()), columns=list(sample.columns))


def _is_text(arrow_type) -> bool:
    """True for string and binary columns (identifiers such as IP addresses)."""
    import pyarrow.types as pt

    return bool(pt.is_string(arrow_type) or pt.is_large_string(arrow_type)
                or pt.is_binary(arrow_type) or pt.is_large_binary(arrow_type))


def training_subsample(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Row positions with ``per_class`` benign and ``per_class`` attack rows (fewer when a class is
    smaller), sorted. Balanced training for every source, drawn with the given seed."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    picks = []
    for cls in (0, 1):
        rows = np.flatnonzero(y == cls)
        if len(rows) == 0:
            raise ValueError(f"no rows of class {cls} to subsample")
        n = min(per_class, len(rows))
        picks.append(rng.choice(rows, size=n, replace=False))
    return np.sort(np.concatenate(picks))
