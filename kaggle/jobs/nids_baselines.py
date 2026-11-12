"""CPU baselines for the network-intrusion module on the shared NetFlow-v2 schema.

Datasets (NF-*-v2 parquet, attached with --datasets, mounted under /kaggle/input/<slug>/):
  unsw_nb15        dhoogla/nfunswnb15v2       (NF-UNSW-NB15-V2)
  cse_cic_ids2018  dhoogla/nfcsecicids2018v2  (NF-CSE-CIC-IDS2018-V2, 641 MB)
  bot_iot          dhoogla/nfbotiotv2         (NF-BoT-IoT-V2, 509 MB)
  ton_iot          dhoogla/nftoniotv2         (NF-ToN-IoT-V2, 209 MB)

Each dataset is read in two streaming passes with pyarrow and capped at --max-rows rows, stratified
by label (benign share kept, attacks water-filled across attack strings). Identifiers (IPs) are
read only for host grouping and are never features. Duplicate feature vectors are dropped before any
split.

Protocols (all models see the same frozen 33-feature schema, except *_live which uses the 19
nfstream-mappable features):
  T0  random stratified 70/15/15 within a dataset (reference only, never the headline)
  T1  cross-dataset, every ordered pair: train and calibrate on A, test on all of B
  T2  leave-one-attack-family-out within a dataset, and across pairs (--t2-cross-models)
  T4  host-disjoint 70/15/15 within a dataset (connected host components, else source-IP groups)
  T5  leave-one-dataset-out: train on the other three datasets (equal rows per source), test on the
      held-out one. Calibration and the K-sensitivity runs use the same source split.
  T1_calib, T5, T2_transfer: home-baseline calibration (see below). Each target gets an
      unlabelled calibration slice of K flows (first K of a seeded random order). The slice is
      removed from the target test set. Labels are never read to fit it; its attack share is
      reported only.
  T2 anomaly: benign-only IsolationForest and autoencoder, plus hybrid scores, with the
      held-out family unseen by every model.

Models: lightgbm (teacher, strong baseline), logreg, mlp (sklearn), lightgbm_live. Every model is
multi-class over the 8 families. The attack score is 1 - P(benign). Thresholds are chosen on the
calibration split only. Confidence intervals are host-group clustered bootstraps.

Home-baseline calibration (--no-calib to skip). Every site (source or target) gets its own
per-feature quantile map, fitted on K unlabelled flows from that site (categorical features pass
through). The source model is trained on source-standardised features, and the target is scored on
target-standardised features. Anomaly scores are converted to a site rank (fraction of the site's
own baseline at or below the flow's score). A hybrid score is max(LightGBM attack probability,
anomaly rank). Each cross-dataset or LODO test is run for K = --calib-k (headline) and
--calib-sensitivity (default 1000 20000).

Student (--no-student to skip): a 64-32 MLP distilled from the live LightGBM teacher (regression on
teacher log-probabilities, scaler folded into the first layer), exported to ONNX with a hand-built
graph, quantised to INT8 with onnxruntime, and checked for size (<= 1 MB), fp32/INT8 parity and
single-thread single-flow latency on this box (x86 Kaggle, so the Pi numbers are measured later).

Flags go through kaggle/launch.sh nids_baselines --job-args="...":
  --data-root PATH     (default: $RAKSHAK_NIDS_DATA, then /kaggle/input)
  --datasets KEY...    subset of unsw_nb15 cse_cic_ids2018 bot_iot ton_iot
  --max-rows INT       per-dataset cap after sampling (default 200000)
  --protocols T0 T1 T2 T4 T5
  --models NAME...     lightgbm logreg mlp lightgbm_live
  --t2-cross-models NAME...   (default lightgbm)
  --calib-k INT        headline baseline size K (default 5000)
  --calib-sensitivity INT...  extra K values for the sensitivity runs (default 1000 20000)
  --no-calib           skip the calibrated, LODO, anomaly and hybrid runs
  --seed INT           (default 0)
  --n-boot INT         bootstrap resamples (default 200; each resample rescans the test split)
  --no-student
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from _common import Units, env_report, record_skip, seed_everything, single_class_reason

from rakshak.eval.metrics import threshold_for_fpr
from rakshak.eval.protocol import evaluate_scores
from rakshak.nids import baseline as B
from rakshak.nids import live as L
from rakshak.nids import schema as S
from rakshak.nids import splits as SP

DATASETS: dict[str, dict[str, str]] = {
    "unsw_nb15": {"slug": "nfunswnb15v2", "token": "UNSW"},
    "cse_cic_ids2018": {"slug": "nfcsecicids2018v2", "token": "CSE"},
    "bot_iot": {"slug": "nfbotiotv2", "token": "BOT"},
    "ton_iot": {"slug": "nftoniotv2", "token": "TON"},
}
MODELS: dict[str, tuple[str, str]] = {  # name -> (feature set, estimator kind)
    "lightgbm": ("full", "lgbm"),
    "logreg": ("full", "logreg"),
    "mlp": ("full", "mlp"),
    "lightgbm_live": ("live", "lgbm"),
}
FEATURE_SETS: dict[str, np.ndarray] = {
    "full": np.arange(len(S.SHARED_FEATURES), dtype=np.int64),
    "live": L.LIVE_COLUMNS,
}
FRACTIONS = (0.70, 0.15, 0.15)  # train, calibration, test
FPR_OPERATING = 0.01
BATCH_ROWS = 1_000_000
BUDGET_BYTES = 1_000_000
CONTROL_FAMILY_INDEX = 0  # benign
LGBM_N_ESTIMATORS = 300  # trees per LightGBM classifier (tests shrink this)
IFOREST_N_ESTIMATORS = 200  # trees per IsolationForest scorer (tests shrink this)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nids_baselines")
    p.add_argument("--data-root", default=None)
    p.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--max-rows", type=int, default=200_000)
    p.add_argument("--protocols", nargs="*", default=["T0", "T1", "T2", "T4", "T5"],
                   choices=["T0", "T1", "T2", "T4", "T5"])
    p.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--t2-cross-models", nargs="*", default=["lightgbm"], choices=list(MODELS))
    p.add_argument("--calib-k", type=int, default=5000)
    p.add_argument("--calib-sensitivity", type=int, nargs="*", default=[1000, 20000])
    p.add_argument("--no-calib", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--no-student", action="store_true")
    return p.parse_args(argv or [])


# ---------------------------------------------------------------------------
# Discovery and inspection
# ---------------------------------------------------------------------------


def discover(root: Path, keys: list[str]) -> dict[str, Path]:
    """Parquet file per dataset key: the Kaggle mount folder first, then a filename token."""
    files = sorted(root.rglob("*.parquet")) if root.is_dir() else []
    found: dict[str, Path] = {}
    for key in keys:
        slug = DATASETS[key]["slug"]
        hits = [p for p in files if any(part.lower() == slug for part in p.parts)]
        if not hits:
            token = DATASETS[key]["token"]
            hits = [p for p in files if token in p.name.upper()]
        if hits:
            found[key] = hits[0]
    return found


def check_feature_csv(parquet: Path) -> dict[str, Any]:
    """Compare the shared schema with the dataset's 'NetFlow v2 Features.csv' (same folder).

    Only tokens of the form NAME_WITH_UNDERSCORES are matched, so the CSV layout does not matter.
    Returns the schema features the CSV lists and the ones it does not.
    """
    csvs = sorted(parquet.parent.glob("*.csv"))
    if not csvs:
        return {"file": None}
    text = csvs[0].read_text(encoding="utf-8", errors="replace")
    tokens = set(re.findall(r"[A-Z][A-Z0-9_]{2,}", text))
    return {
        "file": csvs[0].name,
        "in_schema_present": [f for f in S.SHARED_FEATURES if f in tokens],
        "in_schema_absent": [f for f in S.SHARED_FEATURES if f not in tokens],
    }


# ---------------------------------------------------------------------------
# Streaming sample
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    key: str
    X: np.ndarray  # float32, coerced shared schema (rows, 33)
    y: np.ndarray  # 0 benign, 1 attack
    fam_idx: np.ndarray  # index into S.FAMILIES
    fam: np.ndarray  # family name per row
    groups: np.ndarray | None  # host group per row; None when the parquet has no host columns
    info: dict[str, Any]


def sample_dataset(path: Path, key: str, max_rows: int, seed: int) -> Dataset:
    """Two streaming passes. Pass 1 counts labels and attack strings. Pass 2 keeps rows by
    Bernoulli sampling with the per-stratum probabilities from ``SP.plan_quotas``.

    Column names are resolved against the parquet schema (``S.resolve_columns``). A missing
    feature or label column raises KeyError that lists every missing name and the file's columns.
    Host columns are optional: without both, ``groups`` is None and T4 is skipped.
    """
    pf = pq.ParquetFile(path)
    actual = list(pf.schema_arrow.names)
    feats, miss_feats = S.resolve_columns(actual, S.SHARED_FEATURES)
    labels, miss_labels = S.resolve_columns(actual, S.LABEL_COLUMNS)
    if miss_feats or miss_labels:
        raise KeyError(
            f"{key}: missing required columns: features {miss_feats}, labels {miss_labels}; "
            f"parquet columns are {actual}")
    hosts, _ = S.resolve_columns(actual, S.HOST_COLUMNS)
    has_hosts = len(hosts) == len(S.HOST_COLUMNS)
    resolved = {**feats, **labels, **(hosts if has_hosts else {})}
    rename = {actual_name: wanted for wanted, actual_name in resolved.items()}
    aliases = {wanted: a for wanted, a in resolved.items() if wanted != a}

    attack_counts: Counter[str] = Counter()
    benign_total = 0
    total = 0
    label_cols = [labels[name] for name in S.LABEL_COLUMNS]
    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=label_cols):
        df = batch.to_pandas().rename(columns=rename)
        lab = df["Label"].to_numpy().astype(np.int64)
        total += len(lab)
        benign_total += int((lab == 0).sum())
        attack_names = df["Attack"].fillna("").astype(str)[lab == 1]
        attack_counts.update(attack_names.value_counts().to_dict())
    keep_att, keep_ben = SP.plan_quotas(dict(attack_counts), benign_total, max_rows)

    rng = np.random.default_rng(seed)
    read_cols = ([feats[f] for f in S.SHARED_FEATURES] + label_cols
                 + ([hosts[h] for h in S.HOST_COLUMNS] if has_hosts else []))
    xs, ys, atts, srcs, dsts = [], [], [], [], []
    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=read_cols):
        df = batch.to_pandas().rename(columns=rename)
        lab = df["Label"].to_numpy().astype(np.int64)
        att = df["Attack"].fillna("").astype(str).to_numpy()
        prob = np.ones(len(df))
        benign = lab == 0
        prob[benign] = keep_ben
        attack_rows = np.flatnonzero(~benign)
        prob[attack_rows] = [keep_att.get(a, 1.0) for a in att[attack_rows]]
        keep = rng.random(len(df)) < prob
        if not keep.any():
            continue
        sub = df.loc[keep]
        xs.append(sub[list(S.SHARED_FEATURES)].to_numpy(dtype=np.float64))
        ys.append(lab[keep])
        atts.append(att[keep])
        if has_hosts:
            srcs.append(sub["IPV4_SRC_ADDR"].astype(str).to_numpy())
            dsts.append(sub["IPV4_DST_ADDR"].astype(str).to_numpy())
    if not xs:
        raise ValueError(f"{key}: sampling kept no rows")
    x_raw = np.concatenate(xs)
    y = np.concatenate(ys).astype(np.int64)
    att = np.concatenate(atts)
    if not set(np.unique(y).tolist()) == {0, 1}:
        raise ValueError(f"{key}: sample must hold both classes, got {np.unique(y).tolist()}")

    pairs = [S.map_attack(a, int(lbl)) for a, lbl in zip(att, y, strict=True)]
    fam = np.asarray([f for f, _ in pairs], dtype=object)
    how = Counter(h for _, h in pairs)
    x32 = S.coerce_matrix(x_raw)
    _, first = np.unique(x32, axis=0, return_index=True)
    first = np.sort(first)
    removed = len(x32) - len(first)
    x32, y, att, fam = x32[first], y[first], att[first], fam[first]
    fam_idx = np.asarray([S.FAMILY_INDEX[f] for f in fam], dtype=np.int64)
    if not set(np.unique(y).tolist()) == {0, 1}:
        raise ValueError(f"{key}: after dedupe only one class is left")

    if has_hosts:
        src = np.concatenate(srcs)[first]
        dst = np.concatenate(dsts)[first]
        groups, method = SP.host_groups(src.tolist(), dst.tolist())
        host_info: dict[str, Any] = {
            "method": method, "count": int(groups.max() + 1),
            "largest_share": float(np.bincount(groups).max() / len(groups))}
    else:
        groups = None
        host_info = {"method": "none", "reason": "skipped: no host columns"}
    info = {
        "rows_total": int(total),
        "benign_total": int(benign_total),
        "attack_total": int(total - benign_total),
        "sampled_before_dedupe": int(len(x_raw)),
        "duplicates_removed": int(removed),
        "rows_used": int(len(y)),
        "benign_used": int((y == 0).sum()),
        "attack_used": int((y == 1).sum()),
        "family_counts": {f: int((fam == f).sum()) for f in S.FAMILIES},
        "attack_strings_used": dict(Counter(att[y == 1].tolist()).most_common()),
        "label_mapping_how": dict(how),
        "host_columns": has_hosts,
        "host_groups": host_info,
        "columns": actual,
        "column_aliases": aliases,
    }
    return Dataset(key, x32, y, fam_idx, fam, groups, info)


# ---------------------------------------------------------------------------
# Models and evaluation
# ---------------------------------------------------------------------------


def make_estimator(kind: str, seed: int):
    if kind == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=LGBM_N_ESTIMATORS, learning_rate=0.08, num_leaves=63, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1,
        )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression

        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier

        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=40, early_stopping=True,
            n_iter_no_change=5, batch_size=512, random_state=seed))
    raise KeyError(f"unknown estimator kind {kind!r}")


def fit_predict(kind: str, seed: int, X: np.ndarray, fam: np.ndarray,
                parts: list[np.ndarray]) -> list[np.ndarray]:
    """Fit on (X, fam) and return family probabilities (rows, len(FAMILIES)) for each matrix in
    ``parts``. Classes absent from training get probability 0."""
    classes = np.unique(fam)
    if len(classes) < 2:
        raise ValueError("training split has fewer than two families")
    est = make_estimator(kind, seed).fit(X, np.searchsorted(classes, fam))
    out = []
    for part in parts:
        p = est.predict_proba(part)
        full = np.zeros((len(part), len(S.FAMILIES)))
        full[:, classes] = p
        out.append(full)
    return out


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray
    fam: np.ndarray  # family index
    names: np.ndarray  # family name
    clusters: list[str] | None  # host group per row as strings; None -> iid bootstrap


def take(ds: Dataset, rows: np.ndarray) -> Split:
    clusters = None if ds.groups is None else [str(g) for g in ds.groups[rows]]
    return Split(ds.X[rows], ds.y[rows], ds.fam_idx[rows], ds.fam[rows], clusters)


def family_block(cal: Split, test: Split, s_cal: np.ndarray, pred: np.ndarray | None,
                 s_test: np.ndarray) -> dict[str, Any]:
    """Macro-F1 over families (benign included, and attack families only), and recall per family
    at the calibration threshold for FPR 1%. ``pred`` is None for binary scores, which have no
    family prediction: the macro-F1 fields are then omitted."""
    from sklearn.metrics import accuracy_score, f1_score

    thr = threshold_for_fpr(cal.y, s_cal, FPR_OPERATING)["threshold"]
    per_family = {}
    for name in S.ATTACK_FAMILIES:
        mask = test.names == name
        if not mask.any():
            continue
        recall = float(np.mean(s_test[mask] >= thr)) if np.isfinite(thr) else float("nan")
        per_family[name] = {"n": int(mask.sum()), f"recall_at_fpr_{FPR_OPERATING:g}": recall}
    if pred is None:
        return {"threshold_fpr_1pct": float(thr), "per_family": per_family}
    attack_ids = [S.FAMILY_INDEX[f] for f in S.ATTACK_FAMILIES]
    labels_all = sorted(set(test.fam.tolist()) | set(pred.tolist()))
    labels_attack = [i for i in labels_all if i in attack_ids]
    return {
        "macro_f1": float(f1_score(test.fam, pred, labels=labels_all, average="macro",
                                   zero_division=0)),
        "macro_f1_attack_only": float(f1_score(test.fam, pred, labels=labels_attack,
                                               average="macro", zero_division=0))
        if labels_attack else float("nan"),
        "accuracy": float(accuracy_score(test.fam, pred)),
        "threshold_fpr_1pct": float(thr),
        "per_family": per_family,
    }


FPR_AT_RECALL_TARGETS = (0.90, 0.95)


def result_from_binary(cal: Split, test: Split, s_cal: np.ndarray, s_test: np.ndarray,
                       seed: int, n_boot: int) -> dict[str, Any]:
    """Protocol result from any attack score (higher means attack). No family prediction."""
    res = evaluate_scores(cal.y, s_cal, test.y, s_test, groups={"family": test.names},
                          n_boot=n_boot, seed=seed, cluster_ids=test.clusters)
    res["family"] = family_block(cal, test, s_cal, None, s_test)
    res["fpr_at_recall"] = {f"{r:g}": B.fpr_at_recall(test.y, s_test, r)
                            for r in FPR_AT_RECALL_TARGETS}
    return res


def result_from_scores(cal: Split, test: Split, p_cal: np.ndarray, p_test: np.ndarray,
                       seed: int, n_boot: int) -> dict[str, Any]:
    """Protocol result from family probabilities. The attack score is 1 - P(benign)."""
    s_cal = np.clip(1.0 - p_cal[:, CONTROL_FAMILY_INDEX], 0.0, 1.0)
    s_test = np.clip(1.0 - p_test[:, CONTROL_FAMILY_INDEX], 0.0, 1.0)
    res = evaluate_scores(cal.y, s_cal, test.y, s_test, groups={"family": test.names},
                          n_boot=n_boot, seed=seed, cluster_ids=test.clusters)
    res["family"] = family_block(cal, test, s_cal, p_test.argmax(axis=1), s_test)
    res["fpr_at_recall"] = {f"{r:g}": B.fpr_at_recall(test.y, s_test, r)
                            for r in FPR_AT_RECALL_TARGETS}
    return res


def evaluate_model(kind: str, seed: int, train: Split, cal: Split, test: Split,
                   n_boot: int) -> dict[str, Any]:
    feature_set, estimator = MODELS[kind]
    cols = FEATURE_SETS[feature_set]
    p_cal, p_test = fit_predict(estimator, seed, train.X[:, cols], train.fam,
                                [cal.X[:, cols], test.X[:, cols]])
    return result_from_scores(cal, test, p_cal, p_test, seed, n_boot)


def checked(split_name: str, ds_key: str, y_cal, y_test) -> str | None:
    """Skip reason when the calibration or test split lacks one class, else None."""
    reason = single_class_reason(y_test, f"{ds_key} {split_name} test")
    if reason:
        return reason
    if len(set(np.asarray(y_cal).tolist())) < 2:
        return f"skipped: {ds_key} {split_name} calibration split has one class"
    return None


# ---------------------------------------------------------------------------
# Home-baseline calibration, leave-one-dataset-out, benign-only detectors
# ---------------------------------------------------------------------------
#
# Label discipline: a Site carries labels only in ``train`` and ``cal`` (which fit or pick
# thresholds on the sources). ``base`` is K unlabelled feature rows and is the only input to the
# site transform and the site rank. The target's calibration slice is chosen by row position
# (B.calibration_slice) and is removed from its test split. Attack shares of slices are computed
# for the report after everything is fitted, and nothing reads them before that.

CAT_MASK = np.asarray([f in S.CATEGORICAL for f in S.SHARED_FEATURES], dtype=bool)
ANOMALY_TRAIN_CAP = 50_000
HYBRID_PARTS = ("isolation_forest", "autoencoder")


@dataclass
class Site:
    name: str
    train: Split  # labelled, source side only
    cal: Split  # labelled, thresholds only
    base: np.ndarray  # K unlabelled feature rows (raw schema features) from the site's train side


def concat_splits(parts: list[Split]) -> Split:
    """Labels, families and names of several splits as one split. Features are not kept."""
    return Split(None, np.concatenate([p.y for p in parts]),
                 np.concatenate([p.fam for p in parts]),
                 np.concatenate([p.names for p in parts]), None)


def make_site(name: str, ds: Dataset, train_rows: np.ndarray, cal_rows: np.ndarray,
              base_pool: np.ndarray, k: int, seed: int) -> Site:
    """Site whose baseline is the first K rows of ``base_pool`` in a seeded random order. The
    baseline reads features only."""
    head = B.unlabelled_head(len(base_pool), k, seed)
    return Site(name, take(ds, train_rows), take(ds, cal_rows), ds.X[base_pool[head]])


def lodo_sites(datasets: dict[str, Dataset], partitions: dict[str, dict[str, dict]],
               held_out: str, k: int, seed: int) -> list[Site]:
    """Source sites for leave-one-dataset-out. Train and calibration rows are balanced per source
    (equal counts). The held-out dataset is excluded before any draw. Baselines come from each
    source's full training side, so they do not depend on the balancing."""
    tr = {key: partitions[key]["T0"]["tr"] for key in datasets}
    ca = {key: partitions[key]["T0"]["ca"] for key in datasets}
    bal_tr = SP.lodo_balanced_rows(tr, held_out, seed)
    bal_ca = SP.lodo_balanced_rows(ca, held_out, seed + 1000)
    return [make_site(key, datasets[key], bal_tr[key], bal_ca[key], tr[key], k, seed)
            for key in bal_tr]


def _cap_rows(X: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if len(X) <= cap:
        return X
    idx = np.sort(np.random.default_rng(seed).choice(len(X), size=cap, replace=False))
    return X[idx]


def transfer_variants(sources: list[Site], test: Split, target_base: np.ndarray,
                      calibrate: bool, seed: int, n_boot: int) -> dict[str, Any]:
    """Train on the sources and score one target test split. Returns the variant results.

    ``calibrate=False`` (raw): features as coerced, no site transform. The anomaly variant is the
    raw detector score; the hybrid uses ranks against the raw baselines.
    ``calibrate=True`` (home baseline): each site is quantile-mapped with its own baseline (source
    by its ``base``, target by ``target_base``). The anomaly variant is its site rank.

    Variants: ``lightgbm``, ``isolation_forest``, ``autoencoder`` (anomaly alone), and
    ``hybrid_<detector>`` = max(LightGBM attack probability, that detector's site rank). Thresholds
    are chosen on the pooled source calibration splits only.
    """
    mode = "calib" if calibrate else "raw"

    def transform_for(base_raw: np.ndarray):
        if not calibrate:
            return lambda X: np.asarray(X, dtype=np.float32)
        return B.fit_site_baseline(base_raw, CAT_MASK).transform

    fitted = [(transform_for(s.base), s) for s in sources]
    f_target = transform_for(target_base)
    X_train = np.vstack([f(s.train.X) for f, s in fitted])
    fam_train = np.concatenate([s.train.fam for s in sources])
    benign_train = X_train[np.concatenate([s.train.y for s in sources]) == 0]
    cal_all = concat_splits([s.cal for s in sources])
    X_cal = np.vstack([f(s.cal.X) for f, s in fitted])
    X_test = f_target(test.X)
    p_cal, p_test = fit_predict("lgbm", seed, X_train, fam_train, [X_cal, X_test])
    s_cal_lgbm = np.clip(1.0 - p_cal[:, CONTROL_FAMILY_INDEX], 0.0, 1.0)
    s_test_lgbm = np.clip(1.0 - p_test[:, CONTROL_FAMILY_INDEX], 0.0, 1.0)

    benign_fit = _cap_rows(benign_train, ANOMALY_TRAIN_CAP, seed)
    detectors = {"isolation_forest": B.IsolationForestScorer(
                     seed, n_estimators=IFOREST_N_ESTIMATORS),
                 "autoencoder": B.AutoencoderScorer(seed)}
    variants: dict[str, Any] = {"lightgbm": result_from_scores(cal_all, test, p_cal, p_test,
                                                              seed, n_boot)}
    for name, det in detectors.items():
        det.fit(benign_fit)
        raw_cal, rank_cal = [], []
        for f, s in fitted:
            sc = det.score(f(s.cal.X))
            raw_cal.append(sc)
            rank_cal.append(B.site_rank(det.score(f(s.base)), sc))
        raw_cal, rank_cal = np.concatenate(raw_cal), np.concatenate(rank_cal)
        raw_test = det.score(X_test)
        rank_test = B.site_rank(det.score(f_target(target_base)), raw_test)
        anomaly_cal, anomaly_test = (rank_cal, rank_test) if calibrate else (raw_cal, raw_test)
        variants[name] = result_from_binary(cal_all, test, anomaly_cal, anomaly_test, seed,
                                            n_boot)
        variants[f"hybrid_{name}"] = result_from_binary(
            cal_all, test, np.maximum(s_cal_lgbm, rank_cal),
            np.maximum(s_test_lgbm, rank_test), seed, n_boot)
    return {
        "mode": mode,
        "variants": variants,
        "meta": {"train_rows": int(len(X_train)), "benign_train_rows": int(len(benign_train)),
                 "benign_fit_rows": int(len(benign_fit)), "cal_rows": int(len(cal_all.y)),
                 "test_rows": int(len(test.y)), "target_base_rows": int(len(target_base)),
                 "source_base_rows": {s.name: int(len(s.base)) for s in sources}},
    }


def _calib_ks(args: argparse.Namespace) -> list[int]:
    ks = [args.calib_k]
    for k in args.calib_sensitivity:
        if k not in ks:
            ks.append(k)
    return ks


def _slice_report(ds: Dataset, rows: np.ndarray) -> dict[str, Any]:
    """Size and attack share of a calibration slice. Labels are read here, for the report only,
    after the slice has been used to fit the site transform."""
    return {"n": int(len(rows)), "attack_share": float(np.mean(ds.y[rows]))}


def run_t1_calibrated(ctx, args, datasets, partitions, units, protocols) -> None:
    block = protocols.setdefault("T1_calib", {})
    for a, ds_a in datasets.items():
        tr_a, ca_a = partitions[a]["T0"]["tr"], partitions[a]["T0"]["ca"]
        for b, ds_b in datasets.items():
            if a == b:
                continue
            pair = f"{a}->{b}"
            for k in _calib_ks(args):
                if k >= len(ds_b.y):
                    record_skip(ctx, f"T1_calib {pair} K={k}",
                                f"skipped: K={k} is not below the {len(ds_b.y)} target rows")
                    continue
                slice_rows, rest = B.calibration_slice(len(ds_b.y), k, args.seed)
                source = make_site(a, ds_a, tr_a, ca_a, tr_a, k, args.seed)
                test = take(ds_b, rest)
                reason = checked(f"T1_calib K={k}", pair, source.cal.y, test.y)
                if reason:
                    record_skip(ctx, f"T1_calib {pair} K={k}", reason)
                    continue
                entry = block.setdefault(pair, {}).setdefault(f"K{k}", {})
                entry["slice"] = _slice_report(ds_b, slice_rows)
                for calibrate in (False, True):
                    unit = {"protocol": "T1_calib", "pair": pair, "K": k,
                            "mode": "calib" if calibrate else "raw"}
                    res = units.run(unit, transfer_variants, [source], test,
                                    ds_b.X[slice_rows], calibrate, args.seed, args.n_boot)
                    if res is not None:
                        entry["calib" if calibrate else "raw"] = res
                    _checkpoint(ctx)


def run_t5(ctx, args, datasets, partitions, units, protocols) -> None:
    block = protocols.setdefault("T5", {})
    for held in datasets:
        ds_h = datasets[held]
        n_h = len(ds_h.y)
        for k in _calib_ks(args):
            if k >= n_h:
                record_skip(ctx, f"T5 {held} K={k}",
                            f"skipped: K={k} is not below the {n_h} held-out rows")
                continue
            try:
                sites = lodo_sites(datasets, partitions, held, k, args.seed)
            except ValueError as exc:
                record_skip(ctx, f"T5 {held}", f"skipped: {exc}")
                break
            slice_rows, rest = B.calibration_slice(n_h, k, args.seed)
            test = take(ds_h, rest)
            cal_y = concat_splits([s.cal for s in sites]).y
            reason = checked(f"T5 K={k}", held, cal_y, test.y)
            if reason:
                record_skip(ctx, f"T5 {held} K={k}", reason)
                continue
            entry = block.setdefault(held, {}).setdefault(f"K{k}", {})
            entry["slice"] = _slice_report(ds_h, slice_rows)
            entry["sources"] = [s.name for s in sites]
            for calibrate in (False, True):
                unit = {"protocol": "T5", "held_out": held, "K": k,
                        "mode": "calib" if calibrate else "raw"}
                res = units.run(unit, transfer_variants, sites, test, ds_h.X[slice_rows],
                                calibrate, args.seed, args.n_boot)
                if res is not None:
                    entry["calib" if calibrate else "raw"] = res
                _checkpoint(ctx)


def run_t2_anomaly(ctx, args, datasets, partitions, units, protocols) -> None:
    """Within-dataset, unseen attack family: the held-out family is absent from every fit,
    including the benign-only detectors' baseline (the baseline is the family-free train side)."""
    block = protocols.setdefault("T2_anomaly", {})
    k = args.calib_k
    for key, ds in datasets.items():
        tr, ca, te = (partitions[key]["T0"][x] for x in ("tr", "ca", "te"))
        for fam in S.ATTACK_FAMILIES:
            f = S.FAMILY_INDEX[fam]
            if not (ds.fam_idx[te] == f).any():
                continue
            fam_tr = tr[ds.fam_idx[tr] != f]
            fam_ca = ca[ds.fam_idx[ca] != f]
            fam_te = te[(ds.fam_idx[te] == 0) | (ds.fam_idx[te] == f)]
            site = make_site(key, ds, fam_tr, fam_ca, fam_tr, k, args.seed)
            test = take(ds, fam_te)
            reason = checked(f"T2 anomaly {fam}", key, site.cal.y, test.y)
            if reason:
                record_skip(ctx, f"T2_anomaly {key} hold {fam}", reason)
                continue
            res = units.run({"protocol": "T2_anomaly", "dataset": key, "held_out": fam},
                            transfer_variants, [site], test, site.base, False, args.seed,
                            args.n_boot)
            if res is not None:
                block.setdefault(key, {})[fam] = res
            _checkpoint(ctx)


def run_t2_transfer(ctx, args, datasets, partitions, units, protocols) -> None:
    """Cross-dataset, unseen attack family, at the headline K. The target slice is drawn from all
    target rows (unlabelled) and removed; the test is the target's benign rows plus the held-out
    family, minus the slice."""
    block = protocols.setdefault("T2_transfer", {})
    k = args.calib_k
    for a, ds_a in datasets.items():
        tr_a, ca_a = partitions[a]["T0"]["tr"], partitions[a]["T0"]["ca"]
        for b, ds_b in datasets.items():
            if a == b:
                continue
            if k >= len(ds_b.y):
                record_skip(ctx, f"T2_transfer {a}->{b}",
                            f"skipped: K={k} is not below the {len(ds_b.y)} target rows")
                continue
            slice_rows, rest = B.calibration_slice(len(ds_b.y), k, args.seed)
            for fam in S.ATTACK_FAMILIES:
                f = S.FAMILY_INDEX[fam]
                if not (ds_a.fam_idx[tr_a] == f).any() or not (ds_b.fam_idx == f).any():
                    continue
                fam_tr = tr_a[ds_a.fam_idx[tr_a] != f]
                fam_ca = ca_a[ds_a.fam_idx[ca_a] != f]
                keep = np.intersect1d(np.flatnonzero((ds_b.fam_idx == 0)
                                                     | (ds_b.fam_idx == f)), rest)
                source = make_site(a, ds_a, fam_tr, fam_ca, fam_tr, k, args.seed)
                test = take(ds_b, keep)
                reason = checked(f"T2 transfer {fam}", f"{a}->{b}", source.cal.y, test.y)
                if reason:
                    record_skip(ctx, f"T2_transfer {a}->{b} hold {fam}", reason)
                    continue
                entry = block.setdefault(f"{a}->{b}", {}).setdefault(fam, {})
                for calibrate in (False, True):
                    unit = {"protocol": "T2_transfer", "pair": f"{a}->{b}", "held_out": fam,
                            "mode": "calib" if calibrate else "raw"}
                    res = units.run(unit, transfer_variants, [source], test,
                                    ds_b.X[slice_rows], calibrate, args.seed, args.n_boot)
                    if res is not None:
                        entry["calib" if calibrate else "raw"] = res
                    _checkpoint(ctx)


# ---------------------------------------------------------------------------
# Student: distillation, ONNX export, INT8, parity, latency
# ---------------------------------------------------------------------------


def fold_first_layer(mean: np.ndarray, scale: np.ndarray, W: np.ndarray,
                     b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weights that take raw features: ((x - mean) / scale) @ W + b == x @ W' + b'."""
    W2 = W / scale[:, None]
    b2 = b - (mean / scale) @ W
    return W2, b2


def student_forward_np(X: np.ndarray, layers: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Reference forward pass of the folded student (raw features in, probabilities out)."""
    h = X.astype(np.float64)
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    h = h - h.max(axis=1, keepdims=True)
    e = np.exp(h)
    return e / e.sum(axis=1, keepdims=True)


# onnx.helper stamps the IR version of the installed onnx package (14 for onnx 1.23), and the
# onnxruntime on the Kaggle image loads only up to IR 13 ("Unsupported model IR version: 14").
# IR 8 is the IR that opset 17 is paired with, and every onnxruntime since 1.11 loads it. The
# model is pinned to it so the export does not depend on the onnx version.
ONNX_IR_VERSION = 8
ONNX_OPSET = 17


def build_student_onnx(layers: list[tuple[np.ndarray, np.ndarray]], n_in: int, path: Path,
                       metadata: dict[str, str]) -> None:
    """Write the folded MLP as ONNX: MatMul, Add, Relu, then Softmax. Input 'features' (batch,
    n_in) float32 holds raw live features, output 'probs' (batch, n_families) float32. Batch is a
    symbolic dimension. The IR version is pinned to ONNX_IR_VERSION."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes, inits = [], []
    x = "features"
    for i, (W, b) in enumerate(layers):
        inits.append(numpy_helper.from_array(W.astype(np.float32), f"W{i}"))
        inits.append(numpy_helper.from_array(b.astype(np.float32), f"B{i}"))
        nodes.append(helper.make_node("MatMul", [x, f"W{i}"], [f"mm{i}"]))
        nodes.append(helper.make_node("Add", [f"mm{i}", f"B{i}"], [f"lin{i}"]))
        if i < len(layers) - 1:
            nodes.append(helper.make_node("Relu", [f"lin{i}"], [f"h{i}"]))
            x = f"h{i}"
        else:
            x = f"lin{i}"
    nodes.append(helper.make_node("Softmax", [x], ["probs"], axis=1))
    n_out = layers[-1][0].shape[1]
    graph = helper.make_graph(
        nodes, "rakshak_nids_student", [
            helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch", n_in])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["batch", n_out])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", ONNX_OPSET)],
                              producer_name="rakshak")
    model.ir_version = ONNX_IR_VERSION
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def onnx_session(path: Path, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def onnx_probs(session, X: np.ndarray, batch: int = 65536) -> np.ndarray:
    out = []
    for start in range(0, len(X), batch):
        chunk = X[start:start + batch].astype(np.float32)
        out.append(session.run(None, {"features": chunk})[0])
    return np.concatenate(out) if out else np.zeros((0, len(S.FAMILIES)), np.float32)


def latency_ms(session, X: np.ndarray, n: int = 1000) -> dict[str, float]:
    """Single flow per call, one thread, after a warm-up. Timings are for this machine only."""
    rows = X[np.arange(min(n, len(X))) % len(X)].astype(np.float32)
    for r in rows[:50]:
        session.run(None, {"features": r[None, :]})
    times = []
    for r in rows:
        t0 = time.perf_counter()
        session.run(None, {"features": r[None, :]})
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return {"n": int(len(arr)), "threads": 1, "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)), "mean_ms": float(arr.mean())}


def student_available() -> str | None:
    missing = [m for m in ("onnx", "onnxruntime") if importlib.util.find_spec(m) is None]
    return None if not missing else f"skipped: student needs {missing} (not importable)"


def build_student(ds: Dataset, train: Split, seed: int, out_dir: Path) -> dict[str, Any]:
    """Distil the live teacher into the folded MLP, export FP32 and INT8 ONNX, check them."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    cols = FEATURE_SETS["live"]
    x_tr = train.X[:, cols].astype(np.float64)
    (p_tr,) = fit_predict("lgbm", seed, x_tr, train.fam, [x_tr])  # the teacher, train rows only
    targets = np.log(np.clip(p_tr, 1e-6, 1.0))
    scaler = StandardScaler().fit(x_tr)
    net = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_iter=60,
                       early_stopping=True, n_iter_no_change=6, batch_size=512,
                       random_state=seed).fit(scaler.transform(x_tr), targets)
    layers = [fold_first_layer(scaler.mean_, scaler.scale_, net.coefs_[0], net.intercepts_[0])]
    layers += [(net.coefs_[i], net.intercepts_[i]) for i in range(1, len(net.coefs_))]
    metadata = {
        "rakshak.schema": S.SCHEMA_VERSION,
        "rakshak.features": json.dumps(list(L.LIVE_FEATURES)),
        "rakshak.families": json.dumps(list(S.FAMILIES)),
        "rakshak.source": ds.key,
        "rakshak.kind": "MLP student distilled from live LightGBM (log-prob regression)",
    }
    fp32 = out_dir / f"student_{ds.key}_fp32.onnx"
    int8 = out_dir / f"student_{ds.key}_int8.onnx"
    build_student_onnx(layers, len(cols), fp32, metadata)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    _ensure_metadata(int8, metadata)

    x_check = ds.X[:512][:, cols].astype(np.float64)
    ref = student_forward_np(x_check, layers)
    sess32 = onnx_session(fp32, 1)
    sess8 = onnx_session(int8, 1)
    got32 = onnx_probs(sess32, x_check)
    got8 = onnx_probs(sess8, x_check)
    sklearn_ref = np.exp(net.predict(scaler.transform(x_check)))
    sklearn_ref = sklearn_ref / sklearn_ref.sum(axis=1, keepdims=True)
    parity = {
        "n": int(len(x_check)),
        "fp32_vs_numpy_max_abs": float(np.max(np.abs(got32 - ref))),
        "fp32_vs_sklearn_max_abs": float(np.max(np.abs(got32 - sklearn_ref))),
        "int8_vs_fp32_max_abs": float(np.max(np.abs(got8 - got32))),
        "int8_vs_fp32_argmax_agreement": float(np.mean(got8.argmax(1) == got32.argmax(1))),
    }
    return {
        "layers": [list(W.shape) for W, _ in layers],
        "bytes": {"fp32": int(fp32.stat().st_size), "int8": int(int8.stat().st_size),
                  "budget": BUDGET_BYTES},
        "within_budget": bool(int8.stat().st_size <= BUDGET_BYTES),
        "parity": parity,
        "latency": {"fp32": latency_ms(sess32, ds.X[:, cols]),
                    "int8": latency_ms(sess8, ds.X[:, cols])},
        "paths": {"int8": str(int8), "fp32": str(fp32)},
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
    }


def _ensure_metadata(path: Path, metadata: dict[str, str]) -> None:
    """Quantisation keeps the metadata in current onnx releases. If it did not, write it back."""
    import onnx

    model = onnx.load(str(path))
    have = {e.key for e in model.metadata_props}
    if set(metadata) <= have:
        return
    for key, value in metadata.items():
        if key not in have:
            entry = model.metadata_props.add()
            entry.key, entry.value = key, value
    onnx.save(model, str(path))


def evaluate_student(int8_path: str, cal: Split, test: Split, seed: int,
                     n_boot: int) -> dict[str, Any]:
    """The INT8 student as deployed: probabilities from onnxruntime, thresholds from cal."""
    cols = FEATURE_SETS["live"]
    sess = onnx_session(Path(int8_path), 1)
    p_cal = onnx_probs(sess, cal.X[:, cols]).astype(np.float64)
    p_test = onnx_probs(sess, test.X[:, cols]).astype(np.float64)
    return result_from_scores(cal, test, p_cal, p_test, seed, n_boot)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _f(x: Any, digits: int = 3) -> str:
    if x is None:
        return "-"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "nan" if not np.isfinite(v) else f"{v:.{digits}f}"


def _ci(res: dict, key: str) -> str:
    bs = res.get("bootstrap", {})
    if key not in bs:
        return ""
    lo, hi = bs[key]
    return f" [{_f(lo)}, {_f(hi)}]"


def _row(name: str, res: dict) -> str:
    op = res["operating_points"]["max_f1"]
    tf = res["threshold_free"]
    fam = res.get("family", {})
    r1 = tf["recall_at_cal_fpr"].get("0.01", {}).get("test_recall")
    r01 = tf["recall_at_cal_fpr"].get("0.001", {}).get("test_recall")
    return (f"| {name} | {_f(op['f1'])}{_ci(res, 'f1')} | {_f(op['mcc'])}{_ci(res, 'mcc')} | "
            f"{_f(tf['roc_auc'])} | {_f(tf['pr_auc'])}{_ci(res, 'pr_auc')} | {_f(op['fpr'])} | "
            f"{_f(r01)} | {_f(r1)} | {_f(fam.get('macro_f1'))} |")


_ROW_HEADER = ("| Case | F1 (max-F1 thr) | MCC | ROC-AUC | PR-AUC | FPR @max-F1 | "
               "R @FPR 0.1% (cal thr) | R @FPR 1% (cal thr) | macro-F1 |\n"
               "|---|---|---|---|---|---|---|---|---|")

_CANTONE_REFERENCE = (
    "| Train -> Test (Cantone 2024, CICFlowMeter, 77 features) | best F1 | best MCC "
    "| best AUROC |\n"
    "|---|---|---|---|\n"
    "| CIC-IDS2017 -> CSE-CIC-IDS2018 | 39.12% (LDA) | 35.68% (XGB) | 81.70% (LDA) |\n"
    "| CSE-CIC-IDS2018 -> CIC-IDS2017 | 51.21% (RF) | 44.44% (RF) | 85.06% (LDA) |\n"
    "| LycoS-IDS2017 -> LycoS-Unicas-IDS2018 | 14.42% (XGB) | 10.17% (XGB) | 65.46% (RF) |\n"
    "| LycoS-Unicas-IDS2018 -> LycoS-IDS2017 | 70.46% (LDA) | 60.41% (LDA) | 91.66% (LDA) |\n"
)


_B_HEADER = ("| Case | F1 (max-F1 thr) | MCC | ROC-AUC | PR-AUC | FPR @R=0.90 | "
             "R @FPR 1% (cal thr) |\n|---|---|---|---|---|---|---|")


def _row_b(name: str, res: dict) -> str:
    """Row for a binary-score or family-score result: F1, MCC, ROC-AUC, PR-AUC, FPR at 90% recall
    and recall at FPR 1%."""
    op = res["operating_points"]["max_f1"]
    tf = res["threshold_free"]
    fpr90 = res.get("fpr_at_recall", {}).get("0.9")
    r1 = tf["recall_at_cal_fpr"].get("0.01", {}).get("test_recall")
    return (f"| {name} | {_f(op['f1'])}{_ci(res, 'f1')} | {_f(op['mcc'])}{_ci(res, 'mcc')} | "
            f"{_f(tf['roc_auc'])} | {_f(tf['pr_auc'])}{_ci(res, 'pr_auc')} | {_f(fpr90)} | "
            f"{_f(r1)} |")


CALIB_VARIANTS = ("lightgbm", "isolation_forest", "autoencoder", "hybrid_isolation_forest",
                  "hybrid_autoencoder")


def _calib_headline_k(results: dict[str, Any]) -> int | None:
    meta = results.get("meta", {}).get("calibration")
    return None if not meta else meta.get("headline_K")


def _calib_block_rows(entry: dict[str, Any], names: tuple[str, ...]) -> list[str]:
    rows = []
    for mode in ("raw", "calib"):
        variants = entry.get(mode, {}).get("variants", {})
        for name in names:
            if name in variants:
                rows.append(_row_b(f"{name} {mode}", variants[name]))
    return rows


def _calib_sections(results: dict[str, Any]) -> list[str]:
    protocols = results.get("protocols", {})
    k = _calib_headline_k(results)
    lines: list[str] = []
    t1 = protocols.get("T1_calib", {})
    if t1 and k is not None:
        lines += [f"## T1 home-baseline calibration, headline K={k} (target slice removed "
                  "from test in every row below except the first)", "", _B_HEADER]
        for pair, ks in t1.items():
            entry = ks.get(f"K{k}")
            if not entry:
                continue
            raw_full = protocols.get("T1", {}).get(pair, {}).get("lightgbm")
            if raw_full:
                lines.append(_row_b(f"{pair} / raw T1 (full test, no slice removed)", raw_full))
            for row in _calib_block_rows(entry, CALIB_VARIANTS):
                lines.append(row.replace("| ", f"| {pair} / ", 1))
        lines += ["", "Calibrated rows: each site standardised with its own K-flow unlabelled "
                  "baseline. Raw rows: no transform, same test rows. Anomaly rows use the raw "
                  "score in raw mode and the site rank in calib mode.", ""]
    t5 = protocols.get("T5", {})
    if t5 and k is not None:
        lines += [f"## T5 leave-one-dataset-out, headline K={k}", "", _B_HEADER]
        for held, ks in t5.items():
            entry = ks.get(f"K{k}")
            if not entry:
                continue
            srcs = ", ".join(entry.get("sources", []))
            for row in _calib_block_rows(entry, CALIB_VARIANTS):
                lines.append(row.replace("| ", f"| LODO {held} (train {srcs}) / ", 1))
        lines.append("")
    sens = [("T1 " + p, ks) for p, ks in t1.items()] + [("T5 held " + h, ks)
                                                        for h, ks in t5.items()]
    ks_all = results.get("meta", {}).get("calibration", {}) or {}
    k_values = ks_all.get("K_values", [])
    if sens and k_values:
        lines += ["## K sensitivity (lightgbm calib: F1 / MCC / PR-AUC per baseline size)", "",
                  "| Case | " + " | ".join(f"K={kv}" for kv in k_values) + " |",
                  "|---|" + "---|" * len(k_values)]
        for name, ks in sens:
            cells = []
            for kv in k_values:
                res = ks.get(f"K{kv}", {}).get("calib", {}).get("variants", {}).get("lightgbm")
                if res is None:
                    cells.append("-")
                    continue
                op = res["operating_points"]["max_f1"]
                tf = res["threshold_free"]
                cells.append(f"{_f(op['f1'])} / {_f(op['mcc'])} / {_f(tf['pr_auc'])}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    slices = []
    for name, ks in sens:
        for kv in k_values:
            entry = ks.get(f"K{kv}")
            if entry and "slice" in entry:
                sl = entry["slice"]
                slices.append(f"| {name} | {kv} | {sl['n']} | {_f(sl['attack_share'])} |")
    if slices:
        lines += ["## Calibration slices (attack share is reported, never used to fit)", "",
                  "| Target | K | Slice rows | Attack share of slice |", "|---|---|---|---|"]
        lines += slices + [""]
    return lines


def _t2_anomaly_sections(results: dict[str, Any]) -> list[str]:
    protocols = results.get("protocols", {})
    lines: list[str] = []
    t2a = protocols.get("T2_anomaly", {})
    if t2a:
        lines += ["## T2 unseen attack family, within dataset: benign-only detectors and hybrid",
                  "", "Raw features throughout. The held-out family is absent from every fit "
                  "and from the rank baseline. lightgbm is the T2 model, reused here.", "",
                  _B_HEADER]
        for key, fams in t2a.items():
            for fam, entry in fams.items():
                t2 = protocols.get("T2", {}).get(key, {}).get(fam, {}).get("lightgbm")
                if t2:
                    lines.append(_row_b(f"{key} hold {fam} / lightgbm raw", t2))
                for name, res in entry.get("raw", {}).get("variants", {}).items():
                    if name != "lightgbm":  # the T2 lightgbm row above is the same model
                        lines.append(_row_b(f"{key} hold {fam} / {name} raw", res))
        lines.append("")
    t2t = protocols.get("T2_transfer", {})
    if t2t:
        k = _calib_headline_k(results)
        lines += [f"## T2 unseen attack family, cross-dataset, headline K={k}", "", _B_HEADER]
        for pair, fams in t2t.items():
            for fam, entry in fams.items():
                for mode, names in (("raw", ("lightgbm", "hybrid_isolation_forest")),
                                    ("calib", ("lightgbm", "isolation_forest",
                                               "hybrid_isolation_forest"))):
                    variants = entry.get(mode, {}).get("variants", {})
                    for name in names:
                        if name in variants:
                            lines.append(_row_b(f"{pair} hold {fam} / {name} {mode}",
                                                variants[name]))
        lines.append("")
    return lines


def render_report(results: dict[str, Any]) -> str:
    lines = ["# Network-IDS baselines (shared NetFlow-v2 schema)", ""]
    meta = results.get("meta", {})
    lines += [f"Schema `{meta.get('schema')}`, seed {meta.get('seed')}, max rows per dataset "
              f"{meta.get('max_rows')}, bootstrap resamples {meta.get('n_boot')}.", ""]
    lines += ["## Data", "", "| Dataset | Rows total | Benign / attack used | Duplicates removed | "
              "Host grouping |", "|---|---|---|---|---|"]
    for key, info in results.get("datasets", {}).items():
        groups = info.get("host_groups", {})
        lines.append(f"| {key} | {info.get('rows_total')} | {info.get('benign_used')} / "
                     f"{info.get('attack_used')} | {info.get('duplicates_removed')} | "
                     f"{groups.get('method')} ({groups.get('count')} groups, largest "
                     f"{_f(groups.get('largest_share'))}) |")
    lines += ["", "Labels not mapped by name (`unknown`, `conflict`) are counted in `results.json` "
              "under `datasets.<key>.label_mapping_how`.", ""]
    schema = results.get("schema", {})
    lines += ["## Feature schema", "",
              f"Shared: {len(schema.get('shared', []))} features. Live (nfstream-mappable): "
              f"{len(schema.get('live', []))}.", "",
              "Unmappable live features (the *_live model does not use them):", ""]
    for name, reason in schema.get("unmappable_live", {}).items():
        lines.append(f"- `{name}`: {reason}")
    lines.append("")
    protocols = results.get("protocols", {})
    if "T0" in protocols:
        lines += ["## T0 random split within dataset (reference only)", "", _ROW_HEADER]
        for ds, models in protocols["T0"].items():
            for model, res in models.items():
                lines.append(_row(f"{ds} / {model}", res))
        lines.append("")
    if "T1" in protocols:
        lines += ["## T1 cross-dataset, all ordered pairs", "", _ROW_HEADER]
        for pair, models in protocols["T1"].items():
            for model, res in models.items():
                lines.append(_row(f"{pair} / {model}", res))
        lines += ["", "Reference (Cantone et al. 2024, Table 3). Different datasets, features "
                  "and labels; the metric definition behind their F1 is not verified "
                  "(docs/NETWORK_IDS_PLAN.md section 7, item 5):", "", _CANTONE_REFERENCE, ""]
    if "T4" in protocols:
        lines += ["## T4 host-disjoint within dataset", "", _ROW_HEADER]
        for ds, models in protocols["T4"].items():
            for model, res in models.items():
                lines.append(_row(f"{ds} / {model}", res))
        lines.append("")
    if "T2" in protocols:
        lines += ["## T2 leave-one-attack-family-out", "", _ROW_HEADER]
        for ds, families in protocols["T2"].items():
            for fam, models in families.items():
                for model, res in models.items():
                    lines.append(_row(f"{ds} hold {fam} / {model}", res))
        lines.append("")
    if "T2_cross" in protocols:
        lines += ["## T2 leave-one-attack-family-out, cross-dataset", "", _ROW_HEADER]
        for pair, families in protocols["T2_cross"].items():
            for fam, models in families.items():
                for model, res in models.items():
                    lines.append(_row(f"{pair} hold {fam} / {model}", res))
        lines.append("")
    lines += _calib_sections(results)
    lines += _t2_anomaly_sections(results)
    student = results.get("student", {})
    if student:
        lines += ["## Student (INT8 ONNX, live features)", ""]
        if "skipped" in student:
            lines.append(student["skipped"])
        for src, block in student.get("by_source", {}).items():
            b = block.get("bytes", {})
            lat = block.get("latency", {}).get("int8", {})
            par = block.get("parity", {})
            lines.append(f"- {src}: INT8 {b.get('int8')} bytes (budget {b.get('budget')}, within: "
                         f"{block.get('within_budget')}); FP32 {b.get('fp32')} bytes; INT8 "
                         f"single-thread p50 {_f(lat.get('p50_ms'))} ms, p95 "
                         f"{_f(lat.get('p95_ms'))} ms on this x86 host; parity fp32/numpy max "
                         f"{_f(par.get('fp32_vs_numpy_max_abs'), 6)}, int8/fp32 argmax agreement "
                         f"{_f(par.get('int8_vs_fp32_argmax_agreement'))}.")
        lines.append("")
        lines += [_ROW_HEADER]
        for name, res in student.get("results", {}).items():
            lines.append(_row(name, res))
        lines.append("")
    lines += ["## Metric notes", "",
              "- Binary F1, MCC and PR-AUC treat attacks (any family) as positive. The threshold "
              "for F1 is the max-F1 point on calibration. The recall columns use the threshold "
              "for each FPR on calibration.",
              "- macro-F1 is over the families present in the test split (benign included). The "
              "decision is the argmax of the family probabilities.",
              "- FPR @R=0.90 is the smallest test FPR at which test recall reaches 0.90 (oracle "
              "on the test split, threshold-free). Binary scores have no family prediction, so "
              "the macro-F1 is not reported for them.",
              "- Calibration: a target's K unlabelled flows fit the site transform and are removed "
              "from its test. Attack shares of slices are in the calibration table. Benign-"
              "dominated slices are the deployment assumption. Where a dataset's used rows are "
              "mostly attacks (bot_iot here), the assumption fails and the row says so.",
              "- Hybrid = max(LightGBM attack probability, site rank of the anomaly score). Both "
              "thresholds are chosen on the pooled source calibration splits.",
              "- Bracketed intervals are host-group clustered bootstrap percentiles (95%).",
              "- Feature sets, thresholds and hyper-parameters are fixed before the test splits "
              "are scored. Nothing here was tuned on a test split.", ""]
    return "\n".join(lines)


def _checkpoint(ctx) -> None:
    ctx.work.mkdir(parents=True, exist_ok=True)
    (ctx.work / "report.md").write_text(render_report(ctx.results), encoding="utf-8")


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------


def run(ctx) -> bool:
    args = parse_args(ctx.argv)
    seed_everything(args.seed)
    env = env_report()
    ctx.log(f"env: {env}")
    root = Path(args.data_root or os.environ.get("RAKSHAK_NIDS_DATA", "/kaggle/input"))
    files = discover(root, args.datasets)
    ctx.results["meta"] = {"schema": S.SCHEMA_VERSION, "seed": args.seed,
                           "max_rows": args.max_rows, "n_boot": args.n_boot, "env": env,
                           "data_root": str(root),
                           "calibration": None if args.no_calib else {
                               "headline_K": args.calib_k, "K_values": _calib_ks(args),
                               "transform": "per-feature QuantileTransformer (uniform), fitted "
                                            "on K unlabelled flows per site; categorical "
                                            "features pass through",
                               "rank": "fraction of the site's unlabelled baseline at or below "
                                       "the anomaly score",
                               "slice": "first K of a seeded random order; removed from the "
                                        "target test; attack share reported only",
                               "source_baseline": "first min(K, source train rows) of a seeded "
                                                  "random order of the source's train side"}}
    ctx.results["schema"] = {
        "shared": list(S.SHARED_FEATURES), "live": list(L.LIVE_FEATURES),
        "unmappable_live": L.UNMAPPABLE_LIVE, "log_features": list(S.LOG_FEATURES),
        "families": list(S.FAMILIES),
    }
    units = Units(ctx)
    datasets: dict[str, Dataset] = {}
    partitions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    ctx.results["datasets"] = {}
    for key in args.datasets:
        if key not in files:
            record_skip(ctx, key, f"skipped: no parquet for {key} under {root}")
            continue
        path = files[key]
        columns = pq.read_schema(path).names
        ctx.results.setdefault("parquet_columns", {})[key] = list(columns)
        ctx.log(f"{key}: {path} parquet columns ({len(columns)}): {list(columns)}")
        csv_check = check_feature_csv(path)
        ctx.log(f"{key}: csv check: {csv_check.get('file')}")
        ds = units.run({"stage": "sample", "dataset": key}, sample_dataset, path, key,
                       args.max_rows, args.seed)
        if ds is None:
            continue
        ds.info["feature_csv"] = csv_check
        ctx.results["datasets"][key] = ds.info
        datasets[key] = ds
        partitions[key] = {
            "T0": dict(zip(("tr", "ca", "te"),
                           SP.random_split(len(ds.y), FRACTIONS, args.seed, strata=ds.fam_idx),
                           strict=True)),
            "T4": None if ds.groups is None else dict(zip(
                ("tr", "ca", "te"), SP.group_split(ds.groups, FRACTIONS, args.seed), strict=True)),
        }
        ctx.log(f"{key}: {ds.info['rows_used']} rows, {ds.info['attack_used']} attacks, "
                f"groups {ds.info['host_groups']}")
        _checkpoint(ctx)
    if not datasets:
        ctx.results["summary"] = "no datasets loaded"
        return False

    protocols: dict[str, Any] = {}
    ctx.results["protocols"] = protocols
    models = args.models

    if "T0" in args.protocols:
        block = protocols.setdefault("T0", {})
        for key, ds in datasets.items():
            tr, ca, te = (partitions[key]["T0"][k] for k in ("tr", "ca", "te"))
            reason = checked("T0", key, ds.y[ca], ds.y[te])
            if reason:
                record_skip(ctx, f"T0 {key}", reason)
                continue
            train, cal, test = take(ds, tr), take(ds, ca), take(ds, te)
            for name in models:
                res = units.run({"protocol": "T0", "dataset": key, "model": name},
                                evaluate_model, name, args.seed, train, cal, test, args.n_boot)
                if res is not None:
                    block.setdefault(key, {})[name] = res
                    ctx.log(f"T0 {key} {name}: F1 {res['operating_points']['max_f1']['f1']:.4f}")
                _checkpoint(ctx)

    if "T1" in args.protocols:
        block = protocols.setdefault("T1", {})
        for a, ds_a in datasets.items():
            tr_a, ca_a = partitions[a]["T0"]["tr"], partitions[a]["T0"]["ca"]
            train, cal = take(ds_a, tr_a), take(ds_a, ca_a)
            for b, ds_b in datasets.items():
                if a == b:
                    continue
                test = take(ds_b, np.arange(len(ds_b.y)))
                reason = checked("T1", f"{a}->{b}", cal.y, test.y)
                if reason:
                    record_skip(ctx, f"T1 {a}->{b}", reason)
                    continue
                for name in models:
                    res = units.run({"protocol": "T1", "pair": f"{a}->{b}", "model": name},
                                    evaluate_model, name, args.seed, train, cal, test,
                                    args.n_boot)
                    if res is not None:
                        block.setdefault(f"{a}->{b}", {})[name] = res
                    _checkpoint(ctx)

    if "T1" in args.protocols and not args.no_calib:
        run_t1_calibrated(ctx, args, datasets, partitions, units, protocols)

    if "T4" in args.protocols:
        block = protocols.setdefault("T4", {})
        for key, ds in datasets.items():
            if partitions[key]["T4"] is None:
                record_skip(ctx, f"T4 {key}", "skipped: no host columns")
                continue
            tr, ca, te = (partitions[key]["T4"][k] for k in ("tr", "ca", "te"))
            reason = checked("T4", key, ds.y[ca], ds.y[te])
            if reason:
                record_skip(ctx, f"T4 {key}", reason)
                continue
            train, cal, test = take(ds, tr), take(ds, ca), take(ds, te)
            for name in models:
                res = units.run({"protocol": "T4", "dataset": key, "model": name},
                                evaluate_model, name, args.seed, train, cal, test, args.n_boot)
                if res is not None:
                    block.setdefault(key, {})[name] = res
                _checkpoint(ctx)

    if "T2" in args.protocols:
        within = protocols.setdefault("T2", {})
        cross = protocols.setdefault("T2_cross", {})
        for key, ds in datasets.items():
            tr, ca, te = (partitions[key]["T0"][k] for k in ("tr", "ca", "te"))
            for fam in S.ATTACK_FAMILIES:
                f = S.FAMILY_INDEX[fam]
                fam_tr = tr[ds.fam_idx[tr] != f]
                fam_ca = ca[ds.fam_idx[ca] != f]
                fam_te = te[(ds.fam_idx[te] == 0) | (ds.fam_idx[te] == f)]
                if not (ds.fam_idx[te] == f).any():
                    continue
                reason = checked(f"T2 {fam}", key, ds.y[fam_ca], ds.y[fam_te])
                if reason:
                    record_skip(ctx, f"T2 {key} hold {fam}", reason)
                    continue
                train, cal, test = take(ds, fam_tr), take(ds, fam_ca), take(ds, fam_te)
                for name in models:
                    res = units.run({"protocol": "T2", "dataset": key, "held_out": fam,
                                     "model": name}, evaluate_model, name, args.seed, train,
                                    cal, test, args.n_boot)
                    if res is not None:
                        within.setdefault(key, {}).setdefault(fam, {})[name] = res
                    _checkpoint(ctx)
        for a, ds_a in datasets.items():
            tr_a, ca_a = partitions[a]["T0"]["tr"], partitions[a]["T0"]["ca"]
            for b, ds_b in datasets.items():
                if a == b:
                    continue
                for fam in S.ATTACK_FAMILIES:
                    f = S.FAMILY_INDEX[fam]
                    if not (ds_b.fam_idx == f).any() or not (ds_a.fam_idx[tr_a] == f).any():
                        continue
                    fam_tr = tr_a[ds_a.fam_idx[tr_a] != f]
                    fam_ca = ca_a[ds_a.fam_idx[ca_a] != f]
                    keep_b = np.flatnonzero((ds_b.fam_idx == 0) | (ds_b.fam_idx == f))
                    reason = checked(f"T2 {fam}", f"{a}->{b}", ds_a.y[fam_ca], ds_b.y[keep_b])
                    if reason:
                        record_skip(ctx, f"T2 {a}->{b} hold {fam}", reason)
                        continue
                    train, cal, test = take(ds_a, fam_tr), take(ds_a, fam_ca), take(ds_b, keep_b)
                    for name in args.t2_cross_models:
                        res = units.run({"protocol": "T2_cross", "pair": f"{a}->{b}",
                                         "held_out": fam, "model": name}, evaluate_model, name,
                                        args.seed, train, cal, test, args.n_boot)
                        if res is not None:
                            cross.setdefault(f"{a}->{b}", {}).setdefault(fam, {})[name] = res
                        _checkpoint(ctx)

    if "T5" in args.protocols and not args.no_calib:
        run_t5(ctx, args, datasets, partitions, units, protocols)

    if "T2" in args.protocols and not args.no_calib:
        run_t2_anomaly(ctx, args, datasets, partitions, units, protocols)
        run_t2_transfer(ctx, args, datasets, partitions, units, protocols)

    if not args.no_student:
        skip = student_available()
        if skip:
            ctx.results["student"] = {"skipped": skip}
        else:
            student = ctx.results.setdefault("student", {"by_source": {}, "results": {}})
            out_dir = ctx.work
            for key, ds in datasets.items():
                tr, ca, te = (partitions[key]["T0"][k] for k in ("tr", "ca", "te"))
                train, cal, test = take(ds, tr), take(ds, ca), take(ds, te)
                bundle = units.run({"protocol": "student_build", "dataset": key},
                                   build_student, ds, train, args.seed, out_dir)
                if bundle is None:
                    continue
                student["by_source"][key] = {k: v for k, v in bundle.items() if k != "paths"}
                int8 = bundle["paths"]["int8"]
                targets = [("T0", key, cal, test)]
                targets += [("T1", f"{key}->{other}", cal, take(d, np.arange(len(d.y))))
                            for other, d in datasets.items() if other != key]
                for proto, name, c, t in targets:
                    reason = checked(proto, name, c.y, t.y)
                    if reason:
                        record_skip(ctx, f"{proto} student {name}", reason)
                        continue
                    res = units.run({"protocol": f"student_{proto}", "dataset": key,
                                     "test": name}, evaluate_student, int8, c, t, args.seed,
                                     args.n_boot)
                    if res is not None:
                        student["results"][f"student {name}"] = res
                    _checkpoint(ctx)

    _checkpoint(ctx)
    ctx.results["summary"] = (
        f"{len(datasets)} NIDS datasets, {units.ok} units ok, {units.failed} failed; "
        f"protocols {','.join(args.protocols)}; models {','.join(models)}"
    )
    return units.status()
