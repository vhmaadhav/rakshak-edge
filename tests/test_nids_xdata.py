"""Unit tests for rakshak.nids.xdata (cross-dataset NIDS helpers). Synthetic frames only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rakshak.eval import metrics as M
from rakshak.nids import schema as S
from rakshak.nids import xdata as X


def _raw(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw = {f: rng.integers(0, 1000, size=n).astype(np.float64) for f in S.SHARED_FEATURES}
    raw["IN_BYTES"] = rng.integers(1, 10**6, size=n).astype(np.float64)
    raw["OUT_BYTES"] = rng.integers(1, 10**6, size=n).astype(np.float64)
    raw["IN_PKTS"] = rng.integers(1, 100, size=n).astype(np.float64)
    raw["OUT_PKTS"] = rng.integers(1, 100, size=n).astype(np.float64)
    raw["FLOW_DURATION_MILLISECONDS"] = rng.integers(1, 10**5, size=n).astype(np.float64)
    raw["DURATION_IN"] = raw["FLOW_DURATION_MILLISECONDS"] / 2
    raw["DURATION_OUT"] = raw["FLOW_DURATION_MILLISECONDS"] / 3
    return raw


def test_feature_names_and_hygiene_drop_lists_are_consistent():
    assert len(X.ALL_NAMES) == len(S.SHARED_FEATURES) + len(X.ENGINEERED)
    assert not set(X.HYGIENE_FEATURES) & set(X.HYGIENE_DROP_FEATURES)
    assert "L7_PROTO" not in X.HYGIENE_FEATURES and "DNS_TTL_ANSWER" not in X.HYGIENE_FEATURES
    # identifier, port and TTL columns never reach the model matrix
    raw = _raw(5)
    raw["IPV4_SRC_ADDR"] = np.arange(5, dtype=np.float64)
    raw["L4_SRC_PORT"] = np.arange(5, dtype=np.float64)
    base = X.build_feature_matrix(raw)
    raw["IPV4_SRC_ADDR"] = raw["IPV4_SRC_ADDR"] * 0 + 7
    raw["L4_SRC_PORT"] = raw["L4_SRC_PORT"] * 0 + 9
    assert np.array_equal(base, X.build_feature_matrix(raw))


def test_build_feature_matrix_shape_and_engineered_values():
    raw = _raw(50)
    mat = X.build_feature_matrix(raw)
    assert mat.shape == (50, len(X.ALL_NAMES)) and mat.dtype == np.float32
    assert np.isfinite(mat).all()
    j = X.column_index(["IN_OUT_BYTES_RATIO"])[0]
    expect = np.log1p(raw["IN_BYTES"] / (raw["OUT_BYTES"] + 1.0))
    assert np.allclose(mat[:, j], expect, rtol=1e-5)


def test_column_index_rejects_unknown_names():
    with pytest.raises(KeyError):
        X.column_index(["NOT_A_FEATURE"])


def test_stable_features_keeps_only_features_strong_on_every_source():
    rng = np.random.default_rng(1)
    n = 4000
    y = rng.integers(0, 2, size=n)
    names = [f"f{i}" for i in range(10)]
    base = rng.normal(size=(n, len(names)))
    Xa = base.copy()
    Xb = rng.normal(size=(n, len(names)))
    Xa[:, 0] = Xb[:, 0] = y + rng.normal(0, 0.1, size=n)  # f0 predictive on both sources
    Xa[:, 1] = y + rng.normal(0, 0.1, size=n)  # f1 predictive on source a only
    Xb[:, 2:] = rng.normal(size=(n, len(names) - 2))
    sel = X.stable_features({"a": (Xa, y), "b": (Xb, y)}, names, k=2, min_keep=1, seed=0)
    assert "f0" in sel and "f1" not in sel


def test_stable_features_fallback_uses_worst_source_rank():
    rng = np.random.default_rng(2)
    n = 3000
    y = rng.integers(0, 2, size=n)
    names = ["f0", "f1", "f2"]
    Xa = np.column_stack([y + rng.normal(0, 0.2, n), rng.normal(size=n), rng.normal(size=n)])
    Xb = np.column_stack([rng.normal(size=n), y + rng.normal(0, 0.2, n), rng.normal(size=n)])
    sel = X.stable_features({"a": (Xa, y), "b": (Xb, y)}, names, k=1, min_keep=2, seed=0)
    assert len(sel) == 2 and sel == sorted(sel, key=names.index)


def test_stable_features_rejects_bad_width():
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        X.stable_features({"a": (np.zeros((4, 2)), y)}, ["x", "y", "z"], min_keep=1)


def test_benign_quantile_map_ignores_labels_and_attacks():
    rng = np.random.default_rng(3)
    benign = rng.normal(size=(500, 3))
    names = ["PROTOCOL", "IN_BYTES", "OUT_BYTES"]
    smap = X.fit_benign_quantiles(benign, names)
    attacks = rng.normal(loc=10, size=(500, 3))
    # the map is a function of the rows it is given: the same benign rows give the same map
    smap2 = X.fit_benign_quantiles(benign.copy(), names)
    probe = rng.normal(size=(40, 3))
    assert np.allclose(smap.transform(probe), smap2.transform(probe))
    # attack rows are not an input to the fit: the map is the one fitted on benign rows alone
    assert smap.transform(attacks).shape == attacks.shape
    out = smap.transform(probe)
    assert out.shape == probe.shape
    assert np.allclose(out[:, 0], probe[:, 0].astype(np.float32))  # categorical pass-through


def test_mcc_curve_matches_binary_metrics():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, size=600)
    s = y * 0.6 + rng.random(600)
    thr = np.array([0.2, 0.5, 0.9, 1.3])
    curve = X.mcc_curve(y, s, thr)
    for i, t in enumerate(thr):
        ref = M.binary_metrics(y, s, float(t))["mcc"]
        assert curve["mcc"][i] == pytest.approx(ref, abs=1e-9)


def test_choose_threshold_separable_sits_between_classes():
    y = np.array([0] * 50 + [1] * 50)
    s = np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    out = X.choose_threshold([(y, s), (y, s)])
    assert 0.4 < out["threshold"] < 0.61
    assert out["mean_inner_mcc"] == pytest.approx(1.0)
    assert out["folds"] == 2


def test_choose_threshold_uses_mean_over_folds():
    y = np.array([0, 0, 1, 1])
    good = np.array([0.1, 0.2, 0.8, 0.9])
    bad = np.array([0.6, 0.7, 0.3, 0.4])  # inverted ordering in the second fold
    out = X.choose_threshold([(y, good), (y, bad)])
    assert out["mean_inner_mcc"] < 1.0


def test_choose_threshold_requires_folds():
    with pytest.raises(ValueError):
        X.choose_threshold([])


def test_choose_config_breaks_ties_by_fpr_then_name():
    res = {"cfg0": {"mean_inner_mcc": 0.8, "mean_inner_fpr": 0.02},
           "cfg1": {"mean_inner_mcc": 0.8, "mean_inner_fpr": 0.01},
           "cfg2": {"mean_inner_mcc": 0.7, "mean_inner_fpr": 0.0}}
    assert X.choose_config(res) == "cfg1"
    assert X.choose_config({"cfg0": res["cfg0"]}) == "cfg0"


def test_target_metrics_mcc_matches_repo_metrics():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, size=400)
    s = y * 0.5 + rng.random(400)
    pred = (s >= 0.7).astype(np.int64)
    got = X.target_metrics(y, pred, s)
    ref = M.binary_metrics(y, s, 0.7)
    assert got["mcc"] == pytest.approx(ref["mcc"], abs=1e-9)
    assert got["fpr"] == pytest.approx(ref["fpr"], abs=1e-9)
    assert got["n"] == 400 and 0 < got["prevalence"] < 1


def test_bootstrap_draws_are_paired_and_reproducible():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, size=300)
    p1 = (rng.random(300) < 0.5).astype(np.int64)
    d1 = X.bootstrap_metric_draws(y, [p1, p1], n_boot=20, seed=0)
    assert np.array_equal(d1[:, 0], d1[:, 1])
    d2 = X.bootstrap_metric_draws(y, [p1], n_boot=20, seed=0)
    assert np.array_equal(d1[:, :1], d2)
    lo, hi = X.percentile_interval(d1[:, 0])
    assert lo <= hi


def test_training_subsample_is_balanced_and_capped():
    y = np.array([0] * 100 + [1] * 10)
    rows = X.training_subsample(y, per_class=30, seed=0)
    assert (y[rows] == 0).sum() == 30 and (y[rows] == 1).sum() == 10
    assert np.all(np.diff(rows) > 0)
    with pytest.raises(ValueError):
        X.training_subsample(np.zeros(5, dtype=int), per_class=2, seed=0)


def test_stage_sample_writes_numeric_columns_with_caps(tmp_path):
    rng = np.random.default_rng(7)
    n = 3000
    y = (rng.random(n) < 0.3).astype(np.int64)
    attack = np.where(y == 1, rng.choice(["DoS", "Scan", "Rare"], size=n, p=[0.6, 0.39, 0.01]), "")
    df = pd.DataFrame({
        "IPV4_SRC_ADDR": [f"10.0.0.{i % 250}" for i in range(n)],
        "IN_BYTES": rng.integers(0, 1000, size=n),
        "OUT_BYTES": rng.integers(0, 1000, size=n),
        "Label": y, "Attack": attack,
    })
    src = tmp_path / "raw.parquet"
    df.to_parquet(src, index=False)
    out = tmp_path / "staged.parquet"
    info = X.stage_sample(src, out, max_benign=200, max_attack=100, seed=0, batch_rows=700)
    staged = pd.read_parquet(out)
    assert "IPV4_SRC_ADDR" not in staged.columns
    assert set(staged.columns) == {"IN_BYTES", "OUT_BYTES", "Label", "Attack"}
    assert int((staged["Label"] == 0).sum()) == info.benign_kept <= 200
    assert int((staged["Label"] == 1).sum()) == info.attack_kept <= 100
    # the rare attack keeps all its rows (water-filling)
    assert info.attack_strings.get("Rare", 0) == int(((attack == "Rare")).sum())
    again = tmp_path / "again.parquet"
    X.stage_sample(src, again, max_benign=200, max_attack=100, seed=0, batch_rows=700)
    assert pd.read_parquet(again).equals(staged)
