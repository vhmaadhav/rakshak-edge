"""Fast synthetic-only checks of calibration, split isolation and selective statistics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from rakshak.fusion import (
    FusionModel,
    IsotonicCalibration,
    PlattScaling,
    SelectiveThresholds,
    TemperatureScaling,
    calibration_metrics,
    conformal_fpr,
    feature_matrix,
    risk_coverage,
    search_thresholds,
    selective_metrics,
)
from rakshak.fusion.fuse import FEATURE_NAMES, MODELS
from rakshak.fusion.report import evaluate_fusion, write_report
from rakshak.fusion.selective import HIGH, LOW, campaign_size_buckets, group_coverage


def roundtrip(obj):
    return type(obj).from_dict(json.loads(json.dumps(obj.to_dict(), allow_nan=False)))


def test_temperature_recovers_known_temperature():
    # Exact event rates at each logit remove Monte Carlo noise without a large fit.
    rates = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    true_temperature = 2.5
    scores = np.repeat(np.log(rates / (1 - rates)) * true_temperature, 100)
    y = np.concatenate([np.r_[np.ones(round(p * 100)), np.zeros(100 - round(p * 100))]
                        for p in rates])
    model = TemperatureScaling().fit(scores, y)
    assert model.temperature == pytest.approx(true_temperature, abs=1e-4)
    np.testing.assert_allclose(roundtrip(model).transform(scores), model.transform(scores))


@pytest.mark.parametrize("cls", [PlattScaling, IsotonicCalibration])
def test_calibrator_roundtrip_and_monotonicity(cls):
    s = np.linspace(-3, 3, 80)
    y = np.random.default_rng(3).binomial(1, expit(s))
    model = cls().fit(s, y)
    grid = np.linspace(-5, 5, 100)
    assert np.all(np.diff(model.transform(grid)) >= 0)
    np.testing.assert_allclose(roundtrip(model).transform(grid), model.transform(grid))


def test_perfect_calibration_and_bins():
    p = np.repeat([0.1, 0.5, 0.9], 100)
    y = np.r_[np.ones(10), np.zeros(90), np.ones(50), np.zeros(50),
              np.ones(90), np.zeros(10)]
    stats = calibration_metrics(y, p)
    assert stats["ece_width"] < 1e-12
    assert stats["ece_mass"] < 1e-12
    assert stats["brier"] == pytest.approx(np.mean((p - y) ** 2))
    for name in ("reliability_width", "reliability_mass"):
        assert sum(b["count"] for b in stats[name]) == len(y)
    assert len(stats["reliability_width"]) == 15
    with pytest.raises(ValueError):
        calibration_metrics([0.1], [0.3])


def matrix(n=80):
    rng = np.random.default_rng(17)
    x = np.zeros((n, len(FEATURE_NAMES)))
    x[:, :2] = rng.uniform(size=(n, 2))
    x[:, 2:] = rng.integers(0, 2, size=x[:, 2:].shape)
    x[x[:, 2] == 0, 1] = 0
    y = (x[:, 0] > 0.5).astype(int)
    return x, y


@pytest.mark.parametrize("name", MODELS)
def test_fusion_split_isolation_and_json(name):
    x, y = matrix()
    ids = np.array([f"row{i}" for i in range(len(y))])
    cal = ids[10:70]
    kwargs = dict(ids=ids, cal_ids=cal, train_ids=ids[:10], test_ids=ids[70:])
    model = FusionModel(name).fit(x, y, **kwargs)
    assert model.fit_ids == cal.tolist()
    assert not set(model.fit_ids) & set(ids[70:])
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[:10] = changed_x[70:] = np.nan
    changed_y[:10] = changed_y[70:] = -999
    other = FusionModel(name).fit(changed_x, changed_y, **kwargs)
    np.testing.assert_allclose(other.transform(x), model.transform(x))
    np.testing.assert_allclose(roundtrip(model).transform(x), model.transform(x))
    with pytest.raises(ValueError, match="overlap"):
        FusionModel(name).fit(x, y, **{**kwargs, "cal_ids": ids[10:]})


def test_lightgbm_export_matches_native_and_is_monotone():
    x, y = matrix(120)
    model = FusionModel("lightgbm").fit(x, y, ids=list(range(120)),
                                        cal_ids=list(range(120)), train_ids=[], test_ids=[])
    native = lgb.LGBMClassifier(
        n_estimators=40, num_leaves=7, max_depth=3, learning_rate=0.05,
        min_child_samples=10, n_jobs=1, verbosity=-1, random_state=0,
        monotone_constraints=[1, 1] + [0] * (x.shape[1] - 2),
    ).fit(x, y)
    np.testing.assert_allclose(model.transform(x), native.booster_.predict(x), atol=1e-12)
    for col in (0, 1):
        grid = np.repeat(x[[0]], 31, axis=0)
        grid[:, 2] = 1
        grid[:, col] = np.linspace(0, 1, 31)
        assert np.all(np.diff(model.transform(grid)) >= -1e-12)


def test_missing_url_features_and_baselines():
    x = feature_matrix(["hello", "open https://example.com"], [0.2, 0.4],
                       [np.nan, 0.8], has_url=[False, True])
    np.testing.assert_equal(x[:, :3], [[0.2, 0, 0], [0.4, 0.8, 1]])
    for name, expected in [("mean", [0.2, 0.6]), ("max", [0.2, 0.8])]:
        model = FusionModel(name).fit(x, [0, 1], ids=["a", "b"], cal_ids=["a", "b"],
                                     train_ids=[], test_ids=[])
        np.testing.assert_allclose(model.transform(x), expected)
    with pytest.raises(ValueError):
        feature_matrix(["hello"], [0.2], [0.5], has_url=[False])
    with pytest.raises(ValueError):
        feature_matrix(["https://example.com"], [0.2], [np.nan], has_url=[True])
    with pytest.raises(ValueError, match="disagrees"):
        feature_matrix(["hello"], [0.2], [0.5], has_url=[True])
    flags = feature_matrix(["share your O T P\u200b"], [0.9], [np.nan], has_url=[False])
    assert flags[0, FEATURE_NAMES.index("spacing_fold")] == 1
    assert flags[0, FEATURE_NAMES.index("zero_width_count")] == 1
    assert flags[0, FEATURE_NAMES.index("credential_request")] == 1


def test_threshold_decisions_and_denominators():
    t = SelectiveThresholds(0.2, 0.8)
    p = [0.2, 0.3, 0.8, 0.9]
    assert t.transform(p).tolist() == ["SAFE", "UNCERTAIN", "SCAM", "SCAM"]
    assert roundtrip(t) == t
    stats = selective_metrics([0, 0, 0, 1], p, t)
    assert stats["fpr"] == 0.5  # NOT 1/3: uncertain benign is excluded.
    assert stats["coverage"] == 0.75
    assert np.isnan(selective_metrics([0], [0.5], t)["fpr"])
    with pytest.raises(ValueError):
        SelectiveThresholds(0.5, 0.5)


def test_threshold_constraints_and_infeasible():
    p = np.r_[np.linspace(0, 0.3, 80), np.linspace(0.7, 1, 80)]
    y = np.r_[np.zeros(80), np.ones(80)]
    found = search_thresholds(y, p)
    assert found["status"] == "ok"
    assert found["cal"]["coverage"] == 1
    assert found["cal"]["fpr"] <= 0.01
    assert found["cal"]["miss_rate"] <= 0.05
    assert search_thresholds([0, 1], [0.5, 0.5])["status"] == "infeasible"
    assert search_thresholds([0, 0], [0.1, 0.2])["status"] == "infeasible"


def test_threshold_search_matches_exhaustive_tied_boundary_oracle():
    rng = np.random.default_rng(31)
    for _ in range(20):
        p = rng.choice([0, 0.2, 0.4, 0.8, 1], 18)
        y = rng.integers(0, 2, 18)
        alpha, beta = 0.2, 0.25
        best = -1
        for safe in np.r_[LOW, np.unique(p)]:
            for scam in np.r_[np.unique(p), HIGH]:
                if safe >= scam:
                    continue
                stat = selective_metrics(y, p, SelectiveThresholds(safe, scam))
                if stat["fpr"] <= alpha and stat["miss_rate"] <= beta:
                    best = max(best, stat["decided"])
        found = search_thresholds(y, p, alpha=alpha, beta=beta)
        assert (found["cal"]["decided"] if found["status"] == "ok" else -1) == best


def test_crc_finite_sample_correction_and_abstention_penalty():
    p = np.r_[np.full(90, 0.1), np.linspace(0.5, 0.9, 10)]
    result = conformal_fpr(np.zeros(100), p, t_safe=0.2, alpha=0.05)
    assert result["status"] == "ok"
    t = SelectiveThresholds.from_dict(result["thresholds"])
    pred = t.transform(p)
    loss = np.sum(pred == "SCAM") + 0.05 * np.sum(pred == "UNCERTAIN")
    assert result["corrected_risk"] == pytest.approx((loss + 1) / 101)
    assert result["corrected_risk"] <= 0.05
    assert np.sum(pred == "SCAM") == 3  # Four fails once abstention is penalised.
    assert conformal_fpr([0] * 98, [0.1] * 98, t_safe=0.2)["status"] == "infeasible"
    assert conformal_fpr([0] * 99, [0.1] * 99, t_safe=0.2)["status"] == "ok"
    assert conformal_fpr([0] * 100, [0.8] * 100, t_safe=0.2)["status"] == "infeasible"
    assert conformal_fpr([1], [0.9], t_safe=0.2)["status"] == "infeasible"


def test_risk_coverage_monotone_coverage_not_fabricated_monotone_risk():
    result = risk_coverage([1, 0, 1], [0.01, 0.1, 0.6])
    assert np.all(np.diff(result["curve"]["coverage"]) > 0)
    assert result["curve"]["coverage"][-1] == 1
    np.testing.assert_allclose(result["curve"]["risk"], [1, 0.5, 1 / 3])
    assert result["aurc"] == pytest.approx(np.mean([1, 0.5, 1 / 3]))


def test_group_coverage_and_report(tmp_path):
    y = np.array([0, 1] * 20)
    p = np.where(y, 0.9, 0.1)
    clusters = np.repeat(np.arange(10), 4)
    groups = {key: ["a", "b"] * 20 for key in ("source", "lang", "label_fine")}
    result = evaluate_fusion(y, p, y, p, p, cluster_ids=clusters, groups=groups,
                             crc_y=y, crc_prob=p, n_boot=5)
    assert result["bootstrap"]["method"] == "campaign_cluster"
    assert result["bootstrap"]["n_clusters"] == 10
    assert set(result["selective"]) == {"0.005", "0.01", "0.02"}
    assert result["selective"]["0.01"]["groups"]["source"]["a"]["coverage"] == 1
    payload = {"runs": {"synthetic": {"mean": result}}}
    write_report(payload, tmp_path)
    saved = json.loads((tmp_path / "fusion_metrics.json").read_text())
    assert saved["runs"]["synthetic"]["mean"]["n_test"] == 40
    assert "Before 95% CI" in (tmp_path / "fusion_report.md").read_text()
    assert campaign_size_buckets(["a", "a", "b"]) == ["2-5", "2-5", "1"]
    assert group_coverage([0.1, 0.5], SelectiveThresholds(0.2, 0.8),
                          {"source": ["x", "x"]})["source"]["x"]["coverage"] == 0.5


@pytest.fixture
def job(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("fusion_eval_test", folder / "fusion_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_job_synthetic_end_to_end_partial_and_split_provenance(job, tmp_path, monkeypatch):
    # Each campaign has both labels, so label-blind campaign subdivisions retain both.
    n = 100
    ids = [f"id{i}" for i in range(n)]
    y = np.arange(n) % 2
    sms = pd.DataFrame({"id": ids, "text": ["hello"] * n, "label_binary": y,
                        "label_scam": y, "campaign_id": np.arange(n) // 2,
                        "source": "synthetic", "lang": "en", "label_fine": "test"})
    sms.to_parquet(tmp_path / "sms.parquet")
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "campaign.json").write_text(json.dumps(
        {"train": ids[:20], "cal": ids[20:84], "test": ids[84:]}))
    pd.DataFrame({"id": ids, "score": np.where(y, 0.85, 0.15)}).to_parquet(tmp_path / "m.parquet")
    pd.DataFrame({"id": [], "score": []}).to_parquet(tmp_path / "u.parquet")

    class Context:
        work = tmp_path / "output"
        argv = ["--data-root", str(tmp_path), "--message-scores", str(tmp_path / "m.parquet"),
                "--url-scores", str(tmp_path / "u.parquet"), "--models", "mean", "max",
                "--calibrators", "temperature", "--n-boot", "2"]

        def __init__(self):
            self.results = {}
            self.saves = 0

        def log(self, message):
            pass

        def save(self):
            self.saves += 1
            self.work.mkdir(exist_ok=True)
            (self.work / "results.json").write_text(json.dumps(self.results, allow_nan=False))

    original = job.evaluate_unit

    def fail_max(*args):
        if args[4] == "max":
            raise RuntimeError("synthetic failure")
        return original(*args)

    monkeypatch.setattr(job, "evaluate_unit", fail_max)
    ctx = Context()
    assert job.run(ctx) == "partial"
    assert ctx.saves >= 4
    result = ctx.results["fusion"]["runs"]["campaign"]["mean/temperature"]
    stage_ids = [set(values) for values in result["cal_stage_ids"].values()]
    assert set.union(*stage_ids) == set(ids[20:84])
    assert sum(map(len, stage_ids)) == len(set.union(*stage_ids))
    assert set(result["model"]["fit_ids"]) == stage_ids[0]
    assert not set.union(*stage_ids) & set(ids[84:])
    assert len(ctx.results["failures"]) == 1
    pd.DataFrame({"id": ["x", "x"], "score": [0.1, 0.2]}).to_parquet(tmp_path / "bad.parquet")
    with pytest.raises(ValueError, match="duplicate"):
        job.load_scores(str(tmp_path / "bad.parquet"))
