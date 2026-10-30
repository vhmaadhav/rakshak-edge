"""Tests for the evaluation protocol: metrics against sklearn, thresholds, bootstrap, robustness,
data-contract loading and reports. Tiny synthetic inputs only. Nothing imports torch or
transformers."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from rakshak.eval import data as D
from rakshak.eval import metrics as M
from rakshak.eval.protocol import evaluate_scores, group_metrics
from rakshak.eval.report import render_markdown, sanitize, write_report
from rakshak.eval.robustness import robustness_report
from rakshak.perturb import FAMILIES


def _toy(seed: int = 0, n: int = 400):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    s = np.clip(0.35 * y + rng.normal(0.4, 0.25, size=n), 0.0, 1.0)
    return y, s


# ---------------------------------------------------------------------------
# Metric correctness against sklearn
# ---------------------------------------------------------------------------


def test_fixed_threshold_metrics_match_sklearn():
    y, s = _toy(1)
    thr = 0.5
    pred = (s >= thr).astype(int)
    got = M.binary_metrics(y, s, thr)
    assert got["precision"] == pytest.approx(precision_score(y, pred))
    assert got["recall"] == pytest.approx(recall_score(y, pred))
    assert got["f1"] == pytest.approx(f1_score(y, pred))
    assert got["mcc"] == pytest.approx(matthews_corrcoef(y, pred))
    assert got["accuracy"] == pytest.approx(accuracy_score(y, pred))
    assert got["fpr"] == pytest.approx(((pred == 1) & (y == 0)).sum() / (y == 0).sum())


def test_auc_metrics_match_sklearn():
    y, s = _toy(2)
    assert M.roc_auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert M.pr_auc(y, s) == pytest.approx(average_precision_score(y, s))


def test_max_f1_threshold_reaches_the_best_f1_over_all_thresholds():
    y, s = _toy(3)
    thr, best = M.threshold_max_f1(y, s)
    brute = max(f1_score(y, (s >= t).astype(int)) for t in np.unique(s))
    assert best == pytest.approx(brute)
    assert f1_score(y, (s >= thr).astype(int)) == pytest.approx(best)


def test_ece_and_brier_on_a_hand_checked_case():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.7, 0.2, 0.0])
    # bins (15 equal width): 0.9 -> bin 13, 0.7 -> bin 10, 0.2 -> bin 3, 0.0 -> bin 0
    # each bin holds one point, so ECE = mean |y - p| = (0.1 + 0.3 + 0.2 + 0.0) / 4
    assert M.expected_calibration_error(y, p) == pytest.approx((0.1 + 0.3 + 0.2 + 0.0) / 4)
    assert M.brier_score(y, p) == pytest.approx(np.mean((p - y) ** 2))


def test_ece_rejects_out_of_range_probabilities():
    with pytest.raises(ValueError):
        M.expected_calibration_error([0, 1], [0.2, 1.5])


def test_labels_must_be_binary():
    with pytest.raises(ValueError):
        M.binary_metrics([0, 2], [0.1, 0.9], 0.5)


# ---------------------------------------------------------------------------
# Recall at FPR and threshold selection edge cases
# ---------------------------------------------------------------------------


def test_recall_at_fpr_without_negatives_is_nan():
    out = M.recall_at_fpr([1, 1, 1], [0.1, 0.5, 0.9], 0.01)
    assert math.isnan(out["recall"])
    assert out["resolvable"] is False


def test_recall_at_fpr_with_perfect_separation_is_one_at_zero_fpr():
    y = np.array([0] * 20 + [1] * 20)
    s = np.array([0.1] * 20 + [0.9] * 20)
    out = M.recall_at_fpr(y, s, 0.001)
    assert out["recall"] == pytest.approx(1.0)
    assert out["fpr"] == pytest.approx(0.0)


def test_recall_at_fpr_respects_the_target():
    y, s = _toy(4, n=2000)
    for target in (0.01, 0.05):
        out = M.threshold_for_fpr(y, s, target)
        assert out["fpr"] <= target + 1e-12


def test_recall_at_tiny_fpr_flags_unresolvable_figures():
    y, s = _toy(5, n=300)
    out = M.recall_at_fpr(y, s, 0.0001)
    assert out["n_neg_required"] == math.ceil(10 / 0.0001)
    assert out["resolvable"] is False


def test_recall_at_fpr_handles_tied_scores():
    y = np.array([0, 1, 0, 1, 1, 0])
    s = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    out = M.threshold_for_fpr(y, s, 0.01)
    assert out["fpr"] == pytest.approx(0.0) or out["fpr"] == pytest.approx(1.0)
    assert out["threshold"] == pytest.approx(np.inf) or out["threshold"] == pytest.approx(0.5)


def test_threshold_for_fpr_with_a_single_class_is_nan():
    out = M.threshold_for_fpr([0, 0, 0], [0.1, 0.2, 0.3], 0.01)
    assert math.isnan(out["threshold"])


# ---------------------------------------------------------------------------
# Coverage and bootstrap
# ---------------------------------------------------------------------------


def test_coverage_error_is_zero_for_perfect_predictions_and_full_at_the_end():
    y, s = _toy(6, n=200)
    pred = y == 1
    out = M.coverage_error_curve(y, pred, np.abs(s - 0.5))
    assert out["aurc"] == pytest.approx(0.0)
    assert out["curve"]["coverage"][-1] == pytest.approx(1.0)
    assert len(out["curve"]["coverage"]) <= 101


def test_bootstrap_is_deterministic_for_a_seed_and_changes_with_seed():
    y, s = _toy(7, n=500)
    a = M.bootstrap_intervals(y, s, 0.5, n_boot=60, seed=11)
    b = M.bootstrap_intervals(y, s, 0.5, n_boot=60, seed=11)
    c = M.bootstrap_intervals(y, s, 0.5, n_boot=60, seed=12)
    assert a == b
    assert a["f1"] != c["f1"]
    lo, hi = a["f1"]
    assert lo <= hi


def test_bootstrap_interval_covers_the_point_estimate():
    y, s = _toy(8, n=1200)
    thr = 0.5
    point = M.binary_metrics(y, s, thr)["f1"]
    lo, hi = M.bootstrap_intervals(y, s, thr, n_boot=200, seed=0)["f1"]
    assert lo <= point <= hi


def test_bootstrap_keeps_class_counts_and_needs_both_classes():
    assert M.bootstrap_intervals([1, 1], [0.2, 0.9], 0.5, n_boot=5)["n_boot"] == 0


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_protocol_thresholds_come_from_cal_only():
    y_cal, s_cal = _toy(9, n=300)
    y_test, s_test = _toy(10, n=300)
    res = evaluate_scores(y_cal, s_cal, y_test, s_test, bootstrap=False)
    thr = res["thresholds"]["max_f1"]["threshold"]
    assert res["operating_points"]["max_f1"]["threshold"] == pytest.approx(thr)
    assert M.threshold_max_f1(y_cal, s_cal)[0] == pytest.approx(thr)
    assert res["sizes"]["test"] == 300


def test_protocol_rejects_missing_calibration_positives():
    with pytest.raises(ValueError):
        evaluate_scores([0, 0], [0.1, 0.2], [0, 1], [0.1, 0.9], bootstrap=False)


def test_group_metrics_partition_the_test_set():
    y, s = _toy(11, n=120)
    groups = {"source": ["a"] * 60 + ["b"] * 60}
    rows = group_metrics(y, s, 0.5, groups)["source"]
    assert sum(r["n"] for r in rows) == 120
    assert {r["value"] for r in rows} == {"a", "b"}


def test_coverage_entry_present_in_protocol():
    y_cal, s_cal = _toy(12, n=200)
    y_test, s_test = _toy(13, n=200)
    res = evaluate_scores(y_cal, s_cal, y_test, s_test, bootstrap=False)
    assert "aurc" in res["coverage"]
    assert "recall_at_cal_fpr" in res["threshold_free"]


# ---------------------------------------------------------------------------
# Robustness harness with a dummy keyword model
# ---------------------------------------------------------------------------


def _keyword_model(texts: list[str]) -> np.ndarray:
    return np.array([0.9 if "otp" in t.lower() else 0.05 for t in texts])


def test_robustness_covers_every_family_and_reports_clean_recall():
    pos = ["Your OTP is 1234. Do not share it with anyone."]
    neg = ["Lunch is at 1 pm tomorrow."]
    rep = robustness_report(_keyword_model, pos, neg, 0.5)
    assert set(rep["families"]) == set(FAMILIES)
    assert rep["clean"]["recall"] == pytest.approx(1.0)
    assert rep["clean"]["fpr"] == pytest.approx(0.0)
    assert "fpr_under_attack" in rep["families"]["benign_sentence_insert"]
    assert "fpr_under_attack" not in rep["families"]["homoglyph_swap"]


def test_robustness_detects_evasion_and_bounds_asr():
    pos = ["Your OTP is 1234. Do not share it with anyone."] * 3
    neg = ["Lunch is at 1 pm tomorrow."]
    rep = robustness_report(_keyword_model, pos, neg, 0.5)
    homoglyph = rep["families"]["homoglyph_swap"]
    assert homoglyph["recall_under_attack"] < 1.0
    assert homoglyph["flip_rate"] == pytest.approx(1.0 - homoglyph["recall_under_attack"])
    assert 0.0 <= homoglyph["attack_success_rate"] <= 1.0
    assert rep["summary"]["worst_family"] in FAMILIES


def test_robustness_marks_families_that_change_nothing():
    pos = ["Your OTP is 1234."]
    rep = robustness_report(_keyword_model, pos, [], 0.5, families=["urgency_paraphrase"])
    fam = rep["families"]["urgency_paraphrase"]
    assert fam["changed_rate"] == pytest.approx(0.0)
    assert fam["recall_under_attack"] == pytest.approx(1.0)


def test_robustness_passes_raw_text_to_predict():
    # The harness hands predict raw text, so a predict that canonicalises internally works.
    def canon_predict(texts):
        return _keyword_model([t.replace(" ", "") for t in texts])

    rep = robustness_report(
        canon_predict, ["Your OTP is 1234."], [], 0.5, families=["char_spacing"]
    )
    assert rep["clean"]["recall"] == pytest.approx(1.0)


def test_robustness_rejects_unknown_family_and_subsamples_positives():
    with pytest.raises(KeyError):
        robustness_report(_keyword_model, ["otp"], [], 0.5, families=["nope"])
    pos = [f"otp {i}" for i in range(30)]
    rep = robustness_report(_keyword_model, pos, [], 0.5, families=["leetspeak"],
                            max_positives=10)
    assert rep["n_pos"] == 10


# ---------------------------------------------------------------------------
# Data contract loading (tiny parquet fixtures)
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_root(tmp_path):
    rows = 40
    df = pd.DataFrame(
        {
            "id": [f"id{i:03d}" for i in range(rows)],
            "text": [f"message {i}" for i in range(rows)],
            "label_binary": [i % 2 for i in range(rows)],
            "label_fine": ["scam" if i % 2 else "ham" for i in range(rows)],
            "source": ["s1" if i < 20 else "s2" for i in range(rows)],
            "lang": ["en"] * rows,
            "eval_ok": [i % 10 != 9 for i in range(rows)],
            "synthetic": [i % 13 == 0 for i in range(rows)],
        }
    )
    df.to_parquet(tmp_path / "sms.parquet")
    urls = pd.DataFrame(
        {
            "id": [f"u{i}" for i in range(12)],
            "url": [f"http://x{i}.example" for i in range(12)],
            "label": [i % 2 for i in range(12)],
            "source": ["a"] * 12,
            "reg_domain": [f"x{i}.example" for i in range(12)],
            "url_len": np.arange(12, dtype=float),
            "has_at_sign": np.zeros(12),
            "brand": ["sbi"] * 12,
        }
    )
    urls.to_parquet(tmp_path / "urls.parquet")
    (tmp_path / "splits").mkdir()
    ids = [f"id{i:03d}" for i in range(rows)]
    flat = {"train": ids[:24], "cal": ids[24:30], "test": ids[30:]}
    (tmp_path / "splits" / "flat.json").write_text(json.dumps(flat))
    partitioned = {
        "family": "demo",
        "seed": 5,
        "description": "demo",
        "partitions": {
            "withcal": {"train": ids[:24], "cal": ids[24:30], "test": ids[30:]},
            "nocal": {"train": ids[:30], "test": ids[30:]},
        },
    }
    (tmp_path / "splits" / "parted.json").write_text(json.dumps(partitioned))
    return tmp_path


def test_resolve_data_root_explicit_and_missing(tiny_root, tmp_path):
    assert D.resolve_data_root(tiny_root) == tiny_root
    with pytest.raises(FileNotFoundError):
        D.resolve_data_root(tmp_path / "missing")


def _mount(base, *parts, marker=D.SMS_FILE):
    folder = base.joinpath(*parts)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / marker).write_bytes(b"")
    return folder


@pytest.fixture
def kaggle_input(tmp_path, monkeypatch):
    """A fake /kaggle/input with no known candidate present; the real mount is never read."""
    base = tmp_path / "kaggle_input"
    base.mkdir()
    monkeypatch.delenv(D.ENV_VAR, raising=False)
    monkeypatch.setattr(D, "KAGGLE_INPUT", base)
    monkeypatch.setattr(D, "KAGGLE_DATA", tmp_path / "no_kaggle_data")
    monkeypatch.setattr(D, "REPO_DATA", tmp_path / "no_repo_data")
    return base


def test_resolve_finds_nested_kaggle_mount(kaggle_input):
    folder = _mount(kaggle_input, "datasets", "maadhavvh", "rakshak-data")
    assert D.resolve_data_root() == folder


def test_resolve_prefers_rakshak_data_among_several(kaggle_input):
    _mount(kaggle_input, "aaa-other")  # shallower and sorts first, but is not the dataset
    preferred = _mount(kaggle_input, "zzz", "rakshak-data")
    assert D.resolve_data_root() == preferred


def test_resolve_search_depth_limit(kaggle_input):
    _mount(kaggle_input, "a", "b", "c", "d", "e", "f")  # marker folder 6 levels down
    with pytest.raises(FileNotFoundError):
        D.resolve_data_root()
    shallow = _mount(kaggle_input, "g", "h", "i", "j", "k")  # 5 levels down
    assert D.resolve_data_root() == shallow


def test_resolve_search_uses_marker_argument(kaggle_input):
    folder = _mount(kaggle_input, "x", "y", marker=D.URL_FILE)
    assert D.resolve_data_root(marker=D.URL_FILE) == folder
    with pytest.raises(FileNotFoundError):
        D.resolve_data_root()  # sms.parquet is absent


def test_explicit_path_does_not_fall_back_to_search(kaggle_input, tmp_path):
    _mount(kaggle_input, "x")
    with pytest.raises(FileNotFoundError):
        D.resolve_data_root(tmp_path / "missing")


def test_loaders_and_eval_mask(tiny_root):
    sms = D.load_sms(tiny_root)
    assert sms["id"].is_unique and sms["id"].map(type).eq(str).all()
    mask = D.eval_mask(sms)
    # eval_ok False rows and synthetic rows are excluded from evaluation
    assert not mask[~sms["eval_ok"].to_numpy(dtype=bool)].any()
    assert not mask[sms["synthetic"].to_numpy()].any()
    urls = D.load_urls(tiny_root)
    assert D.url_feature_columns(urls) == ["url_len", "has_at_sign"]


def test_flat_and_partitioned_split_files_become_runs(tiny_root):
    sms = D.load_sms(tiny_root)
    labels = sms["label_binary"].to_numpy()
    mask = D.eval_mask(sms)
    runs = D.load_runs(tiny_root, sms, patterns=None, labels=labels, mask=mask)
    names = [r.name for r in runs]
    assert sorted(names) == ["flat", "parted:nocal", "parted:withcal"]
    nocal = next(r for r in runs if r.name == "parted:nocal")
    assert nocal.notes["cal_carved_from_train"] == len(nocal.cal)
    assert len(nocal.cal) + len(nocal.train) == 30
    for run in runs:
        assert np.intersect1d(run.train, run.test).size == 0
        assert np.intersect1d(run.cal, run.test).size == 0
        assert mask[run.test].all() and mask[run.cal].all()


def test_load_runs_filters_by_pattern(tiny_root):
    sms = D.load_sms(tiny_root)
    runs = D.load_runs(tiny_root, sms, patterns=["parted"], labels=sms["label_binary"].to_numpy(),
                       mask=None)
    assert {r.name.split(":")[0] for r in runs} == {"parted"}


def test_group_values_skip_absent_columns(tiny_root):
    sms = D.load_sms(tiny_root)
    groups = D.group_values(sms, np.arange(5), columns=("source", "missing"))
    assert list(groups) == ["source"]
    assert len(groups["source"]) == 5


def test_canon_text_is_a_string_and_cached():
    assert isinstance(D.canon_text("Your OTP is 1234"), str)
    assert D.canon_of("Your OTP is 1234") is D.canon_of("Your OTP is 1234")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _fake_result(y_cal, s_cal, y_test, s_test):
    res = evaluate_scores(y_cal, s_cal, y_test, s_test, bootstrap=False)
    res["robustness"] = {"summary": {"mean_recall_under_attack": 0.5,
                                     "mean_attack_success_rate": 0.25}}
    return res


def test_write_report_writes_strict_json_and_a_table_per_split(tmp_path):
    y_cal, s_cal = _toy(20, n=200)
    y_test, s_test = _toy(21, n=200)
    payload = {
        "title": "Demo",
        "meta": {"seed": 0},
        "splits": {
            "campaign": {"char_lr/canon": _fake_result(y_cal, s_cal, y_test, s_test)},
            "temporal": {"char_lr/raw": _fake_result(y_cal, s_cal, y_test, s_test)},
        },
    }
    paths = write_report(payload, tmp_path)
    loaded = json.loads(paths["metrics"].read_text())
    assert loaded["splits"]["campaign"]["char_lr/canon"]["sizes"]["test"] == 200
    text = paths["report"].read_text()
    assert "## campaign" in text and "## temporal" in text
    assert "| char_lr/canon |" in text
    assert "0.250" in text  # ASR column


def test_sanitize_turns_nan_and_inf_into_null():
    cleaned = sanitize({"a": float("nan"), "b": np.inf, "c": np.float64(0.5), "d": (1, 2)})
    assert cleaned == {"a": None, "b": None, "c": 0.5, "d": [1, 2]}
    json.dumps(cleaned, allow_nan=False)


def test_render_markdown_marks_missing_robustness():
    y, s = _toy(22, n=150)
    res = evaluate_scores(y, s, y, s, bootstrap=False)
    text = render_markdown({"splits": {"x": {"m": res}}})
    assert "| m |" in text
    assert "n/a" in text


# ---------------------------------------------------------------------------
# Campaign-clustered bootstrap, paired tests and the recompute CLI
# ---------------------------------------------------------------------------


def _clustered_toy(seed: int = 0, n_clusters: int = 60, size: int = 20):
    """Clusters whose rows share one score, so campaign dependence is strong."""
    rng = np.random.default_rng(seed)
    cluster_label = rng.integers(0, 2, size=n_clusters)
    cluster_score = np.clip(0.3 * cluster_label + rng.normal(0.4, 0.3, n_clusters), 0, 1)
    y = np.repeat(cluster_label, size)
    s = np.repeat(cluster_score, size)
    ids = [f"c{c}" for c in np.repeat(np.arange(n_clusters), size)]
    return y, s, ids


def test_bootstrap_defaults_to_campaign_clusters_when_ids_are_given():
    y, s, ids = _clustered_toy(1)
    out = M.bootstrap_intervals(y, s, 0.5, n_boot=40, seed=3, cluster_ids=ids)
    assert out["method"] == "campaign_cluster"
    assert out["n_clusters"] == 60
    iid = M.bootstrap_intervals(y, s, 0.5, n_boot=40, seed=3)
    assert iid["method"] == "iid_stratified"
    assert "n_clusters" not in iid


def test_bootstrap_iid_can_be_forced_and_cluster_needs_ids():
    y, s, ids = _clustered_toy(2)
    forced = M.bootstrap_intervals(y, s, 0.5, n_boot=20, seed=3, cluster_ids=ids, method="iid")
    assert forced["method"] == "iid_stratified"
    with pytest.raises(ValueError):
        M.bootstrap_intervals(y, s, 0.5, n_boot=5, method="campaign_cluster")
    with pytest.raises(ValueError):
        M.bootstrap_intervals(y, s, 0.5, n_boot=5, method="nonsense")


def test_cluster_bootstrap_is_wider_when_rows_within_a_campaign_are_correlated():
    y, s, ids = _clustered_toy(4)
    clustered = M.bootstrap_intervals(y, s, 0.5, n_boot=300, seed=0, cluster_ids=ids)
    iid = M.bootstrap_intervals(y, s, 0.5, n_boot=300, seed=0)
    width = {k: v[1] - v[0] for k, v in clustered.items() if isinstance(v, list)}
    width_iid = {k: v[1] - v[0] for k, v in iid.items() if isinstance(v, list)}
    assert width["pr_auc"] > width_iid["pr_auc"]


def test_cluster_draw_takes_whole_campaigns():
    y = np.array([1, 1, 0, 0, 0, 1, 0])
    ids = ["a", "a", "b", "b", "b", "c", "d"]
    index = M._cluster_index(y, ids)
    rng = np.random.default_rng(0)
    for _ in range(50):
        idx = M._cluster_draw(index, rng)
        counts = np.bincount(idx, minlength=len(y))
        for group in ([0, 1], [2, 3, 4], [5], [6]):
            # every row of a campaign is drawn the same number of times, possibly zero
            assert len(set(counts[group])) == 1


def test_cluster_draw_keeps_positive_and_benign_campaign_strata():
    y = np.array([1, 0, 0, 0, 0, 0])
    ids = ["pos", "ben1", "ben1", "ben2", "ben2", "ben3"]
    index = M._cluster_index(y, ids)
    assert len(index["pos_clusters"]) == 1
    assert len(index["neg_clusters"]) == 3
    idx = M._cluster_draw(index, np.random.default_rng(1))
    assert (y[idx] == 1).sum() >= 1  # the positive stratum always contributes


def test_paired_test_identical_models_has_no_difference():
    y, s, ids = _clustered_toy(5)
    out = M.paired_bootstrap_test(y, s, s, metric="pr_auc", cluster_ids=ids, n_boot=60)
    assert out["delta"] == pytest.approx(0.0)
    assert out["p_value"] == pytest.approx(1.0)


def test_paired_test_detects_a_clearly_better_model():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, size=800)
    good = np.clip(0.5 * y + rng.normal(0.3, 0.2, 800), 0, 1)
    bad = np.clip(0.05 * y + rng.normal(0.3, 0.2, 800), 0, 1)
    ids = [f"g{i // 4}" for i in range(800)]
    out = M.paired_bootstrap_test(y, good, bad, metric="pr_auc", cluster_ids=ids, n_boot=200)
    assert out["delta"] > 0
    assert out["delta_ci"][0] > 0
    assert out["p_value"] < 0.05
    assert out["method"] == "campaign_cluster"


def test_paired_fixed_threshold_metric_needs_thresholds():
    y, s, _ = _clustered_toy(7)
    with pytest.raises(ValueError):
        M.paired_bootstrap_test(y, s, s, metric="f1", n_boot=5)
    out = M.paired_bootstrap_test(y, s, s, metric="f1", threshold_a=0.5, threshold_b=0.5,
                                  n_boot=20)
    assert out["delta"] == pytest.approx(0.0)


def test_protocol_records_the_bootstrap_method_used():
    y_cal, s_cal = _toy(30, n=200)
    y_test, s_test = _toy(31, n=200)
    ids = [f"k{i // 5}" for i in range(200)]
    clustered = evaluate_scores(y_cal, s_cal, y_test, s_test, n_boot=20, cluster_ids=ids)
    assert clustered["bootstrap"]["method"] == "campaign_cluster"
    forced = evaluate_scores(y_cal, s_cal, y_test, s_test, n_boot=20, cluster_ids=ids,
                             bootstrap_method="iid")
    assert forced["bootstrap"]["method"] == "iid_stratified"
    plain = evaluate_scores(y_cal, s_cal, y_test, s_test, n_boot=20)
    assert plain["bootstrap"]["method"] == "iid_stratified"


def test_report_shows_bootstrap_method_and_paired_comparisons(tmp_path):
    y_cal, s_cal = _toy(32, n=200)
    y_test, s_test = _toy(33, n=200)
    res = evaluate_scores(y_cal, s_cal, y_test, s_test, n_boot=10,
                          cluster_ids=[f"k{i // 4}" for i in range(200)])
    cmp_ = M.paired_bootstrap_test(y_test, s_test, s_test, metric="pr_auc", n_boot=10)
    text = render_markdown({"splits": {"x": {"m": res}},
                            "comparisons": {"m vs m": {"pr_auc": cmp_}}})
    assert "campaign_cluster bootstrap" in text
    assert "## Paired comparisons" in text
    assert "| m vs m | pr_auc |" in text


@pytest.fixture()
def tiny_campaign_root(tiny_root):
    sms = pd.read_parquet(tiny_root / "sms.parquet")
    sms["campaign_id"] = [f"camp{i // 4}" for i in range(len(sms))]
    sms.to_parquet(tiny_root / "sms.parquet")
    return tiny_root


def _write_predictions(path, root, cal_ids, test_ids, seed: int, sep: float):
    sms = D.load_sms(root).set_index("id")
    rng = np.random.default_rng(seed)
    rows = []
    for split, ids in (("cal", cal_ids), ("test", test_ids)):
        y = sms.loc[ids, "label_binary"].to_numpy()
        s = 1 / (1 + np.exp(-(sep * y + rng.normal(0, 1, len(y)))))
        rows.append(pd.DataFrame({"id": ids, "split": split, "prob": s,
                                  "label_binary": y}))
    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)


def test_recompute_cli_writes_metrics_ci_method_and_comparison(tiny_campaign_root, tmp_path):
    from rakshak.eval.recompute import main

    ids = [f"id{i:03d}" for i in range(40)]
    cal_ids, test_ids = ids[24:30], ids[30:]
    _write_predictions(tmp_path / "good.parquet", tiny_campaign_root, cal_ids, test_ids, 1, 2.0)
    _write_predictions(tmp_path / "weak.parquet", tiny_campaign_root, cal_ids, test_ids, 2, 0.2)
    out = tmp_path / "out"
    code = main([
        "--scores", str(tmp_path / "good.parquet"), str(tmp_path / "weak.parquet"),
        "--out", str(out), "--data-root", str(tiny_campaign_root), "--n-boot", "30",
        "--compare", "good:weak",
    ])
    assert code == 0
    payload = json.loads((out / "metrics.json").read_text())
    model = payload["splits"]["recomputed"]["good"]
    assert model["bootstrap"]["method"] == "campaign_cluster"
    assert payload["meta"]["bootstrap"] == "cluster"
    assert "good vs weak" in payload["comparisons"]
    assert set(payload["comparisons"]["good vs weak"]) == set(M.PAIRED_METRICS)
    assert (out / "report.md").read_text().count("| good |") >= 1


def test_recompute_cli_iid_option_and_label_check(tiny_campaign_root, tmp_path):
    from rakshak.eval.recompute import main

    ids = [f"id{i:03d}" for i in range(40)]
    path = tmp_path / "model.parquet"
    _write_predictions(path, tiny_campaign_root, ids[24:30], ids[30:], 3, 1.0)
    out = tmp_path / "iid"
    main(["--scores", str(path), "--out", str(out), "--data-root", str(tiny_campaign_root),
          "--n-boot", "10", "--bootstrap", "iid"])
    payload = json.loads((out / "metrics.json").read_text())
    assert payload["splits"]["recomputed"]["model"]["bootstrap"]["method"] == "iid_stratified"


def test_recompute_rejects_ids_missing_from_the_table(tiny_campaign_root, tmp_path):
    from rakshak.eval.recompute import main

    bad = pd.DataFrame({"id": ["nope"], "split": ["test"], "prob": [0.5], "label_binary": [0]})
    bad.to_parquet(tmp_path / "bad.parquet", index=False)
    with pytest.raises(ValueError):
        main(["--scores", str(tmp_path / "bad.parquet"), "--out", str(tmp_path / "o"),
              "--data-root", str(tiny_campaign_root), "--n-boot", "5"])
