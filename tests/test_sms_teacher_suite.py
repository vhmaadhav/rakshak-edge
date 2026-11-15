"""Tests for kaggle/jobs/sms_teacher_suite.py (plan docs/TEACHER_SUITE_PLAN.md).
Synthetic inputs only; no torch, no model, no data."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import sms_retrain as base  # noqa: E402
import sms_teacher_suite as job  # noqa: E402

# ---- threshold rules, calibration split only ----


def test_student_threshold_is_the_shipped_one():
    assert job.STUDENT_TAU == 0.35127517580986023


def test_threshold_pair_on_separable_scores():
    s = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    y = np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)])
    out = job.threshold_pair(s, y)
    assert 0.4 < out["maxf1"] <= 0.6
    assert out["maxf1_cal_f1"] == pytest.approx(1.0)
    assert out["cal1pct"] > 0.4
    assert out["cal1pct_cal_fpr"] <= job.FPR_CAL


def test_cal1pct_respects_the_fpr_target_under_overlap():
    rng = np.random.default_rng(1)
    neg = rng.normal(0.0, 1.0, 2000)
    pos = rng.normal(2.0, 1.0, 2000)
    s = np.concatenate([neg, pos])
    y = np.concatenate([np.zeros(2000, dtype=int), np.ones(2000, dtype=int)])
    out = job.threshold_pair(s, y)
    assert out["cal1pct_cal_fpr"] <= 0.01 + 1e-12
    assert np.isfinite(out["maxf1"]) and np.isfinite(out["cal1pct"])


def test_threshold_pair_fails_without_positives():
    s = np.linspace(0, 1, 20)
    y = np.zeros(20, dtype=int)
    with pytest.raises(ValueError, match="not finite"):
        job.threshold_pair(s, y)


# ---- arms ----


def test_arm_truth_table():
    p_s = np.array([0.9, 0.9, 0.1, 0.1])
    p_t = np.array([0.9, 0.1, 0.9, 0.1])
    f = job.arm_flags(p_s, p_t, tau_t=0.5)
    assert f["T"].tolist() == [True, False, True, False]
    assert f["SandT"].tolist() == [True, False, False, False]
    assert f["SorT"].tolist() == [True, True, True, False]
    assert f["S"].tolist() == [True, True, False, False]


def test_and_is_subset_and_or_is_superset_of_each_arm():
    rng = np.random.default_rng(3)
    p_s, p_t = rng.random(500), rng.random(500)
    f = job.arm_flags(p_s, p_t, tau_t=0.7)
    assert np.all(f["SandT"] <= f["T"]) and np.all(f["SandT"] <= f["S"])
    assert np.all(f["SorT"] >= f["T"]) and np.all(f["SorT"] >= f["S"])


def test_flag_is_inclusive_at_threshold():
    f = job.arm_flags(np.array([job.STUDENT_TAU]), np.array([0.5]), tau_t=0.5)
    assert f["T"][0] and f["SandT"][0] and f["S"][0]


# ---- gates ----


def _gates(fp=0, tp_s=226, tp_p=91, f1=0.99):
    return job.evaluate_gates(fp, 1133, tp_s, 226, tp_p, 91, f1)


@pytest.mark.parametrize(
    ("kwargs", "gate", "expected"),
    [
        ({"fp": 11}, "G1_groundlm_neg_fp", True),
        ({"fp": 12}, "G1_groundlm_neg_fp", False),
        ({"tp_s": 157}, "G2_smishx_smishing_tp", True),
        ({"tp_s": 156}, "G2_smishx_smishing_tp", False),
        ({"tp_p": 87}, "G3_groundlm_pos_recall", True),  # 87/91 = 0.956
        ({"tp_p": 86}, "G3_groundlm_pos_recall", False),  # 86/91 = 0.945
        ({"f1": 0.9713}, "G4_campaign_clean_f1", True),
        ({"f1": 0.97129}, "G4_campaign_clean_f1", False),
        ({"f1": float("nan")}, "G4_campaign_clean_f1", False),
    ],
)
def test_gate_boundaries(kwargs, gate, expected):
    assert _gates(**kwargs)[gate]["pass"] is expected


def test_arm_passes_only_when_all_four_hold():
    assert _gates()["arm_pass"] is True
    assert _gates(fp=12)["arm_pass"] is False
    assert _gates(tp_s=100)["arm_pass"] is False
    assert _gates(tp_p=10)["arm_pass"] is False
    assert _gates(f1=0.97)["arm_pass"] is False


# ---- intervals ----


def test_cluster_ci_perfect_predictor_is_one():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    codes = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    lo, hi = job.cluster_ci(y, y.astype(bool), codes, "f1", n_boot=200, seed=0)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_cluster_ci_is_deterministic_and_ordered():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 300)
    pred = rng.random(300) < 0.6
    codes = rng.integers(0, 40, 300)
    a = job.cluster_ci(y, pred, codes, "f1", n_boot=300, seed=0)
    b = job.cluster_ci(y, pred, codes, "f1", n_boot=300, seed=0)
    assert a == b
    assert a[0] <= a[1]


def test_cluster_ci_precision_with_no_predictions_is_nan():
    y = np.array([1, 0, 1, 0])
    lo, hi = job.cluster_ci(y, np.zeros(4, dtype=bool), np.arange(4), "precision", n_boot=50)
    assert np.isnan(lo) and np.isnan(hi)


def test_cluster_ci_rejects_unknown_metric():
    with pytest.raises(ValueError):
        job.cluster_ci(np.array([1]), np.array([True]), np.array([0]), "auc", n_boot=2)


# ---- contamination ----


def test_clean_mask_excludes_id_and_text_hits():
    ids = np.array(["a", "b", "c"])
    text_hit = np.array([False, False, True])
    clean = job.clean_mask(ids, text_hit, train_ids={"b"})
    assert clean.tolist() == [True, False, False]


# ---- ordering and the paired table ----


def test_threshold_is_written_before_any_test_set_is_scored():
    calls: list[str] = []
    thr_written: list[float] = []

    def score(model, name):
        calls.append(f"score:{name}")
        if name == base.SET_LEGIT_VAL:
            return np.zeros(0)
        if name == base.SET_CAL:
            return np.array([0.1, 0.9])
        return np.array([0.5, 0.6])

    def rule(legit, cal, y):
        return 0.5, False

    def on_threshold(tau, fallback):
        thr_written.append(tau)
        calls.append("threshold")

    out = base.run_seed_pipeline(
        fit=lambda: object(), score=score, rule=rule, y_cal=np.array([0, 1]),
        post_sets=["campaign_test", "groundlm_neg"], on_threshold=on_threshold,
    )
    assert out["events"].index("threshold") < out["events"].index("score:campaign_test")
    assert thr_written == [0.5]


def test_paired_table_counts_and_exact_mcnemar():
    ours = np.array([True, True, True, False, False, False, True])
    ref = np.array([True, False, False, True, False, False, True])
    t = base.paired_table(ours, ref)
    assert (t["both"], t["ours_only"], t["ref_only"], t["neither"]) == (2, 2, 1, 2)
    assert t["mcnemar_exact_p"] == pytest.approx(1.0)  # 2 vs 1 discordant: p = 1.0 two-sided


def test_paired_table_no_discordant_pairs_gives_p_one():
    v = np.array([True, False])
    assert base.paired_table(v, v)["mcnemar_exact_p"] == 1.0


# ---- kernel bundle ----


def test_build_dependencies_list_every_local_import():
    spec = importlib.util.spec_from_file_location(
        "build_kernel_under_test", JOBS.parent / "build_kernel.py"
    )
    build_kernel = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = build_kernel
    spec.loader.exec_module(build_kernel)
    deps = set(build_kernel.JOB_DEPENDENCIES["sms_teacher_suite"])
    tree = ast.parse((JOBS / "sms_teacher_suite.py").read_text())
    local = {"baseline_v0", "sms_retrain", "sms_inbox_retrain", "sms_surrogate_d0",
             "sms_teacher", "sms_teacher_xsource"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for name in imported & local:
        assert f"kaggle/jobs/{name}.py" in deps
    assert "kaggle/jobs/sms_teacher_suite.py" not in deps


# ---- two-band output (amendment 1, 2026-11-12) ----


def test_low95_is_largest_threshold_with_cal_recall_at_least_95_percent():
    s = np.concatenate([np.zeros(40), np.linspace(0.3, 1.0, 100)])
    y = np.concatenate([np.zeros(40, dtype=int), np.ones(100, dtype=int)])
    out = job.threshold_pair(s, y)
    assert out["low95_cal_recall"] >= 0.95
    # k = ceil(0.95 * 100) = 95; the 95th largest positive score is the threshold
    pos = np.sort(s[y == 1])[::-1]
    assert out["low95"] == pytest.approx(pos[94])


def test_band_flags_three_bands():
    p = np.array([0.95, 0.7, 0.4, 0.1])
    b = job.band_flags(p, tau_high=0.8, tau_low=0.3)
    assert b["scam"].tolist() == [True, False, False, False]
    assert b["verify"].tolist() == [False, True, True, False]
    assert b["benign"].tolist() == [False, False, False, True]


def test_band_flags_inverted_thresholds_leave_verify_empty():
    p = np.array([0.95, 0.5, 0.1])
    b = job.band_flags(p, tau_high=0.4, tau_low=0.6)
    assert not b["verify"].any()
    assert b["scam"].tolist() == [True, True, False]


def test_sandt_band_is_gated_by_student():
    p_s = np.array([0.9, 0.1, 0.9])
    p_t = np.array([0.95, 0.7, 0.4])
    b = job.band_for_arm(p_s, p_t, tau_high=0.8, tau_low=0.3, arm="SandT")
    assert b["scam"].tolist() == [True, False, False]
    assert b["verify"].tolist() == [False, False, True]
    assert b["benign"].tolist() == [False, True, False]
    assert np.all(b["scam"] | b["verify"] | b["benign"])
