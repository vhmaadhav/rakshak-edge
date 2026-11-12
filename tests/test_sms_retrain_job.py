"""Pure-function tests for kaggle/jobs/sms_retrain.py. No Kaggle, no torch, no training.

The job imports torch and rakshak.data.retrain_prep only inside run(), so this file imports without
either. The threshold-before-test ordering is checked with a fake pipeline, and build_pools with a
fake builder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import sms_retrain as job  # noqa: E402


def _row(seed, fp=5, tp=200, f1=0.98, nbytes=400_000):
    return {
        "seed": seed,
        "groundlm_fp": fp,
        "smishx_tp": tp,
        "campaign_f1": f1,
        "onnx_bytes": nbytes,
    }


# ---- arguments ----


def test_parse_args_defaults_and_required_arm():
    args = job.parse_args(["--arm", "B"])
    assert args.arm == "B"
    assert args.seeds == [1, 2, 3]
    assert args.epochs == 15 and args.patience == 3 and args.batch_size == 128
    assert args.lr == pytest.approx(2e-3) and args.weight_decay == pytest.approx(1e-4)
    assert args.n_boot == 1000 and args.no_export is False


def test_parse_args_rejects_missing_arm_and_repeated_seed():
    with pytest.raises(SystemExit):
        job.parse_args([])
    with pytest.raises(SystemExit):
        job.parse_args(["--arm", "A", "--seeds", "1", "1"])
    with pytest.raises(SystemExit):
        job.parse_args(["--arm", "Z"])


# ---- gates (plan section 5) ----


def test_gate_passes_on_median_and_majority_of_seeds():
    out = job.gate_eval([_row(1, fp=8), _row(2, fp=10), _row(3, fp=12)])
    g1 = out["G1_groundlm_fp"]
    assert g1["median"] == 10.0
    assert g1["seeds_pass"] == 2 and g1["passes"] is True
    assert out["arm_pass"] is True


def test_gate_fails_when_median_fails():
    out = job.gate_eval([_row(1, fp=3), _row(2, fp=20), _row(3, fp=30)])
    assert out["G1_groundlm_fp"]["seeds_pass"] == 1
    assert out["G1_groundlm_fp"]["median_pass"] is False
    assert out["G1_groundlm_fp"]["passes"] is False
    assert out["arm_pass"] is False


def test_two_of_three_rule_for_one_sided_gates():
    # For a one-sided gate with three seeds, a passing median already means two seeds pass. The
    # check is still applied (it is part of the frozen rule), so both conditions are asserted.
    out = job.gate_eval([_row(1, fp=3), _row(2, fp=11), _row(3, fp=40)])
    assert out["G1_groundlm_fp"]["median_pass"] is True
    assert out["G1_groundlm_fp"]["seeds_pass"] == 2
    assert out["G1_groundlm_fp"]["passes"] is True
    out = job.gate_eval([_row(1, fp=5), _row(2, fp=12), _row(3, fp=13)])
    assert out["G1_groundlm_fp"]["seeds_pass"] == 1
    assert out["G1_groundlm_fp"]["passes"] is False


def test_campaign_f1_median_is_the_middle_seed():
    rows = [_row(1, f1=0.975), _row(2, f1=0.960), _row(3, f1=0.9715)]
    out = job.gate_eval(rows)
    g3 = out["G3_campaign_f1"]
    assert g3["median"] == pytest.approx(0.9715)
    assert g3["seeds_pass"] == 2 and g3["passes"] is True
    out = job.gate_eval([_row(1, f1=0.99), _row(2, f1=0.95), _row(3, f1=0.94)])
    assert out["G3_campaign_f1"]["median"] == pytest.approx(0.95)
    assert out["G3_campaign_f1"]["passes"] is False


def test_gate_g2_and_g4_boundaries():
    assert job.gate_eval([_row(s, tp=157) for s in (1, 2, 3)])["G2_smishx_smishing_tp"]["passes"]
    assert not job.gate_eval([_row(s, tp=156) for s in (1, 2, 3)])["G2_smishx_smishing_tp"][
        "passes"
    ]
    at_limit = job.ONNX_MAX_BYTES
    assert job.gate_eval([_row(s, nbytes=at_limit) for s in (1, 2, 3)])["G4_onnx_bytes"]["passes"]
    assert not job.gate_eval([_row(s, nbytes=at_limit + 1) for s in (1, 2, 3)])["G4_onnx_bytes"][
        "passes"
    ]


def test_missing_value_fails_its_gate_and_the_arm():
    rows = [_row(1), _row(2), {"seed": 3}]  # seed 3 failed: no values at all
    out = job.gate_eval(rows)
    assert out["G1_groundlm_fp"]["median"] is None
    assert out["G1_groundlm_fp"]["passes"] is False
    assert out["arm_pass"] is False


def test_gate_eval_needs_rows():
    with pytest.raises(ValueError):
        job.gate_eval([])


# ---- exact McNemar on paired decisions ----


def test_paired_table_counts_and_exact_mcnemar():
    ours = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 1], dtype=bool)
    ref = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    t = job.paired_table(ours, ref)
    assert t["both"] == 1  # row 0 only
    assert t["ours_only"] + t["ref_only"] + t["both"] + t["neither"] == 10
    assert t["ours_only"] == int((ours & ~ref).sum())
    assert t["ref_only"] == int((~ours & ref).sum())
    from scipy.stats import binomtest

    expect = binomtest(t["ours_only"], t["ours_only"] + t["ref_only"], 0.5).pvalue
    assert t["mcnemar_exact_p"] == pytest.approx(expect)


def test_paired_table_known_p_value_and_no_discordant_pairs():
    # Ten rows where only the reference flags: 10 discordant pairs, all one way.
    ours = np.zeros(10, dtype=bool)
    ref = np.ones(10, dtype=bool)
    t = job.paired_table(ours, ref)
    assert (t["ours_only"], t["ref_only"]) == (0, 10)
    assert t["mcnemar_exact_p"] == pytest.approx(2 * 0.5**10)
    same = job.paired_table(ref, ref)
    assert same["mcnemar_exact_p"] == 1.0


def test_paired_table_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        job.paired_table(np.zeros(3, bool), np.zeros(4, bool))


def test_bucket_counts_per_category():
    cats = ["bank", "bank", "upi", "url"]
    ours = np.array([True, False, True, False])
    ref = np.array([True, True, False, False])
    out = job.bucket_counts(cats, ours, ref)
    assert out["bank"] == {
        "n": 2,
        "ours_flagged": 1,
        "ref_flagged": 2,
        "ours_only": 0,
        "ref_only": 1,
    }
    assert out["upi"]["ours_only"] == 1 and out["url"]["n"] == 1


def test_xlmr_reference_joins_on_idx_and_fails_on_gap():
    gl = pd.DataFrame({"row_key": ["0", "1", "2"], "label_binary": [0, 0, 1]})
    xl = pd.DataFrame({"idx": [2, 0, 1], "pred": [1, 0, 1], "gold": [1, 0, 0]})
    needed = np.array([True, True, True])
    ref, stats = job.xlmr_reference(gl, xl, needed)
    assert ref.tolist() == [False, True, True]  # row 0 -> idx 0, row 1 -> idx 1, row 2 -> idx 2
    assert stats["matched"] == 3 and stats["gold_mismatch"] == 0
    gl2 = pd.DataFrame({"row_key": ["0", "9"], "label_binary": [0, 0]})
    with pytest.raises(RuntimeError):
        job.xlmr_reference(gl2, xl, np.array([True, True]))


def test_float_labels_keeps_missing_as_nan():
    s = pd.Series([0, 1, pd.NA], dtype="Int64")
    out = job.float_labels(s)
    assert out[0] == 0.0 and out[1] == 1.0 and np.isnan(out[2])


def test_expand_keeps_boolean_flags_boolean():
    flags = job.expand(4, np.array([1, 3]), np.array([True, False]), False)
    assert flags.dtype == bool and flags.tolist() == [False, True, False, False]
    scores = job.expand(3, np.array([0]), np.array([0.25]), -1.0)
    assert scores.tolist() == [0.25, -1.0, -1.0]


# ---- threshold before test scoring ----


def test_threshold_is_written_before_any_test_set_is_scored():
    log: list[str] = []

    class FakeModel:
        pass

    def fit():
        log.append("fit")
        return FakeModel()

    def score(model, name):
        assert isinstance(model, FakeModel)
        log.append(f"score:{name}")
        return np.full(4, 0.5)

    def rule(s_legit, s_cal, y_cal):
        log.append("rule")
        return 0.4, False

    def on_threshold(tau, fallback):
        log.append(f"write_threshold:{tau}")

    post = ["campaign_test", "lang_shift", "groundlm", "smsshield_eval", "val_scam"]
    pipe = job.run_seed_pipeline(
        fit=fit,
        score=score,
        rule=rule,
        y_cal=np.zeros(4),
        post_sets=post,
        on_threshold=on_threshold,
    )
    write_at = log.index("write_threshold:0.4")
    for name in post:
        assert log.index(f"score:{name}") > write_at, name
    # Legit validation and campaign cal are scored before the rule runs.
    assert log.index("score:legit_val") < log.index("rule") < write_at
    assert log.index("score:campaign_cal") < log.index("rule")
    assert pipe["tau"] == 0.4 and pipe["fallback"] is False
    assert pipe["events"][0] == "fit" and "threshold" in pipe["events"]


def test_a_test_set_named_as_pre_threshold_set_is_rejected():
    with pytest.raises(ValueError):
        job.run_seed_pipeline(
            fit=lambda: object(),
            score=lambda m, n: np.zeros(2),
            rule=lambda a, b, c: (0.5, True),
            y_cal=np.zeros(2),
            post_sets=["legit_val"],
            on_threshold=lambda t, f: None,
        )


def test_fallback_flag_is_carried_through():
    pipe = job.run_seed_pipeline(
        fit=lambda: object(),
        score=lambda m, n: np.zeros(2),
        rule=lambda a, b, c: (0.2177, True),
        y_cal=np.zeros(2),
        post_sets=[],
        on_threshold=lambda t, f: None,
    )
    assert pipe["fallback"] is True and pipe["tau"] == pytest.approx(0.2177)


# ---- build_pools is called once, with the data seed ----


def test_build_pools_called_once_with_data_seed_and_keywords():
    calls = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return "PREP"

    out = job.build_pools_once("C", {"x": 1}, ["t1", "t2"], 123, ["g1"], fake_build)
    assert out == "PREP"
    assert len(calls) == 1
    assert calls[0] == {
        "arm": "C",
        "sources": {"x": 1},
        "test_only_texts": ["t1", "t2"],
        "n_campaign_train_neg": 123,
        "seed": 0,
        "groundlm_texts": ["g1"],
    }


def test_jaccard_defaults_are_read_from_the_signature():
    def fake_build(
        arm, sources, *, split_jaccard=0.7, cap_jaccard=0.9, seed=0, groundlm_texts=None
    ):
        return None

    assert job.jaccard_defaults(fake_build) == {"split_jaccard": 0.7, "cap_jaccard": 0.9}

    def no_jaccard(arm, seed=0):
        return None

    assert job.jaccard_defaults(no_jaccard) == {}
