"""Pure functions of the url_general job (kaggle/jobs/url_general.py). No torch, no data, no ONNX.

Covers: the train drop by registered domain across both sources, the overlap report, the set
checks, the tau rule on the combined cal rows (with per-source cal FPR), the seed gates, the arm
rule (seed median plus at least 2 of 3 seeds), the exact McNemar p, and the output layouts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import url_general as job  # noqa: E402


def test_frozen_preregistered_values():
    # docs/URL_GENERAL_PLAN.md. Changing any of these needs a new pre-registration.
    assert job.SEEDS == (0, 1, 2)
    assert job.SEED_PASS_MIN == 2
    assert job.UD_SIZES == {"train": 106512, "cal": 15215, "test": 30432}
    assert job.G1_FPR_MAX == 0.015 and job.G1_RECALL_MIN == 0.85
    assert job.G2_FPR_MAX == 0.05
    assert job.G3_AP_MIN == pytest.approx(0.8447)
    assert job.CURRENT_TAU == pytest.approx(0.6308487409530461)


def test_registered_domains_is_empty_without_the_column():
    df = pd.DataFrame({"url": ["http://a.com"]})
    assert job.registered_domains(df).tolist() == [""]
    df2 = pd.DataFrame({"reg_domain": ["a.com", None]})
    assert job.registered_domains(df2).tolist() == ["a.com", ""]


def test_train_keep_mask_drops_only_non_empty_matches():
    keep = job.train_keep_mask(
        np.array(["a.com", "b.com", ""], dtype=object), np.array(["b.com", ""], dtype=object)
    )
    assert keep.tolist() == [True, False, True]


def _example_splits():
    pp_reg = np.array(
        ["a.com", "b.com", "utest.com", "shared.com", "", "t.com", "cal.com"], dtype=object
    )
    ud_reg = np.array(["ud1.com", "t.com", "ud3.com", "utest.com"], dtype=object)
    return job.combined_splits(
        pp_reg,
        pp_train=[0, 1, 2, 3, 4],
        pp_cal=[6],
        pp_test=[5],
        ud_reg=ud_reg,
        ud_train=[0, 1, 2],
        ud_cal=[],
        ud_test=[3],
        n_pp=len(pp_reg),
    )


def test_train_drops_only_rows_whose_domain_is_in_the_url_domain_test_set():
    sp = _example_splits()
    # PP train row 2 (utest.com) is in the url_domain test set, so it is dropped. UD train row 1
    # (t.com) is in the PhreshPhish test set only, so it is kept (amendment 2026-11-12).
    assert sp.dropped_pp == 1 and sp.dropped_ud == 0
    # PP rows keep their positions; UD row p is stored at n_pp + p (n_pp = 7).
    assert sp.train.tolist() == [0, 1, 3, 4, 7, 8, 9]
    assert sp.cal.tolist() == [6]
    assert sp.test_pp.tolist() == [5]
    assert sp.test_ud.tolist() == [10]


def test_phreshphish_train_rows_sharing_a_domain_with_phreshphish_test_are_kept():
    pp_reg = np.array(["t.com", "t.com"], dtype=object)
    ud_reg = np.array(["u.com"], dtype=object)
    sp = job.combined_splits(
        pp_reg,
        pp_train=[0],
        pp_cal=[],
        pp_test=[1],
        ud_reg=ud_reg,
        ud_train=[],
        ud_cal=[],
        ud_test=[0],
        n_pp=2,
    )
    assert sp.train.tolist() == [0]
    assert sp.dropped_pp == 0


def test_training_domains_and_domain_novel_flag():
    sp = _example_splits()
    pp_reg = np.array(
        ["a.com", "b.com", "utest.com", "shared.com", "", "t.com", "cal.com"], dtype=object
    )
    ud_reg = np.array(["ud1.com", "t.com", "ud3.com", "utest.com"], dtype=object)
    doms = job.training_domains(sp, pp_reg, ud_reg, n_pp=7)
    assert doms == {"a.com", "b.com", "shared.com", "ud1.com", "t.com", "ud3.com"}
    flag = job.domain_novel_flag(np.array(["a", "b", "", "c"], dtype=object), [0, 1, 2, 3], {"a"})
    assert flag.tolist() == [False, True, False, True]


def test_novel_subset_reports_ap_and_paired_flags():
    out = job.novel_subset([1, 0], [0.9, 0.1], [0.2, 0.8], tau_new=0.5, tau_old=0.5)
    assert out["ap_new"] == pytest.approx(1.0)
    assert out["ap_old"] == pytest.approx(0.5)
    assert out["new_only"] == 1 and out["old_only"] == 1
    assert out["n_pos"] == 1 and out["n_neg"] == 1


def test_empty_registered_domain_is_kept_in_train():
    sp = _example_splits()
    assert 4 in sp.train.tolist()  # PP row 4 has no registered domain


def test_overlap_report_counts_shared_domains_and_rows():
    cells = {
        "A": np.array(["x.com", "y.com", "", ""], dtype=object),
        "B": np.array(["y.com", "z.com", ""], dtype=object),
    }
    rows = job.domain_overlap(cells)
    assert len(rows) == 1
    assert rows[0] == {"a": "A", "b": "B", "shared_domains": 1, "rows_a": 1, "rows_b": 1}


def test_combined_splits_rejects_a_row_in_two_sets():
    pp_reg = np.array(["a.com", "b.com", "c.com"], dtype=object)
    ud_reg = np.array(["d.com"], dtype=object)
    with pytest.raises(ValueError, match="share rows"):
        job.combined_splits(
            pp_reg,
            pp_train=[0, 1],
            pp_cal=[1],
            pp_test=[2],
            ud_reg=ud_reg,
            ud_train=[],
            ud_cal=[],
            ud_test=[0],
            n_pp=3,
        )


def test_tau_is_set_on_the_combined_cal_benign_rows():
    # PhreshPhish benign 0.000 .. 0.999 (1000 rows); url_domain benign all at 0.95 (1000 rows).
    # Combined benign = 2000, so 1 % allows 20 benign at or above tau. PP alone would allow 10.
    pp_benign = np.arange(1000) / 1000.0
    ud_benign = np.full(1000, 0.95)
    pp_phish = np.full(200, 0.995)
    y = np.concatenate([np.zeros(1000), np.zeros(1000), np.ones(200)]).astype(int)
    s = np.concatenate([pp_benign, ud_benign, pp_phish])
    src = np.array(["PP"] * 1000 + ["UD"] * 1000 + ["PP"] * 200, dtype=object)
    rec = job.calibrate(y, s, src)
    # the UD rows at 0.95 rule out every tau at or below 0.95; the smallest tau with <= 20 benign
    # at or above it is 0.980 (PhreshPhish 0.980 .. 0.999)
    assert rec["tau"] == pytest.approx(0.98)
    assert rec["cal_fp"] == 20
    assert rec["cal_fpr"] == pytest.approx(0.01)
    assert rec["cal_n_benign_PP"] == 1000 and rec["cal_n_benign_UD"] == 1000
    assert rec["cal_fp_PP"] == 20 and rec["cal_fp_UD"] == 0
    assert rec["cal_fpr_PP"] == pytest.approx(0.02)
    assert rec["cal_fpr_UD"] == pytest.approx(0.0)


def test_seed_gates_boundaries():
    ok = job.seed_gates(pp_fpr=0.015, pp_recall=0.85, ud_fpr=0.05, pp_ap1=job.G3_AP_MIN)
    assert ok == {"G1": True, "G2": True, "G3": True, "pass": True}
    assert job.seed_gates(0.0151, 0.9, 0.01, 0.86)["G1"] is False
    assert job.seed_gates(0.01, 0.849, 0.01, 0.86)["G1"] is False
    assert job.seed_gates(0.01, 0.9, 0.0501, 0.86)["G2"] is False
    assert job.seed_gates(0.01, 0.9, 0.01, 0.8446)["G3"] is False
    assert job.seed_gates(0.01, 0.9, 0.01, 0.8446)["pass"] is False


GOOD = {"pp_fpr": 0.01, "pp_recall": 0.9, "ud_fpr": 0.03, "pp_ap1": 0.86}
BAD_UD = {"pp_fpr": 0.01, "pp_recall": 0.9, "ud_fpr": 0.2, "pp_ap1": 0.86}
BAD_PP = {"pp_fpr": 0.02, "pp_recall": 0.9, "ud_fpr": 0.03, "pp_ap1": 0.86}


def test_arm_passes_on_median_and_two_seeds():
    dec = job.arm_decision({0: GOOD, 1: GOOD, 2: BAD_UD})
    assert dec["pass"] is True
    assert dec["seeds_passing"] == 2


def test_arm_fails_with_only_one_seed_passing():
    dec = job.arm_decision({0: GOOD, 1: BAD_PP, 2: BAD_UD})
    assert dec["median_gates"]["G1_fpr"] is True  # the median itself is fine
    assert dec["seeds_passing"] == 1
    assert dec["pass"] is False


def test_arm_fails_when_a_seed_is_missing():
    dec = job.arm_decision({0: GOOD, 1: GOOD})
    assert dec["pass"] is False


def test_arm_fails_when_the_median_misses_a_gate():
    # every seed fails G3 at the median: AP well below the limit
    low = {"pp_fpr": 0.01, "pp_recall": 0.9, "ud_fpr": 0.03, "pp_ap1": 0.80}
    dec = job.arm_decision({0: low, 1: low, 2: low})
    assert dec["median_gates"]["G3"] is False
    assert dec["pass"] is False


def test_mcnemar_exact():
    assert job.mcnemar_exact(0, 0) == 1.0
    assert job.mcnemar_exact(5, 5) == 1.0
    assert job.mcnemar_exact(0, 10) == pytest.approx(2.0 / 1024.0)


def test_paired_counts_on_a_small_example():
    new = [True, True, False, False]
    old = [True, False, True, False]
    y = [1, 0, 0, 1]
    pc = job.paired_counts(new, old, y)
    assert pc["new_only"] == 1 and pc["old_only"] == 1 and pc["both"] == 1
    assert pc["new_fpr"] == pytest.approx(0.5) and pc["old_fpr"] == pytest.approx(0.5)
    assert pc["new_recall"] == pytest.approx(0.5) and pc["old_recall"] == pytest.approx(0.5)


def test_scaler_payload_layout():
    payload = job.scaler_payload(["x", "y"], [0.1, 0.2], [1.0, 2.0])
    assert payload == {
        "columns": ["x", "y"],
        "mean": [0.1, 0.2],
        "std": [1.0, 2.0],
        "clip": 10.0,
        "missing_as": 0.0,
    }


def test_scores_frame_has_one_column_per_seed():
    blocks = {
        "groundlm": {
            "frame": pd.DataFrame({"id": ["a", "b"]}),
            "has": [True, False],
            "scores": {0: np.array([0.9, np.nan]), 2: np.array([0.8, np.nan])},
        }
    }
    out = job.scores_frame(blocks)
    assert list(out.columns) == ["id", "set", "has_url", "score_s0", "score_s2"]
    assert out["set"].tolist() == ["groundlm", "groundlm"]
    assert out["has_url"].tolist() == [True, False]


# ---- score-only mode (no training): file location, threshold read, scaler check, gate ----


def test_parse_args_defaults_to_train_mode():
    assert job.parse_args([]).score_only is False
    assert job.parse_args(["--score-only"]).score_only is True


def test_check_arm_sizes_names_each_mismatch():
    ok = dict(job.EXPECTED_SIZES)
    job.check_arm_sizes(ok)
    bad = dict(ok, train=568923, test_ud=30433)
    with pytest.raises(RuntimeError, match="train 568923 .*test_ud 30433"):
        job.check_arm_sizes(bad)


def test_locate_seed_files_accepts_identical_copies_and_rejects_conflicts(tmp_path):
    names = {
        "best_url_general_s0.onnx": b"onnx-a",
        "url_general_scaler_s0.json": b"{}",
        "thresholds_s0.json": b"{}",
    }
    for name, body in names.items():
        (tmp_path / "kernel_a").mkdir(exist_ok=True)
        (tmp_path / "kernel_a" / name).write_bytes(body)
    found = job.locate_seed_files(0, root=tmp_path)
    assert found["onnx"] == tmp_path / "kernel_a" / "best_url_general_s0.onnx"
    assert set(found) == {"onnx", "scaler", "thresholds"}

    (tmp_path / "copy").mkdir()
    (tmp_path / "copy" / "best_url_general_s0.onnx").write_bytes(b"onnx-b")
    with pytest.raises(RuntimeError, match="different sha256"):
        job.locate_seed_files(0, root=tmp_path)


def test_locate_seed_files_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="best_url_general_s2.onnx"):
        job.locate_seed_files(2, root=tmp_path)


def _threshold_record(**over):
    rec = {
        "seed": 1,
        "tau": job.LOGGED_TAU[1],
        "fit_n": job.EXPECTED_SIZES["train"],
        "cal_n": job.EXPECTED_SIZES["cal"],
    }
    rec.update(over)
    return rec


def test_read_thresholds_checks_seed_sizes_and_logged_tau(tmp_path):
    import json

    path = tmp_path / "thresholds_s1.json"
    path.write_text(json.dumps(_threshold_record()))
    assert job.read_thresholds(path, 1)["tau"] == pytest.approx(0.64630)
    for bad in (
        _threshold_record(seed=0),
        _threshold_record(fit_n=1),
        _threshold_record(cal_n=1),
        _threshold_record(tau=0.7),
        _threshold_record(tau=float("nan")),
    ):
        path.write_text(json.dumps(bad))
        with pytest.raises((RuntimeError, ValueError)):
            job.read_thresholds(path, 1)


def test_read_scaler_checks_columns_and_refit_stats(tmp_path):
    import json

    cols = [f"c{i}" for i in range(job.N_SIDE)]
    mean, std = np.zeros(job.N_SIDE), np.ones(job.N_SIDE)
    path = tmp_path / "url_general_scaler_s0.json"
    path.write_text(json.dumps(job.scaler_payload(cols, mean, std)))
    got_mean, got_std = job.read_scaler(path, cols, mean, std)
    assert got_mean.shape == (job.N_SIDE,) and np.all(got_std == 1.0)
    with pytest.raises(RuntimeError, match="columns differ"):
        job.read_scaler(path, cols[::-1], mean, std)
    with pytest.raises(RuntimeError, match="differ from the scaler refit"):
        job.read_scaler(path, cols, mean + 0.5, std)


def test_gate_adopt_sets_tau_and_forbids_calibration_and_writes():
    import url_threshold as ut

    logged: list[str] = []
    gate = ut.ThresholdGate(logged.append, lambda _r: pytest.fail("score-only must not write"))
    gate.adopt({"tau": 0.91737}, "thresholds_s0.json")
    assert gate.tau == 0.91737
    assert gate.events == ["threshold:read"]
    assert "not recomputed" in logged[0]
    with pytest.raises(RuntimeError, match="calibration set"):
        gate.score_pre("combined_cal", lambda: 0)
    with pytest.raises(RuntimeError, match="already set"):
        gate.adopt({"tau": 0.5}, "again")
    assert gate.score_post("ud_test", lambda: 7) == 7


def test_score_only_refuses_writes_and_is_the_only_threshold_source():
    with pytest.raises(RuntimeError, match="never writes"):
        job._refuse_write({"tau": 0.5})


def _tiny_onnx(path: Path) -> None:
    """A two-input model with the production contract: raw int64 [N,200], side float32 [N,29] in,
    logit float32 [N,1] out. The logit is the sum of the side features."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    node = helper.make_node("ReduceSum", ["side", "axes"], ["logit"], keepdims=1)
    graph = helper.make_graph(
        [node],
        "tiny_url_general",
        [
            helper.make_tensor_value_info("raw", TensorProto.INT64, ["batch", job.URL_MAX_LEN]),
            helper.make_tensor_value_info("side", TensorProto.FLOAT, ["batch", job.N_SIDE]),
        ],
        [helper.make_tensor_value_info("logit", TensorProto.FLOAT, ["batch", 1])],
        initializer=[axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_score_only_scores_a_tiny_onnx_on_the_runtime_path(tmp_path):
    pytest.importorskip("onnx")
    import baseline_v0 as base
    import url_threshold as ut

    onnx_path = tmp_path / "best_url_general_s0.onnx"
    _tiny_onnx(onnx_path)
    from rakshak.url_features import url_features

    cols = list(url_features("http://example.com/"))  # the 29 runtime feature names, in order
    assert len(cols) == job.N_SIDE
    mean = np.zeros(job.N_SIDE, dtype=np.float64)
    std = np.ones(job.N_SIDE, dtype=np.float64)
    session = base._onnx_session(onnx_path)
    urls = ["http://example.com/login", "https://shop.test/parcel?id=1"]
    scores = ut.score_url_strings(session, urls, cols, mean, std)
    assert scores.shape == (2,) and np.isfinite(scores).all()
