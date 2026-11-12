"""Tests for kaggle/jobs/sms_teacher_xsource.py (plan docs/TEACHER_XSOURCE_PLAN.md).
Synthetic inputs only; no torch, no model, no inbox data."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import sms_teacher_xsource as job  # noqa: E402

from rakshak.eval import data as D  # noqa: E402

SENTINEL = "ZQXSENTINEL"
ID_SENTINEL = "ZQXROWID"


# ---- decide: every branch and both boundaries ----


@pytest.mark.parametrize(
    ("a1", "a2", "verdict"),
    [
        (0.98, 0.98, "transfers"),  # both boundaries inclusive
        (1.0, 1.0, "transfers"),
        (0.979999, 0.99, "partial"),  # A1 just under the transfer bar
        (0.99, 0.979999, "partial"),  # A2 just under the transfer bar
        (0.93, 0.99, "partial"),  # A1 exactly at the fail bar: strict < does not fail
        (0.99, 0.85, "partial"),  # A2 exactly at the fail bar: strict < does not fail
        (0.92999, 0.99, "does_not_transfer"),
        (0.99, 0.84999, "does_not_transfer"),
        (0.5, 0.5, "does_not_transfer"),
    ],
)
def test_decide_branches_and_boundaries(a1, a2, verdict):
    out = job.decide(a1, a2)
    assert out["verdict"] == verdict
    assert out["a1"] == a1 and out["a2"] == a2
    assert out["next_plan"]


def test_decide_refuses_undefined_auc():
    with pytest.raises(ValueError, match="undefined"):
        job.decide(None, 0.99)
    with pytest.raises(ValueError, match="undefined"):
        job.decide(0.99, None)


# ---- AUC helpers ----


def test_auc_perfect_separation_is_one():
    assert job.auc_of(np.array([0.9, 0.95, 1.0]), np.array([0.1, 0.2])) == 1.0


def test_auc_reversed_is_zero():
    assert job.auc_of(np.array([0.1, 0.2]), np.array([0.9, 0.95, 1.0])) == 0.0


def test_auc_identical_distributions_near_half():
    rng = np.random.default_rng(3)
    a = job.auc_of(rng.random(3000), rng.random(3000))
    assert abs(a - 0.5) < 0.03


def test_auc_empty_side_is_none():
    assert job.auc_of(np.array([]), np.array([0.1])) is None
    assert job.auc_binary(np.array([1, 1]), np.array([0.2, 0.3])) is None
    assert job.auc_binary(np.array([0, 1]), np.array([0.2, 0.3])) == 1.0


# ---- recall threshold convention ----


def test_keep_threshold_70_and_90_percent_of_ten():
    pos = np.arange(1.0, 11.0)  # 1..10
    # k = ceil(10 * 70 / 100) = 7: the 7th largest is 4, and seven rows are >= 4
    t70 = job.keep_threshold(pos, 70)
    assert t70 == 4.0
    assert int((pos >= t70).sum()) == 7
    # k = 9: the 9th largest is 2
    t90 = job.keep_threshold(pos, 90)
    assert t90 == 2.0
    assert int((pos >= t90).sum()) == 9


def test_keep_threshold_integer_arithmetic_has_no_float_drift():
    # 0.7 * 10 is 7.000000000000001 in floats; the integer ceil must still give 7, not 8
    pos = np.arange(1.0, 11.0)
    assert int((pos >= job.keep_threshold(pos, 70)).sum()) == 7


def test_keep_threshold_rounds_up_and_handles_unit_cases():
    pos = np.array([0.1, 0.2, 0.3])  # k = ceil(3 * 70 / 100) = 3, so the minimum is kept
    assert job.keep_threshold(pos, 70) == 0.1
    assert job.keep_threshold(np.array([0.4]), 90) == 0.4
    assert job.keep_threshold(pos, 100) == 0.1


def test_keep_threshold_rejects_bad_input():
    with pytest.raises(ValueError):
        job.keep_threshold(np.array([]), 70)
    with pytest.raises(ValueError):
        job.keep_threshold(np.array([0.5]), 0)


# ---- bootstrap ----


def test_bootstrap_is_deterministic_with_a_seed():
    rng = np.random.default_rng(1)
    pos = rng.normal(2.0, 1.0, 120)
    neg = rng.normal(0.0, 1.0, 120)
    a = job.bootstrap_auc(pos, neg, 150, seed=0)
    b = job.bootstrap_auc(pos, neg, 150, seed=0)
    assert a == b
    assert a["n_boot"] == 150 and a["seed"] == 0


def test_bootstrap_interval_contains_point_on_easy_data():
    rng = np.random.default_rng(2)
    pos = rng.normal(3.0, 1.0, 150)
    neg = rng.normal(0.0, 1.0, 150)
    out = job.bootstrap_auc(pos, neg, 200, seed=0)
    lo, hi = out["ci95"]
    assert lo <= out["point"] <= hi
    assert 0.9 < out["point"] <= 1.0


def test_bootstrap_perfect_separation_is_degenerate_at_one():
    out = job.bootstrap_auc(np.array([0.8, 0.9, 1.0]), np.array([0.1, 0.2, 0.3]), 50, seed=0)
    assert out["point"] == 1.0
    assert out["ci95"] == [1.0, 1.0]


# ---- contamination ----


def test_contamination_is_exact_match_with_counts():
    texts = ["alpha", "beta", "gamma", "beta "]
    train = {"alpha", "gamma"}
    mask = job.contamination_mask(texts, train, canon=lambda t: t)
    assert mask.tolist() == [True, False, True, False]
    clean = ~mask
    assert [t for t, keep in zip(texts, clean, strict=True) if keep] == ["beta", "beta "]
    assert job.canon_keys(["alpha", "", "gamma"], canon=lambda t: t) == {"alpha", "gamma"}


def test_contamination_uses_canonical_form_of_both_sides():
    text = "Hello there, your account is locked."
    keys = job.canon_keys([text])
    assert job.contamination_mask([text], keys).tolist() == [True]
    assert job.contamination_mask(["an unrelated sentence"], keys).tolist() == [False]
    assert D.canon_text(text) in keys


def test_contamination_mask_empty_input():
    assert job.contamination_mask([], {"x"}).size == 0


# ---- teacher checkpoint and provenance ----


def test_teacher_sha_mismatch_raises_clearly(tmp_path):
    folder = tmp_path / "best_checkpoint"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"not the pinned weights")
    with pytest.raises(ValueError, match="sha256 mismatch") as err:
        job.locate_teacher(tmp_path)
    assert "not the pinned weights" not in str(err.value)


def test_teacher_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="best_checkpoint"):
        job.locate_teacher(tmp_path)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        job.locate_teacher(tmp_path / "absent")


def test_teacher_pinned_copy_is_found(tmp_path, monkeypatch):
    payload = b"synthetic weights"
    folder = tmp_path / "best_checkpoint"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(payload)
    monkeypatch.setattr(job, "TEACHER_SHA", hashlib.sha256(payload).hexdigest())
    assert job.locate_teacher(tmp_path) == folder


def test_teacher_meta_reads_run_and_input_form():
    metrics = {
        "meta": {
            "train_input": "canon",
            "best_checkpoint": {
                "model": "jhu-clsp/mmBERT-base",
                "run": "source_holdout:holdout_x",
                "seed": 0,
                "cal_pr_auc": 0.5,
            },
        }
    }
    meta = job.teacher_meta(metrics)
    assert meta["run"] == "source_holdout:holdout_x"
    assert meta["train_input"] == "canon"
    assert meta["seed"] == 0
    with pytest.raises(ValueError):
        job.teacher_meta({"meta": {"train_input": "canon"}})


def test_resolve_teacher_run_agreement_and_errors():
    assert job.resolve_teacher_run("source_holdout:a", None) == "source_holdout:a"
    assert job.resolve_teacher_run(None, "source_holdout:a") == "source_holdout:a"
    assert job.resolve_teacher_run("source_holdout:a", "source_holdout:a") == "source_holdout:a"
    with pytest.raises(ValueError, match="differs"):
        job.resolve_teacher_run("source_holdout:a", "source_holdout:b")
    with pytest.raises(ValueError, match="no teacher run"):
        job.resolve_teacher_run(None, None)
    with pytest.raises(ValueError, match="split:partition"):
        job.resolve_teacher_run(None, "campaign")


def test_train_input_must_match_teacher_record():
    job.check_train_input("canon", "canon")
    with pytest.raises(ValueError, match="differs"):
        job.check_train_input("canon", "raw")


# ---- argument parsing ----


def test_parse_args_defaults_and_flags():
    args = job.parse_args([])
    assert args.teacher_dir == "/kaggle/input"
    assert args.teacher_run is None
    assert args.train_input == "canon"
    assert args.max_len == 128 and args.eval_batch_size == 128
    assert args.v0_onnx == "/kaggle/input"
    assert args.n_boot == 2000
    args = job.parse_args(
        [
            "--teacher-run",
            "source_holdout:holdout_x",
            "--train-input",
            "raw",
            "--n-boot",
            "10",
            "--max-len",
            "64",
        ]
    )
    assert args.teacher_run == "source_holdout:holdout_x"
    assert args.train_input == "raw" and args.n_boot == 10 and args.max_len == 64


def test_parse_args_rejects_bad_values():
    with pytest.raises(SystemExit):
        job.parse_args(["--train-input", "bogus"])
    with pytest.raises(SystemExit):
        job.parse_args(["--n-boot", "0"])


# ---- summary, report and the privacy rule ----


def _synthetic_inputs(rng: np.random.Generator):
    """Sets with synthetic texts and ids, scores per model. Returns the frames the job would hold
    and the score dict. Texts and ids carry SENTINEL so a leak is visible in the output."""
    n_sx_pos, n_sx_ham, n_gl, n_leg, n_scam, n_held, n_camp = 60, 40, 50, 50, 14, 30, 80
    texts = {}
    labels = {}
    ids = {}
    for name, n, lab in (
        ("smishx", n_sx_pos + n_sx_ham, np.r_[np.ones(n_sx_pos), np.zeros(n_sx_ham)]),
        ("groundlm_legit", n_gl, np.zeros(n_gl)),
        ("test_donor", n_leg + n_scam, np.r_[np.zeros(n_leg), np.ones(n_scam)]),
        ("heldout_legit", n_held, np.zeros(n_held)),
        ("campaign_test", n_camp, np.r_[np.ones(n_camp // 2), np.zeros(n_camp - n_camp // 2)]),
    ):
        texts[name] = [f"{SENTINEL} {name} {i}" for i in range(n)]
        ids[name] = [f"{ID_SENTINEL}{name}{i}" for i in range(n)]
        labels[name] = lab.astype(np.int64)
    scores: dict = {}
    for model, shift in (("teacher", 2.0), ("v0", 1.0)):
        scores[model] = {}
        for name, lab in labels.items():
            base = rng.normal(0.0, 1.0, lab.size) + shift * lab
            scores[model][name] = {
                "scrubbed": base,
                "filled": None
                if name == "smishx" or name == "groundlm_legit"
                else base + rng.normal(0, 0.3, lab.size),
            }
    return texts, ids, labels, scores


def test_summary_payload_contains_no_text_or_id(tmp_path):
    rng = np.random.default_rng(7)
    texts, ids, labels, scores = _synthetic_inputs(rng)
    train_keys = job.canon_keys([texts["groundlm_legit"][0], texts["smishx"][3]], canon=lambda t: t)
    sets = {}
    for name in job.SETS:
        clean = ~job.contamination_mask(texts[name], train_keys, canon=lambda t: t)
        sets[name] = {"labels": labels[name], "clean": clean}
    result = job.summarise(sets, scores, n_boot=20)
    payload = {
        "job": job.JOB,
        "status": "scored",
        "teacher": {
            "model": "synthetic",
            "run": "source_holdout:synthetic",
            "seed": 0,
            "cal_pr_auc": 0.5,
            "train_input": "canon",
            "max_len": 128,
            "eval_batch_size": 128,
        },
        "v0": {"sha256": job.V0_SHA},
        **result,
    }
    dumped = json.dumps(payload, default=str) + job.render_report(payload)
    assert SENTINEL not in dumped
    assert ID_SENTINEL not in dumped
    assert payload["sets"]["smishx"]["contaminated"]["k"] == 1
    assert result["decision"]["verdict"] in {"transfers", "does_not_transfer", "partial"}


def test_summary_views_split_on_contamination():
    rng = np.random.default_rng(9)
    texts, _ids, labels, scores = _synthetic_inputs(rng)
    sets = {}
    for name in job.SETS:
        clean = np.ones(labels[name].size, dtype=bool)
        if name == "smishx":
            clean[:10] = False  # ten positives in the training partition
        sets[name] = {"labels": labels[name], "clean": clean}
    result = job.summarise(sets, scores, n_boot=10)
    all_rows = result["models"]["teacher"]["all"]["rows"]["smishx_positives"]
    clean_rows = result["models"]["teacher"]["clean"]["rows"]["smishx_positives"]
    assert all_rows - clean_rows == 10
    assert result["models"]["teacher"]["all"]["bootstrap"] is None
    assert result["models"]["teacher"]["clean"]["bootstrap"] is not None


def test_render_report_handles_not_scored_status():
    text = job.render_report({"status": "stopped: insufficient data", "reason": "counts only"})
    assert "stopped" in text and "counts only" in text


# ---- kernel bundle ----


def test_build_dependencies_list_every_job_import():
    import ast
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_kernel_under_test", JOBS.parent / "build_kernel.py"
    )
    build_kernel = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = build_kernel
    spec.loader.exec_module(build_kernel)
    deps = set(build_kernel.JOB_DEPENDENCIES["sms_teacher_xsource"])
    tree = ast.parse((JOBS / "sms_teacher_xsource.py").read_text())
    local = {"baseline_v0", "sms_retrain", "sms_inbox_retrain", "sms_surrogate_d0", "sms_teacher"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for name in imported & local:
        assert f"kaggle/jobs/{name}.py" in deps
    assert "kaggle/jobs/sms_teacher_xsource.py" not in deps
    assert "kaggle/jobs/sms_retrain.py" in deps
