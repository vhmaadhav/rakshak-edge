"""Torch- and onnx-free tests for kaggle/jobs/sms_encoder_fold.py and sms_encoder_export.py:
argument
parsing, threshold selection on cal, the manifest builder, node exclusion selection, the parity
gate,
variant selection and the export manifest. Synthetic arrays only; no model, no data."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import sms_encoder_export as exp  # noqa: E402
import sms_encoder_fold as fold  # noqa: E402

SHA = "a" * 64
REV = "b" * 40


# ---- fold: arguments ----


def test_fold_defaults_match_sms_teacher_recipe():
    args = fold.parse_args(["--model", "jhu-clsp/mmBERT-small"])
    assert args.seed == 0
    assert args.split == "source_holdout:holdout_scamshield"
    assert (args.lr, args.epochs, args.patience) == (3e-5, 4, 2)
    assert (args.max_len, args.batch_size, args.eval_batch_size) == (128, 32, 128)
    assert args.weight_decay == 0.01 and args.warmup_ratio == 0.1


def test_fold_rejects_unknown_model():
    with pytest.raises(SystemExit):
        fold.parse_args(["--model", "bert-base"])


def test_fold_pins_full_commit_shas():
    for repo, rev in fold.MODEL_REVISIONS.items():
        assert repo.startswith("jhu-clsp/mmBERT-")
        assert len(rev) == 40 and set(rev) <= set("0123456789abcdef")


def test_fold_rejects_negative_seed():
    with pytest.raises(SystemExit):
        fold.parse_args(["--model", "jhu-clsp/mmBERT-base", "--seed", "-1"])


# ---- fold: probabilities, operating points, thresholds ----


def test_probs_scam_is_softmax_of_class_one():
    logits = np.array([[0.0, 0.0], [-2.0, 2.0], [1000.0, -1000.0]])
    p = fold.probs_scam(logits)
    assert p[0] == pytest.approx(0.5)
    assert p[1] == pytest.approx(1 / (1 + np.exp(-4.0)))
    assert p[2] == pytest.approx(0.0)  # no overflow at large logits


def test_operating_point_counts_alarm_p_ge_tau():
    y = [1, 1, 0, 0, 0]
    p = [0.9, 0.4, 0.5, 0.1, 0.2]
    op = fold.operating_point(y, p, 0.5)
    assert (op["tp"], op["fp"]) == (1, 1)
    assert op["recall"] == pytest.approx(0.5)
    assert op["fpr"] == pytest.approx(1 / 3)
    assert op["precision"] == pytest.approx(0.5)


def test_threshold_candidates_separable_cal():
    y = np.array([1] * 20 + [0] * 200)
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0.8, 1.0, 20), rng.uniform(0.0, 0.3, 200)])
    t = fold.threshold_candidates(y, p)
    assert t["max_f1"]["cal"]["recall"] == pytest.approx(1.0)
    assert t["max_f1"]["cal"]["fp"] == 0
    assert t["cal1pct"]["cal"]["fpr"] <= 0.01
    assert t["cal1pct"]["negatives_above_k"] == 2


def test_cal1pct_respects_fpr_cap_with_ties():
    y = np.array([1] * 5 + [0] * 300)
    p = np.concatenate([np.full(5, 0.99), np.full(300, 0.5)])  # all negatives tied
    t = fold.threshold_candidates(y, p)
    assert t["cal1pct"]["cal"]["fp"] == 0  # tie at the cap is not alarmed (strict above k)
    assert t["cal1pct"]["cal"]["tp"] == 5


def test_threshold_candidates_need_both_classes():
    with pytest.raises(ValueError):
        fold.threshold_candidates([0, 0, 0], [0.1, 0.2, 0.3])


# ---- fold: manifest ----


def _manifest(**over):
    kwargs = dict(
        model="jhu-clsp/mmBERT-small",
        revision=REV,
        seed=1,
        split="s:p",
        checkpoint_sha256=SHA,
        param_count=123,
        train_rows=10,
        train_pos=3,
        cal_rows=4,
        cal_pos=2,
        best_epoch=1,
        cal_pr_auc=0.9,
        history=[{"epoch": 0}],
        hyperparameters={"lr": 3e-5},
        thresholds={"rule": "x"},
        precision="fp16",
        env={},
        seconds=12.345,
    )
    kwargs.update(over)
    return fold.build_manifest(**kwargs)


def test_manifest_fields_and_types():
    m = _manifest()
    assert m["job"] == "sms_encoder_fold" and m["train_input"] == "canon"
    assert m["checkpoint"] == {
        "folder": "best_checkpoint",
        "file": "model.safetensors",
        "sha256": SHA,
        "param_count": 123,
    }
    assert m["rows"] == {"train": 10, "train_pos": 3, "cal": 4, "cal_pos": 2}
    assert m["training"]["seconds"] == 12.3
    assert m["seed"] == 1 and isinstance(m["seed"], int)


@pytest.mark.parametrize("rev", ["main", "A" * 40, "c" * 39])
def test_manifest_refuses_unpinned_revision(rev):
    with pytest.raises(ValueError):
        _manifest(revision=rev)


def test_manifest_refuses_bad_checksum():
    with pytest.raises(ValueError):
        _manifest(checkpoint_sha256="abc")


def test_file_sha256_matches_hashlib(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"rakshak" * 1000)
    assert fold.file_sha256(f) == hashlib.sha256(b"rakshak" * 1000).hexdigest()


# ---- export: node exclusion ----


def _graph(layers=3, head=("classifier",), embed=("embeddings",)):
    nodes = []
    for L in range(layers):
        for part in ("attn/Wqkv", "attn/Wo", "mlp/Wi", "mlp/Wo"):
            nodes.append((f"/model/layers.{L}/{part}/MatMul", "MatMul"))
        nodes.append((f"/model/layers.{L}/norm/LayerNorm", "LayerNormalization"))
    for h in head:
        nodes.append((f"/{h}/MatMul", "MatMul"))
    for e in embed:
        nodes.append((f"/{e}/MatMul", "MatMul"))
    return nodes


def test_boundary_excludes_head_embed_first_and_last_layer():
    excluded, counts = exp.select_excluded_nodes(_graph(), "boundary")
    assert counts["first_layer"] == 4 and counts["last_layer"] == 4
    assert counts["head"] == 1 and counts["embed"] == 1
    assert "/model/layers.0/attn/Wqkv/MatMul" in excluded
    assert "/model/layers.2/mlp/Wo/MatMul" in excluded
    assert "/model/layers.1/attn/Wqkv/MatMul" not in excluded
    assert "/classifier/MatMul" in excluded and "/embeddings/MatMul" in excluded
    assert all(not n.endswith("LayerNorm") for n in excluded)


def test_all_encoder_keeps_middle_layers_quantised_but_not_head_or_embed():
    excluded, counts = exp.select_excluded_nodes(_graph(), "all_encoder")
    assert "/model/layers.0/attn/Wqkv/MatMul" not in excluded
    assert "/classifier/MatMul" in excluded and "/embeddings/MatMul" in excluded
    assert len(excluded) == 2
    assert counts["first_layer"] == 0


def test_boundary_fails_loudly_without_layer_names():
    nodes = [("MatMul_12", "MatMul"), ("/classifier/MatMul", "MatMul")]
    with pytest.raises(ValueError):
        exp.select_excluded_nodes(nodes, "boundary")


def test_head_is_required():
    with pytest.raises(ValueError):
        exp.select_excluded_nodes(_graph(head=()), "all_encoder")


def test_head_pattern_does_not_match_layer_names():
    nodes = [("/model/layers.4/attn/Wo/MatMul", "MatMul"), ("/head/dense/MatMul", "MatMul")]
    excluded, _ = exp.select_excluded_nodes(nodes, "all_encoder")
    assert excluded == ["/head/dense/MatMul"]


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        exp.select_excluded_nodes(_graph(), "everything")


# ---- export: parity and gate ----


def _logits_from_p(p):
    p = np.asarray(p, dtype=np.float64)
    return np.stack([np.log(1 - p), np.log(p)], axis=1)


def test_parity_identical_logits_are_perfect():
    ref = _logits_from_p([0.1, 0.6, 0.9, 0.3])
    s = exp.parity_stats(ref, ref)
    assert s["argmax_agreement"] == 1.0 and s["max_abs_dp"] == 0.0
    assert s["pearson_r"] == pytest.approx(1.0)
    assert exp.passes_gate(s)


def test_parity_detects_flipped_argmax_and_gate_fails():
    ref = _logits_from_p(np.linspace(0.01, 0.99, 200))
    got = _logits_from_p(np.linspace(0.01, 0.99, 200) * 0.5)
    s = exp.parity_stats(ref, got)
    assert s["argmax_agreement"] < 0.995
    assert not exp.passes_gate(s)


def test_gate_thresholds_exact_boundaries():
    base = {"argmax_agreement": 0.995, "max_abs_dp": 0.02}
    assert exp.passes_gate(base)
    assert not exp.passes_gate({**base, "argmax_agreement": 0.9949})
    assert not exp.passes_gate({**base, "max_abs_dp": 0.0201})


def test_parity_rejects_bad_shapes_and_non_finite():
    with pytest.raises(ValueError):
        exp.parity_stats(np.zeros((3, 3)), np.zeros((3, 3)))
    bad = np.array([[0.0, np.nan]])
    with pytest.raises(ValueError):
        exp.parity_stats(bad, bad)


def test_select_variant_smallest_passing_file():
    results = {
        "fp32": {"gate_pass": True, "bytes": 900},
        "int8_boundary": {"gate_pass": True, "bytes": 400},
        "int8_all": {"gate_pass": False, "bytes": 200},
        "fp16_storage": {"gate_pass": True, "bytes": 450},
    }
    assert exp.select_variant(results) == "int8_boundary"


def test_select_variant_none_when_nothing_passes():
    assert exp.select_variant({"fp32": {"gate_pass": False, "bytes": 1}}) is None
    assert exp.select_variant({"int8_all": {"error": "boom"}}) is None


def test_percentile_summary_shape():
    s = exp.percentile_summary([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["p50"] == 3.0 and s["n"] == 5 and s["p95"] > s["p50"]


# ---- export: manifest and arguments ----


def test_export_manifest_status_and_label():
    doc = exp.export_manifest(
        checkpoint_sha256=SHA,
        checkpoint_source="x",
        manifest_check={"found": False},
        files={},
        parity={},
        selected=None,
        latency={},
        versions={},
        notes=[],
    )
    assert doc["status"] == "no_variant_passes" and doc["selected"] is None
    assert doc["latency_ms"]["label"] == "Kaggle CPU, not Pi"
    assert doc["exporter"]["opset"] == 17 and doc["exporter"]["dynamo"] is False
    assert doc["gate"]["argmax_agreement_min"] == 0.995
    ok = exp.export_manifest(
        checkpoint_sha256=SHA,
        checkpoint_source="x",
        manifest_check={},
        files={},
        parity={},
        selected="int8_all",
        latency={},
        versions={},
        notes=[],
    )
    assert ok["status"] == "ok"


def test_export_manifest_rejects_bad_sha():
    with pytest.raises(ValueError):
        exp.export_manifest(
            checkpoint_sha256="xyz",
            checkpoint_source="x",
            manifest_check={},
            files={},
            parity={},
            selected=None,
            latency={},
            versions={},
            notes=[],
        )


def test_export_parse_args_defaults_and_sha_check():
    args = exp.parse_args([])
    assert args.checkpoint_root == "/kaggle/input" and args.expected_sha256 is None
    assert exp.parse_args(["--expected-sha256", SHA]).expected_sha256 == SHA
    with pytest.raises(SystemExit):
        exp.parse_args(["--expected-sha256", "ABC"])


def test_export_constants_match_plan():
    assert exp.PARITY_ROWS == 2000 and exp.GATE_ARGMAX == 0.995 and exp.GATE_MAX_DP == 0.02
    assert exp.OPSET == 17 and exp.MAX_LEN == 128
