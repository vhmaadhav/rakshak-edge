"""Tests for the W1c quantisation variants in kaggle/jobs/sms_encoder_export.py.

Pure parity and naming checks run without onnx. The graph tests build a tiny synthetic ONNX graph
(a few KB) and need onnx, onnxruntime and onnx_ir; they skip when those are missing. No model, no
data, no torch."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import sms_encoder_export as exp  # noqa: E402

# ---- pure: parity in logit space and collapse detection ----


def test_parity_logit_space_fields_present_and_zero_for_identical():
    ref = np.array([[2.0, -1.0], [-0.5, 0.5], [0.1, 0.2]])
    s = exp.parity_stats(ref, ref.copy())
    assert s["argmax_agreement"] == 1.0 and s["max_abs_dp"] == 0.0
    assert s["logit"]["max_abs_dmargin"] == 0.0
    assert s["logit"]["decision_agreement_at_0"] == 1.0
    assert s["logit"]["spearman"] == pytest.approx(1.0)
    assert s["got_constant_prediction"] is False


def test_constant_collapse_is_flagged():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(200, 2))
    got = np.tile([0.0, 1.0], (200, 1))  # predicts class 1 for every row
    s = exp.parity_stats(ref, got)
    assert s["got_constant_prediction"] is True
    assert s["got_positive_rate"] == 1.0
    assert s["argmax_agreement"] == pytest.approx(float(np.mean(ref.argmax(1) == 1)))


def test_logit_margin_gap_detected_when_probs_saturate():
    # Both logits large: probabilities saturate to 0/1, the margin still moves.
    ref = np.array([[30.0, -30.0], [-30.0, 30.0]])
    got = np.array([[30.0, -29.0], [-30.0, 29.0]])
    s = exp.parity_stats(ref, got)
    assert s["max_abs_dp"] < 1e-12
    assert s["logit"]["max_abs_dmargin"] == pytest.approx(1.0)


def test_spearman_of_reversed_margins_is_minus_one():
    ref = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]])
    got = ref[::-1].copy()
    assert exp.parity_stats(ref, got)["logit"]["spearman"] == pytest.approx(-1.0)


def test_rank_gives_positions():
    assert exp._rank(np.array([3.0, 1.0, 2.0])).tolist() == [2.0, 0.0, 1.0]


def test_gate_still_uses_decision_and_probability_only():
    ok = {"argmax_agreement": 0.995, "max_abs_dp": 0.02}
    assert exp.passes_gate(ok)
    assert not exp.passes_gate({"argmax_agreement": 0.994, "max_abs_dp": 0.0})
    assert not exp.passes_gate({"argmax_agreement": 1.0, "max_abs_dp": 0.0201})


# ---- pure: variant set and fp32 keep rules ----


def test_variant_set_has_weight_only_and_fp16_and_fp32_fallback():
    assert exp.VARIANTS[0] == "fp32"
    assert set(exp.NBITS_VARIANTS) == {"nbits8_b32", "nbits8_b128", "nbits4_b32", "nbits4_b128"}
    assert exp.NBITS_VARIANTS["nbits4_b32"] == (4, 32)
    assert exp.NBITS_VARIANTS["nbits8_b128"] == (8, 128)
    assert exp.VARIANTS[-1] == "fp16_storage"
    assert set(exp.INT8_VARIANTS) <= set(exp.VARIANTS)


@pytest.mark.parametrize(
    "name",
    [
        "inner.classifier.weight",
        "inner.model.embeddings.tok_embeddings.weight",
        "head.dense.weight",
    ],
)
def test_embeddings_and_head_stay_fp32_in_fp16_storage(name):
    assert exp.keep_fp32_initializer(name)


@pytest.mark.parametrize(
    "name", ["inner.model.layers.0.attn.Wqkv.weight", "layers.3.mlp.Wi.weight"]
)
def test_encoder_weights_are_converted_in_fp16_storage(name):
    assert not exp.keep_fp32_initializer(name)


def test_select_variant_picks_smallest_passing_including_nbits():
    results = {
        "fp32": {"gate_pass": True, "bytes": 563},
        "int8_all": {"gate_pass": False, "bytes": 437},
        "nbits8_b32": {"gate_pass": True, "bytes": 300},
        "nbits4_b128": {"gate_pass": True, "bytes": 200},
        "fp16_storage": {"gate_pass": True, "bytes": 280},
    }
    assert exp.select_variant(results) == "nbits4_b128"


def test_token_length_summary_counts_real_tokens_only():
    mask = np.array([[1, 1, 1, 0], [1, 1, 1, 1]])
    s = exp.token_length_summary(mask)
    assert s["max"] == 4 and s["mean"] == 3.5


# ---- onnx graph tests on a tiny synthetic model ----

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx_ir")


def _tiny_model(path: Path) -> np.ndarray:
    from onnx import TensorProto as T
    from onnx import helper, numpy_helper

    rng = np.random.default_rng(0)
    V, D = 50, 32

    def init(name, shape, scale=0.3):
        return numpy_helper.from_array(
            (rng.standard_normal(shape) * scale).astype(np.float32), name
        )

    inits = [
        init("embeddings.weight", (V, D)),
        init("model.layers.0.attn.Wq.weight", (D, D)),
        init("model.layers.0.mlp.Wi.weight", (D, 2 * D)),
        init("model.layers.1.attn.Wq.weight", (D, D)),
        init("model.layers.1.mlp.Wo.weight", (2 * D, D)),
        init("classifier.weight", (D, 2)),
        init("classifier.bias", (2,), 0.01),
    ]
    nodes = [
        helper.make_node(
            "Gather", ["embeddings.weight", "input_ids"], ["x0"], axis=0, name="/embeddings/Gather"
        ),
        helper.make_node(
            "MatMul",
            ["x0", "model.layers.0.attn.Wq.weight"],
            ["q0"],
            name="/model/layers.0/attn/Wq/MatMul",
        ),
        helper.make_node("Add", ["x0", "q0"], ["h0"], name="/res0"),
        helper.make_node(
            "MatMul",
            ["h0", "model.layers.0.mlp.Wi.weight"],
            ["f0"],
            name="/model/layers.0/mlp/Wi/MatMul",
        ),
        helper.make_node("Relu", ["f0"], ["f0r"], name="/act0"),
        helper.make_node(
            "MatMul",
            ["f0r", "model.layers.1.mlp.Wo.weight"],
            ["f1"],
            name="/model/layers.1/mlp/Wo/MatMul",
        ),
        helper.make_node(
            "MatMul",
            ["h0", "model.layers.1.attn.Wq.weight"],
            ["q1"],
            name="/model/layers.1/attn/Wq/MatMul",
        ),
        helper.make_node("Add", ["f1", "q1"], ["h1"], name="/res1"),
        helper.make_node("ReduceMean", ["h1"], ["pool"], axes=[1], keepdims=0, name="/pool"),
        helper.make_node(
            "MatMul", ["pool", "classifier.weight"], ["cm"], name="/classifier/MatMul"
        ),
        helper.make_node("Add", ["cm", "classifier.bias"], ["logits"], name="/classifier/Add"),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("input_ids", T.INT64, ["batch", "seq"])],
        [helper.make_tensor_value_info("logits", T.FLOAT, ["batch", 2])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return rng.integers(1, V, (64, 12)).astype(np.int64)


def _logits(path: Path, ids: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, {"input_ids": ids})[0]


def test_fp16_storage_runs_and_keeps_embeddings_and_head_fp32(tmp_path):
    # Regression: the old code did `graph.input[:] = ...`, which raises
    # "TypeError: does not support assignment" on a protobuf repeated field.
    fp32 = tmp_path / "fp32.onnx"
    ids = _tiny_model(fp32)
    out = tmp_path / "fp16.onnx"
    info = exp.fp16_storage(fp32, out)
    assert info["converted_initializers"] == 4  # the four encoder weights, not embed or head
    model = onnx.load(str(out))
    by_name = {i.name: i for i in model.graph.initializer}
    assert by_name["embeddings.weight"].data_type == onnx.TensorProto.FLOAT
    assert by_name["classifier.weight"].data_type == onnx.TensorProto.FLOAT
    assert by_name["model.layers.0.mlp.Wi.weight__fp16"].data_type == onnx.TensorProto.FLOAT16
    assert "model.layers.0.mlp.Wi.weight" not in by_name
    ref = _logits(fp32, ids)
    got = _logits(out, ids)
    assert np.allclose(ref, got, atol=1e-2)


def test_nbits_variants_are_weight_only_and_keep_head_fp32(tmp_path):
    fp32 = tmp_path / "fp32.onnx"
    ids = _tiny_model(fp32)
    ref = _logits(fp32, ids)
    for bits, block in ((8, 32), (4, 128)):
        out = tmp_path / f"nb{bits}_{block}.onnx"
        info = exp.quantise_nbits(fp32, out, bits, block)
        assert info["activations"] == "fp32" and info["symmetric"] is True
        model = onnx.load(str(out))
        nbits = [n for n in model.graph.node if n.op_type == "MatMulNBits"]
        assert len(nbits) == 4  # the four encoder MatMuls
        attr = {a.name: a.i for a in nbits[0].attribute}
        assert attr["bits"] == bits and attr["block_size"] == block
        head = [n for n in model.graph.node if n.name == "/classifier/MatMul"]
        assert head and head[0].op_type == "MatMul"  # head stays fp32
        got = _logits(out, ids)
        assert got.shape == ref.shape and np.isfinite(got).all()
        if bits == 8:
            assert np.abs(got - ref).max() < 0.05
            s = exp.parity_stats(ref, got)
            assert s["argmax_agreement"] >= 0.95
