"""Stand-in SMS student for laptop tests and benchmarks (not the trained model).

The deployed sms_student.onnx is not in the repo or on this laptop (models/ is gitignored). This
builds a small ONNX graph with the deployed model's inputs (raw, canon: int64 (batch, MAX_LEN);
side: float32 (batch, N_RULES)) and its output (logit: float32 (batch,)), with seeded random
weights. ONNX Runtime runs it for real, so featurisation, the batched session.run, the scoring and
the decision all execute the production code path. Verdicts from it say nothing about detection
quality: they only let the old and the new path be compared on the same model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

VOCAB = 512  # byte ids are 0..256 (PAD_ID is 256)


def build_standin(path: Path, seed: int = 0, embed: int = 16) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from rakshak.models.encoding import MAX_LEN, N_RULES

    rng = np.random.default_rng(seed)
    raw_table = rng.normal(0.0, 1.0, (VOCAB, embed)).astype(np.float32)
    canon_table = rng.normal(0.0, 1.0, (VOCAB, embed)).astype(np.float32)
    weight = rng.normal(0.0, 0.5, (2 * embed + N_RULES, 1)).astype(np.float32)
    bias = np.zeros((1,), np.float32)
    initializers = [
        numpy_helper.from_array(raw_table, "raw_table"),
        numpy_helper.from_array(canon_table, "canon_table"),
        numpy_helper.from_array(weight, "W"),
        numpy_helper.from_array(bias, "B"),
        numpy_helper.from_array(np.array([-1], np.int64), "flat"),
    ]
    nodes = [
        helper.make_node("Gather", ["raw_table", "raw"], ["raw_e"], axis=0),
        helper.make_node("ReduceMean", ["raw_e"], ["raw_m"], axes=[1], keepdims=0),
        helper.make_node("Gather", ["canon_table", "canon"], ["canon_e"], axis=0),
        helper.make_node("ReduceMean", ["canon_e"], ["canon_m"], axes=[1], keepdims=0),
        helper.make_node("Concat", ["raw_m", "canon_m", "side"], ["feat"], axis=1),
        helper.make_node("Gemm", ["feat", "W", "B"], ["z"]),
        helper.make_node("Reshape", ["z", "flat"], ["logit"]),
    ]
    inputs = [
        helper.make_tensor_value_info("raw", TensorProto.INT64, ["batch", MAX_LEN]),
        helper.make_tensor_value_info("canon", TensorProto.INT64, ["batch", MAX_LEN]),
        helper.make_tensor_value_info("side", TensorProto.FLOAT, ["batch", N_RULES]),
    ]
    outputs = [helper.make_tensor_value_info("logit", TensorProto.FLOAT, ["batch"])]
    graph = helper.make_graph(nodes, "standin_sms_student", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 9)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path
