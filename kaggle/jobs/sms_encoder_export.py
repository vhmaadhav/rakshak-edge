"""Export one encoder checkpoint to ONNX in several quantised forms, each gated on parity (W1c).

Input: a checkpoint folder (model.safetensors, config.json, tokenizer files) attached as kernel
output. Either a fold job's ``best_checkpoint`` (kaggle/jobs/sms_encoder_fold.py, with its
manifest.json beside it) or the teacher kernel's ``best_checkpoint``. The job searches
--checkpoint-root for model.safetensors, optionally filtered by --expected-sha256.

Export: torch.onnx.export, dynamo=False, opset 17, dynamic batch and sequence, logits output.

Variants (all share the fp32 graph as their source):
- fp32: the export as is. Reference for the fallback.
- int8_boundary, int8_all: onnxruntime quantize_dynamic, QInt8 per-channel weights, reduce_range
  False, MatMul only. DYNAMIC activation quantisation (uint8 per tensor at run time). Kept as a
  record of the failed route. boundary keeps the head, embeddings and first and last encoder
  layers in fp32; all_encoder keeps only the head and embeddings.
- nbits8_b32, nbits8_b128, nbits4_b32, nbits4_b128: WEIGHT-ONLY MatMulNBits from
  onnxruntime.quantization.matmul_nbits_quantizer. Symmetric, block size 32 or 128, 8 or 4 bits.
  Activations stay fp32 and the kernel dequantises weight blocks at run time. The embeddings
  (Gather, not MatMul) and the classifier head are excluded and stay fp32.
- fp16_storage: every float initializer with two or more dimensions, except the embeddings and
  the classifier head, is stored as float16 and cast back to float32 in front of its consumers.
  Compute is fp32. ORT CPU cannot run most ops in native fp16, so this is the runnable fp16 form.

Why the first INT8 variants scored argmax 0.1710 (both, in the 2026-11-12 run). Diagnosis from the
run log and export.json:
- The two graphs are different: the exclusion counts are 14 and 2, and the files are 448.9 MB
  and 437.5 MB. So node exclusion was applied and this is not an identical-graph bug.
- The argmax agreement 0.1710 is the share of cal rows whose fp32 argmax is class 1. So both
  quantised graphs predict scam for (nearly) every row, a constant output. The Pearson r of the
  probabilities is 0.15 and 0.19, not a sign flip (a flip would give agreement near 0, and a
  positive r rules that out).
- The likely cause is run-time activation quantisation: per-tensor dynamic uint8 scales on the
  residual stream, which carries large-magnitude outlier channels. Both INT8 variants quantise the
  middle layers' activations, so both collapse the same way. ORT already leaves activation by
  activation MatMuls (attention QK and PV) in fp32, so those are not the cause.
  The nbits variants test this directly: they quantise weights only, so if they pass, dynamic
  activation quantisation was the cause.
- The fp16_storage failure was a real bug: ``graph.input[:] = ...`` raises "does not support
  assignment" on a protobuf repeated field. Fixed by clearing and extending.

Parity: 2,000 rows drawn with a fixed seed (0) from the cal split of the same fold as the
checkpoint, canonicalised with rakshak.eval.data.canon_text, tokenised at max_len 128. PyTorch fp32
on CPU is the reference. Per variant, the GATE (pre-registered in docs/SOTA_PUSH_PLAN.md W1c) is:
decision agreement (argmax over the two logits, which is the sign of the margin l1 - l0) at least
0.995, AND max |dp| at most 0.02 (dp = softmax probability of class 1). The job has no separate
logit threshold, so margin 0 is the decision point. Also reported, not gated: the logit-space
versions (max and mean |Δmargin|, Spearman of the margins), the positive rate of each side, and a
constant-prediction flag. Probabilities saturate near 0 and 1, so the logit-space numbers are the
more sensitive check.

Selection: among the variants that pass the gate, the smallest file. If none passes, the job
reports "no variant passes" and writes no selection. Nothing is chosen by looking at test rows.

Latency: onnxruntime CPUExecutionProvider, 4 intra-op threads, batch 1, sequence lengths 32, 64 and
128, 5 warm-up runs then 30 timed runs. The parity rows' token-length distribution is recorded in
the notes so the latency can be read against it. This is Kaggle CPU, for reference only. It is NOT
the Raspberry Pi number.

Outputs: one ONNX file per variant that was exported, the tokenizer files, export_manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from _common import env_report, timed

JOB = "sms_encoder_export"
WEIGHTS = "model.safetensors"
MANIFEST_IN = "manifest.json"
EXPORT_MANIFEST = "export_manifest.json"
OPSET = 17
MAX_LEN = 128
THREADS = 4
PARITY_ROWS = 2000
PARITY_SEED = 0
BATCH = 32
GATE_ARGMAX = 0.995
GATE_MAX_DP = 0.02
LAT_SEQ_LENS = (32, 64, 128)
LAT_WARMUP = 5
LAT_RUNS = 30
DEFAULT_SPLIT = "source_holdout:holdout_scamshield"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)
INT8_VARIANTS = ("int8_boundary", "int8_all")
# name -> (bits, block_size) for the weight-only MatMulNBits variants (activations stay fp32).
NBITS_VARIANTS = {
    "nbits8_b32": (8, 32),
    "nbits8_b128": (8, 128),
    "nbits4_b32": (4, 32),
    "nbits4_b128": (4, 128),
}
VARIANTS = ("fp32", *INT8_VARIANTS, *NBITS_VARIANTS, "fp16_storage")
QUANT_OPS = ("MatMul", "Gemm")
SHA_RE = re.compile(r"[0-9a-f]{64}")
# Node-name patterns: exporter names look like "/model/layers.3/attn/Wqkv/MatMul" or
# "/classifier/...".
LAYER_RE = re.compile(r"(?:^|[/.])layers?[./_](\d+)(?=[/._]|$)")
HEAD_RE = re.compile(r"(?:^|[/._])(classifier|head|pooler)(?=[/._]|$)", re.IGNORECASE)
EMBED_RE = re.compile(r"embed", re.IGNORECASE)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=JOB)
    p.add_argument("--checkpoint-root", default="/kaggle/input")
    p.add_argument(
        "--expected-sha256",
        default=None,
        help="sha256 of the model.safetensors to export (checked against the file)",
    )
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--data-root", default=None)
    args = p.parse_args(list(argv or []))
    if args.expected_sha256 is not None and not SHA_RE.fullmatch(args.expected_sha256):
        p.error("--expected-sha256 must be 64 lowercase hex digits")
    return args


# ---- pure helpers (torch- and onnx-free) ----


def select_excluded_nodes(
    nodes: Iterable[tuple[str, str]], mode: str
) -> tuple[list[str], dict[str, int]]:
    """MatMul/Gemm node names to keep in fp32, from (name, op_type) pairs.

    mode "boundary": the head, the embeddings, and the first and last encoder layers.
    mode "all_encoder": the head and the embeddings only.
    Returns the sorted names and the counts per reason. Raises when boundary mode finds no first or
    last layer, so a naming change fails the variant loudly instead of quantising the wrong nodes.
    """
    if mode not in ("boundary", "all_encoder"):
        raise ValueError(f"unknown mode {mode!r}")
    pairs = [(name, op) for name, op in nodes if op in QUANT_OPS]
    layer_of = {}
    for name, _ in pairs:
        match = LAYER_RE.search(name)
        if match:
            layer_of[name] = int(match.group(1))
    if not layer_of:
        raise ValueError("no encoder-layer names found in the graph")
    first, last = min(layer_of.values()), max(layer_of.values())
    counts = {"head": 0, "embed": 0, "first_layer": 0, "last_layer": 0}
    excluded: set[str] = set()
    for name, _ in pairs:
        if HEAD_RE.search(name):
            counts["head"] += 1
            excluded.add(name)
        elif EMBED_RE.search(name):
            counts["embed"] += 1
            excluded.add(name)
        elif mode == "boundary" and name in layer_of:
            if layer_of[name] == first:
                counts["first_layer"] += 1
                excluded.add(name)
            elif layer_of[name] == last:
                counts["last_layer"] += 1
                excluded.add(name)
    if mode == "boundary" and (counts["first_layer"] == 0 or counts["last_layer"] == 0):
        raise ValueError(f"boundary exclusion matched no first or last layer: {counts}")
    if counts["head"] == 0:
        raise ValueError("no classifier head node found in the graph")
    counts["layers_found"] = len(set(layer_of.values()))
    counts["excluded"] = len(excluded)
    return sorted(excluded), counts


def keep_fp32_initializer(name: str) -> bool:
    """Initializers that stay float32 in the fp16 storage variant: embeddings and the head."""
    return bool(EMBED_RE.search(name) or HEAD_RE.search(name))


def parity_stats(ref_logits: np.ndarray, got_logits: np.ndarray) -> dict[str, Any]:
    """Agreement of two logit arrays of shape (n, 2) against the reference.

    Top level (gated): argmax agreement and max/mean |dp| of the class-1 probability, plus the
    Pearson r of the probabilities. ``logit`` holds the logit-space view (margin = l1 - l0):
    max and mean |Δmargin|, Spearman of the margins, and decision agreement at margin 0 (the
    same decision as argmax). The positive rates and the constant-prediction flag are there to
    catch a collapsed graph, which shows as a positive rate of exactly 0 or 1.
    """
    ref = np.asarray(ref_logits, dtype=np.float64)
    got = np.asarray(got_logits, dtype=np.float64)
    if ref.shape != got.shape or ref.ndim != 2 or ref.shape[1] != 2:
        raise ValueError(f"expected two (n, 2) arrays, got {ref.shape} and {got.shape}")
    if not (np.isfinite(ref).all() and np.isfinite(got).all()):
        raise ValueError("non-finite logits in parity")
    p_ref = _softmax_scam(ref)
    p_got = _softmax_scam(got)
    dp = np.abs(p_ref - p_got)
    argmax_ref = ref.argmax(axis=1)
    argmax_got = got.argmax(axis=1)
    argmax = float(np.mean(argmax_ref == argmax_got))
    if np.std(p_ref) > 0 and np.std(p_got) > 0:
        pearson = float(np.corrcoef(p_ref, p_got)[0, 1])
    else:
        pearson = None
    m_ref = ref[:, 1] - ref[:, 0]
    m_got = got[:, 1] - got[:, 0]
    dm = np.abs(m_ref - m_got)
    spearman = _pearson(_rank(m_ref), _rank(m_got))
    pos_got = float(np.mean(argmax_got == 1))
    return {
        "n": int(len(ref)),
        "argmax_agreement": argmax,
        "max_abs_dp": float(dp.max()),
        "mean_abs_dp": float(dp.mean()),
        "pearson_r": pearson,
        "logit": {
            "max_abs_dmargin": float(dm.max()),
            "mean_abs_dmargin": float(dm.mean()),
            "spearman": spearman,
            "decision_agreement_at_0": float(np.mean((m_ref > 0) == (m_got > 0))),
        },
        "ref_positive_rate": float(np.mean(argmax_ref == 1)),
        "got_positive_rate": pos_got,
        "got_constant_prediction": pos_got in (0.0, 1.0),
    }


def _rank(x: np.ndarray) -> np.ndarray:
    """Ranks 0..n-1 (ties broken by position; margins are continuous, so ties are rare)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) > 0 and np.std(b) > 0:
        return float(np.corrcoef(a, b)[0, 1])
    return None


def _softmax_scam(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def passes_gate(stats: dict[str, Any]) -> bool:
    """The pre-registered parity gate: decision agreement >= 0.995 and max |dp| <= 0.02."""
    return stats["argmax_agreement"] >= GATE_ARGMAX and stats["max_abs_dp"] <= GATE_MAX_DP


def select_variant(results: dict[str, dict[str, Any]]) -> str | None:
    """The passing variant with the smallest file, or None. Pure, uses only recorded fields."""
    passing = [
        (rec["bytes"], name)
        for name, rec in results.items()
        if rec.get("gate_pass") and rec.get("bytes") is not None
    ]
    if not passing:
        return None
    return min(passing)[1]


def token_length_summary(mask: np.ndarray) -> dict[str, float]:
    """Distribution of the real (unpadded) token counts of the parity rows."""
    lengths = np.asarray(mask, dtype=np.int64).sum(axis=1)
    return {
        "p50": float(np.percentile(lengths, 50)),
        "p95": float(np.percentile(lengths, 95)),
        "max": int(lengths.max()),
        "mean": round(float(lengths.mean()), 2),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile_summary(values_ms: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "n": int(arr.size),
    }


def export_manifest(
    *,
    checkpoint_sha256: str,
    checkpoint_source: str,
    manifest_check: dict[str, Any],
    files: dict[str, dict[str, Any]],
    parity: dict[str, Any],
    selected: str | None,
    latency: dict[str, Any],
    versions: dict[str, str],
    notes: list[str],
) -> dict[str, Any]:
    """The export_manifest.json document. Pure: arranges what the steps recorded."""
    if not SHA_RE.fullmatch(checkpoint_sha256):
        raise ValueError("checkpoint_sha256 must be 64 lowercase hex digits")
    return {
        "job": JOB,
        "schema": 2,
        "status": "ok" if selected else "no_variant_passes",
        "checkpoint": {
            "sha256": checkpoint_sha256,
            "source": checkpoint_source,
            "manifest_check": manifest_check,
        },
        "exporter": {
            "name": "torch.onnx.export",
            "dynamo": False,
            "opset": OPSET,
            "max_len": MAX_LEN,
        },
        "gate": {
            "argmax_agreement_min": GATE_ARGMAX,
            "max_abs_dp_max": GATE_MAX_DP,
            "rows": PARITY_ROWS,
            "seed": PARITY_SEED,
            "reference": "pytorch fp32, cpu",
        },
        "files": files,
        "parity": parity,
        "selected": selected,
        "latency_ms": {
            "label": "Kaggle CPU, not Pi",
            "threads": THREADS,
            "batch": 1,
            "by_variant": latency,
        },
        "versions": versions,
        "notes": notes,
    }


def tokenizer_ids_for_latency(texts: Sequence[str], tokenizer) -> np.ndarray:
    """One long real token sequence, cut to each latency length in run()."""
    joined = " ".join(texts)
    enc = tokenizer(joined, truncation=True, max_length=max(LAT_SEQ_LENS), return_tensors="np")
    ids = np.asarray(enc["input_ids"], dtype=np.int64)
    if ids.shape[1] < max(LAT_SEQ_LENS):
        pad = tokenizer.pad_token_id or 0
        extra = np.full((1, max(LAT_SEQ_LENS) - ids.shape[1]), pad, dtype=np.int64)
        ids = np.concatenate([ids, extra], axis=1)
    return ids


# ---- ONNX steps (imports inside functions: the module stays importable without onnx) ----


def export_fp32(model, tokenizer, out_path: Path) -> None:
    import torch

    class Wrapped(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapped = Wrapped(model).eval()
    enc = tokenizer(
        ["Your account will be blocked, verify now"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=16,
    )
    torch.onnx.export(
        wrapped,
        (enc["input_ids"], enc["attention_mask"]),
        str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=OPSET,
        dynamo=False,
    )


def quantise_variant(fp32: Path, out: Path, mode: str) -> dict[str, Any]:
    """Dynamic INT8 (activations quantised at run time). Kept as the record of the failed route."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    graph = onnx.load(str(fp32), load_external_data=False)
    nodes = [(n.name, n.op_type) for n in graph.graph.node]
    excluded, counts = select_excluded_nodes(nodes, mode)
    quantize_dynamic(
        str(fp32),
        str(out),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        op_types_to_quantize=["MatMul"],
        nodes_to_exclude=excluded,
    )
    return {"mode": mode, "excluded_nodes": len(excluded), "exclusion_counts": counts}


def require_matmul_nbits():
    """The MatMulNBits quantiser needs onnx_ir (a dependency of ORT >= 1.22). Fail loudly."""
    from onnxruntime.quantization import matmul_nbits_quantizer

    return matmul_nbits_quantizer


def quantise_nbits(fp32: Path, out: Path, bits: int, block_size: int) -> dict[str, Any]:
    """Weight-only MatMulNBits, symmetric, activations fp32. Embeddings (Gather) and the head
    MatMul nodes are excluded and stay fp32."""
    import onnx

    nb = require_matmul_nbits()
    model = onnx.load(str(fp32))
    nodes = [(n.name, n.op_type) for n in model.graph.node]
    excluded, counts = select_excluded_nodes(nodes, "all_encoder")
    quant = nb.MatMulNBitsQuantizer(
        model,
        bits=bits,
        block_size=block_size,
        is_symmetric=True,
        nodes_to_exclude=excluded,
        algo_config=nb.DefaultWeightOnlyQuantConfig(
            block_size=block_size, is_symmetric=True, bits=bits
        ),
    )
    quant.process()
    quant.model.save_model_to_file(str(out), False)
    return {
        "mode": f"nbits{bits}",
        "bits": bits,
        "block_size": block_size,
        "symmetric": True,
        "activations": "fp32",
        "excluded_nodes": len(excluded),
        "exclusion_counts": counts,
    }


def fp16_storage(fp32: Path, out: Path) -> dict[str, Any]:
    """Store float initializers with two or more dimensions as float16 (embeddings and the head
    stay float32), cast to float32 in front of their consumers. The output name is unchanged, so
    no consumer is rewired."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(fp32), load_external_data=False)
    graph = model.graph
    converted: list[str] = []
    new_inits = []
    cast_nodes = []
    for init in graph.initializer:
        if (
            init.data_type == TensorProto.FLOAT
            and len(init.dims) >= 2
            and not keep_fp32_initializer(init.name)
        ):
            arr = numpy_helper.to_array(init).astype(np.float16)
            half_name = f"{init.name}__fp16"
            new_inits.append(numpy_helper.from_array(arr, name=half_name))
            cast_nodes.append(
                helper.make_node(
                    "Cast",
                    [half_name],
                    [init.name],
                    to=TensorProto.FLOAT,
                    name=f"cast__{init.name}",
                )
            )
            converted.append(init.name)
        else:
            new_inits.append(init)
    originals = set(converted)
    kept = [i for i in new_inits if i.name not in originals]
    kept_inputs = [i for i in graph.input if i.name not in originals]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    # A protobuf repeated field does not support slice assignment (the old fp16 failure:
    # "TypeError: does not support assignment"). Clear and extend instead.
    del graph.input[:]
    graph.input.extend(kept_inputs)
    body = list(graph.node)
    del graph.node[:]
    graph.node.extend(cast_nodes + body)
    onnx.save(model, str(out))
    return {"mode": "fp16_storage", "converted_initializers": len(converted)}


def time_session(path: Path, ids: np.ndarray) -> dict[str, Any]:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    result: dict[str, Any] = {}
    for length in LAT_SEQ_LENS:
        x = ids[:, :length]
        mask = np.ones_like(x)
        feed = {"input_ids": x, "attention_mask": mask}
        for _ in range(LAT_WARMUP):
            sess.run(None, feed)
        times = []
        for _ in range(LAT_RUNS):
            t0 = time.perf_counter()
            sess.run(None, feed)
            times.append((time.perf_counter() - t0) * 1000.0)
        result[f"seq{length}"] = percentile_summary(times)
    return result


def onnx_logits(path: Path, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    out = np.zeros((len(ids), 2), dtype=np.float32)
    for start in range(0, len(ids), BATCH):
        sl = slice(start, start + BATCH)
        (logits,) = sess.run(None, {"input_ids": ids[sl], "attention_mask": mask[sl]})
        out[sl] = logits
    return out


def torch_logits(model, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import torch

    out = np.zeros((len(ids), 2), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(ids), BATCH):
            sl = slice(start, start + BATCH)
            logits = model(
                input_ids=torch.from_numpy(ids[sl]), attention_mask=torch.from_numpy(mask[sl])
            ).logits
            out[sl] = logits.float().numpy()
    return out


def parity_inputs(root: Path, split: str, data_root: str | None, tokenizer):
    """2,000 seeded cal rows of the fold, canonicalised and padded to one array per batch."""
    from rakshak.eval import data as D

    sms = D.load_sms(D.resolve_data_root(data_root))
    labels = sms["label_binary"].to_numpy().astype(np.int64)
    stem = split.split(":", 1)[0]
    runs = {
        r.name: r
        for r in D.load_runs(
            D.resolve_data_root(data_root),
            sms,
            patterns=[stem],
            labels=labels,
            mask=D.eval_mask(sms),
        )
    }
    if split not in runs:
        raise ValueError(f"split {split!r} not found")
    cal = runs[split].cal
    if len(cal) < PARITY_ROWS:
        raise ValueError(f"cal has {len(cal)} rows, need {PARITY_ROWS}")
    rng = np.random.default_rng(PARITY_SEED)
    pick = np.sort(rng.choice(len(cal), PARITY_ROWS, replace=False))
    rows = cal[pick]
    texts = [D.canon_text(t) for t in sms["text"].to_numpy(dtype=object)[rows]]
    enc = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="np")
    ids = np.asarray(enc["input_ids"], dtype=np.int64)
    mask = np.asarray(enc["attention_mask"], dtype=np.int64)
    return (
        ids,
        mask,
        texts,
        {
            "rows_in_cal": int(len(cal)),
            "seed": PARITY_SEED,
            "ids_sha256": hashlib.sha256(
                np.asarray(sms["id"].to_numpy()[rows]).astype(str).tobytes()
            ).hexdigest(),
            "token_len": token_length_summary(mask),
        },
    )


def locate_checkpoint(root: Path, expected: str | None) -> Path:
    hits = sorted(p for p in root.rglob(WEIGHTS) if p.is_file())
    if expected:
        hits = [p for p in hits if sha256_file(p) == expected]
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected exactly one {WEIGHTS} under {root} (sha filter {expected!r}), found "
            f"{len(hits)}"
        )
    return hits[0].parent


def check_manifest(folder: Path, sha: str) -> dict[str, Any]:
    """If a fold manifest sits beside the checkpoint folder, its checkpoint sha must match."""
    path = folder.parent / MANIFEST_IN
    if not path.is_file():
        return {"found": False}
    doc = json.loads(path.read_text())
    recorded = doc["checkpoint"]["sha256"]
    if recorded != sha:
        raise ValueError(f"manifest sha256 {recorded[:12]} differs from checkpoint {sha[:12]}")
    return {
        "found": True,
        "sha256_matches": True,
        "model": doc["model"],
        "seed": doc["seed"],
        "revision": doc["revision"],
    }


def run(ctx) -> bool | str:
    args = parse_args(ctx.argv)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    work = Path(ctx.work)
    env = env_report()
    ctx.log(f"env: {env}")
    timings = ctx.results.setdefault("timings", {})
    with timed("locate", timings):
        folder = locate_checkpoint(Path(args.checkpoint_root), args.expected_sha256)
    sha = sha256_file(folder / WEIGHTS)
    manifest_check = check_manifest(folder, sha)
    ctx.log(f"checkpoint {folder} sha256 {sha[:12]} manifest {manifest_check}")

    torch.set_num_threads(THREADS)
    tokenizer = AutoTokenizer.from_pretrained(str(folder))
    model = AutoModelForSequenceClassification.from_pretrained(str(folder)).float().eval()
    params = int(sum(p.numel() for p in model.parameters()))
    ids, mask, texts, parity_meta = parity_inputs(folder, args.split, args.data_root, tokenizer)
    with timed("reference", timings):
        ref = torch_logits(model, ids, mask)

    fp32 = work / "model_fp32.onnx"
    with timed("export_fp32", timings):
        export_fp32(model, tokenizer, fp32)
    files: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {"fp32": fp32}
    results: dict[str, dict[str, Any]] = {}
    notes: list[str] = [f"params {params}", f"parity rows {parity_meta}"]

    def record(name: str, path: Path, extra: dict[str, Any]) -> None:
        stats = parity_stats(ref, onnx_logits(path, ids, mask))
        files[name] = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        results[name] = {
            "gate_pass": passes_gate(stats),
            "parity": stats,
            **extra,
            "bytes": path.stat().st_size,
        }
        lg = stats["logit"]
        ctx.log(
            f"{name}: decision {stats['argmax_agreement']:.4f} max|dp| "
            f"{stats['max_abs_dp']:.4f} max|dmargin| {lg['max_abs_dmargin']:.4f} "
            f"spearman {lg['spearman']} pos ref/got {stats['ref_positive_rate']:.3f}/"
            f"{stats['got_positive_rate']:.3f} gate {results[name]['gate_pass']}"
        )

    def attempt(name: str, build) -> None:
        target = work / f"model_{name}.onnx"
        try:
            with timed(f"quantise_{name}", timings):
                info = build(target)
            paths[name] = target
            record(name, target, {"quantisation": info})
        except Exception as exc:  # one variant failing must not stop the others
            notes.append(f"{name} failed: {type(exc).__name__}: {str(exc)[:300]}")
            ctx.log(notes[-1])
            results[name] = {"gate_pass": False, "error": f"{type(exc).__name__}: {exc}"}

    record("fp32", fp32, {})
    for name, mode in (("int8_boundary", "boundary"), ("int8_all", "all_encoder")):
        attempt(name, lambda out, mode=mode: quantise_variant(fp32, out, mode))
    for name, (bits, block) in NBITS_VARIANTS.items():
        attempt(name, lambda out, bits=bits, block=block: quantise_nbits(fp32, out, bits, block))
    attempt("fp16_storage", lambda out: fp16_storage(fp32, out))

    for tok_file in TOKENIZER_FILES:
        src = folder / tok_file
        if src.is_file():
            shutil.copy(src, work / tok_file)
    latency_ids = tokenizer_ids_for_latency([t for t in texts[:40]], tokenizer)
    latency = {}
    with timed("latency", timings):
        for name, path in paths.items():
            latency[name] = time_session(path, latency_ids)
    selected = select_variant(results)
    import transformers

    versions = {"torch": torch.__version__, "transformers": transformers.__version__}
    try:
        import onnx
        import onnxruntime

        versions.update({"onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__})
    except ImportError as exc:  # recorded, not fatal
        notes.append(f"version lookup: {exc}")
    doc = export_manifest(
        checkpoint_sha256=sha,
        checkpoint_source=str(folder),
        manifest_check=manifest_check,
        files=files,
        parity={"reference": "pytorch fp32 cpu", "variants": results},
        selected=selected,
        latency=latency,
        versions=versions,
        notes=notes,
    )
    (work / EXPORT_MANIFEST).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    ctx.results["export"] = doc
    ctx.results["summary"] = f"selected {selected or 'none'}; " + ", ".join(
        f"{k} decision {v['parity']['argmax_agreement']:.4f}"
        for k, v in results.items()
        if "parity" in v
    )
    ctx.log(ctx.results["summary"])
    ctx.save()
    return bool(selected)
