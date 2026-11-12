"""
Export the pinned SMS teacher to ONNX (fp32 and dynamic INT8), check parity, record reference
latency. CPU kernel. No training, no private data: the parity inputs are the 64 synthetic strings
below, and nothing read from the inbox or the campaign data is used.

Teacher: the ``best_checkpoint`` folder of kernel maadhavvh/rakshak-sms-teacher
(jhu-clsp/mmBERT-base fine-tune, ModernBERT-style sequence classifier, 2 labels),
located under --teacher-dir by the sha256 of its model.safetensors (default
673b970e...). Loaded on CPU, fp32, eval mode.

Export: torch.onnx.export first (dynamo=False), then the dynamo exporter if that fails. Opset 17,
the same as the student export (kaggle/jobs/sms_student.py, rakshak.models.student.export_onnx).
Inputs input_ids and attention_mask (int64, dynamic batch and sequence), output logits (float32,
[batch, 2]). Every attempt is recorded with its exception type and message (model-side messages
only). If no exporter works the job writes export.json with status "export_failed" and returns
"fail"; it does not raise.

Quantisation: onnxruntime.quantization.quantize_dynamic, QInt8 weights, MatMul and Gemm. If it
fails, fp32 only.

Parity: 64 fixed strings, canonicalised with rakshak.eval.data.canon_text (the canon input form
the teacher is trained on), tokenised at max_len 128 with the checkpoint's tokenizer. PyTorch fp32
is the reference for both ONNX files. The INT8 file is also compared with the fp32 ONNX file.

Latency: onnxruntime CPUExecutionProvider, 4 intra-op threads, 1 inter-op thread, sequential
execution, batch 1, sequence lengths 32, 64 and 128 tokens, 5 warmup runs then 30 timed runs. This
is the Kaggle CPU, for reference only. It is not the Raspberry Pi number.

Outputs in the kernel output folder: teacher_fp32.onnx, teacher_int8.onnx, the tokenizer files
(tokenizer.json, tokenizer_config.json, special_tokens_map.json, added_tokens.json when present),
export.json (sizes, sha256, opset, ir_version, exporter, quantised op types, parity, latency,
teacher sha, library versions), report.md, and results.json with ctx.results["summary"].

Launch (CPU kernel: no --gpu; --internet for the pinned pip set; the teacher kernel attached; no
datasets):
  kaggle/launch.sh sms_teacher_export --internet --kernels maadhavvh/rakshak-sms-teacher

Flags (--job-args="FLAGS"):
  --teacher-dir PATH   folder searched for model.safetensors with the pinned sha256
                       (default /kaggle/input)
  --teacher-sha HEX    sha256 of the model.safetensors to export (default: the pinned teacher)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import re
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from _common import env_report, timed

JOB = "sms_teacher_export"
TEACHER_SHA = "673b970e9f4746dbbbb5a00aa84eb056d6750cbc2897ba712ab1aa1b234abc7e"
WEIGHTS = "model.safetensors"
DEFAULT_ROOT = "/kaggle/input"
SHA_RE = re.compile(r"[0-9a-f]{64}")
OPSET = 17
MAX_LEN = 128
THREADS = 4
LAT_SEQ_LENS = (32, 64, 128)
LAT_WARMUP = 5
LAT_RUNS = 30
QUANT_OP_TYPES = ("MatMul", "Gemm")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)
PACKAGES = ("transformers", "torch", "onnx", "onnxruntime", "tokenizers", "numpy")
REQUIRED_KEYS = (
    "job",
    "status",
    "teacher_sha256",
    "exporter",
    "export_attempts",
    "opset",
    "ir_version",
    "max_len",
    "files",
    "quantisation",
    "parity",
    "latency_ms",
    "parity_strings",
    "versions",
    "notes",
)
EXPORT_FAILED = "export_failed"

# 64 synthetic SMS-like strings: scam-style and legitimate-style, English, Hinglish in Roman
# letters, Devanagari, some with URLs and amounts. Invented for this job; no real messages or
# personal data.
SMS_STRINGS: tuple[str, ...] = (
    "Your account will be blocked today. Verify KYC now at http://kyc-update-secure.example.com",
    "Congratulations! You won Rs 25,000 in lucky draw. Claim before 6 pm: bit.ly/claim-prize-x",
    "Dear customer, your SBI YONO account is suspended. Update PAN at"
    " http://sbi-update.example.net",
    "Aapka KYC expire ho raha hai. Turant update karein: https://kyc-verify.example.org",
    "Pay Rs 499 processing fee to receive your parcel. Track: http://dhl-redeliver.example.co",
    "You have been selected for a work from home job, earn Rs 5000 per day. Reply YES",
    "Your electricity connection will be disconnected tonight. Pay Rs 1,850 via"
    " link: http://bill-pay-now.example.in",
    "URGENT: Income tax refund of Rs 12,430 pending. Submit details at"
    " http://itr-refund-claim.example.com",
    "Your Amazon order is on hold. Confirm card details here http://amzn-hold.example.com",
    "Aap ka loan approve ho gaya hai. Rs 2,00,000 transfer ke liye fee bhejiye 999 pe",
    "Free recharge of Rs 300 for loyal users. Click http://free-recharge-offer.example.biz now",
    "Your Aadhaar will be deactivated. Call the helpline and share OTP to avoid block",
    "Final notice: your DTH subscription ends today. Renew at"
    " http://dth-renew-now.example.info for Rs 350",
    "Crypto doubling offer! Invest Rs 10,000 today and get Rs 40,000 in 24 hours. t.me/invest-fast",
    "आपका बैंक खाता बंद हो सकता है। अभी KYC अपडेट करें: http://kyc-hindi.example.com",
    "आपने Rs 5,000 का इनाम जीता है। लिंक पर क्लिक करें http://inam-jeeto.example.com",
    "Your package could not be delivered. Pay Rs 49 customs duty at http://parcel-duty.example.com",
    "Part time job offer: like YouTube videos and earn Rs 800 daily. WhatsApp us to start today",
    "Congrats! Your number won a Rs 1 lakh prize from KBC. Send processing charge to claim",
    "Hi mum, my phone broke, please send Rs 15000 to this new UPI id urgently",
    "Your UPI PIN is needed to receive the cashback of Rs 2000. Enter PIN at"
    " http://cashback-upi.example.com",
    "Verify your streaming payment method now or your account will close:"
    " http://stream-billing.example.com",
    "Loan of Rs 5 lakh pre-approved at 0% interest. Apply in 2 minutes:"
    " http://instant-loan.example.com",
    "Your PAN card is linked to suspicious transactions. Install the attached APK to unblock",
    "Dear user, your EPF withdrawal of Rs 85,000 is ready. Confirm bank details at"
    " http://epf-claim.example.com",
    "Limited time! Buy 1 get 3 free on phones this week. Order at http://deal-phone.example.shop",
    "Your e-challan of Rs 2,500 is unpaid. Pay within 24 hours at"
    " http://challan-pay.example.com to avoid arrest",
    "You are pre-selected for a credit card with Rs 3 lakh limit. Share Aadhaar and OTP to proceed",
    "Your courier fee is pending: pay Rs 99 to release shipment. http://release-parcel.example.com",
    "Team update: your salary account needs re-verification. Login at"
    " http://payroll-verify.example.com",
    "Win a free trip to Goa! Answer 1 question at http://goa-trip-quiz.example.com",
    "Your SIM will be blocked in 2 hours. Send your Aadhaar copy on WhatsApp to the KYC team",
    "Your OTP for login is 482913. Valid for 10 minutes. Do not share it with anyone.",
    "Your order #A1234 has been shipped and will arrive on Tuesday.",
    "Electricity bill of Rs 1,240 for March is generated. Due date 15 April.",
    "Reminder: your appointment is tomorrow at 10:30 AM at the clinic.",
    "Your salary of Rs 45,000 has been credited to your account ending 4521.",
    "Mom, I will reach home by 8 pm. Have dinner ready.",
    "Thanks for paying Rs 600 towards your mobile recharge. Validity till 20 Nov.",
    "Aapka order deliver ho gaya hai. Dhanyavaad!",
    "Kal subah 7 baje meeting hai, please time pe aana.",
    "Your flight 6E 245 is on time. Boarding starts at 06:40 from gate 12.",
    "Your monthly statement is ready. View it in the app under statements.",
    "Happy birthday! Hope you have a wonderful day ahead.",
    "Your password was changed successfully. If this was not you, contact support.",
    "Bus ticket booked: Bengaluru to Chennai, 21 Nov, seat 14. Show this SMS at boarding.",
    "आपका ऑर्डर कल तक पहुँच जाएगा। धन्यवाद।",
    "Dentist appointment confirmed for Friday 5 pm. Reply C to cancel.",
    "Your FD of Rs 1,00,000 matured and has been credited to your savings account.",
    "Payment of Rs 2,350 to the grocery store was successful. Ref 88213.",
    "Please pick up the groceries on your way home, thanks.",
    "College fee receipt for semester 3 is available on the student portal.",
    "Your cab driver is 4 minutes away. Check the vehicle number in the app.",
    "Your gas cylinder booking is confirmed. Expected delivery in 2 days.",
    "Good morning! Team lunch at 1 pm in the cafeteria today.",
    "Your insurance premium of Rs 8,400 is due on 30 Nov. Pay via the app.",
    "Aaj ka match kal shift ho gaya hai, details email pe bhej di hai.",
    "Your train PNR 4521678930 is confirmed for coach S4.",
    "Reminder: parent-teacher meeting on Saturday at 11 am.",
    "Your library book is due on 25 Nov. Renew online on the library website.",
    "Thank you for visiting our store. Your bill is Rs 1,299. Have a nice day.",
    "Gym membership renewed till March 2027. See you at the gym!",
    "Your food order was delivered. Rate your meal in the app.",
    "Check the weather: light rain expected in Chennai this evening.",
)


# ---- pure helpers (importable and unit-tested without torch or onnxruntime) ----


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=JOB)
    p.add_argument("--teacher-dir", default=DEFAULT_ROOT)
    p.add_argument("--teacher-sha", default=TEACHER_SHA)
    args = p.parse_args(list(argv or []))
    if not SHA_RE.fullmatch(args.teacher_sha):
        p.error("--teacher-sha must be 64 lowercase hex digits")
    return args


def sms_strings() -> list[str]:
    """The fixed parity and latency inputs (a copy)."""
    return list(SMS_STRINGS)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(names: Sequence[str] = PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def scam_probability(logits: np.ndarray) -> np.ndarray:
    """P(class 1) from [n, 2] logits, the softmax of sms_teacher.probs_from_logits."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def _argmax_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.argmax(a, axis=1) == np.argmax(b, axis=1)))


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson r, or None when either array is constant or has fewer than two values."""
    if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def logit_parity(ref: np.ndarray, got: np.ndarray) -> dict[str, Any]:
    """Logit-level agreement of ``got`` with ``ref`` (both [n, 2])."""
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    if ref.shape != got.shape or ref.ndim != 2 or ref.shape[0] == 0:
        raise ValueError("logit_parity needs two non-empty arrays of the same [n, classes] shape")
    return {
        "n": int(ref.shape[0]),
        "max_abs_logit_diff": float(np.max(np.abs(ref - got))),
        "argmax_agreement": _argmax_agreement(ref, got),
    }


def probability_parity(ref: np.ndarray, got: np.ndarray) -> dict[str, Any]:
    """Scam-probability agreement of ``got`` with ``ref`` (both [n, 2] logits): max and mean
    absolute probability difference, argmax agreement and Pearson r of the probabilities. Pearson
    is None when a probability vector is constant."""
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    if ref.shape != got.shape or ref.ndim != 2 or ref.shape[0] == 0:
        raise ValueError("probability_parity needs two non-empty arrays of the same [n, 2] shape")
    p_ref = scam_probability(ref)
    p_got = scam_probability(got)
    return {
        "n": int(ref.shape[0]),
        "max_abs_prob_diff": float(np.max(np.abs(p_ref - p_got))),
        "mean_abs_prob_diff": float(np.mean(np.abs(p_ref - p_got))),
        "argmax_agreement": _argmax_agreement(ref, got),
        "pearson_r": _pearson(p_ref, p_got),
    }


def percentile(values: Sequence[float], q: float) -> float:
    if not len(values):
        raise ValueError("percentile of an empty sample")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def latency_summary(samples_ms: Sequence[float]) -> dict[str, Any]:
    """p50, p95, mean and min of latency samples in milliseconds."""
    if not len(samples_ms):
        raise ValueError("latency_summary needs at least one sample")
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "n": int(arr.size),
        "p50": round(percentile(arr, 50), 3),
        "p95": round(percentile(arr, 95), 3),
        "mean": round(float(arr.mean()), 3),
        "min": round(float(arr.min()), 3),
    }


def pad_to(ids: Sequence[int], length: int, pad_id: int) -> tuple[list[int], list[int]]:
    """(ids, attention_mask) of exactly ``length``: truncated, or padded on the right with
    mask 0."""
    if length < 1:
        raise ValueError("length must be positive")
    real = list(ids)[:length]
    mask = [1] * len(real) + [0] * (length - len(real))
    return real + [pad_id] * (length - len(real)), mask


def attempt_record(exporter: str, dynamo: bool, error: BaseException | None) -> dict[str, Any]:
    if error is None:
        return {
            "exporter": exporter,
            "dynamo": dynamo,
            "ok": True,
            "error_type": None,
            "error_message": None,
        }
    return {
        "exporter": exporter,
        "dynamo": dynamo,
        "ok": False,
        "error_type": type(error).__name__,
        "error_message": str(error)[:2000],
    }


def build_export_doc(
    *,
    status: str,
    teacher_sha: str,
    exporter: str | None,
    attempts: list[dict[str, Any]],
    opset: int | None,
    ir_version: int | None,
    max_len: int,
    files: dict[str, dict[str, Any]],
    quantisation: dict[str, Any] | None,
    parity: dict[str, Any] | None,
    latency_ms: dict[str, Any] | None,
    versions: dict[str, Any],
    n_strings: int,
    notes: list[str],
) -> dict[str, Any]:
    doc = {
        "job": JOB,
        "status": status,
        "teacher_sha256": teacher_sha,
        "exporter": exporter,
        "export_attempts": attempts,
        "opset": opset,
        "ir_version": ir_version,
        "max_len": max_len,
        "files": files,
        "quantisation": quantisation,
        "parity": parity,
        "latency_ms": latency_ms,
        "parity_strings": n_strings,
        "versions": versions,
        "notes": notes,
    }
    missing = [k for k in REQUIRED_KEYS if k not in doc]
    if missing:  # programming error, never a data error
        raise ValueError(f"export doc lacks {missing}")
    return doc


def failed_export_doc(
    *, teacher_sha: str, attempts: list[dict[str, Any]], versions: dict[str, Any], max_len: int
) -> dict[str, Any]:
    """The export.json written when no exporter works: status export_failed, every attempt with
    its exception type and message, no files or numbers."""
    return build_export_doc(
        status=EXPORT_FAILED,
        teacher_sha=teacher_sha,
        exporter=None,
        attempts=attempts,
        opset=None,
        ir_version=None,
        max_len=max_len,
        files={},
        quantisation=None,
        parity=None,
        latency_ms=None,
        versions=versions,
        n_strings=len(SMS_STRINGS),
        notes=["no exporter produced an ONNX file; see export_attempts"],
    )


def render_report(doc: dict[str, Any]) -> str:
    """report.md: the numbers from export.json in a short table."""
    lines = [f"# {JOB}", "", f"status: {doc['status']}", f"teacher sha256: {doc['teacher_sha256']}"]
    lines += ["", "## export attempts"]
    for a in doc["export_attempts"]:
        if a["ok"]:
            lines.append(f"- {a['exporter']}: ok")
        else:
            lines.append(f"- {a['exporter']}: {a['error_type']}: {a['error_message']}")
    if doc["status"] == EXPORT_FAILED:
        return "\n".join(lines) + "\n"
    lines += [
        "",
        f"exporter: {doc['exporter']}, opset {doc['opset']}, ir_version {doc['ir_version']}",
    ]
    lines += ["", "## files", "| file | bytes | sha256 |", "| --- | ---: | --- |"]
    for name, info in doc["files"].items():
        lines.append(f"| {name} | {info['bytes']} | {info['sha256'][:16]} |")
    q = doc["quantisation"] or {}
    lines += ["", f"quantised op types: {q.get('quantised_op_types')}"]
    p = doc["parity"] or {}
    if p:
        lines += ["", "## parity (PyTorch fp32 reference)"]
        for key, value in p.items():
            lines.append(f"- {key}: {value}")
    lat = doc["latency_ms"] or {}
    if lat:
        lines += ["", "## reference latency (Kaggle CPU, 4 threads, batch 1, ms)"]
        for variant, by_len in lat.items():
            for seq, summary in by_len.items():
                lines.append(f"- {variant} seq {seq}: p50 {summary['p50']}, p95 {summary['p95']}")
    lines += ["", "## notes"] + [f"- {n}" for n in doc["notes"]]
    return "\n".join(lines) + "\n"


# ---- torch and onnxruntime side (imported inside functions) ----


def locate_teacher_folder(root: Path, sha: str) -> Path:
    from rakshak.data.retrain_prep import locate_pinned

    weights = locate_pinned(root, WEIGHTS, sha)
    folder = weights.parent
    if not (folder / "config.json").is_file() or not (folder / "tokenizer.json").is_file():
        raise FileNotFoundError("the pinned checkpoint folder lacks config.json or tokenizer.json")
    return folder


def _make_wrapper(model):
    import torch

    class TeacherLogits(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

    return TeacherLogits(model).eval()


def _try_export(wrapper, dummy, path: Path, dynamo: bool) -> None:
    import torch

    names = {"input_names": ["input_ids", "attention_mask"], "output_names": ["logits"]}
    params = inspect.signature(torch.onnx.export).parameters
    kwargs: dict[str, Any] = {**names, "opset_version": OPSET}
    if dynamo:
        batch = torch.export.Dim("batch")
        seq = torch.export.Dim("sequence", min=1)
        kwargs["dynamic_shapes"] = ({0: batch, 1: seq}, {0: batch, 1: seq})
        if "external_data" in params:
            kwargs["external_data"] = False
        torch.onnx.export(wrapper, dummy, str(path), dynamo=True, **kwargs)
    else:
        kwargs["dynamic_axes"] = {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        }
        torch.onnx.export(wrapper, dummy, str(path), dynamo=False, **kwargs)


def export_teacher(model, work: Path, attempts: list[dict[str, Any]]) -> tuple[str | None, Path]:
    """Try torch.onnx.export, then the dynamo exporter. Returns (exporter name or None, path)."""
    import torch

    path = work / "teacher_fp32.onnx"
    wrapper = _make_wrapper(model)
    dummy = (
        torch.ones((2, 16), dtype=torch.long),
        torch.ones((2, 16), dtype=torch.long),
    )
    for dynamo, name in ((False, "torch.onnx.export"), (True, "torch.onnx.export(dynamo=True)")):
        try:
            with torch.no_grad():
                _try_export(wrapper, dummy, path, dynamo)
        except Exception as exc:  # every failure is an attempt record, not a crash
            attempts.append(attempt_record(name, dynamo, exc))
            if path.exists():
                path.unlink()
            continue
        attempts.append(attempt_record(name, dynamo, None))
        return name, path
    return None, path


def onnx_summary(path: Path) -> dict[str, Any]:
    """opset, ir_version and the op type counts of the graph (weights not loaded)."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    opset = max(
        (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), default=None
    )
    return {"opset": opset, "ir_version": int(model.ir_version), "op_counts": counts}


def quantise(fp32: Path, int8: Path, fp32_counts: dict[str, int]) -> dict[str, Any]:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    info: dict[str, Any] = {
        "method": "onnxruntime.quantization.quantize_dynamic",
        "weight_type": "QInt8",
        "requested_op_types": list(QUANT_OP_TYPES),
        "quantised_op_types": [],
        "fp32_op_counts": {k: fp32_counts.get(k, 0) for k in QUANT_OP_TYPES},
        "int8_op_counts": {},
        "error_type": None,
        "error_message": None,
    }
    try:
        quantize_dynamic(
            str(fp32),
            str(int8),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=list(QUANT_OP_TYPES),
        )
    except Exception as exc:  # fp32 only, with the error recorded
        info["error_type"] = type(exc).__name__
        info["error_message"] = str(exc)[:2000]
        if int8.exists():
            int8.unlink()
        return info
    int8_counts = onnx_summary(int8)["op_counts"]
    info["int8_op_counts"] = dict(sorted(int8_counts.items()))
    # an op type counts as quantised when the graph lost some of its fp32 nodes of that type
    info["quantised_op_types"] = [
        op for op in QUANT_OP_TYPES if info["fp32_op_counts"][op] > int8_counts.get(op, 0)
    ]
    return info


def _session(path: Path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _ort_logits(session, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = session.run(
        None, {"input_ids": ids.astype(np.int64), "attention_mask": mask.astype(np.int64)}
    )
    return np.asarray(out[0], dtype=np.float32)


def parity_and_latency(model, tokenizer, paths: dict[str, Path], ctx) -> tuple[dict, dict]:
    import torch

    from rakshak.eval import data as D

    texts = [D.canon_text(t) for t in SMS_STRINGS]
    enc = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="np")
    ids = np.asarray(enc["input_ids"], dtype=np.int64)
    mask = np.asarray(enc["attention_mask"], dtype=np.int64)
    with torch.no_grad():
        ref = model(input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(mask))
        ref_logits = ref.logits.float().numpy()
    sessions = {name: _session(path) for name, path in paths.items()}
    got = {name: _ort_logits(sess, ids, mask) for name, sess in sessions.items()}
    parity: dict[str, Any] = {
        "reference": "pytorch fp32 (cpu)",
        "fp32_onnx_vs_pytorch": logit_parity(ref_logits, got["fp32"]),
    }
    if "int8" in got:
        parity["int8_onnx_vs_pytorch"] = probability_parity(ref_logits, got["int8"])
        parity["int8_onnx_vs_fp32_onnx"] = probability_parity(got["fp32"], got["int8"])
    ctx.log(f"parity: {parity}")

    # latency: a real token stream cut to each length, so every timed call has exactly L tokens
    pool = [int(x) for row, m in zip(ids, mask, strict=False) for x in row[m.astype(bool)]]
    latency: dict[str, dict[str, Any]] = {}
    for name, sess in sessions.items():
        latency[name] = {}
        for length in LAT_SEQ_LENS:
            tok, msk = pad_to(pool, length, 0)
            feed_ids = np.asarray([tok], dtype=np.int64)
            feed_mask = np.asarray([msk], dtype=np.int64)
            for _ in range(LAT_WARMUP):
                _ort_logits(sess, feed_ids, feed_mask)
            samples = []
            for _ in range(LAT_RUNS):
                start = time.perf_counter()
                _ort_logits(sess, feed_ids, feed_mask)
                samples.append((time.perf_counter() - start) * 1000.0)
            latency[name][str(length)] = latency_summary(samples)
    ctx.log(f"latency: {latency}")
    return parity, latency


def run(ctx) -> bool | str:
    args = parse_args(ctx.argv)
    ctx.log(f"env: {env_report()}")
    versions = package_versions()
    ctx.log(f"versions: {versions}")
    work = Path(ctx.work)
    timings = ctx.results.setdefault("timings", {})

    with timed("locate_teacher", timings):
        folder = locate_teacher_folder(Path(args.teacher_dir), args.teacher_sha)
    ctx.log(f"teacher folder found with the pinned sha256 {args.teacher_sha[:12]}")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(THREADS)
    tokenizer = AutoTokenizer.from_pretrained(str(folder))
    model = AutoModelForSequenceClassification.from_pretrained(str(folder)).float().eval()
    attempts: list[dict[str, Any]] = []
    with timed("export", timings):
        exporter, fp32_path = export_teacher(model, work, attempts)
    ctx.save()

    if exporter is None:
        doc = failed_export_doc(
            teacher_sha=args.teacher_sha, attempts=attempts, versions=versions, max_len=MAX_LEN
        )
        (work / "export.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        (work / "report.md").write_text(render_report(doc))
        ctx.results["summary"] = f"{EXPORT_FAILED}: no exporter worked ({len(attempts)} attempts)"
        ctx.log(ctx.results["summary"])
        return "fail"

    summary = onnx_summary(fp32_path)
    ctx.log(f"exported with {exporter}: opset {summary['opset']}, ir {summary['ir_version']}")
    int8_path = work / "teacher_int8.onnx"
    with timed("quantise", timings):
        quant = quantise(fp32_path, int8_path, summary["op_counts"])
    ctx.log(f"quantisation: {quant['quantised_op_types']} error {quant['error_type']}")

    paths = {"fp32": fp32_path}
    if int8_path.exists():
        paths["int8"] = int8_path
    with timed("parity_latency", timings):
        parity, latency = parity_and_latency(model, tokenizer, paths, ctx)

    files: dict[str, dict[str, Any]] = {}
    for path in paths.values():
        files[path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    tok_files = {}
    for name in TOKENIZER_FILES:
        src = folder / name
        if src.is_file():
            shutil.copy2(src, work / name)
            tok_files[name] = {
                "bytes": (work / name).stat().st_size,
                "sha256": file_sha256(work / name),
            }
    files.update(tok_files)

    notes = [
        "latency is the Kaggle CPU for reference only; it is not the Raspberry Pi number",
        "the INT8 parity is against PyTorch fp32; the fp32 ONNX comparison is extra",
    ]
    if "int8" not in paths:
        notes.append(f"INT8 not written: {quant['error_type']}")
    doc = build_export_doc(
        status="ok" if "int8" in paths else "fp32_only",
        teacher_sha=args.teacher_sha,
        exporter=exporter,
        attempts=attempts,
        opset=summary["opset"],
        ir_version=summary["ir_version"],
        max_len=MAX_LEN,
        files=files,
        quantisation=quant,
        parity=parity,
        latency_ms=latency,
        versions=versions,
        n_strings=len(SMS_STRINGS),
        notes=notes,
    )
    (work / "export.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    (work / "report.md").write_text(render_report(doc))
    fp = parity["fp32_onnx_vs_pytorch"]["max_abs_logit_diff"]
    ctx.results["summary"] = (
        f"export {doc['status']}: {exporter}, opset {OPSET}, fp32 max|dlogit| {fp:.2e}, "
        f"int8 ops {quant['quantised_op_types']}"
    )
    return True
