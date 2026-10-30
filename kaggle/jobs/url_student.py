"""URL student models on PhreshPhish, trained on a Kaggle T4 GPU (CPU fallback).

Models (one unit per split x model x seed):
- charcnn: byte-level char-CNN (rakshak.models.charcnn, one raw view, max_len 200). The URL feature
  vector (rakshak.url_features columns) is the side input; ``--no-side`` drops it.
- transformer: byte-level transformer teacher (rakshak.models.url_nets.ByteTransformer), 4 to 6
  layers, d_model 256 by default.
- charcnn-kd (``--kd``): the same char-CNN distilled from the transformer's logits on the training
  rows (soft BCE at temperature T, weight alpha). A transformer unit is trained first when it is
  not already in ``--models``.

Splits:
- phreshphish: the official PhreshPhish train split. A ``--cal-fraction`` cal set is carved from it
  with whole registered domains on one side (rakshak.eval.phreshphish.grouped_cal). The models are
  scored on the official test split with the PhreshPhish protocol (AP and P@R=0.90 over the 975
  benchmark subsets, averaged per base rate). The protocol takes no threshold; cal only selects
  the early-stopping checkpoint.
- url_domain: the url_domain split of urls.parquet (train, cal, test as in url_baselines), scored
  with rakshak.eval.protocol.evaluate_scores (threshold on cal, campaign-clustered CIs).

Training: class-weighted BCE (pos_weight = n_neg / n_pos on the training rows), AdamW, fp16 autocast
on CUDA, early stop on cal PR-AUC with the best cal checkpoint kept. ``--max-train`` takes a uniform
subsample of the training rows (after the cal carve) for speed. ``--budget-h`` is a deadline for the
whole job: training stops cleanly at it, the best cal checkpoint so far is kept, and later units are
recorded as skipped.

Caveat: the transformer's distillation targets are in-sample logits (the teacher is trained on the
same rows). Cross-fitted targets would need one teacher per fold and are not implemented here.

Size and speed: fp32 size is the torch.save size of the state dict. INT8 is dynamic quantisation of
the nn.Linear layers only (Conv1d and Embedding stay fp32), with its size and its AP delta on a
fixed sample of the PhreshPhish benchmark rows. CPU latency is one thread per URL, including the
offline URL features, the standardisation and the forward pass.

Flags go through ``kaggle/launch.sh url_student --job-args="..."``:
  --data-root PATH           data root that holds urls_phreshphish.parquet (and urls.parquet)
  --splits GLOB...           url_domain split-file stems (default url_domain)
  --no-url-domain            skip the url_domain split
  --no-phreshphish           skip the PhreshPhish split
  --models {charcnn,transformer}...   (default both)
  --kd                       add the distilled char-CNN (charcnn-kd)
  --side / --no-side         URL feature side input for the char-CNN (default on)
  --cnn-size {base,wide}     char-CNN config (default wide)
  --tf-layers INT            transformer layers (default 4)
  --tf-dim INT               transformer width (default 256)
  --tf-heads INT             transformer heads (default 4)
  --max-len INT              byte length of a URL (default 200)
  --seeds INT...             model seeds (default 0)
  --seed INT                 cal carve and subsample seed (default 0)
  --epochs INT               max epochs (default 8)
  --batch-size INT           (default 512)
  --lr FLOAT                 AdamW learning rate (default 2e-3)
  --weight-decay FLOAT       (default 1e-4)
  --patience INT             early-stop patience on cal PR-AUC (default 3)
  --cal-fraction FLOAT       PhreshPhish cal share of the train split (default 0.05)
  --max-train INT            training rows per split after the cal carve (default all)
  --kd-alpha FLOAT           distillation weight (default 0.5)
  --kd-temp FLOAT            distillation temperature (default 2.0)
  --budget-h FLOAT           deadline for the whole job, in hours (default none)
  --device {auto,cpu,cuda}   (default auto)
  --no-amp                   fp32 training on CUDA
  --no-export                skip ONNX export
  --latency-n INT            URLs timed for CPU latency, one thread (default 300)
  --int8-eval-n INT          benchmark rows for the INT8 AP delta (default 5000)
  --n-boot INT               bootstrap resamples for url_domain (default 1000)

Outputs in /kaggle/working: metrics.json and report.md (protocol tables, model sizes and latency,
url_domain tables), logits_<split>_test.parquet (test logits per unit), best_url_student.onnx and
url_side_scaler.json (the side-feature standardiser the ONNX model expects).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from _common import (
    Units,
    env_report,
    log_input_tree,
    record_skip,
    seed_everything,
    single_class_reason,
)
from torch import nn

from rakshak.eval import data as D
from rakshak.eval import report
from rakshak.eval.metrics import pr_auc
from rakshak.eval.phreshphish import (
    PHRESH_FILE,
    grouped_cal,
    load_phreshphish,
    paper_reference,
    protocol_section,
    protocol_tables,
    subsample_rows,
    subset_metrics,
)
from rakshak.eval.protocol import evaluate_scores
from rakshak.models.charcnn import CONFIGS, CharCNN, OnnxWrapper
from rakshak.models.encoding import (
    apply_standardizer,
    encode_batch,
    encode_bytes,
    fit_standardizer,
)
from rakshak.models.student import latency_ms, param_count, pick_device, state_bytes
from rakshak.models.url_nets import ByteTransformer
from rakshak.url_features import url_features

CLUSTER_COLUMN = "reg_domain"
ONNX_NAME = "best_url_student.onnx"
SCALER_NAME = "url_side_scaler.json"
KD_SUFFIX = "-kd"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="url_student")
    p.add_argument("--data-root", default=None)
    p.add_argument("--splits", nargs="*", default=["url_domain"])
    p.add_argument("--no-url-domain", action="store_true")
    p.add_argument("--no-phreshphish", action="store_true")
    p.add_argument("--models", nargs="*", default=["charcnn", "transformer"],
                   choices=["charcnn", "transformer"])
    p.add_argument("--kd", action="store_true")
    p.add_argument("--side", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cnn-size", default="wide", choices=list(CONFIGS))
    p.add_argument("--tf-layers", type=int, default=4)
    p.add_argument("--tf-dim", type=int, default=256)
    p.add_argument("--tf-heads", type=int, default=4)
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--seeds", nargs="*", type=int, default=[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--cal-fraction", type=float, default=0.05)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--kd-alpha", type=float, default=0.5)
    p.add_argument("--kd-temp", type=float, default=2.0)
    p.add_argument("--budget-h", type=float, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--latency-n", type=int, default=300)
    p.add_argument("--int8-eval-n", type=int, default=5000)
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args(argv or [])
    if not 1 <= args.tf_layers <= 12:
        p.error("--tf-layers must be in [1, 12]")
    if args.budget_h is not None and args.budget_h <= 0:
        p.error("--budget-h must be positive")
    if not 0.0 < args.cal_fraction < 1.0:
        p.error("--cal-fraction must be in (0, 1)")
    return args


@dataclass(frozen=True)
class Spec:
    kind: str  # charcnn | transformer
    key: str  # model key used in the tables and the logits columns
    side: bool  # URL feature side input (char-CNN only)
    kd: bool  # distilled from the transformer teacher


@dataclass
class Dataset:
    """One split as the models see it. Row positions index ``ids``, ``labels`` and ``side``."""

    name: str
    ids: np.ndarray  # uint16 (n_rows, max_len) byte ids, PAD_ID = 256
    labels: np.ndarray  # int64 (n_rows,)
    row_ids: np.ndarray  # str (n_rows,)
    fit: np.ndarray  # training positions (after the cal carve and any subsample)
    cal: np.ndarray
    test: np.ndarray
    side: np.ndarray  # float32 (n_rows, n_side), standardised with the fit rows
    side_cols: list[str]
    side_stats: tuple[np.ndarray, np.ndarray]
    urls: np.ndarray  # str (n_rows,), for the CPU latency timing
    protocol: dict | None = None  # PhreshPhish: bench_full and subsets; None for run splits
    groups: dict = field(default_factory=dict)  # run splits: test groups for the breakdown
    clusters: list | None = None  # run splits: test campaign ids for the bootstrap


def model_keys(args) -> dict[str, str]:
    cnn = f"charcnn-{args.cnn_size}" + ("" if args.side else "-noside")
    return {"charcnn": cnn, "transformer": f"transformer-L{args.tf_layers}-d{args.tf_dim}"}


def unit_specs(args) -> list[Spec]:
    keys = model_keys(args)
    specs: list[Spec] = []
    if "transformer" in args.models or args.kd:
        specs.append(Spec("transformer", keys["transformer"], False, False))
    if "charcnn" in args.models:
        specs.append(Spec("charcnn", keys["charcnn"], args.side, False))
    if args.kd:
        specs.append(Spec("charcnn", keys["charcnn"] + KD_SUFFIX, args.side, True))
    return specs


def make_model(spec: Spec, args, n_side: int) -> nn.Module:
    if spec.kind == "charcnn":
        return CharCNN(CONFIGS[args.cnn_size], views=("raw",), n_side=n_side)
    return ByteTransformer(max_len=args.max_len, d_model=args.tf_dim, layers=args.tf_layers,
                           heads=args.tf_heads)


def forward(model: nn.Module, kind: str, x: torch.Tensor, side: torch.Tensor | None):
    if kind == "transformer":
        return model(x)
    return model({"raw": x}, side)


@torch.inference_mode()
def predict_logits(model, kind, ids, side, rows, device, use_side, batch: int = 1024):
    """Logits (float64) for ``rows`` in order. Eval mode, no gradients."""
    model.eval()
    out = np.zeros(len(rows), dtype=np.float64)
    for start in range(0, len(rows), batch):
        idx = rows[start:start + batch]
        x = torch.as_tensor(ids[idx].astype(np.int64), device=device)
        sd = torch.as_tensor(side[idx], device=device) if use_side else None
        out[start:start + len(idx)] = forward(model, kind, x, sd).double().cpu().numpy()
    return out


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def _cpu_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@dataclass
class TrainCfg:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    patience: int
    seed: int
    kd_alpha: float
    kd_temp: float
    amp: bool


def train_model(model, kind, ds: Dataset, cfg: TrainCfg, device, use_side, deadline,
                teacher_logits: np.ndarray | None, log) -> dict:
    """Fit on ``ds.fit``, early-stop on cal PR-AUC, keep the best cal checkpoint. Stops at the
    deadline (checked before every batch). ``teacher_logits`` aligns with ``ds.fit``."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    y_fit = ds.labels[ds.fit]
    n_pos = int(y_fit.sum())
    n_neg = int(len(y_fit) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"{ds.name}: the training rows need both classes")
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    y_cal = ds.labels[ds.cal]
    n = len(ds.fit)
    best_ap, best_state, best_epoch, stale = -np.inf, None, -1, 0
    history: list[dict] = []
    stopped = None
    start = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(n)
        loss_sum, batches = 0.0, 0
        for s0 in range(0, n, cfg.batch_size):
            if deadline is not None and time.monotonic() >= deadline:
                stopped = "budget"
                break
            pick = order[s0:s0 + cfg.batch_size]
            rows = ds.fit[pick]
            x = torch.as_tensor(ds.ids[rows].astype(np.int64), device=device)
            sd = torch.as_tensor(ds.side[rows], device=device) if use_side else None
            yb = torch.as_tensor(ds.labels[rows], dtype=torch.float32, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.amp):
                logits = forward(model, kind, x, sd)
            logits = logits.float()
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            if teacher_logits is not None:
                t = torch.as_tensor(teacher_logits[pick], dtype=torch.float32, device=device)
                temp = cfg.kd_temp
                kd = F.binary_cross_entropy_with_logits(
                    logits / temp, torch.sigmoid(t / temp)) * temp * temp
                loss = (1.0 - cfg.kd_alpha) * loss + cfg.kd_alpha * kd
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.detach())
            batches += 1
        cal_ap = float(pr_auc(y_cal, sigmoid(predict_logits(
            model, kind, ds.ids, ds.side, ds.cal, device, use_side))))
        history.append({"epoch": epoch, "train_loss": loss_sum / max(batches, 1),
                        "cal_pr_auc": cal_ap, "batches": batches})
        log(f"    epoch {epoch}: loss {loss_sum / max(batches, 1):.4f} cal PR-AUC {cal_ap:.4f}")
        if np.isfinite(cal_ap) and cal_ap > best_ap:
            best_ap, best_state, best_epoch, stale = cal_ap, _cpu_state(model), epoch, 0
        else:
            stale += 1
        if stopped or stale >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_cal_pr_auc": float(best_ap) if best_state is not None else None,
        "epochs_run": len(history),
        "stopped": stopped or ("early_stop" if len(history) < cfg.epochs else "max_epochs"),
        "seconds": round(time.perf_counter() - start, 2),
        "history": history,
    }


def run_unit(ds: Dataset, spec: Spec, seed: int, args, device, deadline, state: dict, log):
    """One (split, model, seed): train, score cal and test, keep what the report needs."""
    use_side = spec.side and ds.side.shape[1] > 0
    n_side = ds.side.shape[1] if use_side else 0
    teacher_logits = None
    if spec.kd:
        teacher_logits = state["teacher"].get((ds.name, seed))
        if teacher_logits is None:
            raise RuntimeError(f"no transformer teacher logits for {ds.name} seed {seed}")
    model = make_model(spec, args, n_side)
    cfg = TrainCfg(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                   weight_decay=args.weight_decay, patience=args.patience, seed=seed,
                   kd_alpha=args.kd_alpha, kd_temp=args.kd_temp, amp=device.type == "cuda"
                   and not args.no_amp)
    info = train_model(model, spec.kind, ds, cfg, device, use_side, deadline, teacher_logits, log)
    if spec.kind == "transformer":
        state["teacher"][(ds.name, seed)] = predict_logits(
            model, spec.kind, ds.ids, ds.side, ds.fit, device, False)
    cal_logits = predict_logits(model, spec.kind, ds.ids, ds.side, ds.cal, device, use_side)
    test_logits = predict_logits(model, spec.kind, ds.ids, ds.side, ds.test, device, use_side)
    s_cal, s_test = sigmoid(cal_logits), sigmoid(test_logits)
    model = model.cpu().eval()
    if ds.protocol is not None:
        scores = np.full(len(ds.labels), np.nan)
        scores[ds.test] = s_test
        result = protocol_section(ds.labels, scores, ds.test, ds.protocol["bench_full"],
                                  ds.protocol["subsets"])
        result["cal_pr_auc"] = float(pr_auc(ds.labels[ds.cal], s_cal))
    else:
        result = evaluate_scores(ds.labels[ds.cal], s_cal, ds.labels[ds.test], s_test,
                                 groups=ds.groups, n_boot=args.n_boot, seed=args.seed,
                                 cluster_ids=ds.clusters)
    params = param_count(model)
    info_out = {"spec": spec.__dict__ | {"use_side": use_side}, "params": params,
                "fp32_bytes": state_bytes(model), "train": info}
    return {"result": result, "info": info_out, "test_logits": test_logits, "model": model}


def build_phreshphish(root: Path, args, ids_all_fn) -> Dataset:
    data = load_phreshphish(root)
    df = data.df
    url = df["url"].astype(str).to_numpy(dtype=object)
    reg = (df["reg_domain"].fillna("").astype(str).to_numpy(dtype=object)
           if "reg_domain" in df.columns else np.full(len(df), "", dtype=object))
    norm = (df["url_norm"].astype(str).to_numpy(dtype=object)
            if "url_norm" in df.columns else url)
    group_key = np.where(reg != "", reg, norm)
    keep, cal = grouped_cal(data.train, data.labels, group_key[data.train], args.cal_fraction,
                            args.seed)
    fit = subsample_rows(keep, args.max_train, args.seed)
    side_all = df[data.feature_cols].fillna(0.0).to_numpy(np.float32)
    mean, std = fit_standardizer(side_all, fit)
    return Dataset(
        name="phreshphish",
        ids=ids_all_fn(list(url)),
        labels=data.labels,
        row_ids=df["id"].astype(str).to_numpy(dtype=object),
        fit=fit,
        cal=cal,
        test=data.test,
        side=apply_standardizer(side_all, mean, std),
        side_cols=list(data.feature_cols),
        side_stats=(mean, std),
        urls=url,
        protocol={"bench_full": data.bench_full, "subsets": data.subsets},
    )


def build_run_datasets(root: Path, args, ids_all_fn, ctx) -> list[Dataset]:
    if not (root / D.URL_FILE).is_file():
        record_skip(ctx, "url_domain", f"{D.URL_FILE} not found under {root}")
        return []
    urls = D.load_urls(root)
    labels = urls["label"].to_numpy().astype(np.int64)
    try:
        runs = D.load_runs(root, urls, patterns=args.splits, labels=labels, mask=None)
    except FileNotFoundError as exc:
        record_skip(ctx, "url_domain", str(exc))
        return []
    feature_cols = D.url_feature_columns(urls)
    side_all = urls[feature_cols].fillna(0.0).to_numpy(np.float32)
    url = urls["url"].astype(str).to_numpy(dtype=object)
    ids = ids_all_fn(list(url))
    out: list[Dataset] = []
    for run_ in runs:
        reason = single_class_reason(labels[run_.test], run_.name)
        if reason:
            record_skip(ctx, run_.name, reason)
            continue
        fit = subsample_rows(run_.train, args.max_train, args.seed)
        mean, std = fit_standardizer(side_all, fit)
        out.append(Dataset(
            name=run_.name,
            ids=ids,
            labels=labels,
            row_ids=urls["id"].astype(str).to_numpy(dtype=object),
            fit=fit,
            cal=run_.cal,
            test=run_.test,
            side=apply_standardizer(side_all, mean, std),
            side_cols=list(feature_cols),
            side_stats=(mean, std),
            urls=url,
            groups=D.group_values(urls, run_.test, columns=("source",)),
            clusters=D.cluster_values(urls, run_.test, CLUSTER_COLUMN),
        ))
    return out


def export_onnx(model: CharCNN, path: Path, n_side: int, max_len: int, opset: int = 17) -> dict:
    """Float char-CNN to ONNX with a dynamic batch. Inputs: ``raw`` int64 (batch, max_len) and,
    when used, ``side`` float32 (batch, n_side). Output ``logit`` (batch,); probability =
    sigmoid(logit)."""
    wrapper = OnnxWrapper(model).cpu().eval()
    names = ["raw"] + (["side"] if n_side else [])
    dummies: list[torch.Tensor] = [torch.zeros(2, max_len, dtype=torch.long)]
    if n_side:
        dummies.append(torch.zeros(2, n_side, dtype=torch.float32))
    dynamic = {name: {0: "batch"} for name in names}
    dynamic["logit"] = {0: "batch"}
    kwargs = {"input_names": names, "output_names": ["logit"], "dynamic_axes": dynamic,
              "opset_version": opset}
    with torch.no_grad():
        try:
            torch.onnx.export(wrapper, tuple(dummies), str(path), dynamo=False, **kwargs)
        except TypeError:  # torch without the dynamo switch
            torch.onnx.export(wrapper, tuple(dummies), str(path), **kwargs)
    return {"path": str(path), "bytes": path.stat().st_size, "inputs": names, "opset": opset}


def onnx_parity(model: CharCNN, path: Path, ds: Dataset, n_side: int, n: int = 64) -> dict:
    """Max absolute logit difference between torch and onnxruntime. Skipped when onnxruntime is
    not importable."""
    try:
        import onnxruntime as ort
    except ImportError:
        return {"skipped": "onnxruntime not importable"}
    rows = ds.test[: min(n, len(ds.test))]
    feed = {"raw": ds.ids[rows].astype(np.int64)}
    if n_side:
        feed["side"] = np.ascontiguousarray(ds.side[rows], dtype=np.float32)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = session.run(None, {i.name: feed[i.name] for i in session.get_inputs()})[0]
    ref = predict_logits(model.cpu().eval(), "charcnn", ds.ids, ds.side, rows, "cpu", n_side > 0)
    return {"n": int(len(rows)), "max_abs_logit_diff": float(np.max(np.abs(got - ref)))}


def int8_copy(model: nn.Module) -> nn.Module:
    """Dynamic INT8 for nn.Linear. The fp32 model is left as it is."""
    from torch.ao.quantization import quantize_dynamic

    return quantize_dynamic(copy.deepcopy(model).cpu().eval(), {nn.Linear}, dtype=torch.qint8)


def cpu_score_fn(model, kind: str, max_len: int, use_side: bool, stats, cols):
    """Per-URL scorer for the latency timing: offline features, standardisation, forward pass."""
    mean, std = stats

    def score(text: str) -> float:
        x = torch.as_tensor(encode_bytes(text, max_len)[None])
        sd = None
        if use_side:
            feats = url_features(text)
            vec = np.asarray([[float(feats.get(c, 0.0)) for c in cols]], dtype=np.float32)
            sd = torch.as_tensor(apply_standardizer(vec, mean, std))
        with torch.inference_mode():
            return float(sigmoid(forward(model, kind, x, sd).double().numpy())[0])

    return score


def protocol_int8_check(fp32, int8, ds: Dataset, n_eval: int, seed: int, use_side: bool,
                        kind: str) -> dict:
    """AP and P@R=0.90 on a fixed sample of the 24% benchmark, fp32 and INT8, both on CPU."""
    bench = ds.protocol["bench_full"]
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(bench, size=min(n_eval, len(bench)), replace=False))
    y = ds.labels[sample]
    p32 = sigmoid(predict_logits(fp32, kind, ds.ids, ds.side, sample, "cpu", use_side))
    p8 = sigmoid(predict_logits(int8, kind, ds.ids, ds.side, sample, "cpu", use_side))
    m32, m8 = subset_metrics(y, p32), subset_metrics(y, p8)
    return {
        "rows": int(len(sample)),
        "fp32_ap": m32["ap"],
        "int8_ap": m8["ap"],
        "ap_delta": float(m8["ap"] - m32["ap"]),
        "fp32_p_at_r90": m32["p_at_r90"],
        "int8_p_at_r90": m8["p_at_r90"],
        "p_at_r90_delta": float(m8["p_at_r90"] - m32["p_at_r90"]),
    }


def size_rows(state: dict, args, phresh: Dataset | None) -> list[list[str]]:
    rows: list[list[str]] = []
    for key, entry in state["kept"].items():
        model, kind, use_side = entry["model"], entry["kind"], entry["use_side"]
        cols = phresh.side_cols
        n = min(args.latency_n, len(phresh.test))
        texts = [phresh.urls[i] for i in phresh.test[:n]]
        fp32_stats = None
        if n:
            fp32_stats = latency_ms(cpu_score_fn(model, kind, args.max_len, use_side,
                                                 phresh.side_stats, cols), texts, n)
        int8 = entry.get("int8")
        q_stats = None
        if n and int8 is not None:
            q_stats = latency_ms(cpu_score_fn(int8["model"], kind, args.max_len, use_side,
                                              phresh.side_stats, cols), texts, n)
        state["sizes"][key] = {
            "params": entry["params"],
            "fp32_bytes": entry["fp32_bytes"],
            "int8_bytes": int8["bytes"] if int8 else None,
            "int8_check": int8["check"] if int8 else None,
            "latency_fp32": fp32_stats,
            "latency_int8": q_stats,
        }
        s = state["sizes"][key]
        rows.append([
            key,
            f"{s['params']:,}",
            f"{s['fp32_bytes'] / 1024:.1f}",
            "n/a" if s["int8_bytes"] is None else f"{s['int8_bytes'] / 1024:.1f}",
            "n/a" if not int8 else f"{int8['check']['ap_delta']:+.4f}",
            "n/a" if fp32_stats is None else f"{fp32_stats['p50_ms']:.2f}",
            "n/a" if fp32_stats is None else f"{fp32_stats['p95_ms']:.2f}",
            "n/a" if q_stats is None else f"{q_stats['p50_ms']:.2f}",
        ])
    return rows


SIZE_HEADER = [
    "Model",
    "Params",
    "fp32 size (KB)",
    "INT8 size (KB)",
    "INT8 AP delta (benchmark sample)",
    "CPU p50 ms (fp32, 1 thread)",
    "CPU p95 ms (fp32, 1 thread)",
    "CPU p50 ms (INT8, 1 thread)",
]


def _tables(state: dict, args, phresh: Dataset | None) -> list[dict]:
    tables: list[dict] = []
    if phresh is not None and state["phresh"]:
        section = {"paper_v2_gte_reference": paper_reference(), "models": state["phresh"]}
        tables += protocol_tables(section)
    if state["sizes"]:
        tables.append({
            "title": "Model size and CPU latency (PhreshPhish, seed of the first unit)",
            "note": ("Size is the torch.save size of the state dict; INT8 is dynamic "
                     "quantisation of the nn.Linear layers only. Latency is one thread per URL, "
                     "including the offline URL features and the standardisation. AP delta is "
                     "fp32 minus INT8 on a fixed benchmark sample."),
            "header": SIZE_HEADER,
            "rows": state["size_rows"],
        })
    return tables


def _checkpoint(ctx, meta: dict, state: dict, phresh: Dataset | None, args) -> None:
    payload = {
        "title": "URL student models",
        "meta": meta,
        "splits": state["splits"],
        "tables": _tables(state, args, phresh),
        "phreshphish": {"models": state["phresh"], "units": state["unit_info"]},
        "models": state["sizes"],
    }
    report.write_report(payload, ctx.work)


def _write_logits(work: Path, state: dict) -> dict[str, str]:
    written: dict[str, str] = {}
    for split, cols in state["logits"].items():
        if not cols:
            continue
        first = next(iter(cols.values()))
        frame = pd.DataFrame({"id": first["id"], "label": first["label"]})
        for col, entry in cols.items():
            frame[col] = entry["logit"]
        stem = split.replace(":", "_")
        path = work / f"logits_{stem}_test.parquet"
        frame.to_parquet(path, index=False)
        written[split] = path.name
    return written


def run(ctx) -> bool:
    args = parse_args(ctx.argv)
    seed_everything(args.seed)
    device = pick_device(args.device)
    env = env_report()
    env["device"] = str(device)
    ctx.log(f"env: {env}")
    deadline = time.monotonic() + args.budget_h * 3600.0 if args.budget_h else None
    try:
        root = D.resolve_data_root(args.data_root, marker=PHRESH_FILE)
    except FileNotFoundError:
        log_input_tree(ctx.log)
        raise
    ctx.log(f"data root: {root}")
    specs_all = unit_specs(args)
    ctx.log(f"units: {[s.key for s in specs_all]} seeds {args.seeds} device {device}")

    def ids_all(texts: list[str]) -> np.ndarray:
        return encode_batch(texts, args.max_len)

    datasets: list[Dataset] = []
    if not args.no_phreshphish:
        if (root / PHRESH_FILE).is_file():
            datasets.append(build_phreshphish(root, args, ids_all))
            ctx.log(f"phreshphish: {len(datasets[-1].fit)} fit, {len(datasets[-1].cal)} cal "
                    f"(grouped), {len(datasets[-1].test)} test")
    if not args.no_url_domain:
        datasets += build_run_datasets(root, args, ids_all, ctx)
    phresh = next((d for d in datasets if d.protocol is not None), None)

    use_amp = device.type == "cuda" and not args.no_amp
    meta = {
        "data_root": str(root),
        "device": str(device),
        "amp": use_amp,
        "models": [s.key for s in specs_all],
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "max_train": args.max_train,
        "cal_fraction": args.cal_fraction,
        "side_features": args.side,
        "kd": args.kd,
        "budget_h": args.budget_h,
        "teacher_note": ("transformer distillation targets are in-sample logits on the training "
                         "rows (no cross-fitting)"),
        "env": env,
    }
    state: dict = {"teacher": {}, "kept": {}, "logits": {}, "splits": {}, "phresh": {},
                   "unit_info": {}, "sizes": {}, "size_rows": [], "cal": {}}
    units = Units(ctx)
    specs = specs_all
    for ds in datasets:
        state["logits"][ds.name] = {}
        block = state["splits"].setdefault(ds.name, {}) if ds.protocol is None else None
        for seed in args.seeds:
            for spec in specs:
                unit = {"split": ds.name, "model": spec.key, "seed": seed}
                label = f"{ds.name}/{spec.key}/s{seed}"
                if deadline is not None and time.monotonic() >= deadline:
                    record_skip(ctx, label, "skipped: budget exhausted")
                    continue
                out = units.run(unit, run_unit, ds, spec, seed, args, device, deadline, state,
                                ctx.log)
                if out is None:
                    continue
                tag = f"{spec.key}/s{seed}"
                state["logits"][ds.name][tag] = {
                    "id": ds.row_ids[ds.test], "label": ds.labels[ds.test],
                    "logit": out["test_logits"]}
                state["unit_info"][label] = out["info"]
                if ds.protocol is not None:
                    state["phresh"][tag] = out["result"]
                    state["cal"].setdefault(spec.key, []).append(
                        out["result"]["cal_pr_auc"])
                    op_ap = out["result"]["full_benchmark_24pct"]["ap"]
                    ctx.log(f"  {label}: cal PR-AUC {out['result']['cal_pr_auc']:.4f}, "
                            f"24% AP {op_ap:.4f}")
                    if seed == args.seeds[0]:
                        state["kept"][spec.key] = {
                            "model": out["model"], "kind": spec.kind, "use_side":
                            spec.side and ds.side.shape[1] > 0, "params": out["info"]["params"],
                            "fp32_bytes": out["info"]["fp32_bytes"], "int8": None, "onnx": None,
                            "spec": spec, "dataset": ds}
                else:
                    block[tag] = out["result"]
                    ctx.log(f"  {label}: PR-AUC {out['result']['threshold_free']['pr_auc']:.4f}")
                _checkpoint(ctx, meta, state, phresh, args)

    if phresh is not None:
        for key, entry in state["kept"].items():
            if entry["kind"] != "charcnn":
                # The transformer is a teacher and is not shipped; torch's TransformerEncoder
                # fast path does not run after dynamic quantisation, so it gets no INT8 row.
                continue
            try:
                q = int8_copy(entry["model"])
                entry["int8"] = {"model": q, "bytes": state_bytes(q), "check": protocol_int8_check(
                    entry["model"], q, phresh, args.int8_eval_n, args.seed, entry["use_side"],
                    entry["kind"])}
            except (RuntimeError, ValueError, TypeError) as exc:
                ctx.log(f"  INT8 failed for {key}: {type(exc).__name__}: {exc}")
        state["size_rows"] = size_rows(state, args, phresh)

    onnx_info = None
    charcnn_keys = [k for k, e in state["kept"].items() if e["kind"] == "charcnn"]
    if charcnn_keys and phresh is not None and not args.no_export:
        best = max(charcnn_keys, key=lambda k: float(np.mean(state["cal"].get(k, [np.nan]))))
        entry = state["kept"][best]
        n_side = phresh.side.shape[1] if entry["use_side"] else 0
        path = ctx.work / ONNX_NAME
        exported = export_onnx(entry["model"], path, n_side, args.max_len)
        parity = onnx_parity(entry["model"], path, phresh, n_side)
        mean, std = phresh.side_stats
        if entry["use_side"]:
            scaler = {"columns": phresh.side_cols, "mean": mean.tolist(), "std": std.tolist(),
                      "clip": 10.0, "missing_as": 0.0}
            (ctx.work / SCALER_NAME).write_text(json.dumps(scaler, indent=2) + "\n")
        onnx_info = {"key": best, "onnx": exported, "parity": parity, "scaler": SCALER_NAME
                     if entry["use_side"] else None, "cal_pr_auc_mean": float(np.mean(
                         state["cal"][best]))}
        ctx.log(f"exported {best}: {exported} parity {parity}")

    logits_files = _write_logits(ctx.work, state)
    meta["best_student"] = onnx_info["key"] if onnx_info else None
    meta["logits"] = logits_files
    _checkpoint(ctx, meta, state, phresh, args)
    ctx.results["onnx"] = onnx_info
    ctx.results["summary"] = (
        f"{len(datasets)} splits, {units.ok} units ok, {units.failed} failed; best student "
        f"{onnx_info['key'] if onnx_info else 'none'}")
    return units.status()
