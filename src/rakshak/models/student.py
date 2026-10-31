"""Training, scoring, INT8 dynamic quantisation, latency, ONNX export and checkpoints for the
student models (char-CNN single and dual view, and the fastText-style baseline).

Training: class-weighted BCE (pos_weight = n_neg / n_pos over the labelled rows), AdamW, early stop
on cal PR-AUC with the best cal checkpoint kept. Optional adversarial augmentation (``adv_p``): each
training row is perturbed with probability p by a random family from rakshak.perturb; the raw view
gets the perturbed text and the canonical view gets canonicalize(perturbed). The teacher logit of a
row is looked up by row index, so a perturbed row keeps the teacher logit of its clean text
(attack-consistency distillation).

Distillation (``teacher``, an (n, 2) array of teacher logits [logit_0, logit_1] aligned with the
training rows, NaN where unknown). Labels are NaN on unlabeled rows. Per row, see
``distillation_loss``: labelled rows get (1 - alpha) BCE + alpha T^2 KL(teacher || student);
unlabeled rows get the KD term only.
"""

from __future__ import annotations

import copy
import io
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rakshak.eval.metrics import pr_auc
from rakshak.models.charcnn import CONFIGS, CharCNN, OnnxWrapper
from rakshak.models.encoding import MAX_LEN, N_RULES
from rakshak.models.fasttext import HashedFastText, pack_bags
from rakshak.models.features import Featurized, featurize
from rakshak.perturb import FAMILIES, perturb

KINDS = ("fasttext", "charcnn", "dual")
SIZES = tuple(CONFIGS)


@dataclass(frozen=True)
class ModelSpec:
    kind: str  # fasttext | charcnn | dual
    size: str | None  # base | wide; None for fasttext
    variant: str  # raw | canon for fasttext and charcnn; dual for dual
    use_rules: bool = True
    buckets: int = 2**17
    dim: int = 16

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}")
        if self.kind == "fasttext":
            if self.variant not in ("raw", "canon"):
                raise ValueError("fasttext takes the raw or canon variant")
        elif self.kind == "charcnn":
            if self.variant not in ("raw", "canon") or self.size not in CONFIGS:
                raise ValueError("charcnn takes raw or canon and a size from CONFIGS")
        elif self.variant != "dual" or self.size not in CONFIGS:
            raise ValueError("dual takes variant dual and a size from CONFIGS")

    @property
    def label(self) -> str:
        return "fasttext" if self.kind == "fasttext" else f"{self.kind}-{self.size}"

    @property
    def hashed(self) -> bool:
        return self.kind == "fasttext"


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 3
    adv_p: float = 0.0
    kd_alpha: float = 0.7
    kd_temp: float = 2.0
    seed: int = 0


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model(spec: ModelSpec) -> nn.Module:
    if spec.kind == "fasttext":
        return HashedFastText(spec.buckets, spec.dim)
    views = ("raw", "canon") if spec.kind == "dual" else (spec.variant,)
    return CharCNN(CONFIGS[spec.size], views=views, n_side=N_RULES if spec.use_rules else 0)


def buckets_for(spec: ModelSpec) -> int | None:
    return spec.buckets if spec.hashed else None


def forward_feats(model: nn.Module, spec: ModelSpec, feats: Featurized, device) -> Tensor:
    """Logits (batch,) for a Featurized batch under the model's spec."""
    if spec.hashed:
        indices, offsets = pack_bags(feats.ngrams(spec.variant))
        return model(torch.as_tensor(indices, device=device),
                     torch.as_tensor(offsets, device=device))
    views = {v: torch.as_tensor(feats.bytes_view(v).astype(np.int64), device=device)
             for v in model.views}
    side = torch.as_tensor(feats.side(spec.variant), device=device) if model.n_side else None
    return model(views, side)


@torch.inference_mode()
def predict_scores(model: nn.Module, spec: ModelSpec, feats: Featurized, device,
                   batch_size: int = 512) -> np.ndarray:
    """Scam probabilities sigmoid(logit), float64, one per row."""
    model.eval()
    out = np.zeros(len(feats), dtype=np.float64)
    for start in range(0, len(feats), batch_size):
        idx = np.arange(start, min(start + batch_size, len(feats)))
        logits = forward_feats(model, spec, feats.take(idx), device)
        out[idx] = torch.sigmoid(logits).double().cpu().numpy()
    return out


def score_texts(model: nn.Module, spec: ModelSpec, texts: Sequence[str], device) -> np.ndarray:
    """Raw message texts to scam probabilities, with featurisation included (the attack harness
    and latency use this)."""
    feats = featurize(texts, buckets=buckets_for(spec))
    return predict_scores(model, spec, feats, device)


def _augment(batch: Featurized, rng: np.random.Generator, p: float,
             families: Sequence[str]) -> Featurized:
    pick = np.nonzero(rng.random(len(batch)) < p)[0]
    if len(pick) == 0:
        return batch
    texts = [
        perturb(batch.texts[j], families[int(rng.integers(len(families)))],
                int(rng.integers(2**31 - 1)))
        for j in pick
    ]
    return batch.replace(pick, featurize(texts, buckets=batch.buckets))


def _state_on_cpu(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def distillation_loss(logits: Tensor, labels: Tensor, teacher: Tensor, *, pos_weight: Tensor,
                      alpha: float, temp: float) -> Tensor:
    """Mean loss over the rows of a batch that carry any signal.

    ``logits`` (B,) are student logits, ``labels`` (B,) are 0/1 with NaN on unlabeled rows, and
    ``teacher`` (B, 2) holds teacher logits [logit_0, logit_1] with NaN where unknown. Per row:

    - labelled, teacher known:    (1 - alpha) BCE(label) + alpha T^2 KL(p_t || p_s)
    - labelled, teacher unknown:  BCE(label)
    - unlabeled, teacher known:   alpha T^2 KL(p_t || p_s)
    - unlabeled, teacher unknown: no term (left out of the mean)

    where p_t = softmax(teacher / T) and p_s = softmax([0, logit] / T) are the two-class
    distributions of teacher and student at temperature T. BCE uses ``pos_weight``. The KL is
    zero exactly when the student logit equals the teacher margin logit_1 - logit_0.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if temp <= 0.0:
        raise ValueError("temperature must be positive")
    lab = (~torch.isnan(labels)).float()
    tea = (~torch.isnan(teacher).any(dim=1)).float()
    bce = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(labels, nan=0.0),
                                             pos_weight=pos_weight, reduction="none")
    student_log = F.log_softmax(torch.stack([torch.zeros_like(logits), logits], dim=1) / temp,
                                dim=1)
    p_teacher = F.softmax(torch.nan_to_num(teacher, nan=0.0) / temp, dim=1)
    kl = F.kl_div(student_log, p_teacher, reduction="none").sum(dim=1)
    coef_bce = lab * (1.0 - alpha * tea)
    coef_kd = alpha * tea
    per_row = coef_bce * bce + coef_kd * (temp * temp) * kl
    signal_rows = (lab + tea - lab * tea).sum().clamp_min(1.0)
    return per_row.sum() / signal_rows


def train_student(spec: ModelSpec, train: Featurized, y_train, cal: Featurized, y_cal,
                  cfg: TrainConfig, *, device, teacher: np.ndarray | None = None,
                  log: Callable[[str], None] = lambda msg: None) -> tuple[nn.Module, dict]:
    """Fit one model on ``train``, early-stop on cal PR-AUC. Returns the model (best cal
    checkpoint, on CPU) and the training history. ``y_train`` is 0/1, NaN on unlabeled rows."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = build_model(spec).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    y = np.asarray(y_train, dtype=np.float64)
    labelled = ~np.isnan(y)
    n_pos = int((y[labelled] == 1).sum())
    n_neg = int((y[labelled] == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("training split needs both classes")
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)
    families = list(FAMILIES)
    if teacher is None:
        teacher = np.full((len(y), 2), np.nan)
    teacher = np.asarray(teacher, dtype=np.float64)
    if teacher.shape != (len(y), 2):
        raise ValueError("teacher logits must be (rows, 2) and aligned with the training rows")
    has_teacher = ~np.isnan(teacher).any(axis=1)
    rows_info = {"rows": int(len(y)), "labelled": int(labelled.sum()),
                 "unlabeled": int((~labelled).sum()), "with_teacher": int(has_teacher.sum()),
                 "labelled_with_teacher": int((labelled & has_teacher).sum())}
    y_cal_arr = np.asarray(y_cal, dtype=np.int64)

    best_pr, best_state, best_epoch, stale = -np.inf, None, -1, 0
    history: list[dict] = []
    start_time = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(y))
        loss_sum, batches = 0.0, 0
        for s in range(0, len(order), cfg.batch_size):
            idx = order[s:s + cfg.batch_size]
            batch = train.take(idx)
            if cfg.adv_p > 0:
                batch = _augment(batch, rng, cfg.adv_p, families)
            yb = torch.as_tensor(y[idx], dtype=torch.float32, device=device)
            tb = torch.as_tensor(teacher[idx], dtype=torch.float32, device=device)
            logits = forward_feats(model, spec, batch, device)
            loss = distillation_loss(logits, yb, tb, pos_weight=pos_weight, alpha=cfg.kd_alpha,
                                     temp=cfg.kd_temp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach())
            batches += 1
        cal_scores = predict_scores(model, spec, cal, device)
        pr = float(pr_auc(y_cal_arr, cal_scores))
        history.append({"epoch": epoch, "train_loss": loss_sum / max(batches, 1),
                        "cal_pr_auc": pr})
        log(f"    epoch {epoch}: loss {loss_sum / max(batches, 1):.4f} cal PR-AUC {pr:.4f}")
        if np.isfinite(pr) and pr > best_pr:
            best_pr, best_state, best_epoch, stale = pr, _state_on_cpu(model), epoch, 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {"best_epoch": best_epoch, "best_cal_pr_auc": float(best_pr)
            if best_state is not None else None,
            "epochs_run": len(history), "seconds": round(time.perf_counter() - start_time, 2),
            "rows": rows_info, "history": history}
    return model.cpu(), info


def param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def state_bytes(model: nn.Module) -> int:
    """Bytes of the serialised state dict (torch.save), the on-disk size of the weights."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def quantize_int8(model: nn.Module, spec: ModelSpec) -> tuple[nn.Module, list[str]]:
    """INT8 dynamic quantisation on CPU. nn.Linear is quantised. For fastText the EmbeddingBag
    table is quantised too, when torch supports it. Conv1d and Embedding stay fp32 in the char-CNN.
    Returns the model and notes on what was quantised."""
    from torch.ao.quantization import float_qparams_weight_only_qconfig, quantize_dynamic

    notes: list[str] = []
    m = copy.deepcopy(model).cpu().eval()
    if spec.hashed:
        try:
            m = quantize_dynamic(m, {nn.EmbeddingBag: float_qparams_weight_only_qconfig},
                                 dtype=torch.quint8)
            notes.append("embedding_bag: uint8 weights")
        except (RuntimeError, TypeError, ValueError) as exc:
            notes.append(f"embedding_bag kept fp32 ({type(exc).__name__})")
    try:
        m = quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
        notes.append("linear: int8 weights")
    except (RuntimeError, TypeError, ValueError) as exc:
        notes.append(f"linear kept fp32 ({type(exc).__name__})")
    return m, notes


def latency_ms(fn: Callable[[str], object], texts: Sequence[str], n: int) -> dict:
    """Per-message wall time on one torch thread, over ``n`` messages cycled from ``texts``. The
    timed call includes canonicalisation, rules, featurisation and the forward pass."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    times = []
    try:
        for i in range(n):
            text = texts[i % len(texts)]
            start = time.perf_counter()
            fn(text)
            times.append((time.perf_counter() - start) * 1000.0)
    finally:
        torch.set_num_threads(prev)
    arr = np.asarray(times)
    return {"n": int(n), "threads": 1, "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)), "mean_ms": float(arr.mean())}


def export_onnx(model: CharCNN, path: Path, opset: int = 17) -> dict:
    """Export the float char-CNN to ONNX with dynamic batch. Inputs are the byte ids per view
    (int64, shape (batch, max_len)) and, if used, ``side`` (float32, (batch, N_RULES)). The output
    ``logit`` has shape (batch,); scam probability = sigmoid(logit)."""
    wrapper = OnnxWrapper(model).cpu().eval()
    names = list(model.views) + (["side"] if model.n_side else [])
    dummies: list[Tensor] = [torch.randint(0, 256, (2, MAX_LEN), dtype=torch.long)
                             for _ in model.views]
    if model.n_side:
        dummies.append(torch.zeros(2, N_RULES, dtype=torch.float32))
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


def onnx_parity(model: CharCNN, path: Path, feats: Featurized, n: int = 64) -> dict:
    """Max absolute logit difference between torch and onnxruntime on ``n`` rows. Skipped when
    onnxruntime cannot be imported."""
    try:
        import onnxruntime as ort
    except ImportError:
        return {"skipped": "onnxruntime not importable"}
    sub = feats.take(np.arange(min(n, len(feats))))
    feed: dict[str, np.ndarray] = {v: sub.bytes_view(v).astype(np.int64) for v in model.views}
    if model.n_side:
        feed["side"] = np.ascontiguousarray(sub.side(model.views[-1]), dtype=np.float32)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = session.run(None, {i.name: feed[i.name] for i in session.get_inputs()})[0]
    with torch.inference_mode():
        views = {v: torch.as_tensor(feed[v]) for v in model.views}
        side = torch.as_tensor(feed["side"]) if model.n_side else None
        ref = model(views, side).cpu().numpy()
    return {"n": int(len(sub)), "max_abs_logit_diff": float(np.max(np.abs(got - ref)))}


def save_checkpoint(path: Path, model: nn.Module, spec: ModelSpec, threshold: float,
                    extra: dict | None = None) -> None:
    torch.save({"spec": asdict(spec), "state_dict": model.state_dict(),
                "threshold": float(threshold), "extra": extra or {}}, path)


def load_checkpoint(path: Path) -> tuple[nn.Module, ModelSpec, float]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    spec = ModelSpec(**blob["spec"])
    model = build_model(spec)
    model.load_state_dict(blob["state_dict"])
    return model.eval(), spec, float(blob["threshold"])


