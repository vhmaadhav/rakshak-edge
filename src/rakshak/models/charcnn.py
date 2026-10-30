"""Byte-level char-CNN student: one view (raw or canonical) or dual (raw and canonical).

Each view has its own embedding and convolution stack. Global max-pooling over valid positions
(padding excluded) gives one vector per width and filter. The views and the rule side features are
concatenated, then an MLP outputs one logit per message. Sigmoid(logit) is the scam probability.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from rakshak.models.encoding import PAD_ID, VOCAB_SIZE


@dataclass(frozen=True)
class StudentConfig:
    name: str
    embed_dim: int
    num_filters: int
    widths: tuple[int, ...]
    hidden: int
    dropout: float = 0.1


# Base: about 46k params (186 KB fp32) single view, 92k (366 KB) dual.
BASE = StudentConfig("base", embed_dim=32, num_filters=64, widths=(3, 4, 5), hidden=64)
# Wide: about 0.68M params (2.7 MB fp32) single view, 1.35M (5.4 MB) dual.
WIDE = StudentConfig("wide", embed_dim=64, num_filters=256, widths=(2, 3, 4, 5, 6), hidden=256)
CONFIGS: dict[str, StudentConfig] = {"base": BASE, "wide": WIDE}


class ByteConv(nn.Module):
    """Embedding, parallel 1-D convolutions and masked global max-pool over one byte view."""

    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, cfg.embed_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList(
            nn.Conv1d(cfg.embed_dim, cfg.num_filters, kernel_size=k) for k in cfg.widths
        )
        self.out_dim = cfg.num_filters * len(cfg.widths)

    def forward(self, ids: Tensor) -> Tensor:
        valid = ids != PAD_ID
        x = self.embed(ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x))
            # A window counts when its first byte is real; all-padding rows pool to zero.
            keep = valid[:, : h.shape[2]].unsqueeze(1)
            h = h.masked_fill(~keep, float("-inf"))
            p = h.max(dim=2).values
            pooled.append(torch.where(torch.isfinite(p), p, torch.zeros_like(p)))
        return torch.cat(pooled, dim=1)


class CharCNN(nn.Module):
    """``views`` is ("canon",), ("raw",) or ("raw", "canon") for dual. ``n_side`` is the number of
    rule side features (0 disables them)."""

    def __init__(self, cfg: StudentConfig, views: tuple[str, ...] = ("canon",),
                 n_side: int = 0):
        super().__init__()
        if not views or any(v not in ("raw", "canon") for v in views):
            raise ValueError(f"views must be drawn from raw and canon, got {views}")
        self.cfg = cfg
        self.views = tuple(views)
        self.n_side = int(n_side)
        self.encoders = nn.ModuleDict({v: ByteConv(cfg) for v in self.views})
        width = sum(enc.out_dim for enc in self.encoders.values()) + self.n_side
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(width, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, 1),
        )

    def forward(self, views: Mapping[str, Tensor], side: Tensor | None = None) -> Tensor:
        """``views[v]`` holds byte ids (batch, max_len) for each view in ``self.views``. Returns
        logits of shape (batch,)."""
        parts = [self.encoders[v](views[v]) for v in self.views]
        if self.n_side:
            if side is None:
                raise ValueError("this model takes rule side features")
            parts.append(side)
        return self.head(torch.cat(parts, dim=1)).squeeze(1)


class OnnxWrapper(nn.Module):
    """Positional-argument view of CharCNN for torch.onnx.export: byte inputs in ``views`` order,
    then the side features when the model has them."""

    def __init__(self, model: CharCNN):
        super().__init__()
        self.model = model

    def forward(self, *inputs: Tensor) -> Tensor:
        m = self.model
        views = {v: inputs[i] for i, v in enumerate(m.views)}
        side = inputs[len(m.views)] if m.n_side else None
        return m(views, side)
