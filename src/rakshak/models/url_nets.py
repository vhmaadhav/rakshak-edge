"""Byte-level transformer for URLs (torch). The teacher for the URL char-CNN student.

Same byte view as ``charcnn``: ids from ``encoding.encode_batch``, PAD_ID masked out of attention
and out of the mean pool. Pre-norm encoder, learned positions, one logit per URL. Torch only
(kaggle/jobs/url_student.py and the tests after ``pytest.importorskip("torch")``).
"""

from __future__ import annotations

from torch import Tensor, nn

from rakshak.models.encoding import PAD_ID, VOCAB_SIZE


class ByteTransformer(nn.Module):
    def __init__(
        self,
        max_len: int = 200,
        d_model: int = 256,
        layers: int = 4,
        heads: int = 4,
        ff_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        self.n_layers = int(layers)
        self.embed = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_ID)
        self.pos = nn.Embedding(max_len, d_model)
        block = nn.TransformerEncoderLayer(
            d_model,
            heads,
            ff_mult * d_model,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, ids: Tensor) -> Tensor:
        """``ids``: int64 (batch, length), length <= max_len, PAD_ID for padding. Returns logits
        of shape (batch,)."""
        if ids.shape[1] > self.max_len:
            raise ValueError(f"length {ids.shape[1]} exceeds max_len {self.max_len}")
        real = ids != PAD_ID
        # Position 0 always attends, so an all-padding row gives finite values, not NaN.
        real = real.clone()
        real[:, 0] = True
        pos = self.pos.weight[: ids.shape[1]].unsqueeze(0)
        h = self.encoder(self.embed(ids) + pos, src_key_padding_mask=~real)
        h = self.norm(h)
        w = real.unsqueeze(-1).to(h.dtype)
        pooled = (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).squeeze(1)
