"""fastText-style supervised model in torch: hashed n-gram embedding bag, mean, then a linear
output. Used where the ``fasttext`` package has no wheel (see the Kaggle notes in
kaggle/jobs/sms_student.py).
"""

from __future__ import annotations

import numpy as np
from torch import Tensor, nn


class HashedFastText(nn.Module):
    def __init__(self, buckets: int = 2**17, dim: int = 16):
        super().__init__()
        self.buckets = int(buckets)
        self.bag = nn.EmbeddingBag(self.buckets, dim, mode="mean")
        self.out = nn.Linear(dim, 1)

    def forward(self, indices: Tensor, offsets: Tensor) -> Tensor:
        return self.out(self.bag(indices, offsets)).squeeze(1)


def pack_bags(bags: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ragged id lists into EmbeddingBag inputs: concatenated indices and bag offsets."""
    if not bags:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    lengths = np.fromiter((len(b) for b in bags), dtype=np.int64, count=len(bags))
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    indices = np.concatenate(bags).astype(np.int64)
    return indices, offsets
