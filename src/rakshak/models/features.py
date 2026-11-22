"""Featurisation of messages into both views (raw and canonical), with rule side features.

``featurize`` is the only place that canonicalises for the student models. Training, the attack
harness, the evaluation and CPU latency all go through it, so they see the same inputs.
``Featurized`` is a row container. ``take`` and ``replace`` build batches, and on-the-fly
adversarial augmentation replaces the perturbed rows of a batch.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from rakshak.canonicalize import Canon, canonicalize, extract_urls
from rakshak.models.encoding import (
    MAX_LEN,
    N_RULES,
    encode_batch,
    ngram_ids,
    rule_flags,
    rule_flags_from,
)

VIEWS = ("raw", "canon")
VARIANTS = ("raw", "canon", "dual")


@dataclass
class Featurized:
    texts: list[str]
    raw_bytes: np.ndarray  # (n, max_len) uint16
    canon_bytes: np.ndarray  # (n, max_len) uint16
    side_raw: np.ndarray  # (n, N_RULES) float32, rules over the unfolded text
    side_canon: np.ndarray  # (n, N_RULES) float32, rules over the canonical text
    buckets: int | None = None
    raw_ngrams: list[np.ndarray] = field(default_factory=list)
    canon_ngrams: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.texts)

    def bytes_view(self, view: str) -> np.ndarray:
        if view == "raw":
            return self.raw_bytes
        if view == "canon":
            return self.canon_bytes
        raise KeyError(f"unknown byte view {view!r}")

    def side(self, variant: str) -> np.ndarray:
        """Side features for a variant: raw uses the unfolded rules, canon and dual the canonical
        ones."""
        return self.side_raw if variant == "raw" else self.side_canon

    def ngrams(self, view: str) -> list[np.ndarray]:
        if self.buckets is None:
            raise ValueError("these features were built without n-grams")
        return self.raw_ngrams if view == "raw" else self.canon_ngrams

    def take(self, idx) -> Featurized:
        idx = np.asarray(idx, dtype=np.int64)
        return Featurized(
            texts=[self.texts[i] for i in idx],
            raw_bytes=self.raw_bytes[idx],
            canon_bytes=self.canon_bytes[idx],
            side_raw=self.side_raw[idx],
            side_canon=self.side_canon[idx],
            buckets=self.buckets,
            raw_ngrams=[self.raw_ngrams[i] for i in idx] if self.buckets else [],
            canon_ngrams=[self.canon_ngrams[i] for i in idx] if self.buckets else [],
        )

    def replace(self, positions, other: Featurized) -> Featurized:
        """A copy with the rows at ``positions`` overwritten by the rows of ``other``."""
        pos = np.asarray(positions, dtype=np.int64)
        if len(pos) != len(other):
            raise ValueError("positions and replacement rows differ in length")
        out = copy.copy(self)
        out.texts = list(self.texts)
        out.raw_bytes = self.raw_bytes.copy()
        out.canon_bytes = self.canon_bytes.copy()
        out.side_raw = self.side_raw.copy()
        out.side_canon = self.side_canon.copy()
        out.raw_ngrams = list(self.raw_ngrams)
        out.canon_ngrams = list(self.canon_ngrams)
        for j, p in enumerate(pos):
            out.texts[p] = other.texts[j]
            out.raw_bytes[p] = other.raw_bytes[j]
            out.canon_bytes[p] = other.canon_bytes[j]
            out.side_raw[p] = other.side_raw[j]
            out.side_canon[p] = other.side_canon[j]
            if self.buckets:
                out.raw_ngrams[p] = other.raw_ngrams[j]
                out.canon_ngrams[p] = other.canon_ngrams[j]
        return out


def with_side_extra(fz: Featurized, extra: np.ndarray) -> Featurized:
    """A copy with ``extra`` (n, k) float32 appended to both side views (raw and canonical).
    Used for the inbox sender features; rows without a sender pass zeros."""
    extra = np.asarray(extra, dtype=np.float32)
    if extra.ndim != 2 or extra.shape[0] != len(fz):
        raise ValueError(f"extra side features must be (rows, k), got {extra.shape}")
    out = copy.copy(fz)
    out.side_raw = np.hstack([fz.side_raw, extra])
    out.side_canon = np.hstack([fz.side_canon, extra])
    return out


def featurize(texts: Sequence[str], *, buckets: int | None = None,
              max_len: int = MAX_LEN, canons: Sequence[Canon] | None = None,
              canon_rules: Sequence[Sequence] | None = None,
              raw_side: bool = True) -> Featurized:
    """Both views, rule flags and (when ``buckets`` is given) hashed n-grams for each message.

    ``canons`` and ``canon_rules`` let a caller that already canonicalised the texts and ran the
    rules on the canonical view (for example the edge verdict path) pass them in, so each is
    computed once per message. The results are the same as when they are computed here.
    ``raw_side=False`` leaves ``side_raw`` at zeros: the serving path never reads it (the SMS
    student takes only the canonical rule flags), so the raw-view rules are skipped there.
    """
    texts = list(texts)
    if canons is None:
        canons = [canonicalize(t) for t in texts]
    elif len(canons) != len(texts):
        raise ValueError("canons and texts differ in length")
    if canon_rules is None:
        side_canon = np.stack([rule_flags(c) for c in canons]) if texts else np.zeros(
            (0, N_RULES), np.float32)
    else:
        if len(canon_rules) != len(texts):
            raise ValueError("canon_rules and texts differ in length")
        side_canon = np.stack([rule_flags_from(r) for r in canon_rules]) if texts else np.zeros(
            (0, N_RULES), np.float32)
    if raw_side:
        raw_canons = [Canon(original=t, normalized=t, urls=extract_urls(t)) for t in texts]
        side_raw = np.stack([rule_flags(c) for c in raw_canons]) if texts else np.zeros(
            (0, N_RULES), np.float32)
    else:
        side_raw = np.zeros((len(texts), N_RULES), np.float32)
    canon_texts = [c.normalized for c in canons]
    out = Featurized(
        texts=texts,
        raw_bytes=encode_batch(texts, max_len),
        canon_bytes=encode_batch(canon_texts, max_len),
        side_raw=side_raw,
        side_canon=side_canon,
        buckets=buckets,
    )
    if buckets is not None:
        out.raw_ngrams = [ngram_ids(t, buckets) for t in texts]
        out.canon_ngrams = [ngram_ids(t, buckets) for t in canon_texts]
    return out
