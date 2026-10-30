"""Byte encoding, rule-flag side features and hashed n-grams. Numpy and the standard library only.

Byte view: the UTF-8 bytes of a message, truncated to ``max_len`` and padded with ``PAD_ID``. Each
byte is one token, so Devanagari (3 bytes per code point) and emoji (4 bytes) take several
positions. The vocabulary is the 256 byte values plus the specials in ``SPECIALS``.

Side features: a multi-hot vector over ``EVIDENCE_TYPES``, one bit per rule that fires on the
canonical form. The raw view uses the rules over the unfolded text, as the baselines do.

Hashed n-grams: fastText-style word unigrams, word bigrams and character 3-5-grams of each word,
hashed with ``zlib.crc32`` so the ids are stable across processes (``hash()`` is salted).
"""

from __future__ import annotations

import re
import zlib

import numpy as np

from rakshak.canonicalize import Canon
from rakshak.rules import EVIDENCE_TYPES, run_rules

SPECIALS: tuple[str, ...] = ("<pad>",)
PAD_ID = 256
VOCAB_SIZE = 256 + len(SPECIALS)
MAX_LEN = 256
N_RULES = len(EVIDENCE_TYPES)
NGRAM_MIN, NGRAM_MAX, NGRAM_CAP = 3, 5, 2000

_WORD = re.compile(r"\w+")
_RULE_INDEX = {name: i for i, name in enumerate(EVIDENCE_TYPES)}


def encode_bytes(text: str, max_len: int = MAX_LEN) -> np.ndarray:
    """UTF-8 bytes of ``text`` as int64 ids, truncated to ``max_len`` and padded with PAD_ID."""
    raw = text.encode("utf-8", errors="replace")[:max_len]
    out = np.full(max_len, PAD_ID, dtype=np.int64)
    if raw:
        out[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return out


def encode_batch(texts, max_len: int = MAX_LEN) -> np.ndarray:
    """Stack of ``encode_bytes`` rows, shape (n, max_len). Stored as uint16 to save memory."""
    out = np.full((len(texts), max_len), PAD_ID, dtype=np.uint16)
    for row, text in enumerate(texts):
        raw = text.encode("utf-8", errors="replace")[:max_len]
        if raw:
            out[row, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return out


def rule_flags(canon: Canon) -> np.ndarray:
    """Multi-hot float32 vector over EVIDENCE_TYPES: 1 where any rule of that type fires."""
    out = np.zeros(N_RULES, dtype=np.float32)
    for ev in run_rules(canon):
        out[_RULE_INDEX[ev.type]] = 1.0
    return out


def _stable_hash(feature: str, buckets: int) -> int:
    return zlib.crc32(feature.encode("utf-8")) % buckets


def ngram_features(text: str) -> list[str]:
    """Word unigrams, word bigrams and character 3-5-grams of each word, in a fixed order."""
    words = _WORD.findall(text.lower())
    feats = [f"w:{w}" for w in words]
    feats += [f"b:{a} {b}" for a, b in zip(words, words[1:], strict=False)]
    for w in words:
        wrapped = f"<{w}>"
        for n in range(NGRAM_MIN, NGRAM_MAX + 1):
            feats += [f"c:{wrapped[i:i + n]}" for i in range(len(wrapped) - n + 1)]
    return feats[:NGRAM_CAP]


def ngram_ids(text: str, buckets: int) -> np.ndarray:
    """Hashed feature ids in [0, buckets). An empty text gets the single id 0, so a bag is never
    empty."""
    feats = ngram_features(text)
    if not feats:
        return np.zeros(1, dtype=np.int32)
    return np.fromiter((_stable_hash(f, buckets) for f in feats), dtype=np.int32,
                       count=len(feats))


STD_CLIP = 10.0


def fit_standardizer(x: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and std over ``rows`` (training rows only). Constant columns get std 1."""
    sub = np.asarray(x, dtype=np.float64)[np.asarray(rows, dtype=np.int64)]
    mean = sub.mean(axis=0)
    std = sub.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(x - mean) / std, NaN and inf set to 0, clipped to +-STD_CLIP. float32, shape of ``x``."""
    z = (np.asarray(x, dtype=np.float64) - mean) / std
    z = np.nan_to_num(z, nan=0.0, posinf=STD_CLIP, neginf=-STD_CLIP)
    return np.clip(z, -STD_CLIP, STD_CLIP).astype(np.float32)
