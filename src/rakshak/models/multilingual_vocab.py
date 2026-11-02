"""Train-ID-only compact multilingual vocabulary and deterministic embedding PCA."""

from collections import Counter

import numpy as np

MODEL = 'jhu-clsp/mmBERT-small'
REVISION = 'abc32620dd4f6ab06f5fbe905dc25f310618e09f'
CARD_SHA256 = 'a56c26f251c188a1c0b783fcddb461e78f754b61ee948975fafbb2b77c863353'
CONFIG_SHA256 = '3b0eea47c4734682370b66dbd73d655f1100387f683d7c69015f33aa1bac1be9'
MAX_TOKENS, DIM, MAX_LEN = 16382, 64, 128


def select_vocabulary(token_rows, row_ids, frozen_train_ids, *, limit=MAX_TOKENS):
    ids = list(map(str, row_ids))
    expected = list(map(str, frozen_train_ids))
    if len(ids) != len(set(ids)) or ids != expected or len(token_rows) != len(ids):
        raise ValueError('vocabulary requires exactly ordered frozen train IDs')
    counts = Counter(int(token) for row in token_rows for token in row)
    if any(token < 0 for token in counts) or not 1 <= limit <= MAX_TOKENS:
        raise ValueError('invalid token IDs or capacity')
    return np.asarray(sorted(counts, key=lambda token: (-counts[token], token))[:limit],
                      dtype=np.int64)


def project_embeddings(selected_rows, *, dim=DIM):
    rows = np.asarray(selected_rows, dtype=np.float64)
    if (rows.ndim != 2 or min(rows.shape[0]-1, rows.shape[1]) < dim
            or not np.isfinite(rows).all()):
        raise ValueError('insufficient finite train-token embeddings for fixed PCA')
    mean = rows.mean(axis=0)
    _, _, components = np.linalg.svd(rows-mean, full_matrices=False)
    components = components[:dim].copy()
    signs = np.sign(components[np.arange(dim), np.abs(components).argmax(axis=1)])
    components *= signs[:, None]
    projected = (rows-mean) @ components.T
    table = np.zeros((len(rows)+2, dim), dtype=np.float32)
    table[2:] = projected
    table[1] = projected.mean(axis=0)
    return table, mean, components


def encode_tokens(token_rows, original_ids, *, max_len=MAX_LEN):
    original_ids = np.asarray(original_ids)
    if (original_ids.ndim != 1 or len(np.unique(original_ids)) != len(original_ids)
            or np.any(original_ids < 0) or max_len < 1):
        raise ValueError('invalid compact mapping')
    mapping = {int(token): index+2 for index, token in enumerate(original_ids)}
    out = np.zeros((len(token_rows), max_len), dtype=np.int64)
    for index, row in enumerate(token_rows):
        tokens = row[:max_len]
        out[index, :len(tokens)] = [mapping.get(int(token), 1) for token in tokens]
    return out
