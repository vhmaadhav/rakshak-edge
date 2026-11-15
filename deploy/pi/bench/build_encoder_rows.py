"""Build the pre-tokenised SMS rows that encoder_bench.py reads with --rows (laptop side).

Takes a seeded sample of the cal split of one fold (the same selection as the W1c parity rows in
kaggle/jobs/sms_encoder_export.py), canonicalises the text, tokenises it with the exported
tokenizer.json and writes one .npz. The Pi then needs no tokenizer and no transformers.

Stored arrays: ids (N x L, padded with pad_id, L <= 128), lengths (kept tokens per row),
full_lengths (tokens before truncation), pad_id, meta_json (split, seed, source hashes, counts).
Truncation keeps the final special token, as encoder_bench.truncate_keep_special does.

Usage (laptop, needs the rakshak package and tokenizers):
  python deploy/pi/bench/build_encoder_rows.py --tokenizer DIR --out FILE.npz
         [--split source_holdout:holdout_scamshield] [--rows 500] [--seed 0] [--data-root DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoder_bench import MAX_LEN, token_length_stats, truncate_keep_special  # noqa: E402

DEFAULT_SPLIT = "source_holdout:holdout_scamshield"
DEFAULT_ROWS = 500
DEFAULT_SEED = 0
PAD_TOKEN_FALLBACK = 0


def pick_rows(n_available: int, n_wanted: int, seed: int) -> np.ndarray:
    """Sorted indices of a seeded sample without replacement. Fails when there are too few rows."""
    if n_wanted < 1:
        raise ValueError("--rows must be at least 1")
    if n_available < n_wanted:
        raise ValueError(f"cal has {n_available} rows, need {n_wanted}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_available, n_wanted, replace=False))


def pack_rows(
    full_ids: Sequence[Sequence[int]], pad_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ids, lengths, full_lengths): truncated to MAX_LEN keeping the final special token, then
    padded on the right to the longest kept row."""
    kept = [truncate_keep_special(ids, MAX_LEN) for ids in full_ids]
    if not kept:
        raise ValueError("no rows to pack")
    width = max(len(k) for k in kept)
    ids = np.full((len(kept), width), pad_id, dtype=np.int64)
    for i, row in enumerate(kept):
        ids[i, : len(row)] = row
    lengths = np.asarray([len(k) for k in kept], dtype=np.int64)
    full = np.asarray([len(f) for f in full_ids], dtype=np.int64)
    return ids, lengths, full


def write_rows(
    path: Path,
    full_ids: Sequence[Sequence[int]],
    pad_id: int,
    meta: dict[str, Any],
) -> dict[str, Any]:
    ids, lengths, full = pack_rows(full_ids, pad_id)
    summary = {
        **meta,
        "rows": int(len(lengths)),
        "max_len": MAX_LEN,
        "token_len": token_length_stats(full.tolist(), int(np.sum(full > MAX_LEN))),
    }
    np.savez(
        path,
        ids=ids,
        lengths=lengths,
        full_lengths=full,
        pad_id=np.int64(pad_id),
        meta_json=np.asarray(json.dumps(summary, sort_keys=True)),
    )
    return summary


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="build_encoder_rows")
    p.add_argument("--tokenizer", required=True, type=Path, help="folder with tokenizer.json")
    p.add_argument("--out", required=True, type=Path, help="the .npz to write")
    p.add_argument("--split", default=DEFAULT_SPLIT, help="fold split whose cal rows are sampled")
    p.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--data-root", default=None, help="as rakshak.eval.data.resolve_data_root")
    args = p.parse_args(None if argv is None else list(argv))

    from tokenizers import Tokenizer

    from rakshak.eval import data as D

    tok_json = args.tokenizer / "tokenizer.json"
    if not tok_json.is_file():
        print(f"build_encoder_rows: {tok_json} not found", file=sys.stderr)
        return 1
    tokenizer = Tokenizer.from_file(str(tok_json))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    config_path = args.tokenizer / "tokenizer_config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    pad_token = config.get("pad_token")
    if isinstance(pad_token, dict):
        pad_token = pad_token.get("content")
    pad_id = tokenizer.token_to_id(pad_token) if isinstance(pad_token, str) else None
    pad_id = PAD_TOKEN_FALLBACK if pad_id is None else int(pad_id)

    root = D.resolve_data_root(args.data_root)
    sms = D.load_sms(root)
    labels = sms["label_binary"].to_numpy().astype(np.int64)
    stem = args.split.split(":", 1)[0]
    runs = {
        r.name: r
        for r in D.load_runs(root, sms, patterns=[stem], labels=labels, mask=D.eval_mask(sms))
    }
    if args.split not in runs:
        print(f"build_encoder_rows: split {args.split!r} not found", file=sys.stderr)
        return 1
    cal = runs[args.split].cal
    pick = pick_rows(len(cal), args.rows, args.seed)
    chosen = cal[pick]
    texts = [D.canon_text(t) for t in sms["text"].to_numpy(dtype=object)[chosen]]
    full_ids = [enc.ids for enc in tokenizer.encode_batch(texts)]
    meta = {
        "split": args.split,
        "seed": args.seed,
        "rows_in_cal": int(len(cal)),
        "tokenizer_json_sha256": _sha256_file(tok_json),
        "ids_sha256": hashlib.sha256(
            np.asarray(sms["id"].to_numpy()[chosen]).astype(str).tobytes()
        ).hexdigest(),
        "pad_id": pad_id,
    }
    summary = write_rows(args.out, full_ids, pad_id, meta)
    print(f"wrote {args.out}: {summary['rows']} rows, token_len {summary['token_len']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
