#!/usr/bin/env python3
"""Build the Hinglish edge eval rows (laptop side; the Pi has no corpus parquet).

    .venv/bin/python deploy/pi/service/build_hinglish_rows.py --out DIR/rows.json

Source: data/processed/sms.parquet, restricted to the lang_shift TEST partition
(data/processed/splits/lang_shift.json, test ids) and to rows that detect_lang() puts in
"hinglish". For that partition detect_lang agrees with the corpus `lang` column on every row
(398 of 398). Seed 0 (numpy default_rng).

- 30 scam rows (label_binary 1) with at least one rule evidence item, spread over the evidence
  types: types are visited from the rarest to the most common, one row per visit, until 30 rows.
  A row can carry several types; its "primary" type is the one it was drawn for.
- 10 benign rows (label_binary 0), drawn at random from the Hinglish test rows (no evidence
  condition, so the benign group is not chosen by the detector or the rules).

The file has the same shape as the English rows.json (row_id, text, label, group), plus the
evidence types and the detect_lang value at build time. The decisions are made on the Pi.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "src")]

from rakshak.canonicalize import canonicalize  # noqa: E402
from rakshak.datasets.lang import detect_lang  # noqa: E402
from rakshak.rules import run_rules  # noqa: E402

SEED = 0
N_SCAM = 30
N_BENIGN = 10


def build() -> dict:
    import pandas as pd

    df = pd.read_parquet(REPO / "data" / "processed" / "sms.parquet")
    split = json.loads((REPO / "data" / "processed" / "splits" / "lang_shift.json").read_text())
    test_ids = set(split["partitions"]["test"])
    test = df[df["id"].isin(test_ids)].copy()
    test["detect"] = test["text"].map(detect_lang)
    hin = test[test["detect"] == "hinglish"].copy()
    hin["types"] = hin["text"].map(lambda t: sorted({e.type for e in run_rules(canonicalize(t))}))
    rng = np.random.default_rng(SEED)

    scam = hin[(hin["label_binary"] == 1) & (hin["types"].map(len) > 0)]
    count = Counter(t for ts in scam["types"] for t in ts)
    order = sorted(count, key=lambda t: (count[t], t))  # rarest first
    chosen: list[str] = []
    primary: dict[str, str] = {}
    pools = {
        t: [i for i, ts in zip(scam["id"], scam["types"], strict=True) if t in ts] for t in order
    }
    for t in order:
        pool = [i for i in pools[t] if i not in chosen]
        if pool and len(chosen) < N_SCAM:
            pick = pool[int(rng.integers(len(pool)))]
            chosen.append(pick)
            primary[pick] = t
    # Second pass: fill the rest of the 30 round-robin from the same type order.
    while len(chosen) < N_SCAM:
        progressed = False
        for t in order:
            pool = [i for i in pools[t] if i not in chosen]
            if pool and len(chosen) < N_SCAM:
                pick = pool[int(rng.integers(len(pool)))]
                chosen.append(pick)
                primary[pick] = t
                progressed = True
        if not progressed:
            break

    rows = []
    by_id = hin.set_index("id")
    for rid in chosen:
        r = by_id.loc[rid]
        rows.append(_row(rid, r, "evidence", primary[rid]))
    benign_pool = hin[hin["label_binary"] == 0]
    bidx = rng.permutation(len(benign_pool))[:N_BENIGN]
    for k in bidx:
        r = benign_pool.iloc[int(k)]
        rows.append(_row(r["id"], r, "benign", None))
    return {
        "seed": SEED,
        "source": "data/processed/sms.parquet, splits/lang_shift.json test, detect_lang hinglish",
        "test_rows": len(test),
        "hinglish_test_rows": len(hin),
        "scam_with_evidence": int(len(scam)),
        "scam_type_coverage": dict(
            Counter(r["primary_type"] for r in rows if r["group"] == "evidence")
        ),
        "rows": rows,
    }


def _row(rid, r, group, primary_type):
    return {
        "row_id": str(rid),
        "text": r["text"],
        "label": int(r["label_binary"]),
        "group": group,
        "primary_type": primary_type,
        "evidence_types": sorted({e.type for e in run_rules(canonicalize(r["text"]))}),
        "source": r["source"],
        "lang_column": r["lang"],
        "detect_lang": detect_lang(r["text"]),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    doc = build()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"rows: {len(doc['rows'])} -> {out}; scam types {doc['scam_type_coverage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
