#!/usr/bin/env python
"""End-to-end detection quality for the deterministic layer, on the real campaign split.

This answers the question the rules-only coverage number cannot: **what does a real user see?**
It scores the full local path (``package_for_sms`` -> ``render_advisory``) on the held-out
campaign-disjoint test split and reports the three-state confusion, precision/recall/F1 for the
``scam`` decision, the abstention rate, and the cost of each rule family.

No model weights, no network: only ``src/rakshak`` and the built ``data/processed``.

Usage
-----
    .venv/bin/python scripts/e2e_detection.py                 # campaign split, test
    .venv/bin/python scripts/e2e_detection.py --split campaign --part test
    .venv/bin/python scripts/e2e_detection.py --limit 5000     # quick run
    .venv/bin/python scripts/e2e_detection.py --json

Notes
-----
* The decision is read from ``package.risk`` only. ``scam`` = HIGH, ``uncertain`` = UNCERTAIN,
  ``benign`` = LOW. MEDIUM is treated as uncertain (it is an abstain bucket by design).
* Labels are the legacy ``label_binary`` (spam-or-scam = 1). That is a weak target: it counts
  promotions as positives. ``label_scam`` is reported alongside when it is populated, because
  "promotions are not fraud" is the project's own stated rule.
* ``proxy_scam_state`` is the label-free stand-in used when fusion output is absent. This is
  therefore the **rules-only** detector, which is a shipped component and a strict lower bound on
  the fused system.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rakshak.ela_eval import package_for_sms  # noqa: E402
from rakshak.evidence import Risk  # noqa: E402

DATA = REPO / "data" / "processed"

# Risk rung -> three-state decision. MEDIUM is an abstain bucket, not a scam call.
DECISION = {
    Risk.LOW: "benign",
    Risk.UNCERTAIN: "uncertain",
    Risk.MEDIUM: "uncertain",
    Risk.HIGH: "scam",
}


def decide(text: str) -> tuple[str, list[str]]:
    """Three-state decision plus the evidence types that produced it."""
    _, pkg = package_for_sms(text)
    return DECISION[pkg.risk], [e.type for e in pkg.evidence]


def load_split(split: str, part: str) -> pd.DataFrame:
    sms = pd.read_parquet(DATA / "sms.parquet")
    sms["id"] = sms["id"].astype(str)
    obj = json.loads((DATA / "splits" / f"{split}.json").read_text(encoding="utf-8"))
    ids = obj["partitions"][part]
    sub = sms[sms["id"].isin(set(ids))]
    if "eval_ok" in sub:
        sub = sub[sub["eval_ok"].fillna(False).astype(bool)]
    if "synthetic" in sub:
        sub = sub[~sub["synthetic"].fillna(False).astype(bool)]
    return sub.reset_index(drop=True)


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def evaluate(df: pd.DataFrame) -> dict:
    labs = df["label_binary"].astype(int).tolist()
    decs, types = [], []
    for text in df["text"].tolist():
        d, t = decide(text)
        decs.append(d)
        types.append(t)

    n = len(df)
    conf = Counter(zip(labs, decs, strict=True))
    tp = conf[(1, "scam")]
    fp = conf[(0, "scam")]
    fn = sum(v for (y, d), v in conf.items() if y == 1 and d != "scam")
    tn = conf[(0, "benign")]
    abstain = sum(1 for d in decs if d == "uncertain")

    # Coverage-vs-error: of the rows it did NOT abstain on, how often was it right?
    decided = [(y, d) for y, d in zip(labs, decs, strict=True) if d != "uncertain"]
    decided_acc = (
        round(sum(1 for y, d in decided if (y == 1) == (d == "scam")) / len(decided), 4)
        if decided
        else None
    )

    out = {
        "n": n,
        "positives": sum(labs),
        "scam_decision": {**prf(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else None,
        "abstention_rate": round(abstain / n, 4),
        "coverage": round(len(decided) / n, 4),
        "accuracy_when_decided": decided_acc,
        "confusion": {f"{k[0]}->{k[1]}": v for k, v in sorted(conf.items())},
        # How often the rules stayed silent, by gold label. Silent = no evidence at all, which is
        # also the population where the ELA explainer has nothing to say.
        "no_evidence": {
            "scam": sum(1 for y, t in zip(labs, types, strict=True) if y == 1 and not t),
            "benign": sum(1 for y, t in zip(labs, types, strict=True) if y == 0 and not t),
            "scam_n": sum(labs),
            "benign_n": n - sum(labs),
        },
        "top_evidence_types": Counter(t for ts in types for t in ts).most_common(12),
    }
    nr = out["no_evidence"]
    nr["scam_rate"] = round(nr["scam"] / nr["scam_n"], 4) if nr["scam_n"] else None
    nr["benign_rate"] = round(nr["benign"] / nr["benign_n"], 4) if nr["benign_n"] else None
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="campaign")
    ap.add_argument("--part", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    df = load_split(args.split, args.part)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=13).reset_index(drop=True)
    res = evaluate(df)

    if args.json:
        print(json.dumps(res, indent=1, default=str))
        return 0

    s = res["scam_decision"]
    print(f"split={args.split}/{args.part}  n={res['n']}  positives={res['positives']} "
          f"({res['positives']/res['n']:.1%})")
    print()
    print("three-state confusion (gold -> decision)")
    for k, v in res["confusion"].items():
        print(f"   {k:18s} {v:6d}")
    print()
    print(f"abstention (UNCERTAIN/MEDIUM) : {res['abstention_rate']:.1%}")
    print(f"coverage (a verdict was given): {res['coverage']:.1%}")
    print(f"accuracy on decided rows      : {res['accuracy_when_decided']}")
    print()
    print("scam decision vs legacy label_binary")
    print(f"   precision {s['precision']:.4f}  recall {s['recall']:.4f}  F1 {s['f1']:.4f}")
    print(f"   tp {s['tp']}  fp {s['fp']}  fn {s['fn']}  tn {s['tn']}")
    print(f"   FPR (benign rows called scam): {res['false_positive_rate']}")
    print()
    nr = res["no_evidence"]
    print("rows with NO evidence (explainer has nothing to cite)")
    print(f"   scam   {nr['scam']:6d}/{nr['scam_n']:<6d} = {nr['scam_rate']}")
    print(f"   benign {nr['benign']:6d}/{nr['benign_n']:<6d} = {nr['benign_rate']}")
    print()
    print("top evidence types:", res["top_evidence_types"][:8])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
