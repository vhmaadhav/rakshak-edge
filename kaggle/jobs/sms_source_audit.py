"""Acceptance report from one completed, hash-pinned source-control kernel. No inference."""

import argparse
import hashlib
import json
from pathlib import Path

from _common import write_metrics

from rakshak.eval.source_gate import source_gate


def run(ctx):
    p = argparse.ArgumentParser()
    p.add_argument("--source-tree", required=True)
    p.add_argument("--arms", nargs="+", default=["balanced", "dro"])
    args = p.parse_args(ctx.argv)
    matches = []
    for path in Path(ctx.input_root).rglob("results.json"):
        result = json.loads(path.read_text())
        if result.get("tree_sha256") == args.source_tree:
            if result.get("status") != "ok" or result.get("failures"):
                raise ValueError("source control must finish successfully without failed units")
            matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError("exactly one completed source-control tree is required")
    root = matches[0]
    path = root / "metrics.json"
    metrics = json.loads(path.read_text())
    report = source_gate(metrics, arms=tuple(args.arms))
    report.update(source_tree=args.source_tree,
                  metrics_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  transfer_pool_audit=json.loads((root / "transfer_pool_audit.json").read_text()))
    write_metrics("metrics.json", report)
    ctx.results["summary"] = str({a: r["decision"] for a, r in report["arms"].items()})
    return True
