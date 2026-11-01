"""Recompute clustered ELA intervals from an attached private completed kernel output.

No model inference, new sample selection, threshold fitting or dataset changes.
"""

import argparse
import hashlib
import json
from pathlib import Path

from _common import write_metrics

from rakshak.eval.audit import clustered_rate, clustered_ratio


def summarize(rows):
    if not rows:
        return {"n": 0}
    groups = [r["campaign_id"] for r in rows]
    binary = {
        "json_parse": [r["json_ok"] for r in rows],
        "verifier_pass": [r["verifier_ok"] for r in rows],
        "unsupported_claim": [r["unsupported_n"] > 0 for r in rows],
        "safe_claim": [r["safe_claim"] for r in rows],
    }
    out = {"n": len(rows), "rates": {
        name: clustered_rate(values, groups) for name, values in binary.items()
    }}
    if all("points_n" in r for r in rows):
        out["point_pass"] = clustered_ratio(
            [r["points_pass_n"] for r in rows], [r["points_n"] for r in rows], groups
        )
    return out


def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-tree", required=True)
    args = parser.parse_args(ctx.argv)
    candidates = []
    for path in Path(ctx.input_root).rglob("generations.jsonl"):
        result_path = path.with_name("results.json")
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text())
        if result.get("tree_sha256") == args.expected_tree:
            candidates.append((path, result))
    if len(candidates) != 1:
        raise ValueError(f"expected one source kernel for the frozen tree, found {len(candidates)}")
    path, result = candidates[0]
    if result.get("status") != "ok":
        raise ValueError("source kernel must have completed successfully")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    clean = [r for r in rows if not r["injected"]]
    arms = sorted({r["arm"] for r in clean})
    reports = {}
    for arm in arms:
        selected = [r for r in clean if r["arm"] == arm]
        reports[arm] = {
            side: summarize([r for r in selected if side == "all" or r["side"] == side])
            for side in ("all", "scam", "benign")
        }
    manifest = json.loads(path.with_name("sample_manifest.json").read_text())
    lang_ids = {r["sample_id"] for r in manifest["rows"] if r["lang_subset"]}
    lang_compare = {
        arm: summarize([r for r in clean if r["arm"] == arm and r["sample_id"] in lang_ids])
        for arm in ("slot", "slot_hinglish")
    }
    write_metrics("metrics.json", {
        "source_tree": args.expected_tree, "source_job": result["job"],
        "generations_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "method": "campaign-cluster percentile; pooled point ratio; 1000 draws, seed 13",
        "arms": reports, "lang_compare": lang_compare,
        "scope": "same frozen campaign sample; descriptive intervals, no tuning or new inference",
    })
    ctx.results["summary"] = f"clustered ELA intervals: {len(clean)} clean rows; {len(arms)} arms"
    return True
