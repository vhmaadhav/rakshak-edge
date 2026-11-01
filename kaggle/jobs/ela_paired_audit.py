"""Paired quality supplement from completed private base/QLoRA outputs. No inference/tuning."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from _common import write_metrics

from rakshak.eval.point_comparison import paired_point_pass
from rakshak.llm.ela_comparison import compare
from rakshak.llm.llama_notice import write_model_notice

BASE_TREE = "48ed512ef46e0abf629fe7e8da89e1dbca68dbd275516f113900e12eb852932d"
BASE_HASH = "8a898ef0d8edcf92598729ef8bf6e94e7bc5674f7d914d854aed02cf799e04d1"


def completed_rows(root, tree):
    matches = []
    for path in Path(root).rglob("generations.jsonl"):
        result_path = path.with_name("results.json")
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text())
        if result.get("tree_sha256") == tree:
            if result.get("status") != "ok":
                raise ValueError("source inference must be complete and successful")
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"expected one completed source for tree {tree}, found {len(matches)}")
    path = matches[0]
    return ([json.loads(line) for line in path.read_text().splitlines()],
            hashlib.sha256(path.read_bytes()).hexdigest(), path.parent)


def run(ctx):
    p = argparse.ArgumentParser()
    p.add_argument("--tuned-tree", required=True)
    args = p.parse_args(ctx.argv)
    base, base_hash, _ = completed_rows(ctx.input_root, BASE_TREE)
    tuned, tuned_hash, tuned_root = completed_rows(ctx.input_root, args.tuned_tree)
    if base_hash != BASE_HASH:
        raise ValueError("base generations hash changed")
    result = compare(base, tuned)  # exact pairing, metadata and point denominators enforced
    # The original run's selected weights are copied verbatim; no training or conversion.
    adapter = ctx.work / "Llama-Rakshak-ELA-QLoRA"
    source_adapter = tuned_root / adapter.name
    if not (source_adapter / "adapter_model.safetensors").is_file():
        raise ValueError("selected source adapter is missing")
    source_adapter_hash = hashlib.sha256(
        (source_adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    shutil.copytree(source_adapter, adapter)
    notice = write_model_notice(adapter)
    if hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest() != (
        source_adapter_hash
    ):
        raise ValueError("adapter changed during notice packaging")
    paired = {}
    for arm in ("slot", "slot_hinglish"):
        b = {str(r["sample_id"]): r for r in base if r["arm"] == arm and not r["injected"]}
        rows = [r for r in tuned if r["arm"] == arm and not r["injected"]]
        paired[arm] = paired_point_pass(
            [b[str(r["sample_id"])]["points_pass_n"] for r in rows],
            [r["points_pass_n"] for r in rows], [r["points_n"] for r in rows],
            [r["campaign_id"] for r in rows])
    write_metrics("metrics.json", {"base_tree": BASE_TREE, "tuned_tree": args.tuned_tree,
                                    "base_generations_sha256": base_hash,
                                    "tuned_generations_sha256": tuned_hash,
                                    "adapter_sha256": source_adapter_hash,
                                    "model_notice": notice,
                                    "paired_quality": paired, "comparison": result,
                                    "scope": "descriptive supplement; unchanged predeclared gates"})
    ctx.results["summary"] = f"paired frozen quality: {paired}; decision {result['decision']}"
    return True
