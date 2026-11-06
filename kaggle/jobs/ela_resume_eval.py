"""Recover frozen FP16 evaluation of a completed QLoRA adapter, without training."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

from _common import write_metrics

from rakshak.llama_source import UNSLOTH_REVISION, resolve_model_dir
from rakshak.llm.llama_notice import write_model_notice

BASE_TREE = "48ed512ef46e0abf629fe7e8da89e1dbca68dbd275516f113900e12eb852932d"
BASE_HASH = "8a898ef0d8edcf92598729ef8bf6e94e7bc5674f7d914d854aed02cf799e04d1"
SOURCE_TREE = "e17dfb992e8b9199712d244fc1f7c474e34f1b3a1b1776a31326514b0bfb034c"
ADAPTER_HASH = "1bdf548fd40ee245ae9a32693af04a89e13448bbe5a9224be7d0808403986e87"
TRAINING_HASH = "1b9c15e3da6edec332ca5172c093a43f70b673d0925827b3f768a1d481ef5fa5"


def training_source(root):
    matches = []
    for path in Path(root).rglob("results.json"):
        result = json.loads(path.read_text())
        if result.get("tree_sha256") == SOURCE_TREE:
            if (result.get("status") != "error" or result.get("error") !=
                    "ImportError: Found an incompatible version of torchao. Found version 0.10.0, "
                    "but only versions above 0.16.0 are supported"):
                raise ValueError("unexpected training completion state")
            folder = path.parent
            for rel, expected in (("training.json", TRAINING_HASH),
                                  ("Llama-Rakshak-ELA-QLoRA/adapter_model.safetensors",
                                   ADAPTER_HASH)):
                if hashlib.sha256((folder / rel).read_bytes()).hexdigest() != expected:
                    raise ValueError("saved training/adapter hash mismatch")
            matches.append(folder)
    if len(matches) != 1:
        raise ValueError("exactly one pinned training source required")
    return matches[0]


def run(ctx):
    argparse.ArgumentParser().parse_args(ctx.argv)
    source = training_source(ctx.input_root)
    # PEFT 0.21.2 checks optional torchao even for ordinary FP16 Linear layers.
    # This experiment uses bitsandbytes, never torchao; remove the incompatible image extra.
    try:
        torchao = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        torchao = None
    if torchao is not None:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
    import llama_ela_eval as evaluation
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    baseline = []
    for path in Path(ctx.input_root).rglob("generations.jsonl"):
        receipt = path.with_name("results.json")
        if receipt.is_file():
            result = json.loads(receipt.read_text())
            if result.get("tree_sha256") == BASE_TREE and result.get("status") == "ok":
                baseline.append(path)
    if len(baseline) != 1 or hashlib.sha256(baseline[0].read_bytes()).hexdigest() != BASE_HASH:
        raise ValueError("frozen successful base output required")
    frozen = json.loads(baseline[0].with_name("sample_manifest.json").read_text())
    model_dir, model_source = resolve_model_dir(lambda: None)
    if model_source["revision"] != UNSLOTH_REVISION or not torch.cuda.is_available():
        raise ValueError("pinned Llama and Kaggle CUDA required")
    adapter = ctx.work / "Llama-Rakshak-ELA-QLoRA"
    shutil.copytree(source / adapter.name, adapter)
    write_model_notice(adapter)
    shutil.copyfile(source / "training.json", ctx.work / "training.json")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    base = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float16).to("cuda")
    tuned = PeftModel.from_pretrained(base, str(adapter)).eval()
    tuned.config.use_cache = True
    ctx.argv = ["--n", "300", "--inject", "150", "--lang-n", "150"]
    outcome = evaluation.run(ctx, model_pair=(tok, tuned), frozen_manifest=frozen)
    if outcome is not True and outcome != "ok":
        raise ValueError("incomplete frozen evaluation; no quality decision")
    final_hash = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    if final_hash != ADAPTER_HASH:
        raise ValueError("adapter weights changed")
    write_metrics("recovery.json", {"source_tree": SOURCE_TREE, "adapter_sha256": ADAPTER_HASH,
                                   "training_sha256": TRAINING_HASH, "removed_torchao": torchao,
                                   "training_repeated": False, "eval_dtype": "fp16",
                                   "source": model_source})
    return True
