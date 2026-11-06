"""Verifier-gated NF4 QLoRA on Kaggle T4, then frozen ELA evaluation. Built with Llama."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
from pathlib import Path

from _common import env_report, write_metrics

from rakshak import prompts
from rakshak.llama_source import UNSLOTH_REVISION, resolve_model_dir
from rakshak.llm import ela_training as T
from rakshak.llm.ela_comparison import compare
from rakshak.llm.llama_notice import write_model_notice

BASE_TREE = "48ed512ef46e0abf629fe7e8da89e1dbca68dbd275516f113900e12eb852932d"
BASE_GENERATIONS = "8a898ef0d8edcf92598729ef8bf6e94e7bc5674f7d914d854aed02cf799e04d1"


def run(ctx):
    # This NF4 experiment does not use torchao; the image extra can break PEFT FP16 reload.
    try:
        importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
    import ela_recompute
    import llama_ela_eval as evaluation
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    p = argparse.ArgumentParser()
    p.add_argument("--data-manifest-sha256", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args(ctx.argv)
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle T4 CUDA required")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    manifests = [p for p in Path(ctx.input_root).rglob("ela_data_manifest.json")
                 if hashlib.sha256(p.read_bytes()).hexdigest() == args.data_manifest_sha256]
    if len(manifests) != 1:
        raise ValueError("expected exactly one hash-pinned private training dataset")
    data_path = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    if (manifest["template_sha256"] != T.template_fingerprint()
            or manifest["model"]["revision"] != UNSLOTH_REVISION):
        raise ValueError("training gate/templates or pinned teacher changed")
    examples = {}
    for name in ("train", "dev"):
        path = data_path / f"ela_{name}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["file_sha256"][path.name]:
            raise ValueError("training dataset file hash mismatch")
        examples[name] = [T.TrainingExample(**json.loads(line))
                          for line in path.read_text().splitlines()]
        if any(e.partition != name for e in examples[name]):
            raise ValueError("dataset partition metadata mismatch")
    T.assert_composition_disjoint(examples["train"] + examples["dev"])
    baseline_paths = []
    for path in Path(ctx.input_root).rglob("generations.jsonl"):
        result_path = path.with_name("results.json")
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if result.get("tree_sha256") == BASE_TREE and result.get("status") == "ok":
                baseline_paths.append(path)
    if len(baseline_paths) != 1:
        raise ValueError("one completed frozen baseline kernel required")
    baseline_path = baseline_paths[0]
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != BASE_GENERATIONS:
        raise ValueError("frozen baseline generations hash mismatch")
    frozen_manifest = json.loads(baseline_path.with_name("sample_manifest.json").read_text())
    packages = T.synthetic_packages(manifest["config"]["packages"], manifest["config"]["seed"])
    package_by_prompt = {prompts.build_points_chat_prompt(pkg, lang): pkg
                         for pkg in packages for lang in ("en", "hinglish")}
    dev = []
    for lang in ("en", "hinglish"):
        candidates = sorted((e for e in examples["dev"] if e.lang == lang),
                            key=lambda e: e.example_id)
        dev.extend(candidates[:24])
    if len(dev) != 48:
        raise ValueError("48 frozen synthetic-dev examples required")
    model_dir, source = resolve_model_dir(lambda: None)
    if source["revision"] != UNSLOTH_REVISION:
        raise ValueError("base must be pinned Llama")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    tok.pad_token_id = tok.eos_token_id
    features, dropped = [], 0
    for example in examples["train"]:
        try:
            features.append(T.completion_features(example, tok, max_length=768))
        except ValueError:
            dropped += 1
    if not features:
        raise ValueError("no complete untruncated targets")

    def collate(rows):
        width = max(len(row["input_ids"]) for row in rows)
        return {key: torch.tensor([row[key] + [pad] * (width - len(row[key])) for row in rows])
                for key, pad in (("input_ids", tok.pad_token_id), ("labels", -100),
                                 ("attention_mask", 0))}

    loader = DataLoader(features, batch_size=2, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), quantization_config=quant,
                                                device_map={"": 0}, dtype=torch.float16)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                           target_modules="all-linear", bias="none",
                                           task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda")
    adapter = ctx.work / "Llama-Rakshak-ELA-QLoRA"

    def dev_score(current_model):
        current_model.eval()
        current_model.config.use_cache = True
        rows = []
        for example in dev:
            package = package_by_prompt[example.prompt]
            text, _, _ = evaluation._generate(tok, current_model, example.prompt,
                                              evaluation.max_new_tokens_for(package), True)
            rows.append(evaluation.score_advisory(text, package, example.lang))
        current_model.config.use_cache = False
        return {"verifier_pass": sum(r["verifier_ok"] and not r["unsupported_n"]
                                     and not r["safe_claim"] for r in rows) / len(rows),
                "point_pass": sum(r["points_pass_n"] for r in rows)
                / sum(r["points_n"] for r in rows), "n": len(rows)}

    history, best, stale = [], -1.0, 0
    base_dev = dev_score(model)
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            # Each full or trailing accumulation window has the correct normalisation.
            window = min(8, len(loader) - (step // 8) * 8)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss; refuse evaluation")
            scaler.scale(loss / window).backward()
            loss_sum += float(loss.detach())
            if (step + 1) % 8 == 0 or step + 1 == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        score = dev_score(model)
        history.append({"epoch": epoch + 1, "loss": loss_sum / len(loader), **score})
        ctx.log(f"epoch {epoch + 1}: {history[-1]}")
        if score["verifier_pass"] > best:
            best, stale = score["verifier_pass"], 0
            model.save_pretrained(adapter)
            tok.save_pretrained(adapter)
        else:
            stale += 1
        write_metrics("training.json", {"history": history, "base_dev": base_dev})
        if stale >= 1:
            break
    notice = write_model_notice(adapter)
    training = {"job": "ela_qlora", "config": vars(args), "model_notice": notice,
                "manifest_sha256": args.data_manifest_sha256,
                "base": source, "history": history, "base_dev": base_dev,
                "selected_by": "synthetic-dev verifier pass only, earliest tie; patience 1",
                "train_rows": len(features), "oversize_rejected": dropped, "env": env_report(),
                "runtime": {p: importlib.metadata.version(p) for p in
                            ("torch", "transformers", "peft", "bitsandbytes", "accelerate")},
                "eval_dtype": "fp16 pinned base plus saved adapter (NF4 training)",
                "dev_ids": [e.example_id for e in dev], "attribution": "Built with Llama"}
    write_metrics("training.json", training)
    del model, opt, loader, features, loss, batch, scaler
    gc.collect()
    torch.cuda.empty_cache()
    # Compare to the prior fp16 base in the same inference dtype; no quantization confound.
    base_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.float16).to("cuda")
    tuned = PeftModel.from_pretrained(base_model, str(adapter)).eval()
    tuned.config.use_cache = True
    ctx.argv = ["--n", "300", "--inject", "150", "--lang-n", "150"]
    ok = evaluation.run(ctx, model_pair=(tok, tuned), frozen_manifest=frozen_manifest)
    if ok is not True and ok != "ok":
        raise ValueError("incomplete frozen evaluation; no gate decision")
    rows = [json.loads(line) for line in (ctx.work / "generations.jsonl").read_text().splitlines()]
    baseline = [json.loads(line) for line in baseline_path.read_text().splitlines()]
    result = compare(baseline, rows)
    result["intervals"] = {
        arm: ela_recompute.summarize([r for r in rows if not r["injected"] and r["arm"] == arm])
        for arm in ("slot", "slot_hinglish", "rank")
    }
    result["baseline_tree"] = BASE_TREE
    result["baseline_generations_sha256"] = BASE_GENERATIONS
    write_metrics("qlora_comparison.json", result)
    ctx.results["summary"] = f"B decision: {result['decision']}; EN {result['language']['slot']}"
    return True
