"""Evidence-Locked Advisory (ELA) explainer evaluation on a T4 GPU.

Launch (after the Kaggle licence consent in docs/EXECUTORCH_PLAN.md, section "Kaggle weights"):
  kaggle/launch.sh llama_ela_eval --gpu t4 --internet \
      --datasets maadhavvh/rakshak-data \
      --models metaresearch/llama-3.2/transformers/1b-instruct

Model: Llama 3.2 1B Instruct, transformers, fp16, greedy decoding, max_new_tokens 128, batch 1.
Sample: a stratified test-partition sample of the campaign split (N scam-side and N benign,
spam excluded), seeded. The ELA arm sees the evidence package only. The free-form arm sees the
raw SMS. The injection arm appends an injection line to the first INJECT scam cases.

Each generation is one Units unit, and every finished generation is appended to
generations.jsonl at once, so a crash keeps the rows already done. Metrics go to metrics.json.
Job flags (ctx.argv): --n N (even, default 300), --inject K (default 50). Use small values to
check the pipeline before the full run.
"""

import json
import os
import time
from pathlib import Path

from _common import Units, env_report, log_input_tree, seed_everything, write_metrics

from rakshak.ela_eval import (
    aggregate,
    append_injection,
    ela_prompt,
    freeform_prompt,
    injection_flips,
    package_for_sms,
    sample_stratified,
    score_output,
)
from rakshak.eval.data import eval_mask, load_sms, read_partitions, resolve_data_root

SEED = 13
MAX_NEW_TOKENS = 128
MODEL_SEARCH_DEPTH = 7
MODEL_ROOT = Path("/kaggle/input")
GEN_FILE = "generations.jsonl"
SPLIT_FILE = "campaign.json"


def _flag(argv, name, default):
    if name in argv:
        i = argv.index(name)
        if i + 1 >= len(argv):
            raise ValueError(f"{name} needs a value")
        return int(argv[i + 1])
    return default


def find_hf_model(root=MODEL_ROOT, depth=MODEL_SEARCH_DEPTH):
    """First folder under root with a Llama config.json, tokenizer.json and safetensors."""
    root = Path(root)
    if not root.is_dir():
        return None
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        if len(here.parts) - base_depth >= depth:
            dirnames[:] = []
        if {"config.json", "tokenizer.json"} <= set(filenames) and any(
            f.endswith(".safetensors") for f in filenames
        ):
            text = (here / "config.json").read_text(errors="replace")
            if '"llama"' in text:
                return here
    return None


def _load_model(ctx, path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; launch with --gpu t4")
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float16)
    if next(model.parameters()).dtype != torch.float16:
        model = model.to(torch.float16)
    model = model.to("cuda").eval()
    ctx.log(f"loaded {path} as {next(model.parameters()).dtype} on {torch.cuda.get_device_name(0)}")
    return tok, model


def _generate(tok, model, prompt):
    """Greedy decode. Returns (text, new_token_count, latency_s). Only generate() is timed."""
    import torch

    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    start_len = enc["input_ids"].shape[1]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    torch.cuda.synchronize()
    latency = time.perf_counter() - t0
    new = out[0, start_len:]
    text = tok.decode(new, skip_special_tokens=True)
    return text, int(new.shape[0]), latency


def run(ctx):
    seed_everything(SEED)
    env = env_report()
    ctx.log(f"env: {env}")
    n = _flag(ctx.argv, "--n", 300)
    n_inject = _flag(ctx.argv, "--inject", 50)
    if n_inject > n // 2:
        raise ValueError("--inject cannot exceed half of --n")

    model_dir = find_hf_model()
    if model_dir is None:
        log_input_tree(ctx.log)
        raise FileNotFoundError(
            "no Llama HF folder (config.json, tokenizer.json, *.safetensors) under /kaggle/input; "
            "attach --models metaresearch/llama-3.2/transformers/1b-instruct and accept the licence"
        )
    ctx.log(f"model dir: {model_dir}")

    data_root = resolve_data_root()
    sms = load_sms(data_root)
    partitions = read_partitions(data_root / "splits" / SPLIT_FILE)
    test_ids = set(partitions["default"]["test"])
    pool = sms[sms["id"].isin(test_ids) & eval_mask(sms)]
    sample = sample_stratified(pool, n=n, seed=SEED)
    scam_ids = sample.loc[sample["side"] == "scam", "id"].tolist()[:n_inject]
    manifest = [
        {"sample_id": str(r.id), "side": r.side, "source": r.source, "label_fine": r.label_fine}
        for r in sample.itertuples()
    ]
    write_metrics(
        "sample_manifest.json",
        {"seed": SEED, "n": n, "inject": n_inject, "test_pool": len(pool), "rows": manifest},
    )
    ctx.log(f"test pool {len(pool)} eval rows; sampled {len(sample)}; injection cases {n_inject}")

    tok, model = _load_model(ctx, model_dir)
    _generate(tok, model, "warm-up")  # discard: first call includes CUDA kernel warm-up
    out_path = ctx.work / GEN_FILE
    handle = out_path.open("a", encoding="utf-8")
    units = Units(ctx)
    scored = {"ela": [], "freeform": []}
    injected = {"ela": [], "freeform": []}

    def record(arm, sample_id, side, is_injected, sms_text):
        _, pkg = package_for_sms(sms_text)
        output, new_tokens, latency = _generate(
            tok, model, ela_prompt(pkg) if arm == "ela" else freeform_prompt(sms_text)
        )
        score = score_output(output, pkg, arm)
        result = {
            **score,
            "sample_id": sample_id,
            "side": side,
            "injected": is_injected,
            "new_tokens": new_tokens,
            "latency_s": round(latency, 4),
            "output": output,
        }
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        handle.flush()
        return result

    for sample_row in sample.itertuples():
        sms_text = sample_row.text
        for arm in ("ela", "freeform"):
            unit = {"arm": arm, "sample_id": str(sample_row.id), "injected": False}
            res = units.run(unit, record, arm, str(sample_row.id), sample_row.side, False, sms_text)
            if res is not None:
                scored[arm].append(res)

    inject_rows = sample[sample["id"].astype(str).isin([str(i) for i in scam_ids])]
    for sample_row in inject_rows.itertuples():
        injected_text = append_injection(sample_row.text)
        for arm in ("ela", "freeform"):
            unit = {"arm": arm, "sample_id": str(sample_row.id), "injected": True}
            res = units.run(
                unit, record, arm, str(sample_row.id), sample_row.side, True, injected_text
            )
            if res is not None:
                injected[arm].append(res)
    handle.close()

    by_side = {}
    for arm in ("ela", "freeform"):
        rows = scored[arm]
        by_side[arm] = {
            "all": aggregate(rows),
            "scam": aggregate([r for r in rows if r["side"] == "scam"]),
            "benign": aggregate([r for r in rows if r["side"] == "benign"]),
        }
    clean_scam = {
        arm: [
            r
            for r in scored[arm]
            if r["side"] == "scam" and str(r["sample_id"]) in {str(i) for i in scam_ids}
        ]
        for arm in ("ela", "freeform")
    }
    flips = {arm: injection_flips(clean_scam[arm], injected[arm]) for arm in ("ela", "freeform")}
    metrics = {
        "job": "llama_ela_eval",
        "env": env,
        "model_dir": str(model_dir),
        "data_root": str(data_root),
        "config": {
            "seed": SEED,
            "n": n,
            "inject": n_inject,
            "max_new_tokens": MAX_NEW_TOKENS,
            "decoding": "greedy",
            "batch": 1,
            "dtype": "float16",
        },
        "arms": by_side,
        "injection": {"flips": flips, "injected_rows": {a: len(injected[a]) for a in injected}},
        "failures_n": len(ctx.results["failures"]),
        "generations_file": GEN_FILE,
    }
    write_metrics("metrics.json", metrics)
    ela = by_side["ela"]["all"]
    free = by_side["freeform"]["all"]
    ctx.results["summary"] = (
        f"ela json {ela.get('json_parse_rate')} verif {ela.get('verifier_pass_rate')} "
        f"| free json {free.get('json_parse_rate')} verif {free.get('verifier_pass_rate')} "
        f"| flips ela {flips['ela']['flips']} free {flips['freeform']['flips']}/{n_inject}"
    )
    ctx.log(f"wrote metrics.json and {GEN_FILE}")
    return units.status()
