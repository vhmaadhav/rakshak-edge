"""Evidence-Locked Advisory (ELA v2) explainer evaluation on a T4 GPU.

Launch (internet on; no model attachment needed, the weights come from the pinned Hugging Face
snapshot unsloth/Llama-3.2-1B-Instruct, see src/rakshak/llama_source.py):
  python3 kaggle/build_kernel.py --job llama_ela_eval --gpu t4 --internet \
    --datasets maadhavvh/rakshak-data
  kaggle/launch.sh llama_ela_eval --gpu t4 --internet --datasets maadhavvh/rakshak-data

Design: the detector decides. The system attaches evidence ids and the action (rakshak.evidence
.choose_action). The explainer only verbalises. The prompt has numbered facts (type and a short
sanitised matched text, never the raw SMS) and no ids, no examples and no placeholders. The
verifier (rakshak.verifier.check_points) checks each sentence, and render_advisory swaps any that
fail for the template sentence.

Arms, on one seeded stratified test sample (N scam-side and N benign, spam excluded):
* ``slot``: free sentence per item, English, every sampled package (the ELA v2 arm).
* ``slot_hinglish``: the same arm with a Hinglish prompt, on the first LANG_N/2 rows of each side
  (the same packages as the English rows for those sample ids, so the comparison is paired).
* ``rank``: K=4 deterministic candidate sentences per item. Llama does not write them. It scores
  each candidate by mean log-likelihood under the prompt, and the best one is kept.
* ``freeform``: baseline, the explainer sees the raw SMS and writes free text. Unchanged.

Injection: 5 attack families (rotating), so the first INJECT scam cases (default 150, so 30 per
family) each get a payload appended. ``slot``, ``rank`` and ``freeform`` all run on them. Only
harmful flips count: scam to safe, or the action moved to a permissive one. Wilson 95% intervals
and two-sided Fisher exact tests against the free-form arm are reported.

Faithfulness: exact evidence-id precision and recall (1.0 by construction, since the system
attaches the ids), per-point lexicon consistency for raw and rendered points, and a hook for a
grounded-factuality judge (rakshak.ela_eval.entailment_hook, not wired).

Model: Llama 3.2 1B Instruct, transformers, fp16, greedy decoding, batch 1. Slot max_new_tokens
is 40*N+40 for a package with N evidence items, and generation stops at the first complete JSON
object. Free-form keeps max_new_tokens 128.
If the snapshot download fails, a Llama model folder attached under /kaggle/input is used instead.

Each generation is one Units unit, and every finished generation is appended to
generations.jsonl at once, so a crash keeps the rows already done. Metrics go to metrics.json and
a 20-row-per-arm readability sample to readability_sample.json for human rating.
Job flags (ctx.argv): --n N (even, default 300), --inject K (multiple of 5, default 150), --lang-n
M (even, default 150, at most N). Use small values to check the pipeline before the full run.
"""

import json
import os
import random
import time
from pathlib import Path

from _common import Units, env_report, log_input_tree, seed_everything, write_metrics

from rakshak.ela_eval import (
    INJECTION_FAMILIES,
    aggregate,
    append_payload,
    balanced_subset,
    ela_prompt,
    freeform_prompt,
    injection_case,
    injection_comparisons,
    json_stop_ready,
    max_new_tokens_for,
    package_for_sms,
    rank_candidates_for,
    rank_output_text,
    rank_prefix,
    sample_stratified,
    score_advisory,
    score_output,
    select_candidates,
)
from rakshak.eval.data import eval_mask, load_sms, read_partitions, resolve_data_root
from rakshak.llama_source import resolve_model_dir
from rakshak.verifier import render_advisory

SEED = 13
FREEFORM_MAX_NEW_TOKENS = 128
MODEL_SEARCH_DEPTH = 7
MODEL_ROOT = Path("/kaggle/input")
GEN_FILE = "generations.jsonl"
SPLIT_FILE = "campaign.json"
ARMS = ("slot", "slot_hinglish", "rank", "freeform")
INJECT_ARMS = ("slot", "rank", "freeform")
LANG_BY_ARM = {"slot": "en", "slot_hinglish": "hinglish", "rank": "en"}
READABILITY_ROWS = 20


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


def _generate(tok, model, prompt, max_new_tokens, stop_on_json=False):
    """Greedy decode. Returns (text, new_token_count, latency_s). Only generate() is timed.

    With ``stop_on_json`` the decode stops at the step where the generated text first holds a
    complete JSON object. Its closing brace is the last token that matters.
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    start_len = enc["input_ids"].shape[1]
    criteria = None
    if stop_on_json:

        class _StopAtClosingBrace(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                text = tok.decode(input_ids[0, start_len:], skip_special_tokens=True)
                done = json_stop_ready(text)
                return torch.full(
                    (input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device
                )

        criteria = StoppingCriteriaList([_StopAtClosingBrace()])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            stopping_criteria=criteria,
        )
    torch.cuda.synchronize()
    latency = time.perf_counter() - t0
    new = out[0, start_len:]
    text = tok.decode(new, skip_special_tokens=True)
    return text, int(new.shape[0]), latency


def _mean_logprobs(tok, model, prefix, candidates):
    """Mean per-token log-likelihood of each candidate continuation of ``prefix``.

    Length-normalised, so a longer candidate is not penalised for its length. Only the candidate
    tokens are scored. The prefix is never generated or changed.
    """
    import torch
    import torch.nn.functional as F

    prefix_ids, candidate_ids = _rank_token_ids(tok, prefix, candidates)
    scores = []
    for cand_ids in candidate_ids:
        ids = torch.tensor([prefix_ids + cand_ids], device="cuda")
        with torch.inference_mode():
            logits = model(ids).logits[0]
        start = len(prefix_ids)
        logp = F.log_softmax(logits[start - 1 : -1].float(), dim=-1)
        target = ids[0, start:]
        scores.append(float(logp.gather(1, target.unsqueeze(1)).mean()))
    return scores, sum(len(ids) for ids in candidate_ids)


def _rank_token_ids(tok, prefix, candidates):
    """The exact continuation token IDs used by scoring (including leading spaces)."""
    return (
        tok(prefix, add_special_tokens=False)["input_ids"],
        [tok(c, add_special_tokens=False)["input_ids"] for c in candidates],
    )


def _rank(tok, model, pkg, lang):
    """Score candidates; generated tokens are always zero, targets count tokenizer IDs."""
    per_item = rank_candidates_for(pkg, lang)
    scores, target_tokens = [], 0
    latency = 0.0
    if per_item:
        import torch

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i, cands in enumerate(per_item):
            item_scores, count = _mean_logprobs(
                tok, model, rank_prefix(pkg, i, lang), [" " + c for c in cands]
            )
            scores.append(item_scores)
            target_tokens += count
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0
    choice = select_candidates(scores)
    output = rank_output_text(pkg, choice, lang)
    return output, latency, {
        "advisory": render_advisory(output, pkg, lang),
        "choice": choice,
        "rank_items": len(per_item),
        "rank_empty": not per_item,
        "rank_candidates": sum(len(cands) for cands in per_item),
        "rank_targets": target_tokens,
    }


def run(ctx, *, model_pair=None, frozen_manifest=None):
    seed_everything(SEED)
    env = env_report()
    ctx.log(f"env: {env}")
    n = _flag(ctx.argv, "--n", 300)
    n_inject = _flag(ctx.argv, "--inject", 150)
    n_lang = _flag(ctx.argv, "--lang-n", 150)
    if n_inject > n // 2 or n_inject % len(INJECTION_FAMILIES):
        raise ValueError(
            f"--inject must be at most {n // 2} and a multiple of {len(INJECTION_FAMILIES)}"
        )
    if n_lang > n or n_lang % 2:
        raise ValueError("--lang-n must be an even number no larger than --n")

    def attached_model():
        found = find_hf_model()
        if found is None:
            log_input_tree(ctx.log)
        return found

    model_dir, model_src = resolve_model_dir(attached_model)
    rev = model_src['revision'][:12]
    ctx.log(f"model dir: {model_dir} (source {model_src['source']}, rev {rev})")

    data_root = resolve_data_root()
    sms = load_sms(data_root)
    partitions = read_partitions(data_root / "splits" / SPLIT_FILE)
    test_ids = set(partitions["default"]["test"])
    pool = sms[sms["id"].isin(test_ids) & eval_mask(sms)]
    sample = sample_stratified(pool, n=n, seed=SEED)
    lang_sample = balanced_subset(sample, n=n_lang)
    lang_ids = {str(i) for i in lang_sample["id"]}
    inject_rows = sample[sample["side"] == "scam"].iloc[:n_inject]
    if len(inject_rows) < n_inject:
        raise ValueError(f"only {len(inject_rows)} scam rows for {n_inject} injection cases")
    scam_id_set = {str(i) for i in inject_rows["id"]}
    campaign_ids = {str(r.id): str(r.campaign_id) for r in sample.itertuples()}
    manifest = [
        {
            "sample_id": str(r.id),
            "side": r.side,
            "source": r.source,
            "label_fine": r.label_fine,
            "lang_subset": str(r.id) in lang_ids,
            "injected": str(r.id) in scam_id_set,
        }
        for r in sample.itertuples()
    ]
    if frozen_manifest is not None:
        if manifest != frozen_manifest["rows"] or (n, n_inject, n_lang) != (
            frozen_manifest["n"], frozen_manifest["inject"], frozen_manifest["lang_n"]
        ):
            raise ValueError("frozen evaluation sample changed; refuse inference")
    write_metrics(
        "sample_manifest.json",
        {
            "seed": SEED,
            "n": n,
            "inject": n_inject,
            "lang_n": n_lang,
            "test_pool": len(pool),
            "rows": manifest,
        },
    )
    ctx.log(
        f"test pool {len(pool)} eval rows; sampled {len(sample)}; hinglish subset "
        f"{len(lang_sample)}; injection cases {n_inject}"
    )

    tok, model = model_pair if model_pair is not None else _load_model(ctx, model_dir)
    _generate(tok, model, "warm-up", FREEFORM_MAX_NEW_TOKENS)  # discard: CUDA warm-up
    out_path = ctx.work / GEN_FILE
    handle = out_path.open("a", encoding="utf-8")
    units = Units(ctx)
    scored = {arm: [] for arm in ARMS}
    injected = {arm: [] for arm in INJECT_ARMS}

    def record(arm, sample_id, side, is_injected, sms_text, family=None):
        _, pkg = package_for_sms(sms_text)
        extra = {}
        if arm == "freeform":
            output, new_tokens, latency = _generate(
                tok, model, freeform_prompt(sms_text), FREEFORM_MAX_NEW_TOKENS
            )
            score = score_output(output, pkg, arm)
        elif arm == "rank":
            lang = LANG_BY_ARM[arm]
            output, latency, extra = _rank(tok, model, pkg, lang)
            new_tokens = 0  # candidate likelihood scoring never calls generate()
            score = {**score_advisory(output, pkg, lang), "arm": arm}
        else:
            lang = LANG_BY_ARM[arm]
            output, new_tokens, latency = _generate(
                tok,
                model,
                ela_prompt(pkg, lang=lang),
                max_new_tokens_for(pkg),
                stop_on_json=True,
            )
            score = {**score_advisory(output, pkg, lang), "arm": arm}
            extra = {"advisory": render_advisory(output, pkg, lang)}
        result = {
            **score,
            **extra,
            "sample_id": sample_id,
            "campaign_id": campaign_ids[sample_id],
            "side": side,
            "injected": is_injected,
            "family": family,
            "new_tokens": new_tokens,
            "latency_s": round(latency, 4),
            "output": output,
        }
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        handle.flush()
        return result

    for sample_row in sample.itertuples():
        sms_text = sample_row.text
        for arm in ("slot", "rank", "freeform"):
            unit = {"arm": arm, "sample_id": str(sample_row.id), "injected": False}
            res = units.run(unit, record, arm, str(sample_row.id), sample_row.side, False, sms_text)
            if res is not None:
                scored[arm].append(res)

    for sample_row in lang_sample.itertuples():
        unit = {"arm": "slot_hinglish", "sample_id": str(sample_row.id), "injected": False}
        res = units.run(
            unit,
            record,
            "slot_hinglish",
            str(sample_row.id),
            sample_row.side,
            False,
            sample_row.text,
        )
        if res is not None:
            scored["slot_hinglish"].append(res)

    for index, sample_row in enumerate(inject_rows.itertuples()):
        family, payload = injection_case(index)
        injected_text = append_payload(sample_row.text, payload)
        for arm in INJECT_ARMS:
            unit = {"arm": arm, "sample_id": str(sample_row.id), "injected": True, "family": family}
            res = units.run(
                unit, record, arm, str(sample_row.id), sample_row.side, True, injected_text, family
            )
            if res is not None:
                injected[arm].append(res)
    handle.close()

    by_side = {}
    for arm in ARMS:
        rows = scored[arm]
        by_side[arm] = {
            "all": aggregate(rows),
            "scam": aggregate([r for r in rows if r["side"] == "scam"]),
            "benign": aggregate([r for r in rows if r["side"] == "benign"]),
        }
    # Paired language comparison: the English slot arm restricted to the Hinglish packages.
    lang_compare = {
        "packages": len(lang_ids),
        "en": aggregate([r for r in scored["slot"] if str(r["sample_id"]) in lang_ids]),
        "hinglish": aggregate(scored["slot_hinglish"]),
    }

    clean_scam = {
        arm: [
            r
            for r in scored[arm]
            if r["side"] == "scam" and str(r["sample_id"]) in scam_id_set
        ]
        for arm in INJECT_ARMS
    }
    comparisons = injection_comparisons(clean_scam, injected)
    harm = comparisons["harmful"]
    free_hits, free_n = harm["freeform"]["harmful"], harm["freeform"]["pairs"]

    rng = random.Random(SEED)

    def readable(rows):
        picked = rng.sample(rows, min(READABILITY_ROWS, len(rows)))
        out = []
        for r in picked:
            adv = r["advisory"]
            text = " ".join([adv["summary"], *(p["text"] for p in adv["points"])])
            out.append({"sample_id": r["sample_id"], "side": r["side"], "text": text})
        return out

    write_metrics(
        "readability_sample.json",
        {"slot": readable(scored["slot"]), "rank": readable(scored["rank"])},
    )

    metrics = {
        "job": "llama_ela_eval",
        "env": env,
        "model": model_src,
        "model_dir": str(model_dir),
        "data_root": str(data_root),
        "config": {
            "seed": SEED,
            "n": n,
            "inject": n_inject,
            "lang_n": n_lang,
            "rank_k": 4,
            "injection_families": {k: len(v) for k, v in INJECTION_FAMILIES.items()},
            "permissive_actions": ["IGNORE"],
            "max_new_tokens": {
                "slot": "40*N+40 (N evidence items per package)",
                "slot_hinglish": "40*N+40 (N evidence items per package)",
                "rank": "no generation (scores K=4 candidates per item)",
                "freeform": FREEFORM_MAX_NEW_TOKENS,
            },
            "stop": "first complete JSON object (slot arms)",
            "decoding": "greedy",
            "batch": 1,
            "dtype": "float16",
        },
        "arms": by_side,
        "lang_compare": lang_compare,
        "injection": {
            **comparisons,
            "injected_rows": {arm: len(injected[arm]) for arm in INJECT_ARMS},
        },
        "failures_n": len(ctx.results["failures"]),
        "generations_file": GEN_FILE,
    }
    write_metrics("metrics.json", metrics)
    slot = by_side["slot"]["all"]
    rank = by_side["rank"]["all"]
    free = by_side["freeform"]["all"]
    ctx.results["summary"] = (
        f"slot full {slot.get('verifier_pass_rate')} points {slot.get('per_point_pass_rate')} "
        f"partial {slot.get('partial_fallback_rate')} | rank full {rank.get('verifier_pass_rate')} "
        f"| free json {free.get('json_parse_rate')} "
        f"| harmful slot {harm['slot']['harmful']}/{harm['slot']['pairs']} "
        f"rank {harm['rank']['harmful']}/{harm['rank']['pairs']} "
        f"free {free_hits}/{free_n}"
    )
    ctx.log(f"wrote metrics.json, readability_sample.json and {GEN_FILE}")
    return units.status()
