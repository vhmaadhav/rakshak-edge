"""Generate training-only ELA targets on Kaggle. Built with Llama.

No corpus is loaded. Existing deterministic templates and pinned Llama paraphrases only;
verifier plus scorer reject unsupported IDs/claims before composition and deduplication.
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict

from _common import env_report, seed_everything, write_metrics

from rakshak import prompts
from rakshak.llama_source import UNSLOTH_REPO, UNSLOTH_REVISION, resolve_model_dir
from rakshak.llm import ela_training as T
from rakshak.rules import EVIDENCE_TYPES

CARD = (f"https://huggingface.co/{UNSLOTH_REPO}/blob/{UNSLOTH_REVISION}/README.md")


def paraphrase_prompt(example):
    point = json.loads(example.target)["points"][0]
    instruction = (
        "Rewrite the provided sentence using different wording in the same language. "
        "Preserve its specific evidence meaning. Do not add facts, identifiers, advice, actions, "
        "or reassurance. Return exactly one JSON object with only the key points, whose value "
        "is an array containing one sentence. No markdown. "
        f"Language: {example.lang}. Sentence: {json.dumps(point, ensure_ascii=False)}"
    )
    return (prompts.BOS + prompts.SOT + "user" + prompts.EOT_HDR + "\n\n" + instruction
            + prompts.EOT + prompts.SOT + "assistant" + prompts.EOT_HDR + "\n\n")


def points_envelope(output):
    """Only wrap a teacher JSON string; never rewrite its sentence or repair malformed JSON."""
    value = json.loads(output)
    return json.dumps({"points": [value]}, ensure_ascii=False) if isinstance(value, str) else output


def run(ctx):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--packages", type=int, default=1200)
    p.add_argument("--attempts", type=int, default=8)
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args(ctx.argv)
    seed_everything(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("generation requires Kaggle CUDA")
    packages = T.synthetic_packages(args.packages, args.seed)
    singletons = packages[1:len(EVIDENCE_TYPES) + 1]
    templates = [T.template_example(pkg, lang, variant=v, seed=args.seed)
                 for pkg in packages for lang in ("en", "hinglish") for v in range(4)]
    bank = defaultdict(list)
    for pkg in singletons:
        for lang in ("en", "hinglish"):
            bank[(pkg.evidence[0].type, lang)] = [
                T.template_example(pkg, lang, variant=v, seed=args.seed) for v in range(4)
            ]
    model_dir, source = resolve_model_dir(lambda: None)
    if source["revision"] != UNSLOTH_REVISION:
        raise ValueError("teacher must be the exact pinned model, no attachment fallback")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float16).to("cuda")
    model.eval()
    counts, rejects = Counter(), Counter()
    accepted = []
    with (ctx.work / "paraphrase_audit.jsonl").open("w") as audit:
        for pkg in singletons:
            for lang in ("en", "hinglish"):
                for attempt in range(args.attempts):
                    example = T.template_example(pkg, lang, variant=attempt, seed=args.seed)
                    prompt = paraphrase_prompt(example)
                    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
                    with torch.inference_mode():
                        output = model.generate(**enc, max_new_tokens=96, do_sample=True,
                                                temperature=0.8, top_p=0.9,
                                                pad_token_id=tok.eos_token_id)
                    text = tok.decode(output[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
                    key = f"{lang}/{pkg.evidence[0].type}"
                    record = {"key": key, "attempt": attempt, "output": text}
                    counts[f"attempted/{lang}"] += 1
                    try:
                        parent = T.llama_paraphrase_example(
                            pkg, lang, points_envelope(text), teacher_repo=UNSLOTH_REPO,
                            teacher_revision=UNSLOTH_REVISION, teacher_seed=args.seed,
                            generated_on="kaggle", seed=args.seed,
                        )
                    except (ValueError, TypeError, KeyError) as exc:
                        reason = str(exc)
                        record.update(kept=False, reason=reason)
                        rejects[reason] += 1
                        counts[f"rejected/{lang}"] += 1
                    else:
                        bank[(pkg.evidence[0].type, lang)].append(parent)
                        accepted.append(parent)
                        record.update(kept=True, example_id=parent.example_id,
                                      envelope="verbatim JSON-string wrapper or original points")
                        counts[f"kept/{lang}"] += 1
                    audit.write(json.dumps(record, ensure_ascii=False) + "\n")
                    audit.flush()
                ctx.log(f"paraphrases {key}: {counts}")
    # Parents are saved in full for composition provenance, even if a composed row deduplicates.
    parents = T.deduplicate_examples([e for pool in bank.values() for e in pool])
    rng = random.Random(args.seed)
    composed = [T.compose_example(pkg, lang,
                                 [rng.choice(bank[(e.type, lang)]) for e in pkg.evidence],
                                 seed=args.seed)
                for pkg in packages for lang in ("en", "hinglish") for _ in range(2)]
    candidates = templates + accepted + composed
    examples = T.deduplicate_examples(candidates)
    T.assert_composition_disjoint(examples)
    hashes = {}
    for name, rows in [("train", [e for e in examples if e.partition == "train"]),
                       ("dev", [e for e in examples if e.partition == "dev"]),
                       ("parents", parents)]:
        path = ctx.work / f"ela_{name}.jsonl"
        path.write_text("".join(json.dumps(e.to_dict(), ensure_ascii=False) + "\n" for e in rows))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        counts[f"unique/{name}"] = len(rows)
    if not counts["unique/dev"] or not accepted:
        raise ValueError("no development examples or no gated paraphrases; stop before training")
    counts["deduplicated"] = len(candidates) - len(examples)
    manifest = {
        "job": "ela_data", "config": vars(args), "env": env_report(), "counts": dict(counts),
        "rejection_reasons": dict(rejects), "file_sha256": hashes,
        "model": source, "licence": "Llama 3.2 Community License (llama3.2)",
        "model_card": CARD, "attribution": "Built with Llama",
        "template_sha256": T.template_fingerprint(),
        "split": "synthetic evidence-composition hash; all languages/order/risk variants grouped",
        "corpus_access": "none; no SMS, evaluation or external rows loaded",
        "target_sources": dict(Counter(e.target_source for e in examples)),
    }
    write_metrics("ela_data_manifest.json", manifest)
    (ctx.work / "README.md").write_text(
        "# Llama-Rakshak ELA synthetic training targets\n\nBuilt with Llama.\n\n"
        f"Teacher: {CARD}\n\nLlama 3.2 Community License. Private, training only. "
        "Existing deterministic templates and pinned Llama paraphrases. No corpus messages. "
        "See ela_data_manifest.json and paraphrase_audit.jsonl for gates and provenance.\n"
    )
    ctx.results["summary"] = f"gated synthetic targets: {dict(counts)}"
    return True
