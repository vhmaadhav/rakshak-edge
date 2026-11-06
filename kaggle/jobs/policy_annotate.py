"""Qwen-only annotation pilot on the pre-frozen 200 training-scope audit IDs.

No classifier fitting, evaluation labels or approved training labels. All text
and proposals stay in private outputs. No paid APIs or local model execution.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
from _common import env_report, seed_everything, write_metrics

from rakshak.eval import data as D
from rakshak.llm import budget, chunks
from rakshak.llm.policy_annotation import (
    ATTEMPTS,
    BATCH_SIZE,
    MAX_NEW_TOKENS,
    MAX_PROMPT_TOKENS,
    MINIMUM_VALID,
    POLICY_INSTRUCTIONS,
    SOURCE_SHA256,
    parse_candidate,
    prompt_manifest,
    system_prompt,
)
from rakshak.llm.policy_training import (
    LICENCE,
    MODEL,
    REVISION,
    annotation_spec,
    assert_proposal_scope,
    ids_hash,
    train_null_ids,
)
from rakshak.llm.validate import check_chunk

SCOPE_SHA = "08b21ff533e5227e071a7345b1bda662209abac3a87d0fcfb0e86bf425041076"


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verified_inputs(input_root: Path):
    candidates = list(input_root.rglob("policy_annotation_scope.json"))
    if len(candidates) != 1 or sha(candidates[0]) != SCOPE_SHA:
        raise ValueError("exact frozen annotation scope required")
    scope = json.loads(candidates[0].read_text())
    root = D.resolve_data_root()
    if sha(root / "sms.parquet") != scope["sms_sha256"]:
        raise ValueError("original SMS hash mismatch")
    for name, expected in scope["split_files"].items():
        if sha(root / "splits" / name) != expected:
            raise ValueError("frozen split hash mismatch")
    sms = D.load_sms(root)
    for fold, parts in scope["frozen_fold_roles"].items():
        if train_null_ids(sms, parts) != scope["eligible_ids"][fold]:
            raise ValueError("training annotation eligibility changed")
    ids = scope["audit_ids"]
    if len(ids) != 200 or len(set(ids)) != 200 or ids_hash(ids) != scope["audit_ids_sha256"]:
        raise ValueError("exact original 200 audit IDs required")
    audit_file = candidates[0].parent / "policy_human_audit_200.parquet"
    if sha(audit_file) != scope["audit_file_sha256"]:
        raise ValueError("original blinded human sample hash mismatch")
    return sms, scope, audit_file


def prepare_tasks(sms, scope, work):
    """Assign each audit ID to its first eligible fold, generate once, never a global label."""
    owners = {}
    for rid in scope["audit_ids"]:
        eligible = [fold for fold, ids in sorted(scope["eligible_ids"].items()) if rid in ids]
        if not eligible:
            raise ValueError("audit ID has no frozen training scope")
        owners[rid] = eligible[0]
    tasks = []
    for fold, parts in sorted(scope["frozen_fold_roles"].items()):
        selected = [rid for rid in scope["audit_ids"] if owners[rid] == fold]
        if not selected:
            continue
        spec = annotation_spec(POLICY_INSTRUCTIONS, allowed_ids=selected)
        folder = work / "chunks" / hashlib.sha256(fold.encode()).hexdigest()[:12]
        exported = chunks.export_spec(spec, folder, sms, parts, chunk_size=1)
        for path in exported:
            row = json.loads(path.read_text())
            tasks.append({"id": row["id"], "text": row["text"], "fold": fold,
                          "task_path": path,
                          "result_path": path.parent.parent / "results" / path.name,
                          "ledger_root": folder, "chunk": path.name})
    if len(tasks) != 200 or {t["id"] for t in tasks} != set(scope["audit_ids"]):
        raise ValueError("chunk inventory changed from the frozen human sample")
    return sorted(tasks, key=lambda item: item["id"])


def pretrained(work):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    folder = Path(snapshot_download(
        MODEL, revision=REVISION, cache_dir="/kaggle/temp/rakshak-policy-model",
        allow_patterns=[*SOURCE_SHA256, "tokenizer.json", "tokenizer_config.json",
                        "generation_config.json", "merges.txt", "vocab.json"]))
    for name, expected in SOURCE_SHA256.items():
        if sha(folder / name) != expected:
            raise ValueError("pinned Qwen source checksum mismatch")
    source = {"model": MODEL, "revision": REVISION, "licence": LICENCE,
              "files": {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
              "compute": "private Kaggle only; no API/provider calls"}
    (work / "pretrained_source.json").write_text(json.dumps(source, indent=2) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True,
                                               trust_remote_code=False, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        folder, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.float16, device_map=0, attn_implementation="sdpa").eval()
    return tokenizer, model, source


def generate(tokenizer, model, tasks, attempt):
    import torch
    from transformers import GenerationConfig

    prompts = [tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt(attempt)},
         {"role": "user", "content": chunks.dump_line({"id": t["id"], "text": t["text"]})}],
        tokenize=False, add_generation_prompt=True) for t in tasks]
    fits = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) <= MAX_PROMPT_TOKENS
            for p in prompts]
    outputs = [None] * len(tasks)
    kept = [i for i, okay in enumerate(fits) if okay]
    if not kept:
        return outputs
    inputs = tokenizer([prompts[i] for i in kept], padding=True, return_tensors="pt",
                       add_special_tokens=False).to(model.device)
    config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1, use_cache=True,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=model.generation_config.eos_token_id)
    with torch.inference_mode():
        generated = model.generate(**inputs, generation_config=config)
    decoded = tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
    for i, raw in zip(kept, decoded, strict=True):
        outputs[i] = raw
    return outputs


def source_archive(src, output):
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(src).rglob("*.py")):
            archive.write(path, arcname=path.relative_to(src))
        archive.write(__file__, arcname="policy_annotate.py")


def run(ctx):
    import torch
    from threadpoolctl import threadpool_limits

    if (not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0)
            or not str(ctx.work).startswith("/kaggle/working")):
        raise ValueError("fixed private Kaggle T4 annotation job required")
    seed_everything(13)
    torch.manual_seed(13)
    torch.set_num_threads(2)
    threadpool_limits(2)
    sms, scope, audit_file = verified_inputs(Path(ctx.input_root))
    tasks = prepare_tasks(sms, scope, ctx.work)
    indexed = sms.set_index("id")
    breakdown = {}
    for fold, eligible in scope["eligible_ids"].items():
        values = indexed.loc[eligible, "label_scam"]
        breakdown[fold] = {"policy_null": len(eligible), "fraud_unknown": int(values.isna().sum()),
                           "fraud_known_0": int(values.eq(0).sum()),
                           "fraud_known_1": int(values.eq(1).sum())}
    recipe = {**prompt_manifest(), "scope_sha256": SCOPE_SHA, "model": MODEL,
              "revision": REVISION, "licence": LICENCE, "source_expected": SOURCE_SHA256,
              "scope_counts": breakdown, "advisory_training_targets": 0,
              "human_review": "PENDING, genuine manual labels required"}
    # Freeze recipe/scopes before model loading or generation. Never change it on outcomes.
    (ctx.work / "annotation_recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    shutil.copyfile(audit_file, ctx.work / audit_file.name)
    source_archive(ctx.src, ctx.work / "annotation_source.zip")
    tokenizer, model, source = pretrained(ctx.work)
    accepted, rejected, attempt_reasons = {}, {}, Counter()
    attempt_log = ctx.work / "annotation_attempts.jsonl"
    for attempt in range(ATTEMPTS):
        pending = [t for t in tasks if t["id"] not in accepted
                   and rejected.get(t["id"]) != "prompt_too_long"]
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start:start + BATCH_SIZE]
            raws = generate(tokenizer, model, batch, attempt)
            for task, raw in zip(batch, raws, strict=True):
                rid = task["id"]
                known = indexed.loc[rid, "label_scam"]
                known = None if pd.isna(known) else int(known)
                obj, reason = (None, "prompt_too_long") if raw is None else parse_candidate(
                    raw, rid, task["text"], known_scam=known)
                if obj is not None:
                    path = task["result_path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(chunks.dump_line(obj) + "\n")
                    # Existing checker is mandatory in addition to the strict contract.
                    checked, _ = check_chunk(task["task_path"], path, "annotate")
                    if not checked.ok:
                        raise ValueError("accepted annotation failed original chunk checker")
                    accepted[rid] = obj
                    rejected.pop(rid, None)
                else:
                    rejected[rid] = reason
                attempt_reasons[reason] += 1
                with attempt_log.open("a") as stream:
                    stream.write(chunks.dump_line({"id": rid, "attempt": attempt + 1,
                                                   "reason": reason, "response": raw}) + "\n")
            ctx.log(f"annotation attempt{attempt + 1}: accepted={len(accepted)}/200")
    fold_counts = {}
    for fold, parts in scope["frozen_fold_roles"].items():
        ids = sorted(set(accepted) & set(scope["eligible_ids"][fold]))
        assert_proposal_scope(sms, parts, ids)
        fold_counts[fold] = len(ids)
        records = [{"id": rid, "fold": fold, "proposal_policy_label": accepted[rid]["policy_label"],
                    "confidence": accepted[rid]["confidence"], "model": MODEL, "revision": REVISION,
                    "approved_for_training": False, "human_reviewed": False} for rid in ids]
        name = hashlib.sha256(fold.encode()).hexdigest()[:12]
        columns = ["id", "fold", "proposal_policy_label", "confidence", "model",
                   "revision", "approved_for_training", "human_reviewed"]
        pd.DataFrame(records, columns=columns).to_parquet(
            ctx.work / f"policy_proposals_{name}.parquet", index=False)
    for task in tasks:
        status = "validated_ok" if task["id"] in accepted else "validated_failed"
        budget.record_chunk(task["ledger_root"], job="POLICY_OPEN",
                            model=f"{MODEL}@{REVISION}", chunk=task["chunk"], messages=1,
                            started=None, finished=budget.now_iso(), status=status)
    kept_rows = [{"id": rid, **obj} for rid, obj in sorted(accepted.items())]
    (ctx.work / "accepted_annotations.jsonl").write_text(
        "".join(chunks.dump_line(obj) + "\n" for obj in kept_rows))
    report = {"technical_gate": "go" if len(accepted) >= MINIMUM_VALID else "kill",
              "accepted": len(accepted), "rejected": len(rejected),
              "attempt_reasons": dict(attempt_reasons),
              "final_rejection_reasons": dict(Counter(rejected.values())),
              "fold_counts": fold_counts,
              "proposed_policy_counts": dict(Counter(o["policy_label"] for o in accepted.values())),
              "recipe_sha256": sha(ctx.work / "annotation_recipe.json"),
              "scope_sha256": SCOPE_SHA, "source": source, "environment": env_report(),
              "approved_for_training": False, "eval_labels_modified": 0,
              "human_accuracy": "UNMEASURABLE: human review pending",
              "source_archive_sha256": sha(ctx.work / "annotation_source.zip"),
              "accepted_file_sha256": sha(ctx.work / "accepted_annotations.jsonl"),
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "benchmark_scope": "training annotation only; no external/test performance claim"}
    write_metrics("policy_annotation.json", report)
    ctx.results["summary"] = (
        f"annotation technical {report['technical_gate']}, accepted={len(accepted)}/200; "
        "human review pending, training approval false")
    return True
