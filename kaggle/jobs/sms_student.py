"""On-device student models for the SMS detector, trained on a Kaggle GPU (CPU fallback).

Models (one unit per split x model x variant x seed):
- fasttext/raw, fasttext/canon: fastText-style supervised model in torch (hashed word unigrams,
  bigrams and char 3-5-grams, mean embedding, linear output). The ``fasttext`` package is not used:
  PyPI has no cp313 wheel for fasttext-wheel 0.9.2 or fasttext 0.9.3 (sdist only), so the torch
  version is the baseline.
- charcnn-<size>/raw, charcnn-<size>/canon: byte-level char-CNN on one view.
- dual-<size>/dual: raw-byte and canonical-byte branches, fused by concatenation with the rule side
  features, then an MLP.
Sizes: base (about 46k params single view, 186 KB fp32; dual about 366 KB) and wide (about 2.7 MB
single, 5.4 MB dual). The char-CNN has no quantised conv layers (INT8 quantises the Linear layers
only, and the fastText EmbeddingBag when torch allows it). Each unit also reports the INT8 size and
metrics, and the delta from fp32.

Each unit trains on the run's train rows, early-stops on cal PR-AUC, sets the threshold on cal, and
scores test with rakshak.eval.protocol.evaluate_scores (campaign-clustered bootstrap, cluster_ids =
campaign_id). Robustness uses the attack harness. Selection uses the mean cal PR-AUC across runs
and seeds, so test is never used for selection. The best char-CNN (student) is exported to
best_student.onnx (opset 17) and best_student.pt (checkpoint with spec and threshold).

Flags go through ``kaggle/launch.sh sms_student --job-args="..."``:
  --data-root PATH           data root (as for sms_baselines)
  --splits GLOB...           split-file stems (default campaign temporal source_holdout lang_shift
                             legacy_*)
  --models {fasttext,charcnn,dual}...   (default all three)
  --sizes {base,wide}...     char-CNN sizes (default both)
  --variants {raw,canon}...  single-view variants for fasttext and charcnn (default both)
  --seeds INT...             training seeds (default 0)
  --seed INT                 bootstrap and subsample seed (default 0)
  --epochs INT               max epochs (default 15)
  --batch-size INT           (default 128)
  --lr FLOAT                 AdamW learning rate (default 2e-3)
  --weight-decay FLOAT       (default 1e-4)
  --patience INT             early-stop patience on cal PR-AUC (default 3)
  --adv-aug P                perturb each training row with probability P (default 0)
  --teacher-logits-dir DIR   folder (or the attached dataset, or its parent) holding the sms_teacher
                             transfer files transfer_<teacher>_s<seed>_<run>.parquet and an optional
                             manifest.json. Enables distillation (default off). Each run takes the
                             file whose run name matches it (never the same file for two runs), and
                             the run's test ids must not appear in it.
  --teacher-seed INT         teacher seed of the transfer files to use (default 0)
  --kd-unlabeled / --no-kd-unlabeled
                             deprecated unsafe pool option; default off, --kd-unlabeled is refused.
                             KD uses only frozen train IDs (see docs/SOURCE_BALANCE.md).
  --kd-alpha FLOAT           distillation weight alpha (default 0.7): labelled rows get
                             (1-alpha) BCE + alpha T^2 KL(teacher || student)
  --kd-temp FLOAT            distillation temperature T (default 2.0)
  --no-rules                 drop the rule side features
  --buckets INT              fastText hash buckets (default 131072)
  --dim INT                  fastText embedding dim (default 16)
  --device {auto,cpu,cuda}   (default auto)
  --no-export                skip ONNX and checkpoint export
  --n-boot INT               bootstrap resamples (default 1000)
  --no-robustness            skip the attack harness
  --robust-seeds INT...      attack seeds (default 0 1 2)
  --max-robust-pos INT       cap scam texts per run in the attack harness (default all)
  --latency-n INT            messages timed for CPU latency, single thread (default 1000)

Outputs in /kaggle/working: metrics.json and report.md (the student block of metrics.json holds
per-unit model size, parameters, INT8 metrics, training history and the export and latency
figures), best_student.onnx and best_student.pt.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from _common import (
    Units,
    env_report,
    log_input_tree,
    record_skip,
    seed_everything,
    single_class_reason,
    timed,
)

from rakshak.eval import data as D
from rakshak.eval import report
from rakshak.eval.protocol import evaluate_scores, headline
from rakshak.eval.robustness import ATTACK_SEEDS, robustness_report
from rakshak.models.charcnn import CONFIGS
from rakshak.models.distill import (
    build_training_pool,
    find_transfer_file,
    load_transfer,
    resolve_teacher_dir,
    verify_manifest,
)
from rakshak.models.features import Featurized, featurize
from rakshak.models.student import (
    ModelSpec,
    TrainConfig,
    export_onnx,
    latency_ms,
    onnx_parity,
    param_count,
    pick_device,
    predict_scores,
    quantize_int8,
    save_checkpoint,
    score_texts,
    state_bytes,
    train_student,
)

DEFAULT_SPLITS = ("campaign", "temporal", "source_holdout", "lang_shift", "legacy_*")
PRIMARY_SPLIT = "campaign"
KINDS = ("fasttext", "charcnn", "dual")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sms_student")
    p.add_argument("--data-root", default=None)
    p.add_argument("--splits", nargs="*", default=list(DEFAULT_SPLITS))
    p.add_argument("--models", nargs="*", default=list(KINDS), choices=KINDS)
    p.add_argument("--sizes", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--variants", nargs="*", default=["raw", "canon"], choices=["raw", "canon"])
    p.add_argument("--seeds", nargs="*", type=int, default=[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--adv-aug", type=float, default=0.0)
    p.add_argument("--teacher-logits-dir", default=None)
    p.add_argument("--teacher-seed", type=int, default=0)
    p.add_argument("--kd-unlabeled", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--kd-alpha", type=float, default=0.7)
    p.add_argument("--kd-temp", type=float, default=2.0)
    p.add_argument("--no-rules", action="store_true")
    p.add_argument("--buckets", type=int, default=2**17)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--no-robustness", action="store_true")
    p.add_argument("--robust-seeds", nargs="*", type=int, default=list(ATTACK_SEEDS))
    p.add_argument("--max-robust-pos", type=int, default=None)
    p.add_argument("--latency-n", type=int, default=1000)
    args = p.parse_args(argv or [])
    if args.kd_unlabeled:
        p.error("--kd-unlabeled is disabled: the broad transfer pool reintroduces excluded rows; "
                "KD training uses frozen train IDs only")
    if not 0.0 <= args.adv_aug <= 1.0:
        p.error("--adv-aug must be in [0, 1]")
    if not 0.0 <= args.kd_alpha <= 1.0:
        p.error("--kd-alpha must be in [0, 1]")
    if args.kd_temp <= 0.0:
        p.error("--kd-temp must be positive")
    return args


def candidates(args) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    rules = not args.no_rules
    for kind in args.models:
        if kind == "fasttext":
            specs += [ModelSpec("fasttext", None, v, rules, args.buckets, args.dim)
                      for v in args.variants]
        elif kind == "charcnn":
            specs += [ModelSpec("charcnn", size, v, rules)
                      for size in args.sizes for v in args.variants]
        else:
            specs += [ModelSpec("dual", size, "dual", rules) for size in args.sizes]
    return specs


@dataclass
class RunData:
    name: str
    train: Featurized
    cal: Featurized
    test: Featurized
    y_train: np.ndarray
    y_cal: np.ndarray
    y_test: np.ndarray
    groups: dict
    cluster_ids: list | None
    pos_test: list[str]
    neg_test: list[str]
    test_texts: list[str]
    teacher: np.ndarray | None  # (rows, 2) teacher logits aligned with train, NaN where unknown
    pool_info: dict | None      # transfer file and pool counts, when distilling


def fit_and_evaluate(rd: RunData, spec: ModelSpec, seed: int, args, device, keep: bool,
                     log) -> dict:
    """One (split, model, variant, seed) unit: train, threshold on cal, score test, then the
    attack harness and the INT8 copy. ``keep`` returns the fp32 and INT8 models on CPU."""
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      weight_decay=args.weight_decay, patience=args.patience,
                      adv_p=args.adv_aug, kd_alpha=args.kd_alpha, kd_temp=args.kd_temp,
                      seed=seed)
    model, train_info = train_student(spec, rd.train, rd.y_train, rd.cal, rd.y_cal, cfg,
                                      device=device, teacher=rd.teacher, log=log)
    model = model.to(device)
    s_cal = predict_scores(model, spec, rd.cal, device)
    s_test = predict_scores(model, spec, rd.test, device)
    result = evaluate_scores(rd.y_cal, s_cal, rd.y_test, s_test, groups=rd.groups,
                             n_boot=args.n_boot, seed=args.seed, cluster_ids=rd.cluster_ids)
    thr = result["thresholds"]["max_f1"]["threshold"]
    if not args.no_robustness:
        result["robustness"] = robustness_report(
            lambda ts: score_texts(model, spec, ts, device), rd.pos_test, rd.neg_test, thr,
            seeds=args.robust_seeds, max_positives=args.max_robust_pos,
            subsample_seed=args.seed,
        )

    cpu_model = model.cpu()
    qmodel, notes = quantize_int8(cpu_model, spec)
    q_result = evaluate_scores(rd.y_cal, predict_scores(qmodel, spec, rd.cal, "cpu"),
                               rd.y_test, predict_scores(qmodel, spec, rd.test, "cpu"),
                               bootstrap=False)
    fp32_head, int8_head = headline(result), headline(q_result)
    delta = {k: (None if fp32_head[k] is None or int8_head[k] is None
                 else float(int8_head[k] - fp32_head[k])) for k in fp32_head}
    info = {
        "spec": spec.__dict__ | {"label": spec.label},
        "params": param_count(cpu_model),
        "fp32_bytes": state_bytes(cpu_model),
        "int8": {"bytes": state_bytes(qmodel), "notes": notes, "headline": int8_head,
                 "delta_vs_fp32": delta},
        "train": train_info,
        "threshold": float(thr),
    }
    out = {"result": result, "info": info, "thr": thr}
    if keep:
        out["model"] = cpu_model
        out["qmodel"] = qmodel
    return out


def _checkpoint(ctx, root, splits: dict, student: dict) -> None:
    report.write_report({"title": "SMS student", "meta": {"data_root": str(root)},
                         "splits": splits, "student": student}, ctx.work)


def _build_run(run_, sms_texts: list[str], full: Featurized, labels, ids, sms,
               teacher_dir, args, cluster_col: str | None) -> RunData:
    """One run's data. Without a teacher folder the training pool is the run's train rows. With
    one, only the labelled frozen train rows receive teacher logits. The inherited extra-row
    pool is disabled because ID exclusion alone did not prevent evaluation contamination."""
    te = run_.test
    if teacher_dir is None:
        train_pos, y_train, teacher, pool_info = (run_.train,
                                                  labels[run_.train].astype(np.float64),
                                                  None, None)
    else:
        tfile = find_transfer_file(teacher_dir, run_.name, args.teacher_seed)
        entry = verify_manifest(teacher_dir, run_.name, tfile)
        pool = build_training_pool(run_, ids, labels, load_transfer(tfile),
                                   include_unlabeled=args.kd_unlabeled)
        train_pos, y_train, teacher = pool.positions, pool.labels, pool.teacher
        pool_info = {"file": tfile.name, "manifest_sha256": (entry or {}).get("sha256"),
                     **pool.stats()}
    return RunData(
        name=run_.name,
        train=full.take(train_pos),
        cal=full.take(run_.cal),
        test=full.take(te),
        y_train=y_train,
        y_cal=labels[run_.cal],
        y_test=labels[te],
        groups=D.group_values(sms, te),
        cluster_ids=(D.cluster_values(sms, te, cluster_col) if cluster_col else None),
        pos_test=[sms_texts[i] for i in te if labels[i] == 1],
        neg_test=[sms_texts[i] for i in te if labels[i] == 0],
        test_texts=[sms_texts[i] for i in te],
        teacher=teacher,
        pool_info=pool_info,
    )


def run(ctx) -> bool:
    args = parse_args(ctx.argv)
    seed_everything(args.seed)
    device = pick_device(args.device)
    env = env_report()
    env["device"] = str(device)
    ctx.log(f"env: {env}")
    try:
        root = D.resolve_data_root(args.data_root)
    except FileNotFoundError:
        log_input_tree(ctx.log)
        raise
    ctx.log(f"data root: {root}")
    sms = D.load_sms(root)
    labels = sms["label_binary"].to_numpy().astype(np.int64)
    sms_texts = [t if isinstance(t, str) else "" for t in sms["text"]]
    ids = sms["id"].to_numpy(dtype=object)
    mask = D.eval_mask(sms)
    runs = D.load_runs(root, sms, patterns=args.splits, labels=labels, mask=mask)
    ctx.log(f"{len(runs)} runs: {[r.name for r in runs]}")
    cluster_col = D.CLUSTER_COLUMNS["sms"] if D.CLUSTER_COLUMNS["sms"] in sms.columns else None
    ctx.log(f"bootstrap clusters: {cluster_col or 'iid'}")

    teacher_dir = None
    if args.teacher_logits_dir:
        teacher_dir = resolve_teacher_dir(args.teacher_logits_dir)
        ctx.log(f"teacher transfer folder: {teacher_dir} (seed {args.teacher_seed}, "
                f"alpha {args.kd_alpha}, T {args.kd_temp}, unlabeled {args.kd_unlabeled})")

    specs = candidates(args)
    ctx.log(f"candidates: {[f'{s.label}/{s.variant}' for s in specs]} seeds {args.seeds}")
    buckets = args.buckets if any(s.hashed for s in specs) else None
    with timed("featurize_s", ctx.results.setdefault("timings", {})):
        full = featurize(sms_texts, buckets=buckets)

    units = Units(ctx)
    primary_name = next((r.name for r in runs if r.name.split(":")[0] == PRIMARY_SPLIT),
                        runs[0].name)
    splits: dict[str, dict] = {}
    student: dict = {"units": {}, "skipped_runs": {}}
    fitted: dict[tuple[str, str], dict] = {}
    teacher_pools: dict[str, dict] = {}
    for run_ in runs:
        ctx.log(f"run {run_.name}: sizes {run_.notes['sizes']}")
        reason = single_class_reason(labels[run_.test], run_.name)
        if reason:
            record_skip(ctx, run_.name, reason)
            continue
        rd = units.run({"split": run_.name, "model": "features", "variant": "-"},
                       _build_run, run_, sms_texts, full, labels, ids, sms, teacher_dir, args,
                       cluster_col)
        if rd is None:
            continue
        if rd.pool_info is not None:
            teacher_pools[run_.name] = rd.pool_info
            ctx.log(f"  teacher pool {run_.name}: {rd.pool_info}")
        block: dict[str, dict] = splits.setdefault(run_.name, {})
        is_primary = run_.name == primary_name
        for spec in specs:
            for seed in args.seeds:
                key = f"{spec.label}/{spec.variant}/s{seed}"
                keep = is_primary and seed == args.seeds[0]
                out = units.run({"split": run_.name, "model": spec.label,
                                 "variant": spec.variant, "seed": seed},
                                fit_and_evaluate, rd, spec, seed, args, device, keep, ctx.log)
                if out is not None:
                    block[key] = out["result"]
                    student["units"].setdefault(run_.name, {})[key] = out["info"]
                    op = out["result"]["operating_points"]["max_f1"]
                    ctx.log(f"  {key}: F1 {op['f1']:.4f} "
                            f"PR-AUC {out['result']['threshold_free']['pr_auc']:.4f} "
                            f"int8 PR-AUC delta "
                            f"{out['info']['int8']['delta_vs_fp32']['pr_auc']}")
                    if keep:
                        fitted[(spec.label, spec.variant)] = {
                            "spec": spec, "model": out["model"], "qmodel": out["qmodel"],
                            "thr": out["thr"], "rd": rd}
                _checkpoint(ctx, root, splits, student)

    # Selection on cal only: mean cal PR-AUC over runs and seeds.
    cal_scores: dict[str, list[float]] = {}
    for block in splits.values():
        for key, result in block.items():
            label_variant = key.rsplit("/", 1)[0]
            cal_scores.setdefault(label_variant, []).append(
                result["cal_threshold_free"]["pr_auc"])
    mean_cal = {k: float(np.mean(v)) for k, v in cal_scores.items()}
    ranking = sorted(((mean_cal[k], k) for k in mean_cal), reverse=True)
    ctx.log(f"selection (mean cal PR-AUC): {ranking[:6]}")

    primary = next((r for r in runs if r.name == primary_name), None)
    latency: dict[str, dict] = {}
    if primary is not None and fitted:
        test_texts = [sms_texts[i] for i in primary.test]
        for (label, variant), entry in fitted.items():
            cpu_spec = entry["spec"]
            for tag, mdl in (("fp32", entry["model"]), ("int8", entry["qmodel"])):
                stats = latency_ms(lambda t, m=mdl, s=cpu_spec: score_texts(m, s, [t], "cpu"),
                                   test_texts, args.latency_n)
                latency[f"{label}/{variant}/{tag}"] = stats
        summary = {k: (round(v["p50_ms"], 3), round(v["p95_ms"], 3))
                   for k, v in latency.items()}
        ctx.log(f"latency p50/p95 ms: {summary}")
    student["latency_cpu_1thread"] = latency
    student["teacher_pools"] = teacher_pools

    onnx_info = None
    student_keys = [k for _, k in ranking
                    if tuple(k.split("/")) in fitted and not k.startswith("fasttext/")]
    if student_keys and not args.no_export:
        best_key = student_keys[0]
        label, variant = best_key.split("/")
        entry = fitted[(label, variant)]
        spec = entry["spec"]
        onnx_path = ctx.work / "best_student.onnx"
        exported = export_onnx(entry["model"], onnx_path)
        parity = onnx_parity(entry["model"], onnx_path, entry["rd"].test)
        save_checkpoint(ctx.work / "best_student.pt", entry["model"], spec, entry["thr"],
                        extra={"run": primary_name, "key": best_key,
                               "cal_pr_auc_mean": mean_cal[best_key]})
        onnx_info = {"key": best_key, "onnx": exported, "parity": parity,
                     "checkpoint": "best_student.pt"}
        ctx.log(f"exported {best_key}: {exported} parity {parity}")
    elif not student_keys:
        ctx.log("no char-CNN candidate has a primary-run fit; nothing exported")
    student["export"] = onnx_info

    payload = {
        "title": "SMS student",
        "meta": {
            "data_root": str(root),
            "runs": len(runs),
            "seeds": args.seeds,
            "device": str(device),
            "bootstrap_clusters": cluster_col or "iid",
            "best_student": onnx_info["key"] if onnx_info else None,
            "teacher_logits_dir": str(teacher_dir) if teacher_dir else None,
            "kd": ({"alpha": args.kd_alpha, "temp": args.kd_temp, "unlabeled": args.kd_unlabeled,
                    "teacher_seed": args.teacher_seed} if teacher_dir else None),
            "env": env,
        },
        "splits": splits,
        "student": student,
    }
    report.write_report(payload, ctx.work)
    best_text = f"{mean_cal[student_keys[0]]:.3f}" if student_keys else "n/a"
    ctx.results["summary"] = (f"{len(runs)} runs, {units.ok} units ok, {units.failed} failed; "
                              f"best student {student_keys[0] if student_keys else 'none'} "
                              f"cal-PR-AUC {best_text}")
    return units.status()
