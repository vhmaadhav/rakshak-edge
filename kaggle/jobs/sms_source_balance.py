"""Three-seed paired ERM / source-balanced / balanced+group-DRO control on frozen splits.

Kaggle only. No test-driven choices. Same dual-base KD student and update count in all arms.
"""

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from _common import Units, env_report, seed_everything, write_metrics

from rakshak.canonicalize import canonicalize
from rakshak.datasets.common import cached_registered_domain
from rakshak.datasets.dedup import assign_campaigns, mark_exact_duplicates
from rakshak.eval import data as D
from rakshak.eval.protocol import evaluate_scores
from rakshak.eval.rule_audit import describe_rate
from rakshak.eval.source_comparison import paired_seed_f1
from rakshak.eval.transfer_audit import audit_extra_rows
from rakshak.models.distill import (
    build_pool,
    build_training_pool,
    find_transfer_file,
    load_transfer,
    verify_manifest,
)
from rakshak.models.features import featurize
from rakshak.models.student import ModelSpec, TrainConfig, predict_scores, train_student

ARMS = ("erm", "balanced", "dro")


def run(ctx, *, arms=ARMS, method_reference=None):
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--teacher-seed", type=int, default=0)
    args = p.parse_args(ctx.argv)
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        raise ValueError("predeclared protocol requires three distinct seeds")
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle CUDA required")
    # GPU-runtime unit invariants before corpus access or training; no local torch install.
    from rakshak.models.source_balance import SourceObjective
    from rakshak.models.student import distillation_loss

    objective = SourceObjective(["easy", "easy", "hard"], 3, "dro", eta=1)
    loss_vector = torch.tensor([0.1, 0.1, 2.0], requires_grad=True)
    objective.reduce(loss_vector, np.arange(3)).backward()
    assert objective.weights()["hard"] > objective.weights()["easy"]
    assert bool((loss_vector.grad > 0).all())
    logits, y = torch.tensor([0.3, -0.2]), torch.tensor([1.0, 0.0])
    teacher = torch.tensor([[0.0, 0.5], [0.0, 0.2]])
    kwargs = {"pos_weight": torch.tensor(2.0), "alpha": 0.7, "temp": 2.0}
    assert torch.allclose(distillation_loss(logits, y, teacher, **kwargs),
                          distillation_loss(logits, y, teacher, reduction="none", **kwargs).mean())
    ctx.log("source objective synthetic torch invariants passed")
    seed_everything(13)
    root = D.resolve_data_root()
    sms = D.load_sms(root)
    labels = sms.label_binary.to_numpy(dtype=np.int64)
    ids = sms.id.to_numpy(dtype=object)
    runs = D.load_runs(root, sms, patterns=["campaign", "source_holdout", "lang_shift", "temporal"],
                       labels=labels, mask=D.eval_mask(sms))
    external = {}
    for name in ("groundlm", "smishx"):
        frame = pd.read_parquet(root / "external" / f"external_{name}.parquet")
        frame = frame[frame.text_available.fillna(False) & D.eval_mask(frame)].copy()
        frame["reg_domains"] = [
            sorted({cached_registered_domain(u) for u in canonicalize(t).urls}) for t in frame.text
        ]
        frame, _ = assign_campaigns(mark_exact_duplicates(frame.reset_index(drop=True)))
        external[name] = (frame, featurize(frame.text.tolist()))
    spec = ModelSpec("dual", "base", "dual")
    feats = featurize(sms.text.fillna("").tolist())
    transfer_paths = list(Path(ctx.input_root).rglob("transfer_*.parquet"))
    folders = sorted({p.parent for p in transfer_paths})
    if len(folders) != 1:
        raise ValueError(f"expected one attached teacher folder, found {len(folders)}")
    teacher_dir = folders[0]
    keys = (sms["dedup_key"].fillna("").astype(str).to_numpy() if "dedup_key" in sms
            else np.asarray([" ".join(D.canon_text(t).lower().split())
                             for t in sms.text.fillna("")]))
    campaigns = sms.campaign_id.fillna("").astype(str)
    if "dup_of" in sms:
        representative_campaign = sms.dup_of.map(dict(zip(sms.id, campaigns, strict=True)))
        campaigns = campaigns.mask(campaigns == "", representative_campaign).fillna("")
    campaign_values = campaigns.to_numpy()
    source_values = sms.source.fillna("missing").astype(str).to_numpy()
    external_keys = {" ".join(D.canon_text(t).lower().split())
                     for frame, _ in external.values() for t in frame.text}
    units, results, comparisons, skipped = Units(ctx), {}, {}, {}
    config = {"args": vars(args), "arms": list(arms), "model": asdict(spec), "env": env_report(),
              "criterion": ("positive ScamShield paired mean-F1 CI; other folds lose "
                            "<= baseline CI half-width"),
              "source_groups": "frozen train IDs only; no additional unlabeled transfer rows",
              "label_endpoint": "legacy spam-inclusive label_binary",
              "selection": "early-stop cal PR-AUC; thresholds cal only; no test selection",
              "reference": method_reference or "https://arxiv.org/abs/1911.08731"}

    def fit(run_, train, y, teacher, sources, arm, seed):
        cfg = TrainConfig(seed=seed, epochs=args.epochs, source_strategy=arm)
        model, history = train_student(spec, train, y, feats.take(run_.cal), labels[run_.cal],
                                      cfg, device="cuda", teacher=teacher, sources=sources,
                                      log=ctx.log)
        model.to("cuda")
        sc = predict_scores(model, spec, feats.take(run_.cal), "cuda")
        st = predict_scores(model, spec, feats.take(run_.test), "cuda")
        clusters = sms.campaign_id.iloc[run_.test].astype(str).tolist()
        metrics = evaluate_scores(labels[run_.cal], sc, labels[run_.test], st,
                                  cluster_ids=clusters, groups=D.group_values(sms, run_.test),
                                  n_boot=args.n_boot, seed=13)
        thr = metrics["thresholds"]["max_f1"]["threshold"]
        name = run_.name.replace(":", "_") + f"_{arm}_s{seed}"
        pd.DataFrame({"id": ids[np.r_[run_.cal, run_.test]],
                      "split": ["cal"] * len(sc) + ["test"] * len(st),
                      "score": np.r_[sc, st]}).to_parquet(ctx.work / f"scores_{name}.parquet")
        ext_metrics = {}
        if run_.name.split(":")[0] == "campaign":
            for name, (frame, features) in external.items():
                scores = predict_scores(model, spec, features, "cuda")
                ext_metrics[name] = {}
                for op in ("max_f1", "fpr_1pct"):
                    threshold = metrics["thresholds"][op]["threshold"]
                    block = {}
                    for label in ("label_binary", "label_scam"):
                        if label not in frame:
                            continue
                        values = frame[label].fillna(-1).to_numpy(dtype=int)
                        block[label] = {}
                        for value, endpoint in ((0, "fpr"), (1, "recall")):
                            mask = values == value
                            block[label][endpoint] = describe_rate(
                                (scores[mask] >= threshold).astype(int),
                                frame.campaign_id[mask].astype(str).tolist(), n_boot=args.n_boot)
                    ext_metrics[name][op] = {"threshold": threshold, "labels": block}
                pd.DataFrame({"id": frame.id, "score": scores}).to_parquet(
                    ctx.work / f"external_{name}_{arm}_s{seed}.parquet")
        del model
        torch.cuda.empty_cache()
        return {"metrics": metrics, "training": history, "config": asdict(cfg),
                "external": ext_metrics}, st >= thr

    pool_audits = {}
    for run_ in runs:
        if len(np.unique(labels[run_.test])) < 2:
            skipped[run_.name] = "single-class test: no F1 comparison"
            continue
        path = find_transfer_file(teacher_dir, run_.name, args.teacher_seed)
        verify_manifest(teacher_dir, run_.name, path)
        transfer = load_transfer(path)
        old_pool = build_pool(run_, ids, labels, transfer, include_unlabeled=True)
        pool_audits[run_.name] = audit_extra_rows(
            old_pool.positions, run_.train, run_.cal, run_.test_all, keys, campaign_values,
            source_values, external_keys=external_keys,
            source_holdout=run_.name.split(":")[0] == "source_holdout")
        write_metrics("transfer_pool_audit.json", pool_audits)
        ctx.log(f"inherited pool membership audit {run_.name}: {pool_audits[run_.name]}")
        pool = build_training_pool(run_, ids, labels, transfer)
        if set(pool.positions) != set(run_.train) or pool.n_unlabeled:
            raise ValueError("C must train on exactly the frozen training IDs")
        del old_pool, transfer
        sources = sms.source.iloc[pool.positions].fillna("missing").astype(str).to_numpy()
        train = feats.take(pool.positions)
        results[run_.name] = {
            "pool": pool.stats(), "source_counts": pd.Series(sources).value_counts().to_dict(),
            "teacher_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "arms": {},
        }
        preds = {}
        for arm in arms:
            preds[arm] = []
            results[run_.name]["arms"][arm] = {}
            for seed in args.seeds:
                out = units.run({"run": run_.name, "arm": arm, "seed": seed}, fit,
                                run_, train, pool.labels, pool.teacher, sources, arm, seed)
                if out is not None:
                    metrics, pred = out
                    results[run_.name]["arms"][arm][str(seed)] = metrics
                    preds[arm].append(pred)
                write_metrics("metrics.json", {"config": config, "runs": results,
                                                "comparisons": comparisons, "skipped": skipped})
        comparisons[run_.name] = {}
        for arm in arms[1:]:
            if len(preds[arm]) == len(preds["erm"]) == 3:
                comparisons[run_.name][arm] = paired_seed_f1(
                    labels[run_.test], preds[arm], preds["erm"],
                    sms.campaign_id.iloc[run_.test].astype(str).tolist(), n_boot=args.n_boot)
        write_metrics("metrics.json", {"config": config, "runs": results,
                                        "comparisons": comparisons, "skipped": skipped})
        ctx.log(f"completed {run_.name}: {comparisons[run_.name]}")
    ctx.results["summary"] = (f"source controls: {len(results)} runs, "
                              f"{len(ctx.results['failures'])} failed units")
    return units.status()
