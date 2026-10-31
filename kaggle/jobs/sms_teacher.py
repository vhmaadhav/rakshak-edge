"""GPU teacher: fine-tune multilingual encoders for scam-SMS detection on one Kaggle T4.

For each model, seed and run (default run: campaign):
- AutoModelForSequenceClassification with 2 labels, fp16 autocast, max_len 128, AdamW, linear
  warmup and decay, class-balanced cross-entropy, gradient clipping at 1.0.
- Early stopping on cal PR-AUC (patience ``--patience``). The best epoch's weights are kept.
- ``--hpo N``: Optuna TPE over lr, epochs, warmup ratio and weight decay. Each trial is a full
  early-stopped fine-tune. The trial with the best cal PR-AUC wins. Trials stop when the run's
  time budget would be exceeded.
- ``--adv-aug p``: each training message is perturbed by a random rakshak family with probability p,
  using a seeded RNG, before canonicalisation. This is adversarial training on the fly.
- Evaluation matches the baselines: protocol metrics, bootstrap CIs, group breakdown and the
  robustness harness, for both inference variants of the same weights. "canon" canonicalises the
  input. "raw" feeds it as received, so it is the same model without the canonicaliser. The model is
  trained on the ``--train-input`` form only. It is not retrained for "raw".
- Logits for train, cal and test go to parquet (for later distillation). Train logits are
  in-sample and therefore overfit. Distillation should use cross-fitted logits where possible.
- Only the best checkpoint across all runs is kept, in the work directory.

Time budget: ``--budget-h`` is split evenly across (model x seed x run). A run trains for 85% of its
share. The rest is evaluation. Compute spent and trial counts are logged in results.json.

Flags go through ``kaggle/launch.sh sms_teacher --gpu t4 --internet --job-args="..."``:
  --models NAME...        Hugging Face model ids (default: mdeberta-v3-base, xlm-roberta-base,
                          muril-base-cased, IndicBERTv2-MLM-only)
  --seeds INT...          seeds (default 0)
  --splits NAME...        split-file stems, as sms_student reads them (default: campaign).
                          campaign, lang_shift, temporal, source_holdout (every fold), legacy_*
                          (glob patterns allowed)
  --hpo N                 Optuna trials per run (default 0 = fixed hyperparameters)
  --budget-h FLOAT        total wall-clock budget in hours (default 8.0)
  --adv-aug FLOAT         probability of perturbing a training message (default 0)
  --train-input canon|raw form the model is trained on (default canon)
  --max-len INT           token limit (default 128)
  --batch-size INT        training batch (default 32)
  --eval-batch-size INT   inference batch (default 128)
  --epochs INT            max epochs without HPO (default 4)
  --lr FLOAT              learning rate without HPO (default 3e-5; ignored with --hpo)
  --weight-decay FLOAT    default 0.01
  --warmup-ratio FLOAT    default 0.1
  --patience INT          early-stopping patience in epochs (default 2)
  --no-fp16               true fp32: no autocast, no GradScaler, no bf16; weights are forced to
                          fp32 after load. The effective precision is logged per unit.
  --transfer-texts        also dump logits for every sms.parquet row outside the split's test
                          partition (train, cal and unlabeled rows) to transfer_*.parquet. Test
                          rows are never written there (checked before the write).
  --data-root PATH        data root (default: $RAKSHAK_DATA, /kaggle/input/rakshak-data,
                          then data/processed, then a search of /kaggle/input)
  --n-boot INT            bootstrap resamples (default 1000; campaign-clustered bootstrap)
  --no-robustness         skip the attack harness
  --robust-seeds INT...   attack seeds (default 0 1 2)
  --max-robust-pos INT    cap on scam rows and on benign rows in the attack harness, each a
                          seeded subsample (default 2000; 0 = all rows)

NaN guard: a non-finite training loss, training logit, gradient norm (fp32 only) or eval logit
fails the unit at once with the step, lr, dtype and last finite loss. Grad clipping at 1.0 and
AdamW eps 1e-6 are always on. Loss and grad norm are logged every 200 steps.
"""

from __future__ import annotations

import argparse
import random
import shutil
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from _common import (
    Units,
    env_report,
    log_input_tree,
    record_skip,
    require_finite,
    seed_everything,
    single_class_reason,
    timed,
)
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from rakshak.canonicalize import canonicalize
from rakshak.eval import data as D
from rakshak.eval import report
from rakshak.eval.metrics import pr_auc
from rakshak.eval.protocol import evaluate_scores
from rakshak.eval.robustness import ATTACK_SEEDS, robustness_report
from rakshak.perturb import FAMILIES, perturb

DEFAULT_MODELS = (
    "microsoft/mdeberta-v3-base",
    "FacebookAI/xlm-roberta-base",
    "google/muril-base-cased",
    "ai4bharat/IndicBERTv2-MLM-only",
)
PRIMARY_SPLIT = "campaign"
TRAIN_FRACTION_OF_RUN = 0.85
BEST_DIR = "best_checkpoint"
NO_DECAY = ("bias", "LayerNorm.weight", "layer_norm", "layernorm")
MAX_GRAD_NORM = 1.0
ADAM_EPS = 1e-6
LOG_EVERY = 200
ROBUST_SAMPLE_SEED = 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sms_teacher")
    p.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    p.add_argument("--seeds", nargs="*", type=int, default=[0])
    p.add_argument("--splits", nargs="*", default=[PRIMARY_SPLIT])
    p.add_argument("--hpo", type=int, default=0)
    p.add_argument("--budget-h", type=float, default=8.0)
    p.add_argument("--adv-aug", type=float, default=0.0)
    p.add_argument("--train-input", choices=("canon", "raw"), default="canon")
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--transfer-texts", action="store_true")
    p.add_argument("--data-root", default=None)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--no-robustness", action="store_true")
    p.add_argument("--robust-seeds", nargs="*", type=int, default=list(ATTACK_SEEDS))
    p.add_argument("--max-robust-pos", type=int, default=2000)
    args = p.parse_args(argv or [])
    if not 0.0 <= args.adv_aug <= 1.0:
        p.error("--adv-aug must be in [0, 1]")
    if args.budget_h <= 0:
        p.error("--budget-h must be positive")
    if args.max_robust_pos < 0:
        p.error("--max-robust-pos must be 0 (all rows) or positive")
    if args.lr <= 0:
        p.error("--lr must be positive")
    return args


class MessageSet(Dataset):
    """Training messages as (model input, label). With probability ``adv_p`` a message is
    perturbed by a random family first. Canonical input is applied after the perturbation."""

    def __init__(self, raw, labels, *, canon: bool, adv_p: float, seed: int):
        self.raw = list(raw)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.canon = canon
        self.adv_p = adv_p
        self.base = [D.canon_text(t) if canon else t for t in self.raw]
        self.rng = random.Random(seed)
        self.families = sorted(FAMILIES)

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, i: int):
        if self.adv_p > 0 and self.rng.random() < self.adv_p:
            family = self.rng.choice(self.families)
            text = perturb(self.raw[i], family, self.rng.randrange(2**31 - 1))
            if self.canon:
                text = canonicalize(text).normalized
        else:
            text = self.base[i]
        return text, int(self.labels[i])


def make_collate(tokenizer, max_len: int):
    def collate(batch):
        texts, labels = zip(*batch, strict=True)
        enc = tokenizer(list(texts), truncation=True, max_length=max_len, padding=True,
                        return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    return collate


def probs_from_logits(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


@torch.no_grad()
def predict_logits(model, tokenizer, texts, cfg, device) -> np.ndarray:
    """Logits for already-prepared texts, batched in length order."""
    model.eval()
    texts = list(texts)
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.zeros((len(texts), 2), dtype=np.float32)
    for start in range(0, len(texts), cfg.eval_batch_size):
        idx = order[start:start + cfg.eval_batch_size]
        enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=cfg.max_len,
                        padding=True, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.fp16):
            logits = model(**enc).logits.float()
        out[idx] = logits.cpu().numpy()
    require_finite(out, f"eval logits over {len(texts)} rows", dtype=cfg.dtype_label)
    return out


def make_scorer(model, tokenizer, cfg, device, variant: str):
    """Raw texts in, scam probabilities out, with canonicalisation for the variant."""

    def score(texts, cached: bool = True) -> np.ndarray:
        texts = list(texts)
        if variant == "canon":
            prepped = [D.canon_of(t).normalized if cached else canonicalize(t).normalized
                       for t in texts]
        else:
            prepped = texts
        return probs_from_logits(predict_logits(model, tokenizer, prepped, cfg, device))

    return score


@dataclass
class Config:
    fp16: bool
    max_len: int
    batch_size: int
    eval_batch_size: int
    patience: int
    adv_aug: float
    train_canon: bool
    log: object  # callable(str); the unit's log line

    @property
    def dtype_label(self) -> str:
        return "fp16 autocast over fp32 weights" if self.fp16 else "fp32 (no autocast, no bf16)"


def train_once(cfg: Config, hp: dict, model_name: str, seed: int, tokenizer, device, data,
               deadline: float) -> dict:
    """Fine-tune once and return the best-epoch model (on CPU) with its cal PR-AUC.

    Raises NonFiniteError at the first non-finite training loss, training logit or (fp32)
    gradient norm, so the unit fails instead of training on NaN.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # .float() forces fp32 weights whatever dtype the checkpoint is stored in.
    model = (AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
             .float().to(device))
    cfg.log(f"    precision: {cfg.dtype_label}; weights {next(model.parameters()).dtype}")
    train_ds = MessageSet(data["train_raw"], data["train_y"], canon=cfg.train_canon,
                          adv_p=cfg.adv_aug, seed=seed)
    loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=make_collate(tokenizer, cfg.max_len),
        generator=torch.Generator().manual_seed(seed),
    )
    decay = [p for n, p in model.named_parameters() if not any(k in n for k in NO_DECAY)]
    no_decay = [p for n, p in model.named_parameters() if any(k in n for k in NO_DECAY)]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": hp["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=hp["lr"], eps=ADAM_EPS,
    )
    total = len(loader) * hp["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(hp["warmup"] * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16)
    counts = np.bincount(data["train_y"], minlength=2).astype(np.float64)
    class_w = len(data["train_y"]) / (2.0 * np.maximum(counts, 1.0))
    loss_fn = CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32, device=device))

    best = {"pr_auc": -1.0, "epoch": -1, "state": None}
    history = []
    bad = 0
    start = time.time()
    out_of_time = False
    step = 0
    last_finite_loss = None
    for epoch in range(hp["epochs"]):
        model.train()
        last_loss = float("nan")
        for batch in loader:
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            lr_now = sched.get_last_lr()[0]
            diag = {"step": step, "lr": lr_now, "dtype": cfg.dtype_label,
                    "last_finite_loss": last_finite_loss}
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.fp16):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            loss_val = float(loss.detach())
            logits_np = logits.detach().float().cpu().numpy()
            require_finite(loss_val, "training loss", **diag)
            require_finite(logits_np, "training logits", **diag)
            last_finite_loss = loss_val
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM))
            # With fp16 the GradScaler skips a step whose grads overflow, which is expected.
            # In fp32 a non-finite grad norm means the weights would go NaN, so it fails.
            if not cfg.fp16:
                require_finite(grad_norm, "gradient norm", **diag)
            scaler.step(opt)
            scaler.update()
            sched.step()
            last_loss = loss_val
            if step % LOG_EVERY == 0:
                cfg.log(f"    step {step} epoch {epoch} lr {lr_now:.3g} loss {loss_val:.4f} "
                        f"grad-norm {grad_norm:.3f} ({cfg.dtype_label})")
            if time.time() > deadline:
                out_of_time = True
                break
        cal_logits = predict_logits(model, tokenizer, data["cal_prepped"], cfg, device)
        ap = pr_auc(data["cal_y"], probs_from_logits(cal_logits))
        history.append({"epoch": epoch, "cal_pr_auc": ap, "last_loss": last_loss})
        if ap > best["pr_auc"]:
            best = {"pr_auc": ap, "epoch": epoch,
                    "state": {k: v.detach().to("cpu", copy=True)
                              for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
        if bad >= cfg.patience or out_of_time:
            break
    model.load_state_dict(best["state"])
    model.to("cpu")
    del opt, scaler, sched
    torch.cuda.empty_cache()
    return {"model": model, "cal_pr_auc": best["pr_auc"], "best_epoch": best["epoch"],
            "history": history, "seconds": round(time.time() - start, 1),
            "stopped_by_budget": out_of_time}


def search_hpo(cfg, n_trials, model_name, short, seed, run_name, tokenizer, device, data,
               job_deadline, budget) -> dict:
    """Optuna TPE over lr, epochs, warmup and weight decay. Each trial is a full fine-tune.

    A trial that would start after ``job_deadline`` (estimated from the mean trial time) ends
    the study. Returns the best trial's {"result", "hp"}, where the result holds the best-epoch
    model.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    trial_secs: list[float] = []
    best: dict = {}

    def objective(trial):
        est = float(np.mean(trial_secs)) if trial_secs else 0.0
        if time.time() + est > job_deadline:
            trial.study.stop()
            raise optuna.TrialPruned("time budget")
        hp = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-4, log=True),
            "epochs": trial.suggest_int("epochs", 2, 5),
            "warmup": trial.suggest_float("warmup", 0.0, 0.2),
            "weight_decay": trial.suggest_float("weight_decay", 1e-3, 0.1, log=True),
        }
        out = train_once(cfg, hp, model_name, seed, tokenizer, device, data, job_deadline)
        trial_secs.append(out["seconds"])
        budget["trials"].append({"model": short, "seed": seed, "run": run_name, "hp": hp,
                                 "cal_pr_auc": out["cal_pr_auc"], "seconds": out["seconds"],
                                 "stopped_by_budget": out["stopped_by_budget"]})
        if "result" not in best or out["cal_pr_auc"] > best["result"]["cal_pr_auc"]:
            best["result"], best["hp"] = out, hp
        return out["cal_pr_auc"]

    study.optimize(objective, n_trials=n_trials, catch=())
    if "result" not in best:
        raise RuntimeError(f"no HPO trial finished for {short} seed {seed}")
    return best


def run(ctx) -> bool:
    args = parse_args(ctx.argv)
    if not torch.cuda.is_available():
        raise RuntimeError("sms_teacher needs a CUDA GPU; launch with --gpu t4")
    device = torch.device("cuda")
    cfg = Config(
        fp16=not args.no_fp16, max_len=args.max_len, batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size, patience=args.patience, adv_aug=args.adv_aug,
        train_canon=args.train_input == "canon", log=ctx.log,
    )
    seed_everything(args.seeds[0])
    env = env_report()
    env["precision"] = cfg.dtype_label
    ctx.log(f"env: {env}")
    try:
        root = D.resolve_data_root(args.data_root)
    except FileNotFoundError:
        log_input_tree(ctx.log)
        raise
    ctx.log(f"data root: {root}")
    sms = D.load_sms(root)
    labels = sms["label_binary"].to_numpy().astype(np.int64)
    texts = sms["text"].to_numpy(dtype=object)
    runs = D.load_runs(root, sms, patterns=args.splits, labels=labels, mask=D.eval_mask(sms))
    n_jobs = len(args.models) * len(args.seeds) * len(runs)
    total_s = args.budget_h * 3600.0
    per_run_s = total_s / n_jobs
    ctx.log(f"splits {[r.name for r in runs]}; {n_jobs} model-seed-run jobs; "
            f"budget {args.budget_h} h = {per_run_s / 60:.1f} min per job; "
            f"lr {args.lr} (no HPO), clip {MAX_GRAD_NORM}, eps {ADAM_EPS}, "
            f"precision {cfg.dtype_label}")
    cluster_col = D.CLUSTER_COLUMNS["sms"] if D.CLUSTER_COLUMNS["sms"] in sms.columns else None
    ctx.log(f"bootstrap clusters: {cluster_col or 'iid'}")

    budget = {"total_budget_h": args.budget_h, "per_job_min": round(per_run_s / 60, 2),
              "jobs": n_jobs, "trials": []}
    timings: dict[str, float] = {}
    ids = sms["id"].to_numpy()
    splits: dict[str, dict] = {}
    best_global = {"cal_pr_auc": -1.0}
    logits_dir = ctx.work
    t_total = time.time()
    units = Units(ctx)

    def checkpoint() -> None:
        report.write_report(
            {"title": "SMS teacher", "meta": {"budget_h": args.budget_h}, "splits": splits},
            ctx.work,
        )

    def dump_transfer(run_, model, tokenizer, short: str, seed: int, safe_run: str) -> None:
        """Teacher logits for every row outside the split's test partition, for distillation on
        a broad transfer set. Test rows are excluded by position and checked again by id before
        the write."""
        tpos = D.transfer_positions(len(sms), run_.test_all)
        D.assert_no_test_rows(ids[tpos], ids[run_.test_all])
        prepped = ([D.canon_text(t) for t in texts[tpos]] if cfg.train_canon
                   else list(texts[tpos]))
        logits = predict_logits(model, tokenizer, prepped, cfg, device)
        membership = np.where(np.isin(tpos, run_.train), "train",
                              np.where(np.isin(tpos, run_.cal), "cal", "unlabeled"))
        frame = pd.DataFrame({"id": ids[tpos], "split": membership,
                              "logit_0": logits[:, 0], "logit_1": logits[:, 1]})
        path = logits_dir / f"transfer_{short}_s{seed}_{safe_run}.parquet"
        frame.to_parquet(path, index=False)
        ctx.log(f"   transfer logits: {len(frame)} rows -> {path.name}")

    def train_model(run_, model_name: str, short: str, seed: int, data: dict) -> dict:
        """Unit: fine-tune one (model, seed) on one run, write its logits, and keep the best
        checkpoint so far. The caller evaluates the returned model per variant."""
        tr, ca, te = run_.train, run_.cal, run_.test
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        key_base = f"{short}-s{seed}"
        ctx.log(f"== {run_.name} {key_base}: train {len(tr)} cal {len(ca)} test {len(te)}")
        job_deadline = time.time() + TRAIN_FRACTION_OF_RUN * per_run_s
        default_hp = {"lr": args.lr, "epochs": args.epochs, "warmup": args.warmup_ratio,
                      "weight_decay": args.weight_decay}
        if args.hpo > 0:
            best_trial = search_hpo(cfg, args.hpo, model_name, short, seed,
                                    run_.name, tokenizer, device, data, job_deadline,
                                    budget)
        else:
            out = train_once(cfg, default_hp, model_name, seed, tokenizer, device,
                             data, job_deadline)
            best_trial = {"result": out, "hp": default_hp}
            budget["trials"].append({"model": short, "seed": seed, "run": run_.name,
                                     "hp": default_hp, "cal_pr_auc": out["cal_pr_auc"],
                                     "seconds": out["seconds"],
                                     "stopped_by_budget": out["stopped_by_budget"]})

        result = best_trial["result"]
        model = result["model"].to(device)
        ctx.log(f"   best hp {best_trial['hp']} cal PR-AUC {result['cal_pr_auc']:.4f} "
                f"epoch {result['best_epoch']} ({result['seconds']} s)")

        train_prep = (
            [D.canon_text(t) for t in texts[tr]] if cfg.train_canon else list(texts[tr])
        )
        logits = {
            "train": predict_logits(model, tokenizer, train_prep, cfg, device),
            "cal": predict_logits(model, tokenizer, data["cal_prepped"], cfg, device),
            "test": predict_logits(
                model, tokenizer,
                [D.canon_text(t) for t in texts[te]] if cfg.train_canon
                else list(texts[te]),
                cfg, device,
            ),
        }
        frame = []
        for split_name, positions in (("train", tr), ("cal", ca), ("test", te)):
            frame.append(pd.DataFrame({
                "id": ids[positions], "split": split_name,
                "logit_0": logits[split_name][:, 0], "logit_1": logits[split_name][:, 1],
                "label_binary": labels[positions],
            }))
        safe_run = run_.name.replace(":", "_")
        logits_path = logits_dir / f"logits_{short}_s{seed}_{safe_run}.parquet"
        pd.concat(frame, ignore_index=True).to_parquet(logits_path, index=False)
        if args.transfer_texts:
            dump_transfer(run_, model, tokenizer, short, seed, safe_run)

        if result["cal_pr_auc"] > best_global["cal_pr_auc"]:
            best_global.update({"cal_pr_auc": result["cal_pr_auc"], "model": model_name,
                                "seed": seed, "run": run_.name})
            ckpt = logits_dir / BEST_DIR
            shutil.rmtree(ckpt, ignore_errors=True)
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            ctx.log(f"   kept as best checkpoint -> {ckpt}")
        return {"model": model, "model_name": model_name, "tokenizer": tokenizer,
                "result": result, "hp": best_trial["hp"]}

    def eval_variant(run_, groups, pos_test, neg_test, short: str, seed: int, variant: str,
                     trained: dict, block: dict) -> None:
        """Unit: protocol metrics and the attack harness for one inference variant."""
        ca, te = run_.cal, run_.test
        key_base = f"{short}-s{seed}"
        result = trained["result"]
        score = make_scorer(trained["model"], trained["tokenizer"], cfg, device, variant)
        s_cal = score(texts[ca])
        s_test = score(texts[te])
        ev = evaluate_scores(
            labels[ca], s_cal, labels[te], s_test,
            groups=groups, n_boot=args.n_boot, seed=seed,
            cluster_ids=(D.cluster_values(sms, te, cluster_col) if cluster_col else None),
        )
        ev["teacher"] = {"model": trained["model_name"], "seed": seed, "hp": trained["hp"],
                         "cal_pr_auc": result["cal_pr_auc"], "best_epoch": result["best_epoch"],
                         "train_input": args.train_input, "variant": variant,
                         "adv_aug": args.adv_aug, "history": result["history"]}
        thr = ev["thresholds"]["max_f1"]["threshold"]
        if not args.no_robustness:
            with timed(f"{run_.name}|{key_base}|{variant}|robust_s", timings):
                # pos_test and neg_test are already capped (seeded) in the run loop.
                ev["robustness"] = robustness_report(
                    lambda ts, s=score: s(ts, cached=False),
                    pos_test, neg_test, thr,
                    seeds=args.robust_seeds, max_positives=None,
                )
        block[f"{key_base}/{variant}"] = ev
        ctx.log(f"   {variant}: F1 {ev['operating_points']['max_f1']['f1']:.4f} "
                f"PR-AUC {ev['threshold_free']['pr_auc']:.4f}")

    for run_ in runs:
        reason = single_class_reason(labels[run_.test], run_.name)
        if reason:
            record_skip(ctx, run_.name, reason)
            continue
        tr, ca, te = run_.train, run_.cal, run_.test
        groups = D.group_values(sms, te)
        data = {
            "train_raw": list(texts[tr]), "train_y": labels[tr],
            "cal_y": labels[ca],
            "cal_prepped": [D.canon_text(t) for t in texts[ca]] if cfg.train_canon
            else list(texts[ca]),
        }
        # The attack harness runs on a capped, seeded sample of scam and benign test rows, so a
        # big hold-out (76k ScamShield rows) does not take hours. Thresholds and the protocol
        # metrics still use every test row.
        pos_test = D.sample_cap(texts[te][labels[te] == 1].tolist(), args.max_robust_pos,
                                ROBUST_SAMPLE_SEED)
        neg_test = D.sample_cap(texts[te][labels[te] == 0].tolist(), args.max_robust_pos,
                                ROBUST_SAMPLE_SEED)
        ctx.log(f"  attack harness rows: {len(pos_test)} scam, {len(neg_test)} benign "
                f"(test has {int(np.sum(labels[te] == 1))} scam, "
                f"{int(np.sum(labels[te] == 0))} benign)")
        block: dict[str, dict] = splits.setdefault(run_.name, {})
        for model_name in args.models:
            short = model_name.split("/")[-1]
            for seed in args.seeds:
                trained = units.run(
                    {"split": run_.name, "model": short, "seed": seed, "variant": "train"},
                    train_model, run_, model_name, short, seed, data,
                )
                if trained is not None:
                    for variant in ("canon", "raw"):
                        units.run(
                            {"split": run_.name, "model": short, "seed": seed,
                             "variant": variant},
                            eval_variant, run_, groups, pos_test, neg_test, short, seed,
                            variant, trained, block,
                        )
                        checkpoint()
                # Drop every reference to the weights so the GPU memory is released.
                trained = None
                torch.cuda.empty_cache()
                checkpoint()

    budget["elapsed_h"] = round((time.time() - t_total) / 3600.0, 3)
    ctx.results["timings"] = timings
    budget["best"] = best_global
    ctx.results["budget"] = budget
    meta = {
        "data_root": str(root), "models": args.models, "seeds": args.seeds,
        "splits": args.splits, "hpo_trials_per_job": args.hpo, "adv_aug": args.adv_aug,
        "train_input": args.train_input, "precision": cfg.dtype_label,
        "lr": args.lr, "max_grad_norm": MAX_GRAD_NORM, "adam_eps": ADAM_EPS,
        "transfer_texts": args.transfer_texts, "max_robust_pos": args.max_robust_pos,
        "bootstrap_clusters": cluster_col or "iid", "best_checkpoint": best_global,
        "env": env,
    }
    report.write_report({"title": "SMS teacher", "meta": meta, "splits": splits}, ctx.work)
    ctx.results["summary"] = (f"{len(runs)} run(s) x {len(args.models)} model(s); "
                              f"{units.ok} units ok, {units.failed} failed; best "
                              f"{best_global.get('model', 'none')} cal-PR-AUC "
                              f"{best_global['cal_pr_auc']:.3f} in {budget['elapsed_h']} h")
    return units.status()
