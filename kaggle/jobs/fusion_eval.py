"""CPU evaluation of frozen, per-message probability scores. Never launch locally.

Required: --message-scores PATH --url-scores PATH (parquet: unique SMS id, score).
Scores must be probabilities from experts trained without cal/test rows. URL scores
must be missing for URL-free messages and present for every message with a URL;
multiple URLs must already be aggregated per message by the frozen URL expert.
Uses label_scam, excluding unknown labels; legacy spam labels are not fraud truth.
All learning uses campaign-disjoint subdivisions of splits/campaign.json's cal.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from _common import Units
from threadpoolctl import threadpool_limits

from rakshak.canonicalize import canonicalize
from rakshak.eval import data as D
from rakshak.eval.report import sanitize
from rakshak.fusion.calibrate import (
    IsotonicCalibration,
    PlattScaling,
    TemperatureScaling,
    logits,
)
from rakshak.fusion.fuse import MODELS, FusionModel, feature_matrix
from rakshak.fusion.report import evaluate_fusion, write_report

CALIBRATORS = {"temperature": TemperatureScaling, "platt": PlattScaling,
               "isotonic": IsotonicCalibration}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="fusion_eval")
    parser.add_argument("--message-scores", required=True)
    parser.add_argument("--url-scores", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--calibrators", nargs="+", choices=CALIBRATORS,
                        default=list(CALIBRATORS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000)
    return parser.parse_args(argv or [])


def load_scores(path: str) -> pd.Series:
    frame = pd.read_parquet(path)
    if not {"id", "score"} <= set(frame) or frame["id"].isna().any():
        raise ValueError("score file requires non-null id and score columns")
    frame["id"] = frame["id"].astype(str)
    if not frame["id"].is_unique:
        raise ValueError("duplicate score IDs; scores must be per message")
    return frame.set_index("id")["score"].astype(float)


def calibration_parts(cal, campaign_ids, *, seed: int) -> list[np.ndarray]:
    """Four partitions assigned by campaign, independent of labels and scores."""
    cal = np.asarray(cal, int)
    campaigns = np.asarray(campaign_ids)[cal]
    unique = np.unique(campaigns)
    if len(unique) < 4:
        raise ValueError("at least four calibration campaigns required")
    np.random.default_rng(seed).shuffle(unique)
    return [cal[np.isin(campaigns, chunk)] for chunk in np.array_split(unique, 4)]


def prepare(sms, run_, message, url, seed: int) -> dict:
    if "label_scam" not in sms:
        raise ValueError("label_scam required: legacy spam labels are not scam truth")
    if run_.notes.get("cal_carved_from_train"):
        raise ValueError("campaign split must explicitly supply cal; never carve from train")
    labels = sms["label_scam"].to_numpy(dtype=float, na_value=np.nan)
    eligible = D.eval_mask(sms) & np.isin(labels, [0, 1])
    cal = run_.cal[eligible[run_.cal]]
    test = run_.test[eligible[run_.test]]
    campaigns = np.array(D.cluster_values(sms, np.arange(len(sms)), "campaign_id"))
    sets = [set(campaigns[pos]) for pos in (run_.train, run_.cal, run_.test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("campaign leakage across train/cal/test")
    positions = np.r_[cal, test]
    ids = sms["id"].iloc[positions].to_numpy()
    texts = sms["text"].iloc[positions].tolist()
    present = np.array([bool(canonicalize(text).urls) for text in texts])
    x = feature_matrix(texts, message.reindex(ids).to_numpy(), url.reindex(ids).to_numpy(),
                       has_url=present)
    local_cal = np.arange(len(cal))
    parts = calibration_parts(local_cal, campaigns[positions], seed=seed)
    local_test = np.arange(len(cal), len(positions))
    y = labels[positions].astype(int)
    for subset in [*parts, local_test]:
        if set(y[subset]) != {0, 1}:
            raise ValueError("each calibration stage and test require both classes")
    return {"x": x, "y": y, "ids": ids, "parts": parts, "test": local_test,
            "train_ids": sms["id"].iloc[run_.train].tolist(),
            "clusters": campaigns[test].tolist(), "groups": D.group_values(sms, test),
            "excluded_unknown_or_ineligible": int(len(run_.cal) + len(run_.test) - len(positions))}


def evaluate_unit(ctx, payload: dict, run_name: str, data: dict, model_name: str,
                  calibration_name: str, args) -> dict:
    fit, calibrate, threshold, crc = data["parts"]
    x, y, ids, test = (data[k] for k in ("x", "y", "ids", "test"))
    with threadpool_limits(limits=1):
        model = FusionModel(model_name).fit(
            x, y, ids=ids, cal_ids=ids[fit], train_ids=data["train_ids"], test_ids=ids[test])
        raw = model.transform(x)
        # All three calibrators consume logits consistently; temperature needs logits.
        calibrator = CALIBRATORS[calibration_name]().fit(logits(raw[calibrate]), y[calibrate])
        prob = calibrator.transform(logits(raw))
        result = evaluate_fusion(
            y[threshold], prob[threshold], y[test], raw[test], prob[test],
            cluster_ids=data["clusters"], groups=data["groups"],
            crc_y=y[crc], crc_prob=prob[crc], n_boot=args.n_boot, seed=args.seed,
        )
    result["model"] = model.to_dict()
    result["calibrator"] = calibrator.to_dict()
    result["calibrator_input"] = "logit(clip(probability, 1e-12, 1-1e-12))"
    result["cal_stage_ids"] = {name: ids[pos].tolist() for name, pos in zip(
        ("fusion", "probability", "threshold", "crc"), data["parts"], strict=True)}
    result["excluded_unknown_or_ineligible"] = data["excluded_unknown_or_ineligible"]
    payload["runs"].setdefault(run_name, {})[f"{model_name}/{calibration_name}"] = result
    ctx.results["fusion"] = sanitize(payload)
    write_report(payload, ctx.work)
    return result


def run(ctx) -> str:
    args = parse_args(ctx.argv)
    units = Units(ctx)
    payload = {"runs": {}, "seed": args.seed, "label": "label_scam",
               "score_provenance": "caller must verify experts never trained on cal/test"}

    def inputs() -> tuple:
        root = D.resolve_data_root(args.data_root)
        sms = D.load_sms(root)
        if "label_scam" not in sms:
            raise ValueError("label_scam required")
        runs = D.load_runs(root, sms, patterns=["campaign"],
                           labels=sms["label_binary"].to_numpy(), mask=D.eval_mask(sms))
        return sms, runs, load_scores(args.message_scores), load_scores(args.url_scores)

    loaded = units.run({"stage": "inputs"}, inputs)
    if loaded is None:
        return units.status()
    sms, runs, message, url = loaded
    for run_ in runs:
        prepared = units.run({"split": run_.name, "stage": "features"},
                             prepare, sms, run_, message, url, args.seed)
        if prepared is None:
            continue
        for model in args.models:
            for calibration in args.calibrators:
                units.run({"split": run_.name, "model": model, "calibration": calibration},
                          evaluate_unit, ctx, payload, run_.name, prepared, model,
                          calibration, args)
    # Input/feature successes alone must not turn an all-model failure into partial success.
    status = units.status() if any(payload["runs"].values()) else "fail"
    ctx.results["summary"] = f"fusion: {units.ok} units ok, {units.failed} failed; {status}"
    ctx.results["status"] = status
    ctx.save()
    return status
