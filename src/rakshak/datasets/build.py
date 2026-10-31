"""Build the SMS and URL corpora and their splits.

    python -m rakshak.datasets.build [--raw-dir data/raw] [--out-dir data/processed]
                                     [--phreshphish PATH_OR_HF_ID]

Writes (under --out-dir, git-ignored):
    sms.parquet, urls.parquet, splits/*.json (id lists only), build_report.json
and prints a short summary. Raw inputs are read, never executed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rakshak.datasets import report
from rakshak.datasets.common import PROCESSED_DIR, RAW_DIR, SEED
from rakshak.datasets.confound import label_from_source_accuracy, source_identity_accuracy
from rakshak.datasets.dedup import minhash_signatures, near_dup_sensitivity
from rakshak.datasets.external import (
    GROUNDLM_ZIP,
    SMISHX_PATH,
    build_external_frame,
    cross_set_overlap,
    load_groundlm,
    load_smishx,
    overlap_report,
    pool_overlap,
)
from rakshak.datasets.loaders import (
    SOURCE_IMC25,
    load_all_sms,
    load_all_urls,
    load_phreshphish,
)
from rakshak.datasets.sms import SMS_COLUMNS, build_sms_frame, cross_source_overlap
from rakshak.datasets.splits import (
    campaign_partition,
    check_disjoint,
    lang_shift_partition,
    legacy_partitions,
    source_holdout_partitions,
    temporal_partition,
    url_domain_partition,
    url_temporal_partition,
)
from rakshak.datasets.urls import build_phreshphish_frame, build_url_frame

EXTERNAL_COLUMNS = [
    "id",
    "source",
    "row_key",
    "text",
    "text_canon",
    "label_binary",
    "label_fine",
    "label_policy",
    "label_scam",
    "policy_negative",
    "category",
    "lang",
    "template_key",
    "dedup_key",
    "text_available",
    "dup_within_set",
    "label_conflict_in_set",
    "eval_ok",
    "pool_exact",
    "pool_near",
    "pool_matches",
]
PHRESHPHISH_FAMILY = "url_phreshphish"
URL_COLUMNS_OUT = [
    "id",
    "url",
    "url_norm",
    "label",
    "source",
    "row_key",
    "reg_domain",
    "date",
    "brand",
]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _assert_eval_clean(frame: pd.DataFrame, ids: list[str], name: str) -> None:
    """Test and cal sets must hold only kept, non-synthetic, eval-eligible rows."""
    sub = frame.set_index("id").loc[ids]
    if sub["dup_of"].notna().any():
        raise AssertionError(f"{name}: duplicate rows leaked into a test/cal partition")
    if sub["synthetic"].fillna(False).astype(bool).any():
        raise AssertionError(f"{name}: synthetic rows leaked into a test/cal partition")
    if not sub["eval_ok"].astype(bool).all():
        raise AssertionError(f"{name}: ineligible rows leaked into a test/cal partition")


def _write_split(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def build_sms(raw_dir: Path, out_dir: Path, rep: dict) -> pd.DataFrame:
    t0 = time.time()
    raw = load_all_sms(raw_dir)
    _log(f"SMS raw rows: {len(raw)} ({time.time() - t0:.1f}s)")
    frame, camp_diag = build_sms_frame(raw)
    _log(f"SMS canonicalised, deduped and clustered ({time.time() - t0:.1f}s)")
    kept = frame[frame["dup_of"].isna()]

    raw_counts = raw["source"].value_counts().to_dict()
    kept_counts = kept["source"].value_counts().to_dict()
    eval_counts = kept[kept["eval_ok"]]["source"].value_counts().to_dict()
    rep["sms_per_source"] = {
        src: {
            "raw": int(raw_counts.get(src, 0)),
            "kept_after_dedup": int(kept_counts.get(src, 0)),
            "exact_duplicates_removed": int(raw_counts.get(src, 0) - kept_counts.get(src, 0)),
            "eval_eligible": int(eval_counts.get(src, 0)),
        }
        for src in raw_counts
    }
    rep["sms_totals"] = {
        "raw": int(len(raw)),
        "kept": int(len(kept)),
        "exact_duplicates": int(len(frame) - len(kept)),
        "label_conflict_groups_rows": int(kept["label_conflict"].sum()),
        "campaigns": camp_diag,
    }
    rep["sms_overlap_pre_dedup"] = cross_source_overlap(frame)
    rep["sms_label_balance"] = report.label_balance(kept)
    rep["sms_label_fine"] = {
        src: report.top_counter(g["label_fine"].tolist(), 10) for src, g in kept.groupby("source")
    }
    rep["sms_lang"] = {
        src: report.top_counter(g["lang"].tolist(), 10) for src, g in kept.groupby("source")
    }
    rep["sms_lang_agreement"] = _lang_agreement(frame)
    rep["campaigns"] = report.campaign_stats(kept)
    _log(f"campaigns: {camp_diag['n_campaigns']}")

    sigs = minhash_signatures(kept["template_key"].tolist())
    rep["near_dup_sensitivity_components"] = near_dup_sensitivity(sigs, [0.5, 0.7, 0.9])
    _log(f"sensitivity done ({time.time() - t0:.1f}s)")

    drop_from_train = build_external(raw_dir, out_dir, kept, rep)

    camp_ids, camp_split = campaign_partition(kept)
    temp_ids = temporal_partition(kept, camp_split)
    lang_ids = lang_shift_partition(kept, camp_split)
    holdouts = source_holdout_partitions(kept)
    legacy = legacy_partitions(kept)

    families: dict[str, tuple[str, dict]] = {
        "campaign": ("Campaign-disjoint 70/10/20; groups = campaign_id.", camp_ids),
        "temporal": (
            "IMC25 smishing by date (70/10/20); ham paired from the campaign split. "
            "Ham has no dates.",
            temp_ids,
        ),
        "lang_shift": (
            "Train/cal English; test Hinglish + Devanagari Hindi (eval-eligible), "
            "campaign-disjoint from train.",
            lang_ids,
        ),
        "source_holdout": ("Leave-one-source-out for UCI, Mishra & Soni, ScamShield.", holdouts),
    }
    for name, parts in legacy.items():
        families[name] = (
            f"Published-style protocol on the dedup pool ({name}); comparability only.",
            parts,
        )

    rep_splits: dict[str, dict] = {}
    for fam, (desc, parts) in families.items():
        before = _train_rows(parts)
        parts = _drop_from_train(parts, drop_from_train)
        blocks = parts if _is_nested(parts) else {"_": parts}
        checks: dict[str, dict] = {}
        for key, block in blocks.items():
            label = f"{fam}/{key}" if key != "_" else fam
            for part in ("test", "cal"):
                if part in block:
                    _assert_eval_clean(frame, block[part], f"{label}/{part}")
            if fam != "temporal" and "train" in block and "test" in block:
                overlap = check_disjoint(kept, block, "campaign_id")
                # Legacy protocols are random by design, so their campaign overlap is recorded.
                if not fam.startswith("legacy") and any(overlap.values()):
                    raise AssertionError(f"{label}: campaign overlap {overlap}")
                checks[key] = overlap
        rep_splits[fam] = {
            "description": desc,
            "seed": SEED,
            "train_rows_removed_external_overlap": before - _train_rows(parts),
            "stats": (
                {k: report.partition_stats(kept, v) for k, v in blocks.items()}
                if _is_nested(parts)
                else report.partition_stats(kept, parts)
            ),
            "campaign_overlap_between_splits": checks,
        }
        _write_split(
            out_dir / "splits" / f"{fam}.json",
            {"family": fam, "seed": SEED, "description": desc, "partitions": parts},
        )
    rep_splits["temporal"]["campaign_overlap_between_splits"] = check_disjoint(
        kept, temp_ids, "campaign_id"
    )
    rep["sms_splits"] = rep_splits
    _log(f"SMS splits written ({time.time() - t0:.1f}s)")

    out = frame[SMS_COLUMNS].copy()
    out.to_parquet(out_dir / "sms.parquet", index=False)
    _log(f"sms.parquet: {len(out)} rows")
    return frame


def build_external(raw_dir: Path, out_dir: Path, kept: pd.DataFrame, rep: dict) -> set[str]:
    """Eval-only external SMS sets, deduplicated against the kept training pool.

    Writes external/<name>.parquet for each set whose raw file is present. Returns the pool ids
    that overlap any external message (exact or near-duplicate). Those rows are dropped from
    every TRAIN partition; the external rows themselves are never removed.
    """
    pool = kept[["id", "dedup_key", "template_key"]]
    built: dict[str, pd.DataFrame] = {}
    if (raw_dir / SMISHX_PATH).is_file():
        built["external_smishx"] = load_smishx(raw_dir)
    if (raw_dir / GROUNDLM_ZIP).is_file():
        built["external_groundlm"] = load_groundlm(raw_dir)
    drop: set[str] = set()
    frames = []
    for name, raw in built.items():
        frame, ids = pool_overlap(build_external_frame(raw), pool)
        drop |= ids
        frames.append(frame)
        (out_dir / "external").mkdir(parents=True, exist_ok=True)
        frame[EXTERNAL_COLUMNS].to_parquet(out_dir / "external" / f"{name}.parquet", index=False)
        _log(f"{name}: {len(frame)} rows written")
    if frames:
        allext = pd.concat(frames, ignore_index=True)
        rep["external"] = {
            **overlap_report(allext, len(pool)),
            "cross_set_exact": cross_set_overlap(allext),
            "pool_rows_dropped_from_train": len(drop),
            "sets_present": sorted(built),
        }
    else:
        rep["external"] = {"sets_present": [], "pool_rows_dropped_from_train": 0}
    return drop


def _train_rows(parts: dict) -> int:
    if _is_nested(parts):
        return sum(_train_rows(v) for v in parts.values())
    return len(parts.get("train", []))


def _drop_from_train(parts: dict, drop: set[str]) -> dict:
    """Remove ``drop`` from every train list (nested folds included); other splits are kept."""
    if not drop:
        return parts
    if _is_nested(parts):
        return {k: _drop_from_train(v, drop) for k, v in parts.items()}
    return {**parts, "train": [i for i in parts["train"] if i not in drop]}


def _is_nested(parts: dict) -> bool:
    """True for {name: {split: ids}} (folds / sources), False for {split: ids}."""
    return bool(parts) and all(isinstance(v, dict) for v in parts.values())


def _lang_agreement(frame: pd.DataFrame) -> dict:
    kept = frame[frame["dup_of"].isna()]
    imc = kept[(kept["source"] == SOURCE_IMC25) & kept["src_lang"].notna()]
    en_src = imc["src_lang"] == "en"
    det_en = imc["lang"] == "en"
    out = {
        "imc25_en_source_rows": int(en_src.sum()),
        "imc25_en_detector_recall": float((det_en & en_src).sum() / max(1, en_src.sum())),
        "imc25_non_en_source_rows": int((~en_src).sum()),
        "imc25_non_en_flagged_en": float((det_en & ~en_src).sum() / max(1, (~en_src).sum())),
    }
    scam = kept[(kept["source"] == "scamshield") & kept["src_lang"].notna()]
    out["scamshield_lang_agreement"] = float((scam["lang"] == scam["src_lang"]).mean())
    out["scamshield_lang_crosstab"] = report.language_agreement(
        kept[kept["source"] == "scamshield"]
    ).get("scamshield", {})
    return out


def build_urls(raw_dir: Path, out_dir: Path, rep: dict) -> None:
    """General URL pool (PhishTank, OpenPhish, LegitPhish). PhreshPhish is built separately."""
    t0 = time.time()
    raw = load_all_urls(raw_dir)
    _log(f"URL raw rows: {len(raw)} ({time.time() - t0:.1f}s)")
    urls, accounting = build_url_frame(raw)
    _log(f"URL corpus kept: {len(urls)} ({time.time() - t0:.1f}s)")
    rep["url_accounting"] = accounting
    rep["url_label_balance"] = report.label_balance(urls, label="label")
    # Sanity check for the LegitPhish label convention: phish-leaning tokens should be more
    # frequent among label-1 rows (phishing) than label-0 rows if the mapping is right.
    lp = urls[urls["source"] == "legitphish_mendeley"]
    rep["legitphish_label_sanity"] = {
        str(lab): {
            "n": int(len(g)),
            "mean_suspicious_token_count": float(g["suspicious_token_count"].mean()),
            "mean_tld_risky": float(g["tld_risky"].mean()),
            "mean_has_ip": float(g["ip_literal"].mean()),
        }
        for lab, g in lp.groupby("label")
    }
    urls["group"] = urls["reg_domain"].where(urls["reg_domain"] != "", urls["url_norm"])
    dom_ids, _dom_split = url_domain_partition(urls)
    temp_ids = url_temporal_partition(urls)
    labels_by_id = urls.set_index("id")["label"]
    for fam, parts, desc in (
        (
            "url_domain",
            dom_ids,
            "70/10/20 grouped by registered domain (reg_domain, else normalised URL).",
        ),
        (
            "url_temporal",
            temp_ids,
            "Time-ordered 70/10/20 over dated rows of the general pool (PhishTank submission "
            "date). PhreshPhish is its own family (url_phreshphish).",
        ),
    ):
        single = [
            f"{split}: single class"
            for split in ("train", "test")
            if labels_by_id.reindex(parts[split]).nunique() < 2
        ]
        if single:
            rep.setdefault("url_splits_skipped", {})[fam] = single
            _log(f"{fam} not written: {single}")
            continue
        check = check_disjoint(urls, parts, "group")
        stats = report.partition_stats(urls, parts)
        _write_split(
            out_dir / "splits" / f"{fam}.json",
            {"family": fam, "seed": SEED, "description": desc, "partitions": parts},
        )
        rep.setdefault("url_splits", {})[fam] = {
            "description": desc,
            "stats": stats,
            "reg_domain_overlap": check,
        }
    drop = {"row_order", "source_rank", "dup_of", "label_conflict", "group"}
    urls_out = urls[
        URL_COLUMNS_OUT + [c for c in urls.columns if c not in URL_COLUMNS_OUT and c not in drop]
    ]
    urls_out.to_parquet(out_dir / "urls.parquet", index=False)
    _log(f"urls.parquet: {len(urls_out)} rows, {urls_out.shape[1]} columns")


PHRESH_META_COLUMNS = ["lang", "split", "bench_benign", "bench_full", "bench_subsets"]
PHRESH_SUBSET_RATES = {
    "benchmark-005": "0.05%",
    "benchmark-010": "0.1%",
    "benchmark-050": "0.5%",
    "benchmark-100": "1%",
    "benchmark-500": "5%",
}


def build_phreshphish(src: Path, out_dir: Path, rep: dict) -> None:
    """Official PhreshPhish release as its own split family (url_phreshphish).

    urls_phreshphish.parquet keeps every train and test row. splits/url_phreshphish.json holds
    the official train and test id lists. There is no cal partition: the PhreshPhish protocol
    (AP and precision at 90% recall) takes no threshold.
    """
    if not src.exists():
        # A missing release is a normal state on a fresh checkout: the corpus is 5 GB and is
        # git-ignored. Fail with an actionable message instead of a bare pyarrow traceback, and
        # record it in the report so a run that skips this family is obviously incomplete.
        raise SystemExit(
            f"PhreshPhish release not found at {src}.\n"
            "It is not committed (about 5 GB). Either fetch it, or skip this family:\n"
            "  python -m rakshak.datasets.build --skip-urls\n"
            "  python -m rakshak.datasets.build --phreshphish /path/to/phreshphish_full\n"
            "Note: the Hugging Face parquet copy does NOT work here. Its columns are "
            "(sha256, url, label, target, date, lang, lang_score, html); this loader needs the "
            "official release with split/bench_* columns, produced by the Kaggle extract job "
            "kaggle/jobs/phreshphish_extract.py."
        )
    raw = load_phreshphish(src)
    urls = build_phreshphish_frame(raw)
    urls["group"] = urls["reg_domain"].where(urls["reg_domain"] != "", urls["url_norm"])
    train = urls[urls["split"] == "train"]
    test = urls[urls["split"] == "test"]
    desc = (
        "PhreshPhish official release split (train, test). Its own family: no other URL source "
        "is mixed in. No cal partition (the protocol takes no threshold)."
    )
    _write_split(
        out_dir / "splits" / f"{PHRESHPHISH_FAMILY}.json",
        {
            "family": PHRESHPHISH_FAMILY,
            "seed": SEED,
            "description": desc,
            "partitions": {"official": {"train": sorted(train["id"]), "test": sorted(test["id"])}},
        },
    )
    subset_ids: set[str] = set()
    for ids in urls["bench_subsets"]:
        subset_ids.update(ids)
    per_rate: dict[str, int] = {}
    for sid in subset_ids:
        rate = PHRESH_SUBSET_RATES.get(sid.split("/")[0], sid.split("/")[0])
        per_rate[rate] = per_rate.get(rate, 0) + 1
    stray = sum(
        1
        for ids, full in zip(urls["bench_subsets"], urls["bench_full"], strict=True)
        if ids and not full
    )
    rep["url_phreshphish"] = {
        "rows": int(len(urls)),
        "rows_by_split_label": {
            f"{s}|{'phish' if lab == 1 else 'benign'}": int(n)
            for (s, lab), n in urls.groupby(["split", "label"]).size().items()
        },
        "date_range": {
            s: [str(g["date"].dropna().min()), str(g["date"].dropna().max())]
            for s, g in urls.groupby("split")
        },
        "benchmark_benign_rows": int(urls["bench_benign"].sum()),
        "benchmark_benign_outside_test": int(
            (urls["bench_benign"] & (urls["split"] != "test")).sum()
        ),
        "benchmark_full_rows": int(urls["bench_full"].sum()),
        "benchmark_full_phish_rows": int((urls["bench_full"] & (urls["label"] == 1)).sum()),
        "benchmark_subsets_per_rate": dict(sorted(per_rate.items())),
        "subset_phish_rows_outside_full": int(stray),
        "url_norm_shared_train_test": len(set(train["url_norm"]) & set(test["url_norm"])),
        "reg_domain_shared_train_test": len(
            (set(train["reg_domain"]) & set(test["reg_domain"])) - {""}
        ),
        "duplicate_sha256": int(urls["row_key"].duplicated().sum()),
        "duplicate_url_norm": int(urls["url_norm"].duplicated().sum()),
    }
    cols = URL_COLUMNS_OUT + PHRESH_META_COLUMNS
    feats = [c for c in urls.columns if c not in cols and c not in ("group",)]
    urls[cols + feats].to_parquet(out_dir / "urls_phreshphish.parquet", index=False)
    _log(f"urls_phreshphish.parquet: {len(urls)} rows")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rakshak.datasets.build")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    ap.add_argument(
        "--phreshphish",
        type=Path,
        default=RAW_DIR / "phreshphish_full",
        help="PhreshPhish release: folder or parquet file (default: data/raw/phreshphish_full)",
    )
    ap.add_argument("--skip-urls", action="store_true")
    ap.add_argument("--skip-confound", action="store_true")
    args = ap.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rep: dict = {"seed": SEED}
    t0 = time.time()
    frame = build_sms(args.raw_dir, out_dir, rep)
    kept = frame[frame["dup_of"].isna()]
    if not args.skip_confound:
        rep["source_identity"] = source_identity_accuracy(kept)
        rep["label_from_source"] = label_from_source_accuracy(kept)
        _log(f"confound probes done ({time.time() - t0:.1f}s)")
    if not args.skip_urls:
        build_urls(args.raw_dir, out_dir, rep)
        build_phreshphish(args.phreshphish, out_dir, rep)
    rep["elapsed_s"] = round(time.time() - t0, 1)
    with (out_dir / "build_report.json").open("w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2, default=_jsonable)
    _log(f"wrote {out_dir / 'build_report.json'} in {rep['elapsed_s']}s")
    return 0


def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"not serialisable: {type(obj)}")


if __name__ == "__main__":
    raise SystemExit(main())
