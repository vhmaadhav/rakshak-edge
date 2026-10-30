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
from rakshak.datasets.loaders import SOURCE_IMC25, load_all_sms, load_all_urls
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
from rakshak.datasets.urls import build_url_frame

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


def build_urls(raw_dir: Path, out_dir: Path, phreshphish: str | None, rep: dict) -> None:
    t0 = time.time()
    raw = load_all_urls(raw_dir, phreshphish)
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
    for fam, parts, desc in (
        (
            "url_domain",
            dom_ids,
            "70/10/20 grouped by registered domain (reg_domain, else normalised URL).",
        ),
        (
            "url_temporal",
            temp_ids,
            "Time-ordered 70/10/20 over dated rows (PhishTank submission date, PhreshPhish date).",
        ),
    ):
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rakshak.datasets.build")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    ap.add_argument(
        "--phreshphish",
        default=None,
        help="Local parquet dir/file or HF id to stream (default: shards in raw dir)",
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
        build_urls(args.raw_dir, out_dir, args.phreshphish, rep)
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
