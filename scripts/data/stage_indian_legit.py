"""Stage licensed Indian SMS legit/scam sources into data/processed/external/indian_legit/.

Writes one parquet per source (columns: id, text, label Int64, source, group) and stage_report.json.
label: 1 scam, 0 legit, <NA> unknown. Excluded sources (junioralive, IIITD, vinit9638, flatmars,
Gandharv) are never read here. Run with `python -I`.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "indian_legit"
OUT = ROOT / "data" / "processed" / "external" / "indian_legit"
GROUNDLM_ZIP = (
    ROOT
    / "data"
    / "raw"
    / "zenodo_20480151_scamshield_v4"
    / "scamshield_emnlp_artifact_v4_2026-05-31.zip"
)
GROUNDLM_MEMBER = "scamshield_emnlp_artifact_v4/predictions/xlmr_cleaned_test_probs.csv"
TAU = 0.5
MAX_LEN = 320


def sid(source: str, text: str) -> str:
    return hashlib.sha1(f"{source}\x1f{text}".encode()).hexdigest()


def frame(rows: list[dict]) -> pd.DataFrame:
    cols = ["id", "text", "label", "source", "group"]
    df = pd.DataFrame(rows, columns=cols)
    df["label"] = df["label"].astype("Int64")
    return df


def stage_finee(report: dict) -> pd.DataFrame:
    rep = {
        "rows_in": 0,
        "no_user_turn": 0,
        "too_long": 0,
        "email_or_statement": 0,
        "icici_dropped": 0,
        "exact_dup_dropped": 0,
    }
    seen: set[str] = set()
    out: list[dict] = []
    for split in ("train", "valid", "test"):
        path = RAW / "finee" / f"{split}.jsonl"
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rep["rows_in"] += 1
                rec = json.loads(line)
                user = next((m["content"] for m in rec["messages"] if m["role"] == "user"), None)
                if not user or "\n\n" not in user:
                    rep["no_user_turn"] += 1
                    continue
                text = user.split("\n\n", 1)[1].strip()
                if len(text) > MAX_LEN:
                    rep["too_long"] += 1
                    continue
                low = text.lower()
                if re.search(r"subject:|statement|dear (customer|valued)|\n.*\n.*\n", low):
                    rep["email_or_statement"] += 1
                    continue
                if "icici" in low:
                    rep["icici_dropped"] += 1
                    continue
                if text in seen:
                    rep["exact_dup_dropped"] += 1
                    continue
                seen.add(text)
                out.append(
                    {
                        "id": sid("finee_synth", text),
                        "text": text,
                        "label": 0,
                        "source": "finee_synth",
                        "group": None,
                    }
                )
    rep["rows_out"] = len(out)
    report["finee_synth"] = rep
    return frame(out)


def stage_cloveai(report: dict) -> pd.DataFrame:
    df = pd.read_parquet(RAW / "cloveai_india_spam_sms" / "train.parquet")
    texts = df["text"].astype(str)
    polarity = {
        "label_1_examples": texts[df["label"] == 1].head(5).tolist(),
        "label_0_examples": texts[df["label"] == 0].head(5).tolist(),
        "label_1_share": float((df["label"] == 1).mean()),
        "decision": "label 1 = spam/scam (prize, blocked-account, link bait, dating); "
        "label 0 = genuine transactional/personal",
    }
    report["cloveai_polarity"] = polarity
    seen: set[str] = set()
    out = []
    dropped = 0
    for text, lab in zip(texts, df["label"], strict=False):
        if text in seen:
            dropped += 1
            continue
        seen.add(text)
        out.append(
            {
                "id": sid("cloveai_india_spam_sms", text),
                "text": text,
                "label": int(lab),
                "source": "cloveai_india_spam_sms",
                "group": None,
            }
        )
    report["cloveai_india_spam_sms"] = {
        "rows_in": int(len(df)),
        "exact_dup_dropped": dropped,
        "rows_out": len(out),
    }
    return frame(out)


def stage_skyaara(report: dict) -> pd.DataFrame:
    rows = json.loads(
        (RAW / "skyaara" / "indian-payment-sms-dataset.json").read_text(encoding="utf-8")
    )["rows"]
    rep = {"rows_in": len(rows), "phishing_like_dropped": 0, "dropped_ids": []}
    out = []
    for r in rows:
        if r["id"].startswith("phishing-"):
            rep["phishing_like_dropped"] += 1
            rep["dropped_ids"].append(r["id"])
            continue
        group = r.get("variant_of") or r["id"]
        out.append(
            {
                "id": sid("skyaara", r["raw_sms"]),
                "text": r["raw_sms"],
                "label": 0,
                "source": "skyaara",
                "group": group,
            }
        )
    rep["rows_out"] = len(out)
    report["skyaara"] = rep
    return frame(out)


def stage_ridham(report: dict) -> pd.DataFrame:
    df = pd.read_csv(
        RAW / "ridham115_csv" / "synthetic_audited.csv", dtype=str, keep_default_na=False
    )
    mapping = {"scam": 1, "safe": 0}
    out, unknown = [], 0
    for _, r in df.iterrows():
        lab = mapping.get(r["label"].strip().lower())
        if lab is None:
            unknown += 1
            continue
        out.append(
            {
                "id": sid("ridham115", r["text"]),
                "text": r["text"],
                "label": lab,
                "source": "ridham115",
                "group": None,
            }
        )
    report["ridham115"] = {
        "rows_in": len(df),
        "unmapped_label": unknown,
        "rows_out": len(out),
        "scam": sum(o["label"] == 1 for o in out),
        "safe": sum(o["label"] == 0 for o in out),
    }
    return frame(out)


def stage_groundlm(report: dict) -> pd.DataFrame:
    with zipfile.ZipFile(GROUNDLM_ZIP) as z:
        raw = z.read(GROUNDLM_MEMBER)
    df = pd.read_csv(io.BytesIO(raw))
    out = pd.DataFrame(
        {
            "idx": df["idx"].astype("int64"),
            "prob": df["prob_scam"].astype("float64"),
            "pred": (df["prob_scam"] >= TAU).astype("int64"),
            "gold": df["label_cleaned"].astype("int64"),
        }
    )
    report["groundlm_xlmr_probs"] = {
        "rows": len(out),
        "tau": TAU,
        "pred_matches_file_pred": int((out["pred"] == df["pred"]).sum()),
        "gold_positives": int(out["gold"].sum()),
    }
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "skipped": {
            "smsshield_eval": "hf ganmoor-ai-labs/sms-shield not accessible (401; "
            "hf_fs: repository not found for this account)",
            "ieee_ham": "data/raw/ieee_dataport/ absent",
        }
    }
    stages = {
        "finee_synth": stage_finee,
        "cloveai_india_spam_sms": stage_cloveai,
        "skyaara": stage_skyaara,
        "ridham115": stage_ridham,
    }
    for name, fn in stages.items():
        df = fn(report)
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        report[name]["file"] = str((OUT / f"{name}.parquet").relative_to(ROOT).as_posix())
    gdf = stage_groundlm(report)
    gdf.to_parquet(OUT / "groundlm_xlmr_probs.parquet", index=False)
    report["groundlm_xlmr_probs"]["file"] = str(
        (OUT / "groundlm_xlmr_probs.parquet").relative_to(ROOT).as_posix()
    )
    (OUT / "stage_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "cloveai_polarity"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
