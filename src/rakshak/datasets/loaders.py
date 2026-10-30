"""Readers for the raw SMS and URL sources.

Every reader returns a DataFrame with the same SMS columns (see ``SMS_RAW_COLUMNS``). Raw
files are untrusted data: they are parsed with pandas / pyarrow / json only and never
executed. Zip members are read in memory and never extracted to disk.

Label mapping (documented in docs/DATACARD.md):
  UCI              ham -> ham (0); spam -> spam (1)
  Mishra & Soni    ham -> ham; spam/Spam -> spam; smishing/Smishing -> smishing (1 for both)
  IMC25            every row is a user-reported smishing message -> smishing (1)
  ScamShield       is_scam 0 -> ham; is_scam 1 with intent "General Spam / Telemarketing"
                   -> spam; any other scam intent -> scam
  anmol scam-ham   ham -> ham; spam -> spam
  bolewara         0 -> ham; 1 -> scam (its English scam rows are UCI spam copies)
"""

from __future__ import annotations

import bz2
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

SMS_RAW_COLUMNS = [
    "source",
    "subsource",
    "row_key",
    "text",
    "label_binary",
    "label_fine",
    "source_split",
    "date",
    "synthetic",
    "augmentation_only",
    "src_lang",
]

SOURCE_UCI = "uci_sms_spam"
SOURCE_MISHRA = "mishra_soni_smsphishing"
SOURCE_IMC25 = "imc25_fishing_for_smishing"
SOURCE_SCAMSHIELD = "scamshield"
SOURCE_ANMOL = "anmol_scam_ham_india"
SOURCE_BOLEWARA = "bolewara_hinglish_scam"

SMS_SOURCES = [
    SOURCE_UCI,
    SOURCE_MISHRA,
    SOURCE_IMC25,
    SOURCE_SCAMSHIELD,
    SOURCE_ANMOL,
    SOURCE_BOLEWARA,
]

_SCAMSHIELD_LANG = {"English": "en", "Hindi": "hi_deva", "Hinglish": "hinglish"}


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=SMS_RAW_COLUMNS)
    df["synthetic"] = pd.array(df["synthetic"].tolist(), dtype="boolean")
    df["augmentation_only"] = df["augmentation_only"].astype(bool)
    return df


def load_uci(raw_dir: Path) -> pd.DataFrame:
    with zipfile.ZipFile(raw_dir / "uci_sms_spam" / "smsspamcollection.zip") as zf:
        lines = zf.read("SMSSpamCollection").decode("utf-8").splitlines()
    rows = []
    for i, line in enumerate(lines):
        label, _, text = line.partition("\t")
        binary = 1 if label == "spam" else 0
        rows.append(
            dict(
                source=SOURCE_UCI,
                subsource="",
                row_key=f"{i:05d}",
                text=text,
                label_binary=binary,
                label_fine="spam" if binary else "ham",
                source_split=None,
                date=None,
                synthetic=None,
                augmentation_only=False,
                src_lang=None,
            )
        )
    return _frame(rows)


_MISHRA_FINE = {"ham": "ham", "spam": "spam", "smishing": "smishing"}


def load_mishra(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "mishra_soni_mendeley_smsphishing" / "Dataset_5971.zip"
    with zipfile.ZipFile(path) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        data = pd.read_csv(
            io.BytesIO(zf.read(member)),
            encoding="utf-8",
            encoding_errors="replace",
            dtype=str,
            keep_default_na=False,
        )
    rows = []
    for i, (lab, text) in enumerate(zip(data["LABEL"], data["TEXT"], strict=True)):
        fine = _MISHRA_FINE[lab.strip().lower()]
        rows.append(
            dict(
                source=SOURCE_MISHRA,
                subsource="",
                row_key=f"{i:05d}",
                text=text,
                label_binary=0 if fine == "ham" else 1,
                label_fine=fine,
                source_split=None,
                date=None,
                synthetic=None,
                augmentation_only=False,
                src_lang=None,
            )
        )
    return _frame(rows)


def load_imc25(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "imc25_fishing_for_smishing" / "final_dataset_output.csv"
    data = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    rows = []
    for i, rec in enumerate(data.itertuples(index=False)):
        date = rec.time if rec.time and rec.time[:4].isdigit() else None
        rows.append(
            dict(
                source=SOURCE_IMC25,
                subsource=rec.scam_type,
                row_key=f"{i:06d}",
                text=rec.text,
                label_binary=1,
                label_fine="smishing",
                source_split=None,
                date=date,
                synthetic=False,
                augmentation_only=False,
                src_lang=("en" if rec.language == "English" else "non_en")
                if rec.language
                else None,
            )
        )
    return _frame(rows)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_scamshield(raw_dir: Path) -> pd.DataFrame:
    base = raw_dir / "hf_sidzzz07_scamshield"
    rows = []
    for split in ("train", "val", "test"):
        for rec in _read_jsonl(base / f"{split}.jsonl"):
            is_scam = int(rec["is_scam"])
            intent = rec.get("head2_scam_intent", "")
            if not is_scam:
                fine = "ham"
            elif intent == "General Spam / Telemarketing":
                fine = "spam"
            else:
                fine = "scam"
            sub = rec.get("source_dataset", "")
            if sub == "Synthetic_Tier_C":
                synthetic: bool | None = True
            elif str(rec["id"]).startswith("real_"):
                synthetic = False
            else:
                synthetic = None
            rows.append(
                dict(
                    source=SOURCE_SCAMSHIELD,
                    subsource=sub,
                    row_key=str(rec["id"]),
                    text=rec["text"],
                    label_binary=is_scam,
                    label_fine=fine,
                    source_split=split,
                    date=None,
                    synthetic=synthetic,
                    augmentation_only=False,
                    src_lang=_SCAMSHIELD_LANG.get(rec.get("language", "")),
                )
            )
    return _frame(rows)


def load_anmol(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "hf_anmolshrivastav_scam_ham_india" / "scam_hum_india.csv"
    data = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    rows = []
    for i, (text, lab) in enumerate(zip(data["text"], data["label"], strict=True)):
        binary = 1 if lab.strip().lower() == "spam" else 0
        rows.append(
            dict(
                source=SOURCE_ANMOL,
                subsource="",
                row_key=f"{i:05d}",
                text=text,
                label_binary=binary,
                label_fine="spam" if binary else "ham",
                source_split=None,
                date=None,
                synthetic=None,
                augmentation_only=True,
                src_lang=None,
            )
        )
    return _frame(rows)


def load_bolewara(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "hf_bolewara_hinglish_scam" / "data.json"
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    rows = []
    for i, rec in enumerate(payload["rows"]):
        binary = int(rec["label"])
        rows.append(
            dict(
                source=SOURCE_BOLEWARA,
                subsource="",
                row_key=f"{i:05d}",
                text=rec["text"],
                label_binary=binary,
                label_fine="scam" if binary else "ham",
                source_split=None,
                date=None,
                synthetic=None,
                augmentation_only=True,
                src_lang=None,
            )
        )
    return _frame(rows)


def load_all_sms(raw_dir: Path) -> pd.DataFrame:
    frames = [
        load_uci(raw_dir),
        load_mishra(raw_dir),
        load_imc25(raw_dir),
        load_scamshield(raw_dir),
        load_anmol(raw_dir),
        load_bolewara(raw_dir),
    ]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# URL sources
# ---------------------------------------------------------------------------

URL_RAW_COLUMNS = ["source", "row_key", "url", "label", "date", "brand"]

SOURCE_PHISHTANK = "phishtank_online_valid"
SOURCE_OPENPHISH = "openphish_community_feed"
SOURCE_LEGITPHISH = "legitphish_mendeley"
SOURCE_PHRESHPHISH = "phreshphish"

PHRESHPHISH_HF_ID = "phreshphish/phreshphish"


def load_phishtank(raw_dir: Path) -> pd.DataFrame:
    with bz2.open(
        raw_dir / "phishtank_online_valid" / "online-valid.json.bz2", "rt", encoding="utf-8"
    ) as fh:
        records = json.load(fh)
    rows = []
    for rec in records:
        sub = rec.get("submission_time") or ""
        rows.append(
            dict(
                source=SOURCE_PHISHTANK,
                row_key=str(rec["phish_id"]),
                url=rec["url"],
                label=1,
                date=sub[:10] or None,
                brand=rec.get("target") or None,
            )
        )
    return pd.DataFrame(rows, columns=URL_RAW_COLUMNS)


def load_openphish(raw_dir: Path) -> pd.DataFrame:
    lines = (
        (raw_dir / "openphish_community_feed" / "feed.txt").read_text(encoding="utf-8").splitlines()
    )
    rows = [
        dict(
            source=SOURCE_OPENPHISH,
            row_key=f"{i:05d}",
            url=line.strip(),
            label=1,
            date=None,
            brand=None,
        )
        for i, line in enumerate(lines)
        if line.strip()
    ]
    return pd.DataFrame(rows, columns=URL_RAW_COLUMNS)


def load_legitphish(raw_dir: Path) -> pd.DataFrame:
    data = pd.read_csv(
        raw_dir / "legitphish_mendeley" / "url_features_extracted1.csv",
        dtype={"URL": str},
        encoding="utf-8",
    )
    rows = []
    for i, (url, cls) in enumerate(zip(data["URL"], data["ClassLabel"], strict=True)):
        if pd.isna(cls) or not isinstance(url, str):
            continue
        # Convention confirmed by the build's suspicious-token sanity check (report.json):
        # ClassLabel 0 = phishing, 1 = legitimate.
        label = 1 if int(cls) == 0 else 0
        rows.append(
            dict(
                source=SOURCE_LEGITPHISH,
                row_key=f"{i:06d}",
                url=url.strip(),
                label=label,
                date=None,
                brand=None,
            )
        )
    return pd.DataFrame(rows, columns=URL_RAW_COLUMNS)


def load_phreshphish(src: str | Path) -> pd.DataFrame:
    """PhreshPhish URL, label, date and target columns from local parquet or HF streaming.

    ``src`` is either a directory / parquet file on disk, or a Hugging Face dataset id
    (e.g. ``phreshphish/phreshphish``). The HF path streams rows with the ``datasets``
    library (not a project dependency: install it in the Kaggle job) and keeps only the
    four columns, so the 145 GB train split never has to be materialised locally.
    """
    path = Path(src)
    cols = ["sha256", "url", "label", "target", "date"]
    if path.exists():
        import pyarrow.parquet as pq

        files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
        frames = []
        for f in files:
            table = pq.read_table(f, columns=cols)
            frames.append(table.to_pandas())
        records = pd.concat(frames, ignore_index=True).to_dict("records")
    else:
        from datasets import load_dataset  # optional; only used for the full HF dataset

        streams = []
        for split in ("train", "test"):
            ds = load_dataset(str(src), split=split, streaming=True)
            ds = ds.select_columns(cols)
            streams.extend({**rec, "_split": split} for rec in ds)
        records = streams
    rows = []
    for rec in records:
        label = {"phish": 1, "benign": 0}.get(str(rec["label"]))
        if label is None:
            continue
        date = rec.get("date")
        date_s = (
            str(date)[:10] if date is not None and str(date) not in ("", "NaT", "None") else None
        )
        target = rec.get("target")
        rows.append(
            dict(
                source=SOURCE_PHRESHPHISH,
                row_key=str(rec["sha256"]),
                url=rec["url"],
                label=label,
                date=date_s,
                brand=None if target in (None, "None", "") else str(target),
            )
        )
    return pd.DataFrame(rows, columns=URL_RAW_COLUMNS)


def load_all_urls(raw_dir: Path, phreshphish_src: str | Path | None) -> pd.DataFrame:
    frames = [load_phishtank(raw_dir), load_openphish(raw_dir), load_legitphish(raw_dir)]
    src = phreshphish_src if phreshphish_src is not None else raw_dir / "hf_phreshphish_shards"
    frames.append(load_phreshphish(src))
    return pd.concat(frames, ignore_index=True)
