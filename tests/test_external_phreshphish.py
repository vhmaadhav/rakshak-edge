"""Tests for the label taxonomy, external SMS sets, the PhreshPhish release and the protocol
helpers. Tiny in-memory or tmp_path fixtures only; no raw data is needed."""

from __future__ import annotations

import datetime
import io
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rakshak.datasets import build as build_mod
from rakshak.datasets.common import stable_id
from rakshak.datasets.external import (
    GROUNDLM_MEMBER,
    GROUNDLM_ZIP,
    SMISHX_PATH,
    build_external_frame,
    load_groundlm,
    load_smishx,
    overlap_report,
    pool_overlap,
)
from rakshak.datasets.loaders import (
    PHRESHPHISH_FILE,
    SMS_RAW_COLUMNS,
    SOURCE_GROUNDLM,
    SOURCE_IMC25,
    SOURCE_MISHRA,
    SOURCE_SCAMSHIELD,
    SOURCE_SMISHX,
    SOURCE_UCI,
    load_phreshphish,
)
from rakshak.datasets.sms import build_sms_frame
from rakshak.datasets.taxonomy import policy_for
from rakshak.datasets.urls import build_phreshphish_frame
from rakshak.eval.external import policy_fpr
from rakshak.eval.phreshphish import precision_at_recall, protocol_over_subsets

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import url_baselines  # noqa: E402  (kaggle job module, importable without its dependencies)


def _sms_row(
    source: str, text: str, label_fine: str, label_binary: int, row_key: str, **extra
) -> dict:
    base = {c: None for c in SMS_RAW_COLUMNS}
    base.update(
        source=source,
        subsource="",
        row_key=row_key,
        text=text,
        label_binary=label_binary,
        label_fine=label_fine,
        source_split=None,
        date=None,
        synthetic=None,
        augmentation_only=False,
        src_lang=None,
    )
    base.update(extra)
    return base


def _sms_raw(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=SMS_RAW_COLUMNS)
    frame["synthetic"] = pd.array(frame["synthetic"].tolist(), dtype="boolean")
    return frame


# ---------------------------------------------------------------------------
# taxonomy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "fine", "conflict", "expected"),
    [
        (SOURCE_MISHRA, "smishing", False, ("scam", 1)),
        (SOURCE_MISHRA, "spam", False, ("promotion", 0)),
        (SOURCE_MISHRA, "ham", False, (None, 0)),
        (SOURCE_IMC25, "smishing", False, ("scam", 1)),
        (SOURCE_SCAMSHIELD, "scam", False, ("scam", 1)),
        (SOURCE_SCAMSHIELD, "spam", False, (None, None)),
        (SOURCE_UCI, "spam", False, (None, None)),
        (SOURCE_UCI, "ham", False, (None, 0)),
        (SOURCE_SMISHX, "legitimate", False, (None, 0)),
        (SOURCE_SMISHX, "spam", False, ("promotion", 0)),
        (SOURCE_SMISHX, "smishing", False, ("scam", 1)),
        (SOURCE_GROUNDLM, "fraud", False, ("scam", 1)),
        (SOURCE_GROUNDLM, "safe", False, (None, 0)),
        (SOURCE_UCI, "spam", True, ("unclear", None)),
        ("some_unknown_source", "spam", False, (None, None)),
    ],
)
def test_policy_for_fills_only_what_the_source_label_determines(source, fine, conflict, expected):
    got = policy_for(source, fine, conflict)
    assert (got.label_policy, got.label_scam) == expected


def test_sms_frame_adds_policy_columns_and_keeps_label_binary():
    texts = {
        "scam": "Your parcel is held, pay Rs 50 at http://a.example/x",
        "promo": "Flat 20% off on shoes this weekend only",
        "ham": "Are we still meeting for lunch on Friday",
        "uci_spam": "Congratulations you won a prize call now",
    }
    rows = [
        _sms_row(SOURCE_MISHRA, texts["scam"], "smishing", 1, "1"),
        _sms_row(SOURCE_MISHRA, texts["promo"], "spam", 1, "2"),
        _sms_row(SOURCE_UCI, texts["ham"], "ham", 0, "3"),
        _sms_row(SOURCE_UCI, texts["uci_spam"], "spam", 1, "4"),
    ]
    frame, _ = build_sms_frame(_sms_raw(rows))
    by_text = {r["text"]: r for r in frame.to_dict("records")}
    assert by_text[texts["scam"]]["label_policy"] == "scam"
    assert by_text[texts["scam"]]["label_scam"] == 1
    assert by_text[texts["promo"]]["label_policy"] == "promotion"
    assert by_text[texts["promo"]]["label_scam"] == 0
    assert pd.isna(by_text[texts["ham"]]["label_policy"])
    assert by_text[texts["ham"]]["label_scam"] == 0
    assert pd.isna(by_text[texts["uci_spam"]]["label_policy"])
    assert pd.isna(by_text[texts["uci_spam"]]["label_scam"])
    assert [by_text[texts[k]]["label_binary"] for k in ("scam", "promo", "ham", "uci_spam")] == [
        1,
        1,
        0,
        1,
    ]
    assert str(frame["label_scam"].dtype) == "Int8"


# ---------------------------------------------------------------------------
# external sets
# ---------------------------------------------------------------------------


def _write_smishx(raw: Path, rows: list[tuple[str, str]]) -> None:
    path = raw / SMISHX_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "SMS,label,if_URL,if_phone,if_email\n" + "".join(
        f'"{t}",{lab},0,0,0\n' for t, lab in rows
    )
    path.write_text(body, encoding="utf-8")


def _write_groundlm(raw: Path, rows: list[tuple[str, str, str, str]]) -> None:
    """rows: (idx, label_cleaned, template_bucket, text_or_hash)."""
    header = "idx,language,source,label_cleaned,pred,text,template_bucket\n"
    body = "".join(
        f'{i},en_in,cloveai_india,{lab},{lab},"{text}",{bucket}\n' for i, lab, bucket, text in rows
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(GROUNDLM_MEMBER, header + body)
    path = raw / GROUNDLM_ZIP
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())


def test_load_smishx_maps_labels_and_keeps_legacy_binary(tmp_path):
    _write_smishx(
        tmp_path,
        [
            ("Your OTP is 1234", "legitimate"),
            ("Win a prize", "spam"),
            ("Click http://x.example now", "smishing"),
        ],
    )
    raw = load_smishx(tmp_path)
    assert list(raw["label_fine"]) == ["legitimate", "spam", "smishing"]
    assert list(raw["label_binary"]) == [0, 1, 1]
    assert set(raw["source"]) == {SOURCE_SMISHX}


def test_load_groundlm_reads_member_in_memory_and_flags_hash_only_rows(tmp_path):
    _write_groundlm(
        tmp_path,
        [
            ("0", "1", "UPI/Payment", "Pay Rs 500 to verify UPI PIN"),
            ("1", "0", "Bank/account", "Your a/c credited Rs 900"),
            ("2", "0", "Other", "sha256:" + "ab" * 32),
        ],
    )
    raw = load_groundlm(tmp_path)
    assert list(raw["label_fine"]) == ["fraud", "safe", "safe"]
    assert list(raw["policy_negative"]) == [False, True, True]
    assert pd.isna(raw["text"].iloc[2])
    frame = build_external_frame(raw)
    assert list(frame["text_available"]) == [True, True, False]
    assert list(frame["eval_ok"]) == [True, True, False]


def test_external_within_set_duplicates_and_label_conflicts_are_not_eval_eligible(tmp_path):
    _write_smishx(
        tmp_path,
        [
            ("Your account is blocked, verify at http://a.example", "smishing"),
            ("your  ACCOUNT is blocked, verify at http://a.example", "smishing"),
            ("Free coupon for you", "spam"),
            ("Free coupon for you", "legitimate"),
        ],
    )
    frame = build_external_frame(load_smishx(tmp_path))
    # Row 1 repeats row 0 after case and spacing folding; row 3 repeats row 2 with another label.
    assert list(frame["dup_within_set"]) == [False, True, False, True]
    assert list(frame["label_conflict_in_set"]) == [False, False, True, True]
    assert list(frame["eval_ok"]) == [True, False, False, False]
    assert frame.loc[2, "label_policy"] == "unclear"


def test_pool_overlap_finds_exact_and_near_duplicates_and_reports_pool_ids():
    pool_rows = [
        _sms_row(
            SOURCE_UCI,
            "Your SBI account 123 is blocked. Update KYC at http://a.example/x today",
            "spam",
            1,
            "p1",
        ),
        _sms_row(SOURCE_UCI, "Are we still meeting for lunch on Friday", "ham", 0, "p2"),
        _sms_row(
            SOURCE_UCI, "Completely unrelated text about the weather in Pune today", "ham", 0, "p3"
        ),
    ]
    pool, _ = build_sms_frame(_sms_raw(pool_rows))
    pool = pool.loc[pool["dup_of"].isna()].reset_index(drop=True)
    ext_raw = pd.DataFrame(
        [
            # exact: same canonical text, different case and spacing
            {
                "source": SOURCE_SMISHX,
                "row_key": "e1",
                "text": "are we still   meeting for lunch on friday",
                "label_fine": "legitimate",
                "label_binary": 0,
                "policy_negative": None,
                "category": None,
            },
            # near: same template after masking digits and URL, different digits
            {
                "source": SOURCE_SMISHX,
                "row_key": "e2",
                "text": "Your SBI account 987 is blocked. Update KYC at http://b.example/y today",
                "label_fine": "smishing",
                "label_binary": 1,
                "policy_negative": None,
                "category": None,
            },
            # unrelated
            {
                "source": SOURCE_SMISHX,
                "row_key": "e3",
                "text": "Your flight to Delhi is on time",
                "label_fine": "legitimate",
                "label_binary": 0,
                "policy_negative": None,
                "category": None,
            },
        ]
    )
    frame, pool_ids = pool_overlap(build_external_frame(ext_raw), pool)
    assert list(frame["pool_exact"]) == [True, False, False]
    assert list(frame["pool_near"]) == [False, True, False]
    assert pool_ids == {pool.loc[0, "id"], pool.loc[1, "id"]}
    rep = overlap_report(frame, len(pool))
    assert rep["sets"][SOURCE_SMISHX]["pool_exact"] == 1
    assert rep["sets"][SOURCE_SMISHX]["pool_near_only"] == 1
    assert rep["sets"][SOURCE_SMISHX]["pool_any"] == 2


def test_pool_overlap_on_empty_pool_is_a_no_op(tmp_path):
    _write_smishx(tmp_path, [("hello there", "legitimate")])
    frame, ids = pool_overlap(
        build_external_frame(load_smishx(tmp_path)),
        pd.DataFrame(columns=["id", "dedup_key", "template_key"]),
    )
    assert ids == set()
    assert not frame["pool_exact"].any()


def test_drop_from_train_touches_only_train_lists_and_handles_folds():
    flat = {"train": ["a", "b", "c"], "cal": ["b"], "test": ["d"]}
    out = build_mod._drop_from_train(flat, {"b"})
    assert out == {"train": ["a", "c"], "cal": ["b"], "test": ["d"]}
    nested = {
        "fold_0": {"train": ["a", "b"], "test": ["x"]},
        "fold_1": {"train": ["b"], "test": ["y"]},
    }
    out = build_mod._drop_from_train(nested, {"b"})
    assert out["fold_0"] == {"train": ["a"], "test": ["x"]}
    assert out["fold_1"] == {"train": [], "test": ["y"]}
    assert build_mod._train_rows(nested) == 3
    assert build_mod._drop_from_train(flat, set()) is flat


# ---------------------------------------------------------------------------
# PhreshPhish release
# ---------------------------------------------------------------------------


def _phresh_table(tmp_path: Path) -> Path:
    rows = [
        ("s1", "https://a.example/login", "phish", "BrandA", "train", False, False, []),
        ("s2", "https://b.example/", "benign", None, "train", False, False, []),
        (
            "s3",
            "http://c.example/pay?id=7",
            "phish",
            "BrandC",
            "test",
            False,
            True,
            ["benchmark-005/0"],
        ),
        ("s4", "https://d.example/help", "benign", None, "test", True, True, []),
        ("s5", "http://e.example/x", "phish", "BrandE", "test", False, False, []),
    ]
    table = pa.table(
        {
            "sha256": [r[0] for r in rows],
            "url": [r[1] for r in rows],
            "label": [r[2] for r in rows],
            "target": [r[3] for r in rows],
            "date": pa.array([datetime.date(2025, 1, 2)] * len(rows), type=pa.date32()),
            "lang": ["en"] * len(rows),
            "split": [r[4] for r in rows],
            "bench_benign": [r[5] for r in rows],
            "bench_full": [r[6] for r in rows],
            "bench_subsets": pa.array([r[7] for r in rows], type=pa.list_(pa.string())),
        }
    )
    path = tmp_path / PHRESHPHISH_FILE
    pq.write_table(table, path)
    return path


def test_load_phreshphish_keeps_official_split_and_benchmark_flags(tmp_path):
    raw = load_phreshphish(_phresh_table(tmp_path))
    assert len(raw) == 5
    assert list(raw["label"]) == [1, 0, 1, 0, 1]
    assert list(raw["split"]) == ["train", "train", "test", "test", "test"]
    assert raw["bench_benign"].tolist() == [False, False, False, True, False]
    assert raw.loc[2, "bench_subsets"] == ["benchmark-005/0"]
    assert pd.isna(raw.loc[1, "brand"])


def test_load_phreshphish_rejects_unknown_labels(tmp_path):
    path = _phresh_table(tmp_path)
    table = pq.read_table(path).to_pandas()
    table.loc[0, "label"] = "maybe"
    pq.write_table(pa.Table.from_pandas(table, preserve_index=False), path)
    with pytest.raises(ValueError, match="label"):
        load_phreshphish(path)


def test_phreshphish_frame_has_keys_and_features_and_no_dedup(tmp_path):
    frame = build_phreshphish_frame(load_phreshphish(_phresh_table(tmp_path)))
    assert frame["id"].is_unique
    assert frame.loc[0, "id"] == stable_id("phreshphish", "s1")
    assert "url_norm" in frame and "reg_domain" in frame
    assert len(frame) == 5  # nothing dropped: the release is the protocol


def test_phreshphish_subsets_combine_benign_list_with_each_subset_phish():
    df = pd.DataFrame(
        {
            "bench_benign": [False, True, True, False, False],
            "bench_subsets": [
                ["benchmark-005/0"],
                [],
                [],
                ["benchmark-005/0", "benchmark-100/3"],
                ["benchmark-100/3"],
            ],
        }
    )
    subsets = dict(url_baselines.phreshphish_subsets(df))
    # Benign benchmark rows are 1 and 2 in every subset; phish rows come from bench_subsets.
    assert sorted(subsets["0.05%"].tolist()) == [0, 1, 2, 3]
    assert sorted(subsets["1%"].tolist()) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# protocol helpers
# ---------------------------------------------------------------------------


def test_precision_at_recall_picks_best_precision_reaching_target():
    y = np.array([1, 1, 0, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.6, 0.2])
    # Recall 0.9 needs all three positives, so the cut-off is 0.6 and precision is 3 / 4.
    assert precision_at_recall(y, s, 0.9) == pytest.approx(0.75)
    assert precision_at_recall(y, s, 0.5) == pytest.approx(1.0)
    assert np.isnan(precision_at_recall(np.zeros(4), np.arange(4.0)))


def test_protocol_over_subsets_averages_per_rate_with_sample_std():
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    s = np.array([0.9, 0.2, 0.1, 0.8, 0.3, 0.7, 0.6, 0.05])
    subsets = [("1%", np.array([0, 1, 2])), ("1%", np.array([3, 4, 5])), ("5%", np.arange(8))]
    out = protocol_over_subsets(y, s, subsets)
    assert list(out) == ["1%", "5%"]
    assert out["1%"]["n_subsets"] == 2
    assert out["1%"]["ap_mean"] == pytest.approx(
        np.mean([1.0, 1.0]) if False else out["1%"]["ap_mean"]
    )  # exact values checked below
    # Subset [0,1,2]: its one positive is ranked first, AP 1. Subset [3,4,5]: both positives sit
    # above the negative, AP 1. Mean 1, sample std 0.
    assert out["1%"]["ap_mean"] == pytest.approx(1.0)
    assert out["1%"]["ap_std"] == pytest.approx(0.0)
    assert out["5%"]["n_subsets"] == 1
    assert math.isnan(out["5%"]["ap_std"])  # one subset: sample std is undefined


def test_policy_fpr_global_and_per_category():
    scores = np.array([0.9, 0.7, 0.2, 0.8, 0.1, 0.4])
    negative = np.array([False, True, True, True, True, True])
    cats = ["Bank/account", "Bank/account", "Bank/account", "UPI/Payment", "UPI/Payment", None]
    out = policy_fpr(scores, 0.5, negative, cats)
    assert out["n_negatives"] == 5
    assert out["false_positives"] == 2  # 0.7 (bank) and 0.8 (upi)
    assert out["global_fpr"] == pytest.approx(0.4)
    assert out["meets_target"] is False
    assert out["by_category"]["Bank/account"] == {"fp": 1, "n": 2, "fpr": 0.5}
    assert out["by_category"]["UPI/Payment"] == {"fp": 1, "n": 2, "fpr": 0.5}
    assert out["by_category"]["(missing)"]["n"] == 1
    assert (
        policy_fpr(scores, 0.5, np.zeros(6, bool))["global_fpr"]
        != policy_fpr(scores, 0.5, np.zeros(6, bool))["global_fpr"]
    )  # NaN without negatives
