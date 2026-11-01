"""Teacher transfer files for distillation: per-run file matching, the test-id guard, membership
checks, the pool (labelled train rows plus unlabeled rows), manifest sha256 checks and the folder
resolver. Torch-free, so these run in the local venv."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from rakshak.eval import data as D
from rakshak.models import distill as K

TEACHER = "mmBERT-base"


def _sms_ids(n: int = 20) -> np.ndarray:
    return np.asarray([str(i) for i in range(n)], dtype=object)


def _run(name: str, train, cal, test, test_all=None) -> D.Run:
    test = np.asarray(test, dtype=np.int64)
    return D.Run(name=name, train=np.asarray(train, dtype=np.int64),
                 cal=np.asarray(cal, dtype=np.int64), test=test,
                 test_all=test if test_all is None else np.asarray(test_all, dtype=np.int64),
                 notes={})


def _frame(rows) -> pd.DataFrame:
    """rows: (id, split, logit_0, logit_1)."""
    return pd.DataFrame(rows, columns=["id", "split", "logit_0", "logit_1"])


def _write(folder, run_safe: str, frame: pd.DataFrame, seed: int = 0, teacher: str = TEACHER):
    path = folder / f"transfer_{teacher}_s{seed}_{run_safe}.parquet"
    frame.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------- file matching


def test_each_run_gets_its_own_transfer_file(tmp_path):
    runs = ["campaign", "lang_shift", "temporal", "source_holdout:holdout_scamshield",
            "source_holdout:holdout_uci_sms_spam"]
    for run in runs:
        _write(tmp_path, K.safe_run_name(run), _frame([("0", "train", 0.0, 1.0)]))
    got = {run: K.find_transfer_file(tmp_path, run).name for run in runs}
    assert got["campaign"] == f"transfer_{TEACHER}_s0_campaign.parquet"
    assert got["lang_shift"] == f"transfer_{TEACHER}_s0_lang_shift.parquet"
    assert got["source_holdout:holdout_scamshield"] == \
        f"transfer_{TEACHER}_s0_source_holdout_holdout_scamshield.parquet"
    # Distinct runs never resolve to the same file.
    assert len(set(got.values())) == len(runs)


def test_source_holdout_fold_names_map_colon_to_underscore(tmp_path):
    _write(tmp_path, "source_holdout_holdout_scamshield", _frame([("0", "train", 0.0, 1.0)]))
    _write(tmp_path, "source_holdout_holdout_uci_sms_spam", _frame([("0", "train", 0.0, 1.0)]))
    path = K.find_transfer_file(tmp_path, "source_holdout:holdout_scamshield")
    assert path.name.endswith("_source_holdout_holdout_scamshield.parquet")
    assert K.safe_run_name("source_holdout:holdout_scamshield") == \
        "source_holdout_holdout_scamshield"


def test_no_file_or_wrong_seed_is_an_error(tmp_path):
    _write(tmp_path, "campaign", _frame([("0", "train", 0.0, 1.0)]), seed=0)
    with pytest.raises(FileNotFoundError):
        K.find_transfer_file(tmp_path, "temporal")
    with pytest.raises(FileNotFoundError):
        K.find_transfer_file(tmp_path, "campaign", teacher_seed=1)
    assert K.find_transfer_file(tmp_path, "campaign", teacher_seed=0).exists()


def test_two_files_for_one_run_are_ambiguous(tmp_path):
    _write(tmp_path, "campaign", _frame([("0", "train", 0.0, 1.0)]), teacher="modelA")
    _write(tmp_path, "campaign", _frame([("0", "train", 0.0, 1.0)]), teacher="modelB")
    with pytest.raises(FileNotFoundError, match="2 teacher transfer files"):
        K.find_transfer_file(tmp_path, "campaign")


# ---------------------------------------------------------------- pool building


def test_pool_has_labelled_train_rows_then_unlabeled_rows():
    ids = _sms_ids()
    labels = np.arange(20) % 2
    run = _run("campaign", train=[0, 1, 2, 3], cal=[4, 5], test=[6, 7], test_all=[6, 7, 8])
    transfer = _frame([
        ("0", "train", 0.0, 1.0), ("1", "train", 1.0, 0.0), ("2", "train", 0.0, 2.0),
        ("3", "train", 0.0, 0.5),
        ("4", "cal", 0.0, 1.0), ("5", "cal", 0.0, 1.0),
        ("9", "unlabeled", 0.2, 0.1), ("10", "unlabeled", -1.0, 3.0),
    ])
    pool = K.build_pool(run, ids, labels, transfer, include_unlabeled=True)
    assert pool.n_labelled == 4 and pool.n_unlabeled == 2
    assert pool.positions.tolist() == [0, 1, 2, 3, 9, 10]
    assert pool.labels[:4].tolist() == labels[[0, 1, 2, 3]].tolist()
    assert np.isnan(pool.labels[4:]).all()
    assert pool.teacher[0].tolist() == [0.0, 1.0]
    assert pool.teacher[4].tolist() == [0.2, 0.1]
    stats = pool.stats()
    assert stats["labelled_with_teacher"] == 4 and stats["unlabeled_with_teacher"] == 2

    no_unl = K.build_pool(run, ids, labels, transfer, include_unlabeled=False)
    assert no_unl.positions.tolist() == [0, 1, 2, 3] and no_unl.n_unlabeled == 0


def test_test_id_in_transfer_file_is_refused():
    ids = _sms_ids()
    labels = np.zeros(20, dtype=np.int64)
    run = _run("campaign", train=[0, 1], cal=[2], test=[5], test_all=[5, 6])
    transfer = _frame([("0", "train", 0.0, 1.0), ("1", "train", 0.0, 1.0),
                       ("2", "cal", 0.0, 1.0), ("6", "unlabeled", 0.0, 1.0)])
    with pytest.raises(ValueError, match="test ids"):
        K.build_pool(run, ids, labels, transfer, include_unlabeled=True)


def test_training_pool_never_reintroduces_the_broad_unlabeled_rows():
    ids = _sms_ids()
    labels = np.arange(20) % 2
    run = _run("campaign", train=[0, 1], cal=[2], test=[3])
    # Row 4 could be an excluded copy of test row 3: ID-only guards cannot establish safety.
    transfer = _frame([("0", "train", 0.0, 1.0), ("1", "train", 1.0, 0.0),
                       ("2", "cal", 0.0, 1.0), ("4", "unlabeled", 0.0, 1.0)])
    pool = K.build_training_pool(run, ids, labels, transfer)
    assert pool.positions.tolist() == [0, 1] and pool.n_unlabeled == 0
    assert np.isfinite(pool.labels).all()
    with pytest.raises(ValueError, match="unlabeled KD training is disabled"):
        K.build_training_pool(run, ids, labels, transfer, include_unlabeled=True)


def test_file_from_another_split_fails_membership_check():
    """A campaign file used for a lang_shift run has different train rows: refused."""
    ids = _sms_ids()
    labels = np.zeros(20, dtype=np.int64)
    run = _run("lang_shift", train=[0, 1, 2], cal=[3], test=[9])
    campaign_file = _frame([("0", "train", 0.0, 1.0), ("1", "train", 0.0, 1.0),
                            ("5", "train", 0.0, 1.0), ("3", "cal", 0.0, 1.0)])
    with pytest.raises(ValueError, match="train rows differ"):
        K.build_pool(run, ids, labels, campaign_file, include_unlabeled=True)


def test_cal_membership_mismatch_is_refused():
    ids = _sms_ids()
    labels = np.zeros(20, dtype=np.int64)
    run = _run("temporal", train=[0, 1], cal=[2, 3], test=[9])
    frame = _frame([("0", "train", 0.0, 1.0), ("1", "train", 0.0, 1.0), ("2", "cal", 0.0, 1.0)])
    with pytest.raises(ValueError, match="cal rows differ"):
        K.build_pool(run, ids, labels, frame, include_unlabeled=True)


def test_unlabeled_rows_must_be_known_and_ids_unique():
    ids = _sms_ids()
    labels = np.zeros(20, dtype=np.int64)
    run = _run("campaign", train=[0, 1], cal=[2], test=[9])
    unknown = _frame([("0", "train", 0.0, 1.0), ("1", "train", 0.0, 1.0),
                      ("2", "cal", 0.0, 1.0), ("zz", "unlabeled", 0.0, 1.0)])
    with pytest.raises(ValueError, match="not in sms.parquet"):
        K.build_pool(run, ids, labels, unknown, include_unlabeled=True)
    clash = _frame([("0", "train", 0.0, 1.0), ("1", "train", 0.0, 1.0),
                    ("2", "cal", 0.0, 1.0), ("1", "unlabeled", 0.0, 1.0)])
    with pytest.raises(ValueError, match="duplicate ids"):
        K.build_pool(run, ids, labels, clash, include_unlabeled=True)


def test_every_train_row_must_appear_in_the_teacher_file():
    """The file's train membership must equal the run's train partition, so a teacher file that
    lacks a train row is refused rather than leaving that row without a teacher logit."""
    ids = _sms_ids()
    labels = np.array([1, 0] * 10, dtype=np.int64)
    run = _run("campaign", train=[0, 1, 2], cal=[3], test=[9])
    frame = _frame([("0", "train", 0.0, 1.0), ("2", "train", 0.0, 1.0), ("3", "cal", 0.0, 1.0)])
    with pytest.raises(ValueError, match="train rows differ"):
        K.build_pool(run, ids, labels, frame, include_unlabeled=True)


# ---------------------------------------------------------------- loading and manifest


def test_load_transfer_checks_schema_and_values(tmp_path):
    path = _write(tmp_path, "campaign", _frame([("0", "train", 0.0, 1.0)]))
    df = K.load_transfer(path)
    assert df["id"].tolist() == ["0"] and df["split"].tolist() == ["train"]
    bad = _write(tmp_path, "temporal", _frame([("0", "holdout", 0.0, 1.0)]))
    with pytest.raises(ValueError, match="unknown split"):
        K.load_transfer(bad)
    nan = _write(tmp_path, "lang_shift", _frame([("0", "train", np.nan, 1.0)]))
    with pytest.raises(ValueError, match="non-finite"):
        K.load_transfer(nan)


def test_manifest_sha256_is_checked(tmp_path):
    path = _write(tmp_path, "campaign", _frame([("0", "train", 0.0, 1.0)]))
    assert K.verify_manifest(tmp_path, "campaign", path) is None  # no manifest, no check
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"teacher": TEACHER, "seed": 0,
                "splits": {"campaign": {"file": path.name, "sha256": digest}}}
    (tmp_path / K.MANIFEST).write_text(json.dumps(manifest))
    assert K.verify_manifest(tmp_path, "campaign", path)["sha256"] == digest
    manifest["splits"]["campaign"]["sha256"] = "0" * 64
    (tmp_path / K.MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sha256"):
        K.verify_manifest(tmp_path, "campaign", path)
    with pytest.raises(ValueError, match="no entry"):
        K.verify_manifest(tmp_path, "temporal", path)


def test_resolve_teacher_dir_accepts_folder_or_one_level_down(tmp_path):
    (tmp_path / "rakshak-teacher-logits").mkdir()
    _write(tmp_path / "rakshak-teacher-logits", "campaign", _frame([("0", "train", 0.0, 1.0)]))
    assert K.resolve_teacher_dir(tmp_path / "rakshak-teacher-logits") == \
        tmp_path / "rakshak-teacher-logits"
    assert K.resolve_teacher_dir(tmp_path) == tmp_path / "rakshak-teacher-logits"
    with pytest.raises(FileNotFoundError):
        K.resolve_teacher_dir(tmp_path / "missing")
