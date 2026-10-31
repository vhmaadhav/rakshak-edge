"""Torch-free checks for the sms_teacher changes: split iteration, the transfer-set exclusion of
test rows, the attack-harness row cap, and the non-finite (NaN) detector.

sms_teacher imports torch at module level, which this environment does not install. So these
tests exercise the helpers the teacher calls: rakshak.eval.data and kaggle/jobs/_common.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import _common  # noqa: E402

from rakshak.eval import data as D  # noqa: E402

N = 40
IDS = [str(i) for i in range(N)]
SYNTHETIC = {5, 27}  # 5 sits in a train partition, 27 in the campaign test partition


def _split_bodies() -> dict[str, dict]:
    """Split files as the teacher reads them: flat and partitioned, with the same row ranges."""
    return {
        "campaign": {"seed": 0, "train": IDS[0:20], "cal": IDS[20:25], "test": IDS[25:40]},
        "temporal": {"seed": 0, "train": IDS[0:20], "test": IDS[30:40]},
        "legacy_a": {"seed": 0, "train": IDS[0:20], "test": IDS[20:40]},
        "legacy_b": {"seed": 0, "train": IDS[20:40], "test": IDS[0:20]},
        "source_holdout": {"seed": 1, "partitions": {
            "fold_a": {"train": IDS[10:40], "test": IDS[0:10]},
            "fold_b": {"train": IDS[0:10] + IDS[20:40], "test": IDS[10:20]},
            "fold_c": {"train": IDS[0:20] + IDS[30:40], "test": IDS[20:30]},
        }},
    }


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    sms = pd.DataFrame({
        "id": IDS,
        "text": [f"message {i}" for i in range(N)],
        "label_binary": [i % 2 for i in range(N)],
        "campaign_id": [f"c{i // 4}" for i in range(N)],
        "eval_ok": [True] * N,
        "synthetic": [i in SYNTHETIC for i in range(N)],
    })
    sms.to_parquet(tmp_path / "sms.parquet", index=False)
    (tmp_path / "splits").mkdir()
    for name, body in _split_bodies().items():
        (tmp_path / "splits" / f"{name}.json").write_text(json.dumps(body), encoding="utf-8")
    return tmp_path


@pytest.fixture()
def table(root: Path) -> pd.DataFrame:
    return D.load_sms(root)


def _runs(root: Path, table: pd.DataFrame, patterns):
    labels = table["label_binary"].to_numpy().astype(np.int64)
    return D.load_runs(root, table, patterns=patterns, labels=labels, mask=D.eval_mask(table))


def _by_name(runs):
    return {r.name: r for r in runs}


# Split iteration (what --splits selects, as sms_student does)


def test_default_campaign_selects_only_the_campaign_split(root, table):
    assert [r.name for r in _runs(root, table, ["campaign"])] == ["campaign"]


def test_source_holdout_yields_every_fold(root, table):
    names = [r.name for r in _runs(root, table, ["source_holdout"])]
    assert names == ["source_holdout:fold_a", "source_holdout:fold_b", "source_holdout:fold_c"]


def test_legacy_glob_matches_each_legacy_file(root, table):
    names = [r.name for r in _runs(root, table, ["legacy_*"])]
    assert names == ["legacy_a", "legacy_b"]


def test_several_patterns_are_unioned(root, table):
    names = [r.name for r in _runs(root, table, ["temporal", "legacy_a"])]
    assert names == ["legacy_a", "temporal"]


def test_unmatched_split_raises(root, table):
    with pytest.raises(FileNotFoundError):
        _runs(root, table, ["lang_shift"])


def test_every_fold_keeps_train_test_disjoint(root, table):
    for run in _runs(root, table, ["source_holdout"]):
        assert not set(run.train) & set(run.test_all)


def test_carved_calibration_when_split_has_no_cal(root, table):
    run = _by_name(_runs(root, table, ["temporal"]))["temporal"]
    assert len(run.cal) > 0
    assert not set(run.cal) & set(run.train)


# Transfer set: every row outside the split's test partition, never a test row


def test_transfer_excludes_the_whole_test_partition(root, table):
    run = _by_name(_runs(root, table, ["campaign"]))["campaign"]
    tpos = D.transfer_positions(len(table), run.test_all)
    assert not set(tpos.tolist()) & set(run.test_all.tolist())
    assert set(tpos.tolist()) | set(run.test_all.tolist()) == set(range(N))
    assert set(run.train.tolist()) <= set(tpos.tolist())
    assert set(run.cal.tolist()) <= set(tpos.tolist())


def test_transfer_excludes_test_rows_the_eval_mask_dropped(root, table):
    run = _by_name(_runs(root, table, ["campaign"]))["campaign"]
    masked_test = 27
    assert masked_test in run.test_all
    assert masked_test not in run.test
    assert masked_test not in D.transfer_positions(len(table), run.test_all)


def test_transfer_ids_pass_the_id_check_for_every_fold(root, table):
    ids = table["id"].to_numpy()
    for run in _runs(root, table, ["source_holdout", "legacy_*"]):
        tpos = D.transfer_positions(len(table), run.test_all)
        D.assert_no_test_rows(ids[tpos], ids[run.test_all])


def test_assert_no_test_rows_raises_on_overlap():
    with pytest.raises(ValueError, match="test ids"):
        D.assert_no_test_rows(["1", "2"], ["2", "3"])


def test_assert_no_test_rows_compares_ids_as_strings():
    with pytest.raises(ValueError):
        D.assert_no_test_rows([1, 2], ["2"])


def test_assert_no_test_rows_passes_when_disjoint():
    D.assert_no_test_rows(["1", "4"], ["2", "3"])


# Attack-harness cap


def test_sample_cap_is_seeded_ordered_and_bounded():
    items = list(range(100))
    first = D.sample_cap(items, 10, 0)
    assert len(first) == 10
    assert first == D.sample_cap(items, 10, 0)
    assert first == sorted(first)
    assert D.sample_cap(items, 10, 1) != first


@pytest.mark.parametrize("cap", [None, 0, 100, 500])
def test_sample_cap_keeps_everything_when_not_needed(cap):
    items = list(range(100))
    assert D.sample_cap(items, cap, 0) == items


# Non-finite detector


class FakeCtx:
    def __init__(self):
        self.results: dict = {}
        self.lines: list[str] = []

    def log(self, msg):
        self.lines.append(str(msg))

    def save(self):
        pass


def test_require_finite_passes_on_finite_arrays():
    _common.require_finite(np.ones((3, 2)), "eval logits")
    _common.require_finite(0.25, "training loss", step=1)


def test_require_finite_diagnostic_names_step_lr_dtype_and_last_loss():
    with pytest.raises(_common.NonFiniteError) as exc:
        _common.require_finite(
            np.array([[0.1, np.nan]]), "training logits", step=412, lr=2.5e-5,
            dtype="fp32 (no autocast, no bf16)", last_finite_loss=0.6931,
        )
    message = str(exc.value)
    assert "training logits" in message
    assert "step 412" in message
    assert "lr 2.5e-05" in message
    assert "dtype fp32 (no autocast, no bf16)" in message
    assert "last finite loss 0.6931" in message
    assert "1 of 2 values" in message


def test_require_finite_flags_an_inf_loss_on_the_first_step():
    with pytest.raises(_common.NonFiniteError) as exc:
        _common.require_finite(float("inf"), "training loss", step=1, lr=3e-5,
                               dtype="fp16 autocast over fp32 weights", last_finite_loss=None)
    assert "step 1" in str(exc.value)
    assert "last finite loss n/a" in str(exc.value)


def test_non_finite_error_is_a_runtime_error():
    assert issubclass(_common.NonFiniteError, RuntimeError)


def test_non_finite_unit_fails_and_the_job_goes_on():
    ctx = FakeCtx()
    units = _common.Units(ctx)

    def diverge():
        _common.require_finite(float("nan"), "training loss", step=7, lr=1e-5,
                               dtype="fp32 (no autocast, no bf16)", last_finite_loss=0.5)

    assert units.run({"model": "mdeberta", "seed": 0}, diverge) is None
    assert units.run({"model": "xlmr", "seed": 0}, lambda: "ok") == "ok"
    assert units.failed == 1 and units.ok == 1
    failure = ctx.results["failures"][0]
    assert failure["error"].startswith("NonFiniteError: non-finite training loss at step 7")
    assert units.status() == "partial"
