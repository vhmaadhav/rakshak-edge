"""Invented messages only; no external files/models or corpus inference."""

import json

import pandas as pd
import pytest

from rakshak.datasets.loaders import SOURCE_GROUNDLM
from rakshak.llm import budget, chunks
from rakshak.llm.policy_training import (
    MODEL,
    REVISION,
    annotation_spec,
    assert_proposal_scope,
    inventory,
    train_null_ids,
)


def fixture():
    frame = pd.DataFrame({"id": [f"id{i}" for i in range(8)],
                          "text": ["Invented message " + str(i) for i in range(8)],
                          "source": ["training"] * 7 + [SOURCE_GROUNDLM],
                          "dup_of": [None] * 4 + ["id0", None, None, None],
                          "label_policy": [None, "promotion", None, None, None, None, None, None],
                          "synthetic": [False] * 5 + [True, False, False],
                          "lang": ["en"] * 8, "label_fine": ["ham"] * 8,
                          "campaign_id": [f"campaign{i}" for i in range(8)]})
    parts = {"train": {"id0", "id1", "id4", "id5", "id6", "id7"},
             "cal": {"id2"}, "test": {"id3"}}
    return frame, parts


def test_selector_preserves_existing_labels_and_excludes_raw_cal_test_external_synthetic():
    frame, parts = fixture()
    before = frame.copy(deep=True)
    assert train_null_ids(frame, parts) == ["id0", "id6"]
    pd.testing.assert_frame_equal(frame, before)
    for rid in ["id1", "id2", "id3", "id4", "id5", "id7"]:
        with pytest.raises(ValueError, match="outside"):
            assert_proposal_scope(frame, parts, [rid])


def test_same_id_cannot_carry_a_training_proposal_into_another_folds_test():
    frame, parts = fixture()
    other = {"train": {"id3"}, "cal": {"id2"}, "test": {"id0", "id6"}}
    assert_proposal_scope(frame, parts, ["id0"])
    with pytest.raises(ValueError, match="outside"):
        assert_proposal_scope(frame, other, ["id0"])
    result = inventory(frame, {"first": parts, "other": other})
    assert result["eligible_ids"]["other"] == ["id3"]
    assert result["approved_for_training"] is False
    assert result["eval_labels_modified"] == 0


def test_scope_refuses_missing_or_overlapping_ids_and_duplicate_proposals():
    frame, parts = fixture()
    for bad in [{**parts, "test": {"id0"}}, {**parts, "train": {"unknown"}}]:
        with pytest.raises(ValueError):
            train_null_ids(frame, bad)
    with pytest.raises(ValueError, match="outside"):
        assert_proposal_scope(frame, parts, ["id0", "id0"])


def test_open_model_spec_reuses_chunk_contract_and_records_true_model(tmp_path):
    frame, parts = fixture()
    spec = annotation_spec("Synthetic policy instructions", allowed_ids=["id0", "id2", "id3"])
    paths = chunks.export_spec(spec, tmp_path, frame, parts, chunk_size=1)
    tasks = [json.loads(line) for p in paths for line in p.read_text().splitlines()]
    assert tasks == [{"id": "id0", "text": "Invented message 0"}]
    assert budget.read_ledger(tmp_path)[0]["model"] == f"{MODEL}@{REVISION}"
    assert set(tasks[0]) == {"id", "text"}
    with pytest.raises(ValueError, match="positive"):
        chunks.export_spec(spec, tmp_path, frame, parts, chunk_size=0)
