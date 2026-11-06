"""A frozen quality decision must bind exactly one successful source output."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def test_saved_output_selector_refuses_partial_or_ambiguous_source_trees(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "kaggle/jobs"))
    spec = importlib.util.spec_from_file_location("audit_input_fixture",
                                                root / "kaggle/jobs/ela_paired_audit.py")
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    source = tmp_path / "source"
    source.mkdir()
    path = source / "generations.jsonl"
    path.write_text(json.dumps({"sample_id": "synthetic-fixture"}) + "\n")
    receipt = source / "results.json"
    receipt.write_text(json.dumps({"tree_sha256": "fixture-tree", "status": "ok"}))
    rows, digest, folder = job.completed_rows(tmp_path, "fixture-tree")
    assert rows == [{"sample_id": "synthetic-fixture"}]
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest() and folder == source
    receipt.write_text(json.dumps({"tree_sha256": "fixture-tree", "status": "partial"}))
    with pytest.raises(ValueError, match="complete and successful"):
        job.completed_rows(tmp_path, "fixture-tree")
    receipt.write_text(json.dumps({"tree_sha256": "fixture-tree", "status": "ok"}))
    second = tmp_path / "duplicate"
    second.mkdir()
    (second / "generations.jsonl").write_bytes(path.read_bytes())
    (second / "results.json").write_bytes(receipt.read_bytes())
    with pytest.raises(ValueError, match="expected one completed source"):
        job.completed_rows(tmp_path, "fixture-tree")
