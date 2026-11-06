"""Inventory writes only a blinded review sample and leaves labels untouched."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from rakshak.eval.clean_kd import FOLDS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle" / "jobs"))
import policy_inventory as job  # noqa: E402


def test_private_audit_is_blinded_and_every_fold_has_original_train_binding(tmp_path, monkeypatch):
    root, work = tmp_path / "input", tmp_path / "output"
    root.mkdir()
    work.mkdir()
    frame = pd.DataFrame({"id": [f"id{i}" for i in range(202)],
                          "text": [f"Invented message {i}" for i in range(202)],
                          "source": "training", "dup_of": None, "label_policy": None,
                          "label_binary": 0, "lang": "en", "label_fine": "ham",
                          "campaign_id": [f"c{i}" for i in range(202)]})
    frame.to_parquet(root / "sms.parquet", index=False)
    runs = [SimpleNamespace(name=name, train=np.arange(200), cal=np.array([200]),
                            test_all=np.array([201]), notes={"missing_ids": {"train": 0}})
            for name in sorted(FOLDS)]
    monkeypatch.setattr(job.D, "resolve_data_root", lambda: root)
    monkeypatch.setattr(job.D, "load_runs", lambda *args, **kwargs: runs)
    context = SimpleNamespace(work=work, results={})
    monkeypatch.setattr(job, "write_metrics",
                        lambda name, obj: (work / name).write_text(json.dumps(obj)))
    assert job.run(context)
    audit = pd.read_parquet(work / "policy_human_audit_200.parquet")
    assert len(audit) == 200
    assert set(audit) == {"id", "text", "human_label", "human_reviewed"}
    assert audit.human_label.isna().all() and not audit.human_reviewed.any()
    assert not set(audit.id) & {"id200", "id201"}
    plan = json.loads((work / "policy_annotation_scope.json").read_text())
    assert set(plan["folds"]) == FOLDS
    assert not plan["approved_for_training"]
    assert plan["eval_labels_modified"] == 0
    pd.testing.assert_frame_equal(pd.read_parquet(root / "sms.parquet"), frame)
