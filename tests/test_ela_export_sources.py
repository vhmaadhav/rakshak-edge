"""A passing model audit and unchanged selected weights are mandatory for export."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def test_export_refuses_failed_gate_and_changed_adapter(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo / 'kaggle/jobs'))
    spec = importlib.util.spec_from_file_location('conditional_export_fixture',
                                                repo / 'kaggle/jobs/ela_export.py')
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    audit, tuned = tmp_path/'audit', tmp_path/'tuned'
    audit.mkdir()
    tuned.mkdir()
    for folder, tree in ((audit, job.AUDIT_TREE), (tuned, job.TUNED_TREE)):
        (folder/'results.json').write_text(json.dumps({'tree_sha256': tree, 'status': 'ok'}))
    metrics = audit/'metrics.json'
    metrics.write_text(json.dumps({'comparison': {'decision': 'kill'}}))
    monkeypatch.setattr(job, 'AUDIT_HASH', hashlib.sha256(metrics.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='forbidden'):
        job.sources(tmp_path)
    metrics.write_text(json.dumps({'comparison': {'decision': 'go'}}))
    monkeypatch.setattr(job, 'AUDIT_HASH', hashlib.sha256(metrics.read_bytes()).hexdigest())
    adapter = audit/'Llama-Rakshak-ELA-QLoRA'
    adapter.mkdir()
    weights = adapter/'adapter_model.safetensors'
    weights.write_bytes(b'synthetic fixture weights')
    monkeypatch.setattr(job, 'ADAPTER_HASH', hashlib.sha256(weights.read_bytes()).hexdigest())
    generations = tuned/'generations.jsonl'
    generations.write_text('{}\n')
    monkeypatch.setattr(job, 'GENERATIONS_HASH',
                        hashlib.sha256(generations.read_bytes()).hexdigest())
    assert job.sources(tmp_path) == (adapter, tuned)
    weights.write_bytes(b'changed')
    with pytest.raises(ValueError, match='adapter hash'):
        job.sources(tmp_path)
