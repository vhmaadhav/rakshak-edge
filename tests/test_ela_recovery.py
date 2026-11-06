"""Recovery must refuse changed weights and unrelated failed runs."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def test_recovery_binds_failure_training_and_weights(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'kaggle/jobs'))
    spec = importlib.util.spec_from_file_location('recovery_fixture',
                                                root / 'kaggle/jobs/ela_resume_eval.py')
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    source = tmp_path / 'source'
    adapter = source / 'Llama-Rakshak-ELA-QLoRA'
    adapter.mkdir(parents=True)
    weights = adapter / 'adapter_model.safetensors'
    weights.write_bytes(b'synthetic-weight-fixture')
    training = source / 'training.json'
    training.write_text('{}')
    monkeypatch.setattr(job, 'ADAPTER_HASH', hashlib.sha256(weights.read_bytes()).hexdigest())
    monkeypatch.setattr(job, 'TRAINING_HASH', hashlib.sha256(training.read_bytes()).hexdigest())
    result = {'tree_sha256': job.SOURCE_TREE, 'status': 'error',
              'error': 'ImportError: Found an incompatible version of torchao. '
                       'Found version 0.10.0, '
                       'but only versions above 0.16.0 are supported'}
    receipt = source / 'results.json'
    receipt.write_text(json.dumps(result))
    assert job.training_source(tmp_path) == source
    weights.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash mismatch'):
        job.training_source(tmp_path)
    result['error'] = 'unrelated failure'
    receipt.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='completion state'):
        job.training_source(tmp_path)
