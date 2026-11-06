"""Saved diagnostic scores cannot silently drop or duplicate eligible messages."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def test_saved_score_id_binding_and_aggregate_only_training_profile(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'kaggle/jobs'))
    spec = importlib.util.spec_from_file_location('diagnostic_fixture',
                                                root / 'kaggle/jobs/sms_transactional_audit.py')
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    frame = pd.DataFrame({'id': ['b', 'a'], 'text': ['OTP fixture', 'delivery fixture'],
                          'source': ['synthetic-fixture', 'synthetic-fixture'],
                          'label_scam': [0, None]})
    path = tmp_path / 'scores.parquet'
    pd.DataFrame({'id': ['a', 'b'], 'score': [.2, .8]}).to_parquet(path)
    assert job.aligned_scores(path, frame).tolist() == [.8, .2]
    pd.DataFrame({'id': ['a', 'a'], 'score': [.2, .8]}).to_parquet(path)
    with pytest.raises(ValueError, match='exactly'):
        job.aligned_scores(path, frame)
    result = job.training_profile(frame)
    assert result['fraud_negative_n'] == 1 and result['unknown_policy_n'] == 1
    assert result['category_counts']['otp'] == 1 and 'text' not in result
