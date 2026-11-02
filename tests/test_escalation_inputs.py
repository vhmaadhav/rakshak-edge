"""A saved v1 result cannot be silently applied to different external IDs or thresholds."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rakshak.fusion.selective import SelectiveThresholds


def test_saved_policy_inputs_exact_ids_and_parent_decisions(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo/'kaggle/jobs'))
    spec = importlib.util.spec_from_file_location('policy_input_fixture',
                                                repo/'kaggle/jobs/evidence_escalation.py')
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    frame = pd.DataFrame({'id': ['b', 'a'], 'text': ['synthetic', 'fixture'],
                          'label_scam': [0, 1], 'label_binary': [0, 1],
                          'campaign_id': ['g1', 'g2']})
    source = tmp_path/'scores.parquet'
    pd.DataFrame({'id': ['a', 'b'], 'student': [.9, .1], 'fused': [.9, .1],
                  'state': ['SCAM', 'SAFE']}).to_parquet(source)
    aligned = job.bound_scores(source, frame)
    assert aligned.fused.tolist() == [.1, .9]
    class URL:
        def score(self, texts):
            return np.full(len(texts), np.nan), np.zeros(len(texts), dtype=bool)
    threshold = SelectiveThresholds(.2, .8)
    report, decisions = job.audit(frame, aligned, threshold, .5, URL(), .9, 20)
    assert decisions.state.tolist() == ['SAFE', 'UNCERTAIN']
    assert report['label_scam']['recall_alert']['rate'] == 1
    assert report['label_scam']['recall_scam']['rate'] == 0
    changed = aligned.copy()
    changed['state'] = 'SAFE'
    with pytest.raises(ValueError, match='baseline decisions'):
        job.audit(frame, changed, threshold, .5, URL(), .9, 20)
    with pytest.raises(ValueError, match='eligible population'):
        job.bound_scores(source, frame.iloc[:1])
    poisoned = pd.read_parquet(source)
    poisoned.loc[0, 'student'] = 1.1
    poisoned.to_parquet(source)
    with pytest.raises(ValueError, match='probabilities'):
        job.bound_scores(source, frame)
