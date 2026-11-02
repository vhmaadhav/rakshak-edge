"""No policy outcome file may be read until every fold's cal-only URL threshold freezes."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from rakshak.eval import data as D
from rakshak.fusion.escalation_audit import FOLDS


def test_policy_freezes_all_cal_thresholds_before_outcome_files(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo/'kaggle/jobs'))
    spec = importlib.util.spec_from_file_location('policy_order_fixture',
                                                repo/'kaggle/jobs/evidence_escalation.py')
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    sms = pd.DataFrame({'id': [f'id{i}' for i in range(24)],
                        'text': [f"{'positive' if i%2 else 'ordinary'} https://example.com"
                                 for i in range(24)],
                        'label_scam': [0, 1]*12, 'label_binary': [0, 1]*12,
                        'campaign_id': [str(i//2) for i in range(24)]})
    runs = [D.Run(name, np.arange(8), np.arange(8, 16), np.arange(16, 24),
                  np.arange(16, 24), {}) for name in FOLDS]
    state = {'fusion': {'thresholds': {'t_safe': .2, 't_scam': .8}},
             'student_cal_1pct_threshold': .5}
    models = {name: (tmp_path/name.replace(':', '_'), state, '2'*64) for name in FOLDS}
    monkeypatch.setattr(job, 'v1_inputs', lambda *a: models)
    monkeypatch.setattr(job.D, 'resolve_data_root', lambda: tmp_path)
    monkeypatch.setattr(job.D, 'load_sms', lambda _: sms)
    monkeypatch.setattr(job.D, 'load_runs', lambda *a, **k: runs)
    cal_calls = []
    class URL:
        def score(self, texts):
            if not (tmp_path/'frozen_candidate_policy.json').exists():
                cal_calls.append(texts)
            return np.asarray([.9 if text.startswith('positive') else .1 for text in texts]), \
                np.ones(len(texts), dtype=bool)
    monkeypatch.setattr(job, 'FrozenURLExpert', lambda _: URL())
    monkeypatch.setattr(job, 'frozen_url_folder', lambda _: tmp_path)
    monkeypatch.setattr(job, 'mark_exact_duplicates', lambda frame: frame)
    monkeypatch.setattr(job, 'assign_campaigns', lambda frame: (frame, {}))
    def parquet(path):
        frozen = json.loads((tmp_path/'frozen_candidate_policy.json').read_text())
        assert set(frozen['url_cal_thresholds']) == FOLDS
        assert len(cal_calls) == 6
        path = Path(path)
        if path.name.startswith('external_') and path.name.endswith('_v1.parquet'):
            frame = sms.iloc[:8]
        elif path.name == 'test_v1.parquet':
            frame = sms.iloc[16:24]
        else:
            return sms.iloc[:8].assign(text_available=True)
        values = np.where(frame.label_scam.to_numpy(), .9, .1)
        return pd.DataFrame({'id': frame.id.to_numpy(), 'student': values, 'fused': values,
                             'state': np.where(values >= .8, 'SCAM', 'SAFE')})
    monkeypatch.setattr(job.pd, 'read_parquet', parquet)
    monkeypatch.setenv('RAKSHAK_WORK', str(tmp_path))
    ctx = SimpleNamespace(argv=['--v1-tree', '1'*64, '--v1-manifest-sha256', '2'*64],
                          src=repo/'src', work=tmp_path, input_root=tmp_path,
                          results={'git_sha': '3'*40, 'tree_sha256': '4'*64})
    assert job.run(ctx)
    result = json.loads((tmp_path/'metrics.json').read_text())
    assert set(result['internal']) == FOLDS
    assert result['decision']['decision'] == 'go'  # synthetic fixture only, no benchmark claim
