"""Synthetic full-run check: candidate scores cannot be silently replaced by the control."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from rakshak.eval import data as D


def test_generic_fusion_uses_distinct_expert_scores_with_context_features_disabled(tmp_path,
                                                                                 monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo/'kaggle/jobs'))
    spec = importlib.util.spec_from_file_location('fusion_roles_fixture',
                                                repo/'kaggle/jobs/fusion_transactional.py')
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    output, source = tmp_path/'out', tmp_path/'source'
    output.mkdir()
    source.mkdir()
    monkeypatch.setenv('RAKSHAK_WORK', str(output))
    n = 160
    y = np.arange(n) % 2
    ids = np.asarray([f'fixture{i}' for i in range(n)])
    sms = pd.DataFrame({'id': ids, 'text': ['ordinary synthetic fixture']*n,
                        'label_scam': y, 'label_binary': y, 'campaign_id': np.arange(n)//2})
    runs = [D.Run(name, np.arange(24), np.arange(24, 120), np.arange(120, n),
                  np.arange(120, n), {}) for name in job.REQUIRED_RUNS]
    meta = {'runs': {run.name: {'arms': {arm: {str(seed): {'metrics': {
        'thresholds': {'fpr_1pct': {'threshold': .5}}}} for seed in (1, 2, 3)}
        for arm in ('erm', 'hardneg')}} for run in runs}}
    (source/'metrics.json').write_text(json.dumps(meta))
    for run in runs:
        positions = np.r_[run.cal, run.test]
        for seed in (1, 2, 3):
            for arm in ('erm', 'hardneg'):
                scores = (np.where(y[positions], .9, .1) if arm == 'erm'
                          else np.full(len(positions), .5))
                pd.DataFrame({'id': ids[positions], 'score': scores,
                              'split': ['cal']*len(run.cal)+['test']*len(run.test)}).to_parquet(
                    source/f"scores_{run.name.replace(':', '_')}_{arm}_s{seed}.parquet")
    external_root = tmp_path/'external'
    external_root.mkdir()
    ext = sms.iloc[:8].copy()
    ext['text_available'] = True
    for name in ('groundlm', 'smishx'):
        ext.to_parquet(external_root/f'external_{name}.parquet')
        for seed in (1, 2, 3):
            for arm in ('erm', 'hardneg'):
                pd.DataFrame({'id': ext.id, 'score': np.where(ext.label_scam, .9, .1)}).to_parquet(
                    source/f'external_{name}_{arm}_s{seed}.parquet')
    monkeypatch.setattr(job.D, 'resolve_data_root', lambda: tmp_path)
    monkeypatch.setattr(job.D, 'load_sms', lambda _: sms)
    monkeypatch.setattr(job.D, 'load_runs', lambda *a, **k: runs)
    monkeypatch.setattr(job, 'mark_exact_duplicates', lambda frame: frame)
    monkeypatch.setattr(job, 'assign_campaigns', lambda frame: (frame, {}))
    monkeypatch.setattr(job, 'frozen_url_folder', lambda _: source)
    monkeypatch.setattr(job, 'FrozenURLExpert', lambda _: SimpleNamespace(
        score=lambda texts: (np.full(len(texts), np.nan), np.zeros(len(texts), dtype=bool))))
    observed = {}
    def gate(internal, external, states):
        observed.update(states=states)
        return {'decision': 'unmeasurable'}
    ctx = SimpleNamespace(argv=['--n-boot', '2'], input_root=str(tmp_path), work=output, results={})
    source_info = {'folder': source, 'tree': 'fixture', 'metrics_sha256': 'fixture'}
    assert job.run(ctx, source_info=source_info,
                   model_arms={'baseline': 'erm', 'transactional': 'hardneg'},
                   feature_roles={'baseline': False, 'transactional': False}, gate=gate)
    for block in observed['states'].values():
        for seed in block.values():
            assert seed['baseline']['status'] == 'ok'
            assert seed['transactional']['status'] == 'infeasible'
    report = json.loads((output/'metrics.json').read_text())
    assert report['transactional_feature_roles'] == {'baseline': False, 'transactional': False}
