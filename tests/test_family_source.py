"""Hash-bound clean cohort and cal-only file filtering use synthetic fixtures only."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def job_module(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo/'kaggle/jobs'))
    spec = importlib.util.spec_from_file_location('family_source_fixture',
                                                repo/'kaggle/jobs/family_calibration.py')
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    return job


def test_clean_cohort_rejects_changed_checkpoint(tmp_path, monkeypatch):
    job = job_module(monkeypatch)
    metrics = {'skipped': {}, 'config': {
        'teacher_arms': {'erm': False, 'kd': True},
        'args': {'epochs': 15, 'n_boot': 1000, 'seeds': [1, 2, 3], 'teacher_seed': 0}}, 'runs': {}}
    for fold in job.FOLDS:
        metrics['runs'][fold] = {'arms': {'erm': {}}}
        for seed in ('1', '2', '3'):
            name = f'student_{fold.replace(":", "_")}_erm_s{seed}.pt'
            (tmp_path/name).write_bytes(b'synthetic checkpoint fixture')
            metrics['runs'][fold]['arms']['erm'][seed] = {'checkpoint': {
                'file': name, 'sha256': hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()},
                'teacher_enabled': False, 'training': {'rows': {'with_teacher': 0, 'unlabeled': 0}}}
    path = tmp_path/'metrics.json'
    path.write_text(json.dumps(metrics))
    monkeypatch.setattr(job, 'SOURCE_METRICS', job.sha256(path))
    (tmp_path/'results.json').write_text(json.dumps({
        'job': 'sms_clean_kd', 'tree_sha256': job.SOURCE_TREE, 'status': 'ok'}))
    assert job.clean_source(tmp_path)[1] == metrics
    next(tmp_path.glob('*.pt')).write_bytes(b'changed fixture')
    with pytest.raises(ValueError, match='checkpoint hash'):
        job.clean_source(tmp_path)


def test_cached_score_read_filters_out_every_test_row(tmp_path, monkeypatch):
    job = job_module(monkeypatch)
    frame = pd.DataFrame({'id': ['cal1', 'test-only', 'cal0'],
                          'split': ['cal', 'test', 'cal'], 'score': [.4, np.nan, .2]})
    path = tmp_path/'scores.parquet'
    frame.to_parquet(path)
    read = pd.read_parquet
    def filtered(path, **kwargs):
        assert kwargs['filters'] == [('split', '==', 'cal')]
        selected = read(path, **kwargs)
        assert 'test-only' not in set(selected.id)
        return selected
    monkeypatch.setattr(job.pd, 'read_parquet', filtered)
    assert job.aligned_cal(path, ['cal0', 'cal1']).tolist() == [.2, .4]
    with pytest.raises(ValueError, match='cached cal IDs'):
        job.aligned_cal(path, ['cal0', 'test-only'])


def test_family_bundle_includes_its_actual_shared_matrix_code():
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('family_builder', repo/'kaggle/build_kernel.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert 'kaggle/jobs/fusion_v1.py' in builder.bundle_names('family_calibration')
