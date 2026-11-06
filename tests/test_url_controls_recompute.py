import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def module(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo/'kaggle/jobs'))
    spec = importlib.util.spec_from_file_location('url_recovery_unit',
                                                 repo/'kaggle/jobs/url_controls_recompute.py')
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    return job


def test_exact_saved_source_and_complete_ordered_cache_guards(tmp_path, monkeypatch):
    job = module(monkeypatch)
    frozen = {'tree_sha256': job.PARENT_TREE, 'parent_sha256': job.CONTROL_SHA}
    manifest = tmp_path/'evaluation_manifest.json'
    manifest.write_text(json.dumps(frozen))
    (tmp_path/'results.json').write_text(json.dumps({'job': 'url_controls_eval',
        'tree_sha256': job.PARENT_TREE, 'status': 'error',
        'error': "FileNotFoundError: no split files match ['url_temporal'] under synthetic"}))
    file = tmp_path/'synthetic_scores.parquet'
    pd.DataFrame({'id': ['a', 'b'], 'split': ['test']*2, 'logit': [-100., 100.]}).to_parquet(file)
    monkeypatch.setattr(job, 'SCORE_HASHES',
                        {manifest.name: job.sha(manifest), file.name: job.sha(file)})
    assert job.source(tmp_path) == (tmp_path, frozen)
    np.testing.assert_array_equal(job.cached_scores(file, ['a', 'b'], 'test'),
                                   1/(1+np.exp(-np.asarray([-60., 60.]))))
    with pytest.raises(ValueError, match='ordered original'):
        job.cached_scores(file, ['b', 'a'], 'test')
    file.write_bytes(b'changed synthetic cache')
    with pytest.raises(ValueError, match='immutable saved'):
        job.source(tmp_path)


def test_recovery_bundle_has_helpers_without_forward_or_training(monkeypatch):
    job = module(monkeypatch)
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('url_recompute_builder_unit',
                                                 repo/'kaggle/build_kernel.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert {'kaggle/jobs/url_controls_recompute.py', 'kaggle/jobs/url_controls_eval.py',
            'kaggle/jobs/url_controls_cal.py', 'kaggle/jobs/url_student.py',
            'kaggle/jobs/url_teacher_kd.py'} <= set(builder.bundle_names('url_controls_recompute'))
    text = Path(job.__file__).read_text()
    assert all(call not in text for call in ('predict_logits(', 'make_model(', 'train_model('))


def test_saved_manifest_adapter_keeps_three_thresholded_control_reports(monkeypatch):
    from types import SimpleNamespace

    from rakshak.eval import protocol

    job = module(monkeypatch)
    actual = protocol.evaluate_scores
    monkeypatch.setattr(protocol, 'evaluate_scores',
                        lambda *a, **kw: actual(*a, **{**kw, 'bootstrap': False}))
    ds = SimpleNamespace(name='synthetic', labels=np.asarray([0, 1]*4),
                         cal=np.arange(4), test=np.arange(4, 8),
                         groups={}, clusters=['a', 'b', 'c', 'd'], protocol=None)
    thresholds = {'max_f1': .5, 'cal_fpr_1pct': .9}
    manifest = {'controls': {'synthetic': {str(seed): {'thresholds': thresholds}
                                         for seed in (1, 2, 3)}}, 'unrelated': 'metadata'}
    scores = {f's{seed}': np.asarray([.1, .8, .6, .95]) for seed in (1, 2, 3)}
    cal = {seed: np.asarray([.1, .8, .2, .95]) for seed in (1, 2, 3)}
    result = job.recover_report(ds, scores, cal, manifest)
    assert set(result['seeds']) == {'1', '2', '3'}
    for row in result['seeds'].values():
        assert row['thresholds'] == thresholds
        assert row['fixed_operating_points']['max_f1']['tp'] == 2
        assert row['fixed_operating_points']['max_f1']['fp'] == 1
        assert row['fixed_operating_points']['cal_fpr_1pct']['tp'] == 1
        assert row['fixed_operating_points']['cal_fpr_1pct']['fp'] == 0


def test_missing_temporal_file_is_unmeasurable_without_inventing_a_split(tmp_path, monkeypatch):
    module(monkeypatch)
    from url_controls_eval import temporal_status

    result = temporal_status(tmp_path)
    assert result == {'status': 'UNMEASURABLE: url_temporal file absent from frozen inputs',
                      'n': None, 'positives': None}
    assert not list(tmp_path.rglob('*.json'))
