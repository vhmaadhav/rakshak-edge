import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')


def modules(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo/'kaggle/jobs'))
    import url_student

    spec = importlib.util.spec_from_file_location('url_controls_job',
                                                 repo/'kaggle/jobs/url_controls_cal.py')
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    return job, url_student


def dataset():
    return SimpleNamespace(name='synthetic', fit=np.asarray([0, 1]), cal=np.asarray([2, 3]),
                           test=np.asarray([4, 5]), row_ids=np.asarray(list('abcdef')),
                           ids=np.zeros((6, 8), dtype=int), labels=np.asarray([0, 1]*3),
                           side=np.zeros((6, 2), dtype=np.float32), side_cols=['a', 'b'],
                           side_stats=(np.zeros(2), np.ones(2)))


def test_control_freezes_cal_only_and_initialization_is_seeded(tmp_path, monkeypatch):
    job, base = modules(monkeypatch)
    ds = dataset()
    args = base.parse_args(['--models', 'charcnn'])
    states, accesses = [], []

    def make(spec, args, n_side):
        model = torch.nn.Linear(2, 1)
        states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
        return model

    def fit(model, kind, actual, cfg, device, side, deadline, teacher, log):
        assert actual is ds and teacher is None and deadline is None
        assert cfg.seed == 2 and cfg.epochs == 8 and cfg.batch_size == 512
        return {'best_epoch': 0, 'best_cal_pr_auc': .8}

    def score(model, kind, ids, side, rows, device, use_side):
        np.testing.assert_array_equal(rows, ds.cal)
        accesses.append(rows.copy())
        return np.asarray([-.2, .3])

    monkeypatch.setattr(base, 'make_model', make)
    monkeypatch.setattr(base, 'train_model', fit)
    monkeypatch.setattr(base, 'predict_logits', score)
    for index in (1, 2):
        work = tmp_path/str(index)
        work.mkdir()
        torch.manual_seed(80+index)
        torch.randn(31*index)
        ctx = SimpleNamespace(work=work, log=lambda text: None,
                              results={'git_sha': 'fixture', 'tree_sha256': 'fixture-tree'})
        info = job.train_control(ctx, ds, 2, args)
        assert info['metadata']['teacher_enabled'] is False
        cached = pd.read_parquet(work/info['cal_logits']['file'])
        assert cached.id.tolist() == ['c', 'd'] and set(cached['split']) == {'cal'}
        blob = torch.load(work/info['checkpoint']['file'], weights_only=True)
        for key, value in states[-1].items():
            torch.testing.assert_close(value, blob['state_dict'][key], rtol=0, atol=0)
    assert len(accesses) == 2
    for key, value in states[0].items():
        torch.testing.assert_close(value, states[1][key], rtol=0, atol=0)


def test_control_rejects_test_id_overlap_before_model_construction(tmp_path, monkeypatch):
    job, base = modules(monkeypatch)
    ds = dataset()
    ds.row_ids[4] = ds.row_ids[0]
    monkeypatch.setattr(base, 'make_model', lambda *a: pytest.fail('model must not be built'))
    ctx = SimpleNamespace(work=tmp_path)
    with pytest.raises(ValueError, match='IDs must remain disjoint'):
        job.train_control(ctx, ds, 1, base.parse_args(['--models', 'charcnn']))
