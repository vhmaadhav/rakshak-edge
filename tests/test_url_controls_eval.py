import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')


def modules(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo/'kaggle/jobs'))
    from test_url_controls_cal import dataset
    from test_url_controls_cal import modules as control_modules

    control, base = control_modules(monkeypatch)
    spec = importlib.util.spec_from_file_location('url_controls_eval_unit',
                                                 repo/'kaggle/jobs/url_controls_eval.py')
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    return repo, control, base, evaluation, dataset()


def test_restore_original_control_checks_cal_only_and_rejects_tampered_checkpoint(
        tmp_path, monkeypatch):
    _, control, base, evaluation, ds = modules(monkeypatch)
    if not torch.cuda.is_available():
        pytest.skip('synthetic checkpoint CUDA recovery runs on Kaggle')
    args = base.parse_args(['--models', 'charcnn'])
    monkeypatch.setattr(base, 'make_model', lambda *a: torch.nn.Linear(2, 1))
    monkeypatch.setattr(base, 'train_model', lambda *a: {'best_epoch': 0, 'best_cal_pr_auc': .8})
    seen = []

    def predict(model, kind, ids, side, rows, device, use_side):
        np.testing.assert_array_equal(rows, ds.cal)
        seen.append(rows.copy())
        return np.asarray([-.2, .3])

    monkeypatch.setattr(base, 'predict_logits', predict)
    from types import SimpleNamespace
    ctx = SimpleNamespace(work=tmp_path, log=lambda text: None,
                          results={'git_sha': 'synthetic', 'tree_sha256': 'synthetic-tree'})
    entry = control.train_control(ctx, ds, 1, args)
    restored, cal, record = evaluation.restore_cal(ds, 1, entry, tmp_path, args)
    assert len(seen) == 2 and record['cal_max_abs_logit_error'] == 0
    assert set(record['thresholds']) == {'max_f1', 'cal_fpr_1pct'}
    np.testing.assert_allclose(cal, base.sigmoid(np.asarray([-.2, .3])))
    del restored
    path = tmp_path/entry['checkpoint']['file']
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='checkpoint hash'):
        evaluation.restore_cal(ds, 1, entry, tmp_path, args)
    assert len(seen) == 2


def test_control_eval_actual_bundle_includes_all_recovery_helpers(monkeypatch):
    repo, _, _, evaluation, _ = modules(monkeypatch)
    spec = importlib.util.spec_from_file_location('url_controls_eval_builder_unit',
                                                 repo/'kaggle/build_kernel.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    files = set(builder.bundle_names('url_controls_eval'))
    assert {'kaggle/jobs/url_controls_cal.py', 'kaggle/jobs/url_student.py',
            'kaggle/jobs/url_teacher_kd.py', 'kaggle/jobs/url_controls_eval.py'} <= files
    source = Path(evaluation.__file__).read_text()
    freeze = source.index("'evaluation_manifest.json').open('x')")
    assert freeze < source.index('logits = base.predict_logits', source.index('def run(ctx):'))
    assert 'train_model(' not in source and 'load_pretrained(' not in source
