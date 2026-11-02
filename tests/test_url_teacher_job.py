import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')


def job_module(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo/'kaggle/jobs'))
    spec = importlib.util.spec_from_file_location('url_teacher_job_unit',
                                                 repo/'kaggle/jobs/url_teacher_kd.py')
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    return job


def test_synthetic_teacher_training_scores_cal_only_and_averages_partial_update(
        tmp_path, monkeypatch):
    from rakshak.models import url_byt5

    job = job_module(monkeypatch)
    if not torch.cuda.is_available():
        pytest.skip('synthetic CUDA trainer test runs on Kaggle')
    ds = SimpleNamespace(fit=np.arange(18), cal=np.asarray([18, 19]),
                         labels=np.asarray([0, 1]*11),
                         side=np.asarray([[i/30, (i % 2)*.3] for i in range(22)],
                                         dtype=np.float32))
    ids = np.zeros((22, 2), dtype=int)
    initial = torch.nn.Linear(2, 1)

    class TinyTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 1)
            self.layer.load_state_dict(initial.state_dict())

        def forward(self, ids, side):
            return self.layer(side).squeeze(1)

    seen = []

    def cal_only(model, ids, side, rows):
        np.testing.assert_array_equal(rows, ds.cal)
        seen.append(rows.copy())
        return np.asarray([-.5, .5])

    monkeypatch.setattr(url_byt5, 'predict', cal_only)
    actual, history = job.train_teacher(TinyTeacher(), ds, ids, time.monotonic()+60,
                                        lambda message: None)
    assert len(seen) == 4 and len(history['history']) == 4
    assert all(entry['optimizer_updates'] == 1 for entry in history['history'])
    reference = TinyTeacher().cuda()
    optimizer = torch.optim.AdamW(reference.parameters(), lr=5e-5, weight_decay=.01)
    with torch.autocast('cuda', dtype=torch.float16):
        values = reference(torch.tensor(ids[ds.fit], device='cuda'),
                           torch.tensor(ds.side[ds.fit], device='cuda'))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        values.float(), torch.tensor(ds.labels[ds.fit], dtype=torch.float32, device='cuda'))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
    optimizer.step()
    for key, value in actual.state_dict().items():
        # AMP matrix batch sizes can change rounding; this verifies a whole final partial update.
        torch.testing.assert_close(value, reference.cpu().state_dict()[key], rtol=1e-5, atol=1e-6)
    path = tmp_path/'teacher.pt'
    entry = job.save_model(actual, path, {'no_test_predictions': True})
    assert len(entry['sha256']) == 64 and entry['bytes'] > 0
    reloaded = torch.load(path, weights_only=True)
    for key, value in actual.state_dict().items():
        torch.testing.assert_close(value, reloaded['state_dict'][key], rtol=0, atol=0)


def test_real_teacher_dependencies_are_bundled_and_target_creation_follows_gate(monkeypatch):
    job = job_module(monkeypatch)
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('url_teacher_builder_unit',
                                                 repo/'kaggle/build_kernel.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    bundle = set(builder.bundle_names('url_teacher_kd'))
    assert {'kaggle/jobs/url_controls_cal.py', 'kaggle/jobs/url_student.py',
            'kaggle/jobs/url_teacher_kd.py'} <= bundle
    source = Path(job.__file__).read_text()
    assert source.index("'eligibility.json').open('x')") < source.index(
        "if gate['decision'] == 'go':")
    guarded = source.split("if gate['decision'] == 'go':", 1)[1]
    assert 'targets = predict(model, teacher_ids, ds.side, ds.fit)' in guarded
    assert 'ds.test' not in source and 'evaluate_scores(' not in source
