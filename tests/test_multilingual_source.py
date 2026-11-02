import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')


def test_pinned_bin_loader_is_weights_only_and_rejects_changed_weight(tmp_path, monkeypatch):
    import huggingface_hub
    from transformers import AutoTokenizer

    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo/'kaggle/jobs'))
    spec = importlib.util.spec_from_file_location('multilingual_source_job',
                                                 repo/'kaggle/jobs/sms_multilingual.py')
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    folder, work = tmp_path/'pinned', tmp_path/'output'
    folder.mkdir()
    work.mkdir()
    (folder/'README.md').write_text('license: mit\n')
    (folder/'config.json').write_text('{}')
    weight = folder/'pytorch_model.bin'
    tensor = torch.arange(80*70).reshape(80, 70).float()
    torch.save({'model.embeddings.tok_embeddings.weight': tensor}, weight)
    monkeypatch.setattr(job, 'CARD_SHA256', job.sha(folder/'README.md'))
    monkeypatch.setattr(job, 'CONFIG_SHA256', job.sha(folder/'config.json'))
    monkeypatch.setattr(job, 'WEIGHT_SHA256', job.sha(weight))
    monkeypatch.setattr(job, 'EMBEDDING_SHAPE', (80, 70))

    def snapshot(model, *, revision, allow_patterns):
        assert model == job.MODEL and revision == job.REVISION
        assert 'pytorch_model.bin' in allow_patterns
        return str(folder)

    def save_tokenizer(destination):
        destination.mkdir()
        (destination/'tokenizer.json').write_text('{}')

    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot)
    tokenizer = SimpleNamespace(save_pretrained=save_tokenizer)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda path, **kw: tokenizer)
    original_load, calls = torch.load, []

    def checked_load(path, **kwargs):
        assert kwargs == {'map_location': 'cpu', 'weights_only': True}
        calls.append(path)
        return original_load(path, **kwargs)

    monkeypatch.setattr(torch, 'load', checked_load)
    actual_tokenizer, rows, source = job.pretrained(work)
    assert actual_tokenizer is tokenizer and len(calls) == 1
    np.testing.assert_array_equal(rows, tensor.numpy())
    assert source['licence'] == 'MIT' and source['transformer_layers_deployed'] == 0
    assert source['files']['pytorch_model.bin'] == job.WEIGHT_SHA256
    weight.write_bytes(b'changed-weight')
    with pytest.raises(ValueError, match='weight hash changed'):
        job.pretrained(work)
    assert len(calls) == 1


def test_source_archive_accepts_old_timestamps_and_excludes_artifacts(tmp_path, monkeypatch):
    import os
    import zipfile

    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo/'kaggle/jobs'))
    spec = importlib.util.spec_from_file_location('multilingual_archive_job',
                                                 repo/'kaggle/jobs/sms_multilingual.py')
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    source = tmp_path/'src'
    source.mkdir()
    module = source/'fixture.py'
    module.write_text('constant = 1\n')
    job_file = tmp_path/'job.py'
    job_file.write_text('def run(ctx): pass\n')
    os.utime(module, (0, 0))
    os.utime(job_file, (0, 0))
    (source/'weights.pt').write_bytes(b'never-archive')
    first, second = tmp_path/'first.zip', tmp_path/'second.zip'
    job.write_source_archive(source, job_file, first)
    job.write_source_archive(source, job_file, second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert set(archive.namelist()) == {'fixture.py', 'sms_multilingual.py'}
        assert archive.read('fixture.py') == module.read_bytes()
        assert all(item.date_time[0] == 1980 for item in archive.infolist())
    (source/'linked.py').symlink_to(module)
    with pytest.raises(ValueError, match='symlink refused'):
        job.write_source_archive(source, job_file, tmp_path/'unsafe.zip')
