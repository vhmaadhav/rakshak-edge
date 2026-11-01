"""Synthetic allowlist fixture: changed weights fail, unlisted row files stay local."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def publisher():
    path = Path(__file__).resolve().parents[1]/'kaggle/publish_clean_kd.py'
    spec = importlib.util.spec_from_file_location('clean_kd_publish', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path, module):
    source = tmp_path/'source'
    source.mkdir()
    (source/'results.json').write_text(json.dumps({
        'job': 'sms_clean_kd', 'status': 'ok', 'git_sha': 'a'*40, 'tree_sha256': 'b'*64}))
    metrics = {'skipped': {}, 'runs': {}, 'config': {
        'args': {'epochs': 15, 'n_boot': 1000, 'seeds': [1, 2, 3], 'teacher_seed': 0},
        'teacher_arms': {'erm': False, 'kd': True}},
        'clean_kd_gate': {'decision': 'kill', 'missing': []},
        'teacher_target_scope': 'synthetic test: single teacher, not OOF'}
    for fold in module.FOLDS:
        metrics['runs'][fold] = {'arms': {}}
        for arm in ('erm', 'kd'):
            seeds = {}
            for seed in ('1', '2', '3'):
                filename = f'student_{fold.replace(":", "_")}_{arm}_s{seed}.pt'
                (source/filename).write_bytes(b'synthetic model fixture')
                seeds[seed] = {'checkpoint': {
                    'file': filename, 'sha256': hashlib.sha256((source/filename).read_bytes())
                    .hexdigest()}, 'teacher_enabled': arm == 'kd', 'training': {
                        'rows': {'with_teacher': 10 if arm == 'kd' else 0, 'rows': 10}}}
            metrics['runs'][fold]['arms'][arm] = seeds
    (source/'metrics.json').write_text(json.dumps(metrics))
    (source/'unlisted_scores.parquet').write_bytes(b'synthetic row data must stay local')
    push = tmp_path/'push.log'
    push.write_text('Kernel version 1 successfully pushed.\n')
    return source, push


def test_archive_is_private_allowlist_with_recorded_negative_gate(tmp_path):
    module = publisher()
    source, push = fixture(tmp_path, module)
    target = tmp_path/'stage'
    result = module.stage(source, target, push)
    assert len(result['models']) == 36
    assert result['experiment_gate']['decision'] == 'kill'
    assert result['shipment_approved'] is False
    assert not (target/'unlisted_scores.parquet').exists()
    assert json.loads((target/'dataset-metadata.json').read_text())['isPrivate'] is True
    with pytest.raises(FileExistsError):
        module.stage(source, target, push)


def test_changed_checkpoint_is_rejected_before_staging(tmp_path):
    module = publisher()
    source, push = fixture(tmp_path, module)
    next(source.glob('*.pt')).write_bytes(b'changed synthetic fixture')
    with pytest.raises(ValueError, match='checkpoint hash'):
        module.stage(source, tmp_path/'stage', push)
    assert not (tmp_path/'stage').exists()
