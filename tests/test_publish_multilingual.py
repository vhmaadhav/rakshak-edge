import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1]/'kaggle/publish_multilingual.py'
    spec = importlib.util.spec_from_file_location('multilingual_publisher_unit', path)
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    return job


def fixture(tmp_path):
    job = module()
    source = tmp_path/'source'
    source.mkdir()
    (source/'inference_source.zip').write_bytes(b'synthetic source archive')
    (source/'pretrained_source.json').write_text('{}')
    tokenizer = source/'tokenizer'
    tokenizer.mkdir()
    for name in ('tokenizer.json', 'tokenizer_config.json', 'SOURCE_README.md'):
        (tokenizer/name).write_text('synthetic source fixture')
    runs = {}
    for fold in job.FOLDS:
        stem = fold.replace(':', '_')
        vocab = f'vocabulary_{stem}.npz'
        (source/vocab).write_bytes(b'synthetic vocabulary fixture')
        info = {'fold': fold, 'tree_sha256': job.TREE, 'rows': 2, 'train_ids': ['a', 'b'],
                'source_revision': job.REVISION, 'vocabulary_sha256': job.digest(source/vocab),
                'inference_source_sha256': job.digest(source/'inference_source.zip')}
        (source/f'vocabulary_{stem}.json').write_text(json.dumps(info))
        block = {'vocabulary': {'file': vocab, 'sha256': job.digest(source/vocab)}, 'arms': {}}
        for arm in job.ARMS:
            block['arms'][arm] = {}
            for seed in ('1', '2', '3'):
                name = f'student_{stem}_{arm}_s{seed}.pt'
                (source/name).write_bytes(b'synthetic checkpoint fixture')
                block['arms'][arm][seed] = {
                    'checkpoint': {'file': name, 'sha256': job.digest(source/name), 'bytes': 32},
                    'training': {'rows': 2, 'teacher_rows': 2 if arm == 'multilingual_kd' else 0}}
        runs[fold] = block
    metrics = {'runs': runs, 'comparisons': dict.fromkeys(job.FOLDS, {}),
               'config': {'arms': sorted(job.ARMS), 'seeds': [1, 2, 3], 'n_boot': 1000,
                          'pretrained': {'licence': 'MIT', 'revision': job.REVISION},
                          'inference_source_sha256': job.digest(source/'inference_source.zip')},
               'multilingual_gate': {'missing': [], 'decision': 'go', 'checks': {'pass': True},
                                     'candidate': 'multilingual_kd', 'ablation_promotable': False}}
    (source/'metrics.json').write_text(json.dumps(metrics))
    (source/'results.json').write_text(json.dumps({'status': 'ok', 'job': 'sms_multilingual',
                                                   'tree_sha256': job.TREE, 'git_sha': 'fixture'}))
    (source/'raw_sms.parquet').write_text('never upload synthetic row fixture')
    log = tmp_path/'push.log'
    log.write_text('Kernel version 1 successfully pushed')
    return job, source, log


def test_private_multilingual_archive_keeps_all_arms_and_excludes_rows(tmp_path):
    job, source, log = fixture(tmp_path)
    target = tmp_path/'archive'
    manifest = job.stage(source, target, log)
    assert len(manifest['models']) == 54 and manifest['shipment_approved'] is False
    assert {item['arm'] for item in manifest['models']} == job.ARMS
    assert json.loads((target/'dataset-metadata.json').read_text())['isPrivate'] is True
    assert not list(target.rglob('*.parquet'))
    assert (target/'tokenizer/SOURCE_README.md').is_file()
    assert 'excludes tokenizer' in manifest['size_scope']


def test_multilingual_archive_refuses_teacher_scope_or_weight_tampering(tmp_path):
    job, source, log = fixture(tmp_path)
    file = source/'student_campaign_multilingual_kd_s1.pt'
    file.write_bytes(b'changed synthetic checkpoint')
    with pytest.raises(ValueError, match='hash or path'):
        job.stage(source, tmp_path/'bad', log)
    assert not (tmp_path/'bad').exists()
