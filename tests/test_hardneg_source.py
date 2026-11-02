"""A partial or differently generated training run cannot enter the fused comparison."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from rakshak.fusion.hardneg_gate import FOLDS
from rakshak.models.transactional_negatives import fingerprint


def test_hardneg_source_binds_training_protocol_and_receipt(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('hardneg_source_fixture',
                                                repo/'kaggle/jobs/fusion_hardneg.py')
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    tree = '2'*64
    receipt = {'status': 'ok', 'tree_sha256': tree, 'failures': []}
    audit = {'original_rows': 100, 'original_binary_negative_n': 80,
             'synthetic_rows': 20, 'teacher_synthetic_rows': 0,
             'generator_seed': 20261101, 'template_sha256': fingerprint()}
    arms = {'erm': {str(seed): {'training': {'rows': {'rows': 100}}} for seed in (1, 2, 3)},
            'hardneg': {str(seed): {'training_augmentation': audit,
                        'training': {'rows': {'rows': 120, 'with_teacher': 100}}}
                       for seed in (1, 2, 3)}}
    meta = {'config': {'args': {'seeds': [1, 2, 3], 'epochs': 15, 'teacher_seed': 0,
                                'n_boot': 1000}, 'arms': ['erm', 'hardneg']},
            'runs': {fold: {'arms': copy.deepcopy(arms)} for fold in FOLDS}}
    results = tmp_path/'results.json'
    results.write_text(json.dumps(receipt))
    metrics = tmp_path/'metrics.json'
    metrics.write_text(json.dumps(meta))
    digest = hashlib.sha256(metrics.read_bytes()).hexdigest()
    assert job.validated_source(tmp_path, tree, digest)['folder'] == tmp_path
    with pytest.raises(ValueError, match='metrics hash'):
        job.validated_source(tmp_path, tree, '0'*64)
    receipt['status'] = 'partial'
    results.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='successful'):
        job.validated_source(tmp_path, tree, digest)
    receipt['status'] = 'ok'
    results.write_text(json.dumps(receipt))
    for fold in FOLDS:
        meta['runs'][fold]['arms']['hardneg']['1']['training_augmentation'][
            'teacher_synthetic_rows'] = 1
    metrics.write_text(json.dumps(meta))
    new_digest = hashlib.sha256(metrics.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='provenance'):
        job.validated_source(tmp_path, tree, new_digest)


def test_v1_reference_is_bound_to_experts_and_reporting_receipt(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('hardneg_v1_fixture',
                                                repo/'kaggle/jobs/fusion_hardneg.py')
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    weights = tmp_path/'fixture.onnx'
    weights.write_bytes(b'synthetic weight file; never inference')
    manifest = {'schema': 'rakshak-fusion-v1', 'fitting': {'tree_sha256': '2'*64},
                'artifacts': {'student': {'file': weights.name,
                            'sha256': hashlib.sha256(weights.read_bytes()).hexdigest()}}}
    path = tmp_path/'fusion_v1.json'
    path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path/'results.json').write_text(json.dumps({'status': 'ok', 'tree_sha256': '2'*64}))
    metrics = {'artifact_gate': 'go', 'manifest_sha256': digest,
               'external': {'groundlm': {'fixture': True}}, 'safety_gate': {'decision': 'kill'}}
    metrics_path = tmp_path/'metrics.json'
    metrics_path.write_text(json.dumps(metrics))
    metrics_digest = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    actual = job.validated_v1_reference(tmp_path, digest, metrics_digest)
    assert actual['safety_gate']['decision'] == 'kill'  # archival reference != safety approval
    weights.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='expert changed'):
        job.validated_v1_reference(tmp_path, digest, metrics_digest)
