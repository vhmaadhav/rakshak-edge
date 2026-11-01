"""Synthetic artifact tampering tests; never load a detector or real messages."""

import copy
import json

import numpy as np
import pytest

from rakshak.fusion.cal_only import fit_cal_only
from rakshak.fusion.frozen_bundle import artifact, load_manifest, source_archive, write_manifest
from rakshak.fusion.fuse import FEATURE_NAMES


def test_frozen_bundle_rejects_mutations_and_overwrite(tmp_path):
    source = tmp_path/'source'
    (source/'rakshak').mkdir(parents=True)
    code = source/'rakshak/rules.py'
    code.write_text('# synthetic fixture\n')
    archive = tmp_path/'inference_source.zip'
    source_archive(source, archive)
    files = {'inference_source': artifact(archive)}
    for key in ('student_checkpoint', 'student_onnx', 'url_onnx', 'url_scaler'):
        path = tmp_path/key
        path.write_bytes(key.encode())
        files[key] = artifact(path)
    n = 160
    x = np.zeros((n, len(FEATURE_NAMES)))
    y = np.arange(n) % 2
    x[:, 0] = np.where(y, .9, .1)
    state = fit_cal_only(x, y, [str(i) for i in range(n)], np.arange(n)//2,
                         train_ids=['train'], test_ids=['test'])
    assert state['status'] == 'ok'
    provenance = {'git_sha': '1'*40, 'tree_sha256': '2'*64,
                  'kernel': 'maadhavvh/rakshak-fixture', 'kernel_version': 1}
    mapping = {'student': 'student_onnx', 'url': 'url_onnx', 'rules': 'inference_source'}
    meta = {'schema': 'rakshak-fusion-v1', 'description': 'newly fitted (not previously shipped)',
            'shipment_approved': False, 'alpha': .01, 'beta': .05, 'fusion': state,
            'student_cal_1pct_threshold': .5,
            'fitting': provenance, 'artifacts': files,
            'experts': {name: {**provenance, 'artifact_sha256': files[key]['sha256']}
                        for name, key in mapping.items()}}
    checkpoint_meta = copy.deepcopy(meta)
    checkpoint_meta.update(student_format='pytorch_checkpoint', measurement_reference_only=True,
                           arm_android_portability='UNVERIFIED')
    del checkpoint_meta['artifacts']['student_onnx']
    checkpoint_meta['experts']['student']['artifact_sha256'] = files['student_checkpoint']['sha256']
    checkpoint_path = tmp_path/'checkpoint_reference.json'
    write_manifest(checkpoint_path, checkpoint_meta)
    assert load_manifest(checkpoint_path, source_root=source) == checkpoint_meta
    checkpoint_meta['arm_android_portability'] = 'verified'
    with pytest.raises(ValueError, match='unverified portability'):
        write_manifest(tmp_path/'misleading.json', checkpoint_meta)
    target = tmp_path/'fusion_v1.json'
    digest = write_manifest(target, meta)
    assert len(digest) == 64
    assert load_manifest(target, source_root=source) == meta
    with pytest.raises(FileExistsError):
        write_manifest(target, meta)
    changed = copy.deepcopy(meta)
    changed['experts']['student']['kernel_version'] = None
    with pytest.raises(ValueError, match='kernel/version'):
        write_manifest(tmp_path/'bad.json', changed)
    changed = copy.deepcopy(meta)
    changed['fusion']['thresholds'] = {**changed['fusion']['thresholds'], 't_scam': .99}
    with pytest.raises(ValueError, match='threshold state'):
        write_manifest(tmp_path/'bad.json', changed)
    code.write_text('# modified inference\n')
    with pytest.raises(ValueError, match='inference source'):
        load_manifest(target, source_root=source)
    (tmp_path/'student_onnx').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='artifact'):
        load_manifest(target)
    assert json.loads(target.read_text()) == meta


def test_no_positive_roc_boundary_has_exact_portable_probability_semantics():
    from rakshak.fusion.frozen_bundle import probability_threshold

    threshold = probability_threshold(float('inf'))
    assert 1 < threshold < float('inf')
    assert not (np.asarray([0., .5, 1.]) >= threshold).any()
    with pytest.raises(ValueError):
        probability_threshold(float('nan'))
