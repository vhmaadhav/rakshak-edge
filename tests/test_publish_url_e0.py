import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1]/'kaggle/publish_url_e0.py'
    spec = importlib.util.spec_from_file_location('url_publisher_unit', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture(tmp_path, monkeypatch):
    mod = module()
    control, evaluation, failed = [tmp_path/name for name in ('control', 'evaluation', 'failed')]
    for path in (control, evaluation, failed):
        path.mkdir()
    entries = {}
    for ds in ('phreshphish', 'url_domain'):
        entries[ds] = {}
        for seed in ('1', '2', '3'):
            name = f'url_control_{ds}_s{seed}.pt'
            (control/name).write_bytes(b'synthetic checkpoint fixture')
            entries[ds][seed] = {'checkpoint': {'file': name, 'sha256': mod.sha(control/name)}}
    (control/'inference_source.zip').write_bytes(b'synthetic source fixture')
    manifest = {'complete': True, 'no_test_predictions': True, 'controls': entries,
                'tree_sha256': mod.CONTROL_TREE,
                'source_archive': {'file': 'inference_source.zip',
                                   'sha256': mod.sha(control/'inference_source.zip')}}
    (control/'cal_controls_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(mod, 'CONTROL_MANIFEST_SHA', mod.sha(control/'cal_controls_manifest.json'))
    (control/'results.json').write_text(json.dumps({'status': 'ok', 'job': 'url_controls_cal',
                                                   'tree_sha256': mod.CONTROL_TREE}))
    frozen = {'controls': {ds: {seed: {'parent': value} for seed, value in rows.items()}
                            for ds, rows in entries.items()},
              'parent_sha256': mod.CONTROL_MANIFEST_SHA, 'shipment_approved': False,
              'tree_sha256': mod.PARENT_EVAL_TREE}
    (evaluation/'evaluation_manifest.json').write_text(json.dumps(frozen))
    recovery = {'tree_sha256': mod.EVAL_TREE, 'parent_tree': mod.PARENT_EVAL_TREE,
                'no_model_execution': True}
    (evaluation/'recovery_manifest.json').write_text(json.dumps(recovery))
    (evaluation/'original_eval_results.json').write_text(json.dumps({'status': 'error'}))
    metrics = {'recovery_manifest_sha256': mod.sha(evaluation/'recovery_manifest.json'),
               'teacher_gate': 'UNMEASURABLE', 'kd_gate': 'UNMEASURABLE',
               'manifest_sha256': mod.sha(evaluation/'evaluation_manifest.json'),
               'controls': {ds: {'seeds': rows} for ds, rows in entries.items()}}
    (evaluation/'metrics.json').write_text(json.dumps(metrics))
    (evaluation/'results.json').write_text(json.dumps(
        {'status': 'ok', 'job': 'url_controls_recompute', 'tree_sha256': mod.EVAL_TREE}))
    (failed/'results.json').write_text(json.dumps({'status': 'error',
        'tree_sha256': mod.TEACHER_TREE,
        'error': 'FloatingPointError: nonfinite FP32 teacher loss; stop fixed arm'}))
    (evaluation/'raw_urls.parquet').write_bytes(b'never copy this synthetic row fixture')
    return mod, control, evaluation, failed


def test_private_allowlist_excludes_rows_and_requires_exact_checkpoint_hash(tmp_path, monkeypatch):
    mod, control, evaluation, failed = fixture(tmp_path, monkeypatch)
    dest = tmp_path/'staged'
    archive = mod.stage(control, evaluation, failed, dest)
    assert len(archive['files']) == 15 and archive['shipment_approved'] is False
    assert json.loads((dest/'dataset-metadata.json').read_text())['isPrivate'] is True
    assert not list(dest.glob('*.parquet'))
    assert (dest/'failed_teacher_results.json').is_file()
    path = control/'url_control_phreshphish_s1.pt'
    path.write_bytes(b'tampered synthetic fixture')
    with pytest.raises(ValueError, match='hash or path'):
        mod.stage(control, evaluation, failed, tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_publisher_refuses_metadata_symlink_before_staging(tmp_path, monkeypatch):
    mod, control, evaluation, failed = fixture(tmp_path, monkeypatch)
    file = evaluation/'metrics.json'
    real = evaluation/'metrics-real.json'
    file.rename(real)
    file.symlink_to(real)
    with pytest.raises(ValueError, match='regular allowlisted'):
        mod.stage(control, evaluation, failed, tmp_path/'staged')
    assert not (tmp_path/'staged').exists()
