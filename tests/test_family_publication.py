import importlib.util
import json
from pathlib import Path

import pytest

from rakshak.fusion.hardneg_gate import FOLDS


def publisher():
    spec = importlib.util.spec_from_file_location('family_publisher',
                                                 Path(__file__).parents[1]/
                                                 'kaggle/publish_family_cal.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_source(tmp_path, module):
    source = tmp_path/'source'
    source.mkdir()

    def entry(name):
        path = source/name
        path.write_bytes(b'synthetic-model-fixture')
        return {'file': name, 'sha256': module.sha(path)}

    manifest = {'schema': 'rakshak-template-cal-c1', 'shipment_approved': False,
                'fitting_kernel_version': 1, 'git_sha': 'synthetic-sha', 'tree_sha256': 'tree',
                'states': {fold: {str(s): {} for s in (1, 2, 3)} for fold in FOLDS},
                'student_provenance': {
                    fold: {str(s): {'checkpoint': entry(f'{fold.replace(":", "_")}_{s}.pt')}
                           for s in (1, 2, 3)} for fold in FOLDS},
                'other_artifacts': {name: entry(name) for name in (
                    'url.onnx', 'url_side_scaler.json', 'inference_source.zip')}}
    (source/'frozen_template_cal.json').write_text(json.dumps(manifest))
    result = {'status': 'ok', 'git_sha': 'synthetic-sha', 'tree_sha256': 'tree', 'failures': []}
    (source/'results.json').write_text(json.dumps(result))
    metrics = {'manifest_sha256': module.sha(source/'frozen_template_cal.json'),
               'internal': dict.fromkeys(FOLDS), 'external': {'groundlm': {}, 'smishx': {}},
               'decision': {'decision': 'kill'}}
    (source/'metrics.json').write_text(json.dumps(metrics))
    (source/'scores_external.parquet').write_bytes(b'private-row-fixture-never-upload')
    log = tmp_path/'push.log'
    log.write_text('Kernel version 1 successfully pushed.\n')
    return source, log


def test_publication_excludes_row_scores_and_marks_private(tmp_path):
    module = publisher()
    source, log = make_source(tmp_path, module)
    destination = tmp_path/'stage'
    archive = module.stage(source, destination, log)
    assert archive['experiment_gate']['decision'] == 'kill'
    assert not archive['shipment_approved']
    assert len(archive['files']) == 24
    assert not (destination/'scores_external.parquet').exists()
    metadata = json.loads((destination/'dataset-metadata.json').read_text())
    assert metadata['isPrivate'] is True and metadata['id'].startswith('maadhavvh/')


def test_publication_rejects_changed_weight_before_staging(tmp_path):
    module = publisher()
    source, log = make_source(tmp_path, module)
    next(source.glob('*.pt')).write_bytes(b'altered')
    with pytest.raises(ValueError, match='changed or unsafe'):
        module.stage(source, tmp_path/'stage', log)
    assert not (tmp_path/'stage').exists()


def test_publication_script_imports_without_scientific_dependencies():
    import subprocess
    import sys

    script = Path(__file__).parents[1]/'kaggle/publish_family_cal.py'
    code = ('import runpy,sys; sys.modules["numpy"]=None; '
            'module=runpy.run_path(sys.argv[1]); assert len(module["FOLDS"])==6')
    subprocess.run([sys.executable, '-I', '-c', code, str(script)], check=True,
                    capture_output=True, text=True)
