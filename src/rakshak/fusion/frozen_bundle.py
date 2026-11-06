"""Immutable private fusion references, bound to exact experts and inference source."""

import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np

from rakshak.fusion.calibrate import PlattScaling
from rakshak.fusion.fuse import FEATURE_NAMES, FusionModel
from rakshak.fusion.selective import SelectiveThresholds

HEX64 = re.compile(r'[0-9a-f]{64}')
HEX40 = re.compile(r'[0-9a-f]{40}')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_archive(source_root, output):
    """Save deterministic source-only zip; never collect a corpus, credentials or weights."""
    source_root = Path(source_root)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((source_root/'rakshak').rglob('*.py')):
            entry = zipfile.ZipInfo(path.relative_to(source_root).as_posix())
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
    return {'file': Path(output).name, 'sha256': sha256(output)}


def artifact(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError('nonempty inference artifact required')
    return {'file': path.name, 'sha256': sha256(path), 'bytes': path.stat().st_size}



def probability_threshold(value):
    """Represent ROC's +infinity boundary portably for probabilities bounded by one."""
    value = float(value)
    if value == float('inf'):
        return float(np.nextafter(1., np.inf))
    if not np.isfinite(value) or not 0 <= value <= np.nextafter(1., np.inf):
        raise ValueError('invalid probability decision threshold')
    return value

def validate_manifest(manifest):
    if (manifest['schema'] != 'rakshak-fusion-v1'
            or manifest['description'] != 'newly fitted (not previously shipped)'
            or manifest['shipment_approved'] is not False):
        raise ValueError('explicit newly fitted reference scope required')
    if manifest['alpha'] != .01 or manifest['beta'] != .05:
        raise ValueError('frozen alpha/beta changed')
    if manifest.get('student_format') not in (None, 'onnx', 'pytorch_checkpoint'):
        raise ValueError('unknown student reference format')
    if (manifest.get('student_format') == 'pytorch_checkpoint'
            and (manifest.get('measurement_reference_only') is not True
                 or manifest.get('arm_android_portability') != 'UNVERIFIED')):
        raise ValueError('checkpoint must disclose measurement-only/unverified portability')
    probability_threshold(manifest['student_cal_1pct_threshold'])
    state = manifest['fusion']
    if (state['status'] != 'ok' or state['model']['features'] != list(FEATURE_NAMES)
            or state['threshold_selection']['alpha'] != manifest['alpha']
            or state['threshold_selection']['beta'] != manifest['beta']):
        raise ValueError('feasible baseline state with matching constraints required')
    groups = state['cal_stage_ids']
    if set(groups) != {'fusion', 'probability', 'threshold', 'crc'}:
        raise ValueError('four fixed calibration stages required')
    flattened = [item for values in groups.values() for item in values]
    if not all(groups.values()) or len(flattened) != len(set(flattened)):
        raise ValueError('nonempty disjoint calibration stages required')
    if set(state['model']['fit_ids']) != set(groups['fusion']):
        raise ValueError('stacker fitting IDs differ from declared calibration stage')
    if (state['model']['name'] != 'logistic'
            or len(state['model']['state']['weights']) != len(FEATURE_NAMES)):
        raise ValueError('baseline logistic stacker shape required')
    FusionModel.from_dict(state['model'])
    calibrator = PlattScaling.from_dict(state['calibrator'])
    if not all(float('-inf') < value < float('inf')
               for value in (calibrator.weight, calibrator.intercept,
                             *state['model']['state']['weights'],
                             state['model']['state']['intercept'])):
        raise ValueError('finite fusion parameters required')
    SelectiveThresholds.from_dict(state['thresholds'])
    if state['thresholds'] != state['threshold_selection']['thresholds']:
        raise ValueError('threshold state mismatch')
    if set(manifest['experts']) != {'student', 'url', 'rules'}:
        raise ValueError('all three exact experts required')
    for entry in [manifest['fitting'], *manifest['experts'].values()]:
        if (not HEX40.fullmatch(entry['git_sha'])
                or not HEX64.fullmatch(entry['tree_sha256'])
                or not entry['kernel'].startswith('maadhavvh/rakshak-')
                or type(entry['kernel_version']) is not int or entry['kernel_version'] < 1):
            raise ValueError('exact git/bundle/kernel/version required')
    student_key = ('student_checkpoint' if manifest.get('student_format') == 'pytorch_checkpoint'
                   else 'student_onnx')
    for key in {'student_checkpoint', student_key, 'url_onnx', 'url_scaler', 'inference_source'}:
        item = manifest['artifacts'][key]
        if (Path(item['file']).name != item['file'] or item['file'] in ('', '.', '..')
                or not HEX64.fullmatch(item['sha256']) or item['bytes'] < 1):
            raise ValueError('safe file name and exact artifact hash required')
    expected = {'student': student_key, 'url': 'url_onnx', 'rules': 'inference_source'}
    for expert, key in expected.items():
        if manifest['experts'][expert]['artifact_sha256'] != manifest['artifacts'][key]['sha256']:
            raise ValueError('expert hash differs from bundled artifact')
    return manifest


def write_manifest(path, manifest):
    validate_manifest(manifest)
    path = Path(path)
    # Exclusive creation: a frozen version cannot be overwritten after outcomes.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return sha256(path)


def load_manifest(path, *, source_root=None):
    path = Path(path)
    manifest = validate_manifest(json.loads(path.read_text()))
    for entry in manifest['artifacts'].values():
        item = path.parent/entry['file']
        if item.is_symlink() or not item.is_file() or sha256(item) != entry['sha256']:
            raise ValueError('frozen artifact missing or modified')
    if source_root is not None:
        archive_path = path.parent/manifest['artifacts']['inference_source']['file']
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                parts = Path(name)
                if parts.is_absolute() or '..' in parts.parts or parts.suffix != '.py':
                    raise ValueError('source archive contains an unsafe path')
                current = Path(source_root)/parts
                if not current.is_file() or current.read_bytes() != archive.read(name):
                    raise ValueError('current inference source differs from frozen version')
    return manifest
