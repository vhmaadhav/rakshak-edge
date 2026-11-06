#!/usr/bin/env python3
"""Create one immutable PRIVATE model bundle, never a new version of an existing v1.

Run with the Python interpreter of the installed Kaggle CLI. No training/inference.
Only the allowlisted inference files and JSON receipts are uploaded; no SMS rows.
"""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

DATASET = 'maadhavvh/rakshak-fusion-v1-20261101'
KEYS = {'student_checkpoint', 'student_onnx', 'url_onnx', 'url_scaler', 'inference_source'}
COMPANIONS = {'lang_shift', 'temporal', 'source_holdout:holdout_scamshield',
              'source_holdout:holdout_uci_sms_spam',
              'source_holdout:holdout_mishra_soni_smsphishing'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stage(source, destination, push_log):
    source, destination = Path(source), Path(destination)
    manifest_path = source/'fusion_v1.json'
    manifest = json.loads(manifest_path.read_text())
    result = json.loads((source/'results.json').read_text())
    metrics = json.loads((source/'metrics.json').read_text())
    required = (KEYS - {'student_onnx'} if manifest.get('student_format') == 'pytorch_checkpoint'
                else KEYS)
    version = re.findall(r'Kernel version (\d+) successfully pushed', Path(push_log).read_text())
    if (result['status'] != 'ok' or result.get('failures') or metrics['artifact_gate'] != 'go'
            or metrics['manifest_sha256'] != digest(manifest_path)
            or manifest['fitting']['tree_sha256'] != result['tree_sha256']
            or manifest['fitting']['git_sha'] != result['git_sha']
            or version != [str(manifest['fitting']['kernel_version'])]
            or manifest['shipment_approved'] is not False
            or manifest['schema'] != 'rakshak-fusion-v1'
            or set(manifest['artifacts']) != required
            or set(manifest.get('validation_references', {})) != COMPANIONS):
        raise ValueError('complete hash-bound frozen v1 and actual pushed version required')
    files = ['fusion_v1.json', 'results.json', 'metrics.json']
    for item in manifest['artifacts'].values():
        path = source/item['file']
        if (Path(item['file']).name != item['file'] or path.is_symlink()
                or not path.is_file() or digest(path) != item['sha256']):
            raise ValueError('frozen inference file missing or modified')
        files.append(item['file'])
    for name, reference in manifest.get('validation_references', {}).items():
        relative = Path(reference['manifest'])
        if (relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 2
                or relative.name != 'fusion_v1.json'):
            raise ValueError('unsafe validation manifest path')
        child_path = source/relative
        child = json.loads(child_path.read_text())
        if (digest(child_path) != reference['sha256'] or child['fold'] != name
                or child['fitting'] != manifest['fitting']
                or set(child['artifacts']) != required or child['validation_references']):
            raise ValueError('fold-validation manifest provenance mismatch')
        files.append(str(relative))
        for item in child['artifacts'].values():
            filename = Path(item['file'])
            path = child_path.parent/filename
            if (filename.name != item['file'] or path.is_symlink()
                    or not path.is_file() or digest(path) != item['sha256']):
                raise ValueError('fold-validation artifact mismatch')
            files.append(str(relative.parent/filename))
    destination.mkdir(parents=True, exist_ok=False)
    for name in files:
        (destination/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, destination/name)
    (destination/'dataset-metadata.json').write_text(json.dumps({
        'id': DATASET, 'title': 'Rakshak frozen fusion v1 20261101',
        'licenses': [{'name': 'other'}], 'isPrivate': True,
        'description': 'PRIVATE mixed-licence reference. Newly fitted (not previously shipped). '
                       'Safety gate may fail: consult metrics.json. Shipment not approved. '
                       'Immutable manifest SHA256 '+digest(manifest_path)}, indent=2)+'\n')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--push-log', required=True, type=Path)
    p.add_argument('--stage', required=True, type=Path)
    p.add_argument('--publish', action='store_true', help='explicitly create PRIVATE dataset once')
    args = p.parse_args()
    stage(args.source, args.stage, args.push_log)
    if not args.publish:
        print('Validated allowlisted model bundle staged; no upload requested.')
        return
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

    api = KaggleApi()
    api.authenticate()
    response = api.dataset_create_new(str(args.stage), public=False,
                                      convert_to_csv=False, dir_mode='zip')
    if getattr(response, 'error', None) or getattr(response, 'status', None) == 'error':
        raise RuntimeError('private dataset creation rejected; do not overwrite frozen v1')
    with api.build_kaggle_client() as client:
        request = ApiGetDatasetRequest()
        request.owner_slug, request.dataset_slug = DATASET.split('/')
        metadata = client.datasets.dataset_api_client.get_dataset(request)
        if (metadata.ref != DATASET or metadata.owner_ref != request.owner_slug
                or metadata.is_private is not True or metadata.current_version_number < 1):
            raise RuntimeError('private owner/version publication could not be verified')
    print(f'Private immutable model bundle verified: {DATASET}; manifest '
          f'{digest(args.stage/"fusion_v1.json")}')


if __name__ == '__main__':
    main()
