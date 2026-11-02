#!/usr/bin/env python3
"""Stage one hash-bound PRIVATE family-calibration research archive; no row scores."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

FOLDS = {'campaign', 'lang_shift', 'temporal', 'source_holdout:holdout_scamshield',
         'source_holdout:holdout_uci_sms_spam', 'source_holdout:holdout_mishra_soni_smsphishing'}

DATASET = 'maadhavvh/rakshak-template-cal-c1-20261102'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stage(source, destination, push_log):
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source/'frozen_template_cal.json').read_text())
    metrics = json.loads((source/'metrics.json').read_text())
    result = json.loads((source/'results.json').read_text())
    version = re.findall(r'Kernel version (\d+) successfully pushed', Path(push_log).read_text())
    if (result['status'] != 'ok' or result.get('failures') or version != ['1']
            or manifest['schema'] != 'rakshak-template-cal-c1'
            or manifest['shipment_approved'] is not False
            or manifest['fitting_kernel_version'] != 1
            or manifest['git_sha'] != result['git_sha']
            or manifest['tree_sha256'] != result['tree_sha256']
            or metrics['manifest_sha256'] != sha(source/'frozen_template_cal.json')
            or set(manifest['states']) != FOLDS
            or set(manifest['student_provenance']) != FOLDS
            or set(metrics['internal']) != FOLDS
            or set(metrics['external']) != {'groundlm', 'smishx'}):
        raise ValueError('complete exact frozen family-calibration cohort required')
    entries = list(manifest['other_artifacts'].values())
    if {item['file'] for item in entries} != {
            'url.onnx', 'url_side_scaler.json', 'inference_source.zip'}:
        raise ValueError('unexpected nonstudent artifact inventory')
    for fold, seeds in manifest['student_provenance'].items():
        if set(seeds) != {'1', '2', '3'} or set(manifest['states'][fold]) != set(seeds):
            raise ValueError('all three fixed seeds required')
        entries.extend(info['checkpoint'] for info in seeds.values())
    names = ['frozen_template_cal.json', 'metrics.json', 'results.json']
    for item in entries:
        filename = item['file']
        path = source/filename
        if (Path(filename).name != filename or path.is_symlink() or not path.is_file()
                or sha(path) != item['sha256']):
            raise ValueError('model artifact changed or unsafe path')
        names.append(filename)
    if len(names) != len(set(names)) or len(entries) != 21:
        raise ValueError('unique 18 student and 3 expert/source artifacts required')
    if any((source/name).is_symlink() for name in names):
        raise ValueError('archive symlink refused')
    destination.mkdir(parents=True, exist_ok=False)
    for name in names:
        shutil.copyfile(source/name, destination/name)
    archive = {'schema': 'rakshak-template-cal-c1-private-archive',
               'shipment_approved': False, 'experiment_gate': metrics['decision'],
               'scope': 'in-sample intervention selection; independent Indian validation OPEN',
               'hf_publish': 'HF publish pending',
               'files': {name: {'sha256': sha(destination/name),
                                'bytes': (destination/name).stat().st_size} for name in names}}
    (destination/'archive_manifest.json').write_text(json.dumps(archive, indent=2)+'\n')
    (destination/'dataset-metadata.json').write_text(json.dumps({
        'id': DATASET, 'title': 'Rakshak template calibration c1 20261102',
        'licenses': [{'name': 'other'}], 'isPrivate': True,
        'description': 'PRIVATE mixed-licence research cohort. Consult unchanged go/kill result. '
                       'No shipment approval. SMS/row-score parquets excluded. '
                       'HF publish pending.'},
        indent=2)+'\n')
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--stage', required=True, type=Path)
    parser.add_argument('--push-log', required=True, type=Path)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    stage(args.source, args.stage, args.push_log)
    if not args.publish:
        print('Hash-bound allowlist staged; no upload requested.')
        return
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

    api = KaggleApi()
    api.authenticate()
    response = api.dataset_create_new(str(args.stage), public=False,
                                      convert_to_csv=False, dir_mode='zip')
    if getattr(response, 'error', None) or getattr(response, 'status', None) == 'error':
        raise RuntimeError('private create rejected; do not repeat upload')
    with api.build_kaggle_client() as client:
        request = ApiGetDatasetRequest()
        request.owner_slug, request.dataset_slug = DATASET.split('/')
        metadata = client.datasets.dataset_api_client.get_dataset(request)
        receipt = {'ref': metadata.ref, 'owner': metadata.owner_ref,
                   'isPrivate': metadata.is_private, 'version': metadata.current_version_number,
                   'totalBytes': metadata.total_bytes,
                   'manifest_sha256': sha(args.stage/'archive_manifest.json')}
        if (receipt['ref'] != DATASET or receipt['owner'] != 'maadhavvh'
                or receipt['isPrivate'] is not True or receipt['version'] != 1):
            raise RuntimeError('private owner/version verification failed')
        (args.stage.parent/'family-privacy-verification.json').write_text(
            json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
