#!/usr/bin/env python3
"""Private hash-bound six-control archive; no raw URL or row-score files."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

DATASET = 'maadhavvh/rakshak-url-e0-20261102'
CONTROL_TREE = 'ebcc1bb75c6e9668a789928946c7247e66b560305d5e62e38598fe7d675f715b'
TEACHER_TREE = '84d35409ac4ac48138a2a38673bd14efd7682b40808d64a9b80762d3f7e5e64d'
PARENT_EVAL_TREE = '4fdfe6994123ec28645a3383841f9e251bd27e5e2cacf91235aa3db2bb65a3ec'
EVAL_TREE = '17576395e1c2fc76e32d4970c15ca5c180844da3527a40542680e07c6d34b7c2'
CONTROL_MANIFEST_SHA = '88b20cd71febccdc1ed81d8c2a9ec0f1b984b9eed9f44c389f357b5bfcc4ac7d'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(folder, name):
    path = folder/name
    if path.is_symlink() or not path.is_file():
        raise ValueError('regular allowlisted metadata file required')
    return json.loads(path.read_text())


def stage(control, evaluation, failed_teacher, destination):
    control, evaluation, failed_teacher, destination = map(
        Path, (control, evaluation, failed_teacher, destination))
    manifest = read(control, 'cal_controls_manifest.json')
    outcome = read(evaluation, 'metrics.json')
    frozen = read(evaluation, 'evaluation_manifest.json')
    recovery = read(evaluation, 'recovery_manifest.json')
    original = read(control, 'results.json')
    evaluated = read(evaluation, 'results.json')
    failed = read(failed_teacher, 'results.json')
    if (original['status'] != 'ok' or original['tree_sha256'] != CONTROL_TREE
            or sha(control/'cal_controls_manifest.json') != CONTROL_MANIFEST_SHA
            or original['job'] != 'url_controls_cal'
            or evaluated['job'] != 'url_controls_recompute'
            or manifest['tree_sha256'] != CONTROL_TREE
            or evaluated['tree_sha256'] != EVAL_TREE
            or manifest['complete'] is not True or manifest['no_test_predictions'] is not True
            or evaluated['status'] != 'ok' or evaluated.get('failures')
            or outcome['teacher_gate'] != 'UNMEASURABLE' or outcome['kd_gate'] != 'UNMEASURABLE'
            or outcome['manifest_sha256'] != sha(evaluation/'evaluation_manifest.json')
            or frozen['parent_sha256'] != sha(control/'cal_controls_manifest.json')
            or frozen['tree_sha256'] != PARENT_EVAL_TREE
            or recovery['tree_sha256'] != evaluated['tree_sha256']
            or recovery['parent_tree'] != PARENT_EVAL_TREE
            or recovery['no_model_execution'] is not True
            or outcome['recovery_manifest_sha256'] != sha(evaluation/'recovery_manifest.json')
            or frozen['shipment_approved'] is not False
            or failed['status'] != 'error' or failed['tree_sha256'] != TEACHER_TREE
            or failed['error'] != 'FloatingPointError: nonfinite FP32 teacher loss; stop fixed arm'
            or set(manifest['controls']) != {'phreshphish', 'url_domain'}
            or set(outcome['controls']) != {'phreshphish', 'url_domain'}):
        raise ValueError('complete frozen controls/evaluation/exact numerical failure required')
    entries = [(control, 'cal_controls_manifest.json', 'control_manifest.json'),
               (control, 'results.json', 'control_results.json'),
               (evaluation, 'evaluation_manifest.json', 'evaluation_manifest.json'),
               (evaluation, 'metrics.json', 'metrics.json'),
               (evaluation, 'results.json', 'evaluation_results.json'),
               (evaluation, 'recovery_manifest.json', 'recovery_manifest.json'),
               (evaluation, 'original_eval_results.json', 'original_eval_results.json'),
               (failed_teacher, 'results.json', 'failed_teacher_results.json')]
    source = manifest['source_archive']
    if source['file'] != 'inference_source.zip':
        raise ValueError('exact original source archive required')
    artifacts = [source]
    for ds, seeds in manifest['controls'].items():
        if set(seeds) != {'1', '2', '3'} or set(outcome['controls'][ds]['seeds']) != set(seeds):
            raise ValueError('all three seeds on both frozen datasets required')
        for seed, block in seeds.items():
            if frozen['controls'][ds][seed]['parent'] != block:
                raise ValueError('evaluation used a different control artifact')
            if block['checkpoint']['file'] != f'url_control_{ds}_s{seed}.pt':
                raise ValueError('only six original checkpoint names allowed')
            artifacts.append(block['checkpoint'])
    for entry in artifacts:
        name = entry['file']
        path = control/name
        if (Path(name).name != name or path.is_symlink() or not path.is_file()
                or sha(path) != entry['sha256']):
            raise ValueError('allowlisted model/source hash or path changed')
        entries.append((control, name, name))
    if len(entries) != 15 or len({name for _, _, name in entries}) != len(entries):
        raise ValueError('exact six-checkpoint/source/aggregate inventory required')
    if any((root/name).is_symlink() for root, name, _ in entries):
        raise ValueError('archive symlink refused')
    destination.mkdir(parents=True, exist_ok=False)
    for root, name, target in entries:
        shutil.copyfile(root/name, destination/target)
    archive = {'schema': 'rakshak-url-e0-private-archive-v1', 'shipment_approved': False,
               'teacher_gate': 'UNMEASURABLE', 'kd_gate': 'UNMEASURABLE',
               'hf_publish': 'HF publish pending',
               'scope': 'URL-only; repeated benchmark/in-sample intervention selection',
               'files': {name: {'sha256': sha(destination/name),
                                'bytes': (destination/name).stat().st_size}
                         for _, _, name in entries}}
    (destination/'archive_manifest.json').write_text(json.dumps(archive, indent=2)+'\n')
    (destination/'dataset-metadata.json').write_text(json.dumps({
        'id': DATASET, 'title': 'Rakshak URL three seed controls 20261102',
        'licenses': [{'name': 'other'}], 'isPrivate': True,
        'description': 'PRIVATE URL research controls. Teacher failed numerical stability; '
                       'no KD or stronger teacher claim. No raw URLs or row-score parquets. '
                       'HF publish pending. No deployment approval.'}, indent=2)+'\n')
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--failed-teacher', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    stage(args.control, args.evaluation, args.failed_teacher, args.stage)
    if not args.publish:
        print('Exact allowlist staged; no upload requested.')
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
        (args.stage.parent/'url-privacy-verification.json').write_text(
            json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
