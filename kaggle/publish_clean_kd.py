#!/usr/bin/env python3
"""Create a private, immutable archive of the matched KD experiment's 36 checkpoints.

Only receipts, aggregate metrics and explicitly hash-listed models are staged.
No model loading, message reads, row-score uploads, training or evaluation.
"""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

DATASET = 'maadhavvh/rakshak-clean-kd-20261101'
FOLDS = {'campaign', 'lang_shift', 'temporal', 'source_holdout:holdout_scamshield',
         'source_holdout:holdout_uci_sms_spam', 'source_holdout:holdout_mishra_soni_smsphishing'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stage(source, destination, push_log):
    source, destination = Path(source), Path(destination)
    receipt = json.loads((source/'results.json').read_text())
    metrics = json.loads((source/'metrics.json').read_text())
    version = re.findall(r'Kernel version (\d+) successfully pushed', Path(push_log).read_text())
    if (receipt['job'] != 'sms_clean_kd' or receipt['status'] != 'ok' or receipt.get('failures')
            or version != ['1'] or metrics['skipped'] or set(metrics['runs']) != FOLDS
            or metrics['config']['args'] != {'epochs': 15, 'n_boot': 1000,
                                             'seeds': [1, 2, 3], 'teacher_seed': 0}
            or metrics['config']['teacher_arms'] != {'erm': False, 'kd': True}
            or metrics['clean_kd_gate']['missing']
            or metrics['clean_kd_gate']['decision'] not in ('go', 'kill')):
        raise ValueError('complete matched three-seed six-fold experiment required')
    models = []
    for fold, block in metrics['runs'].items():
        if set(block['arms']) != {'erm', 'kd'}:
            raise ValueError('both matched arms required')
        for arm, seeds in block['arms'].items():
            if set(seeds) != {'1', '2', '3'}:
                raise ValueError('three seeds per arm required')
            for seed, item in seeds.items():
                entry = item['checkpoint']
                filename = f'student_{fold.replace(":", "_")}_{arm}_s{seed}.pt'
                path = source/filename
                rows = item['training']['rows']
                if (entry['file'] != filename or path.is_symlink() or not path.is_file()
                        or digest(path) != entry['sha256'] or path.stat().st_size == 0
                        or item['teacher_enabled'] != (arm == 'kd')
                        or rows['with_teacher'] != (rows['rows'] if arm == 'kd' else 0)):
                    raise ValueError('checkpoint hash or actual teacher coverage mismatch')
                models.append({'fold': fold, 'arm': arm, 'seed': int(seed), **entry,
                               'teacher_enabled': item['teacher_enabled']})
    manifest = {'schema': 'rakshak-clean-kd-v1', 'shipment_approved': False,
                'experiment_gate': metrics['clean_kd_gate'],
                'teacher_target_scope': metrics['teacher_target_scope'],
                'fitting': {'git_sha': receipt['git_sha'], 'tree_sha256': receipt['tree_sha256'],
                            'kernel': 'maadhavvh/rakshak-sms-clean-kd-kd1-20261101',
                            'kernel_version': 1},
                'metrics_sha256': digest(source/'metrics.json'), 'models': models,
                'exposure': 'single teacher, train IDs only, not OOF; rule-feature design and '
                            'GroundLM/SmishX intervention feedback in-sample; not a fused claim',
                'hf_publish': 'pending'}
    destination.mkdir(parents=True, exist_ok=False)
    for name in ['results.json', 'metrics.json', *[item['file'] for item in models]]:
        shutil.copyfile(source/name, destination/name)
    (destination/'model_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (destination/'dataset-metadata.json').write_text(json.dumps({
        'id': DATASET, 'title': 'Rakshak clean KD matched checkpoints 20261101',
        'licenses': [{'name': 'other'}], 'isPrivate': True,
        'description': 'PRIVATE mixed-licence research archive. Consult the recorded gate; '
                       'shipment not approved. HF publish pending.'}, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--stage', required=True, type=Path)
    parser.add_argument('--push-log', required=True, type=Path)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    stage(args.source, args.stage, args.push_log)
    if not args.publish:
        print('Hash-validated allowlist staged; no upload requested.')
        return
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

    api = KaggleApi()
    api.authenticate()
    response = api.dataset_create_new(str(args.stage), public=False,
                                      convert_to_csv=False, dir_mode='zip')
    if getattr(response, 'error', None) or getattr(response, 'status', None) == 'error':
        raise RuntimeError('private create rejected; do not overwrite an immutable archive')
    # One owner-record read. A processing-time denial needs a separate later verification,
    # never a repeat upload or a status-polling loop.
    with api.build_kaggle_client() as client:
        request = ApiGetDatasetRequest()
        request.owner_slug, request.dataset_slug = DATASET.split('/')
        metadata = client.datasets.dataset_api_client.get_dataset(request)
        if (metadata.ref != DATASET or metadata.owner_ref != request.owner_slug
                or metadata.is_private is not True or metadata.current_version_number != 1):
            raise RuntimeError('private owner/version publication not verified')
    print(f'Private checkpoint archive verified: {DATASET} version1')


if __name__ == '__main__':
    main()
