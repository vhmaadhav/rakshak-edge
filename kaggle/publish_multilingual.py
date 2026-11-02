#!/usr/bin/env python3
"""Private multilingual research archive: exact models/tokenizer/vocab, no SMS/row scores."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

DATASET = 'maadhavvh/rakshak-multilingual-m3-20261102'
TREE = '93ee0115cc22af3a2bedeaacc82df2c10ebb52ed13f2e30387b72747da90d12e'
REVISION = 'abc32620dd4f6ab06f5fbe905dc25f310618e09f'
FOLDS = {'campaign', 'lang_shift', 'temporal', 'source_holdout:holdout_scamshield',
         'source_holdout:holdout_uci_sms_spam', 'source_holdout:holdout_mishra_soni_smsphishing'}
ARMS = {'byte_nokd', 'multilingual_nokd', 'multilingual_kd'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stage(source, destination, push_log):
    source, destination = Path(source), Path(destination)
    result = json.loads((source/'results.json').read_text())
    metrics = json.loads((source/'metrics.json').read_text())
    gate, config = metrics['multilingual_gate'], metrics['config']
    version = re.findall(r'Kernel version (\d+) successfully pushed', Path(push_log).read_text())
    if (result['job'] != 'sms_multilingual' or result['status'] != 'ok' or result.get('failures')
            or result['tree_sha256'] != TREE or version != ['1']
            or set(metrics['runs']) != FOLDS or set(metrics['comparisons']) != FOLDS
            or set(config['arms']) != ARMS or config['seeds'] != [1, 2, 3]
            or config['n_boot'] != 1000 or config['pretrained']['licence'] != 'MIT'
            or config['pretrained']['revision'] != REVISION or gate['missing']
            or gate['decision'] not in ('go', 'kill') or gate['candidate'] != 'multilingual_kd'
            or gate['ablation_promotable'] is not False
            or (gate['decision'] == 'go' and not all(gate['checks'].values()))):
        raise ValueError('complete original fixed multilingual cohort required')
    files = {'results.json': digest(source/'results.json'),
             'metrics.json': digest(source/'metrics.json'),
             'pretrained_source.json': digest(source/'pretrained_source.json'),
             'inference_source.zip': config['inference_source_sha256']}
    models = []
    for fold, block in metrics['runs'].items():
        stem = fold.replace(':', '_')
        vocabulary = block['vocabulary']
        if vocabulary['file'] != f'vocabulary_{stem}.npz' or set(block['arms']) != ARMS:
            raise ValueError('all original arms and fold vocabulary required')
        files[vocabulary['file']] = vocabulary['sha256']
        name = f'vocabulary_{stem}.json'
        info = json.loads((source/name).read_text())
        if (info['fold'] != fold or info['tree_sha256'] != TREE
                or info['vocabulary_sha256'] != vocabulary['sha256']
                or info['source_revision'] != REVISION
                or info['inference_source_sha256'] != config['inference_source_sha256']
                or len(info['train_ids']) != info['rows']):
            raise ValueError('train-only frozen vocabulary manifest changed')
        files[name] = digest(source/name)
        for arm, seeds in block['arms'].items():
            if set(seeds) != {'1', '2', '3'}:
                raise ValueError('three seeds per arm required')
            for seed, item in seeds.items():
                entry, training = item['checkpoint'], item['training']
                name = f'student_{stem}_{arm}_s{seed}.pt'
                if (entry['file'] != name or training['rows'] != info['rows']
                        or training['teacher_rows'] != (info['rows'] if arm == 'multilingual_kd'
                                                       else 0)
                        or (arm == 'multilingual_kd' and entry['bytes'] > 8*1024*1024)):
                    raise ValueError('original train-only teacher/model inventory changed')
                files[name] = entry['sha256']
                models.append({'fold': fold, 'arm': arm, 'seed': int(seed), **entry})
    tokenizer = source/'tokenizer'
    allowed = {'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
               'SOURCE_README.md', 'LICENSE', 'LICENSE.txt', 'LICENSE.md'}
    names = {p.name for p in tokenizer.iterdir()}
    if (not {'tokenizer.json', 'tokenizer_config.json', 'SOURCE_README.md'} <= names
            or not names <= allowed):
        raise ValueError('exact saved tokenizer/card/licence allowlist required')
    for name in sorted(names):
        files['tokenizer/'+name] = digest(tokenizer/name)
    for name, expected in files.items():
        path = source/name
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError('archive model/source hash or path mismatch')
    if len(models) != 54:
        raise ValueError('all54 original checkpoints required')
    destination.mkdir(parents=True, exist_ok=False)
    for name in files:
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, target)
    manifest = {'schema': 'rakshak-multilingual-m3-private-v1', 'shipment_approved': False,
                'experiment_gate': gate, 'models': models,
                'fitting': {'git_sha': result['git_sha'], 'tree_sha256': TREE,
                            'kernel': 'maadhavvh/rakshak-sms-multilingual-m3-20261102',
                            'kernel_version': 1},
                'files': {name: {'sha256': value, 'bytes': (destination/name).stat().st_size}
                          for name, value in files.items()},
                'tokenizer_bytes': sum((tokenizer/name).stat().st_size for name in names),
                'size_scope': '8MiB gate covers classifier tensors only, excludes tokenizer/vocab',
                'exposure': ('in-sample intervention selection; '
                             'not an independent Indian/fused claim'),
                'teacher_scope': 'original train IDs only, single teacher0, not OOF/ensemble',
                'hf_publish': 'HF publish pending'}
    (destination/'model_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (destination/'dataset-metadata.json').write_text(json.dumps({
        'id': DATASET, 'title': 'Rakshak multilingual research checkpoints 20261102',
        'licenses': [{'name': 'other'}], 'isPrivate': True,
        'description': 'PRIVATE mixed-licence research cohort. MIT mmBERT-small-derived compact '
                       'embeddings; source card/tokenizer included. Allthree arms, not ablation '
                       'selection. No SMS or row-score parquets; shipment not approved. '
                       'HF publish pending.'}, indent=2)+'\n')
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
