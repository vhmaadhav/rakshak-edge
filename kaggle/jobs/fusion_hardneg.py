"""Frozen three-seed hard-negative comparison through the unchanged fusion schema."""

import argparse
import hashlib
import json
from pathlib import Path

from rakshak.fusion.hardneg_gate import FOLDS, hardneg_gate
from rakshak.models.transactional_negatives import fingerprint


def validated_source(root, tree, digest):
    matches = []
    for path in Path(root).rglob('results.json'):
        receipt = json.loads(path.read_text())
        if receipt.get('tree_sha256') == tree:
            if receipt.get('status') != 'ok' or receipt.get('failures'):
                raise ValueError('complete successful hard-negative source required')
            if hashlib.sha256(path.with_name('metrics.json').read_bytes()).hexdigest() != digest:
                raise ValueError('source metrics hash mismatch')
            matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError('one exact completed source tree required')
    folder = matches[0]
    metrics = json.loads((folder/'metrics.json').read_text())
    args = metrics['config']['args']
    if (args['seeds'] != [1, 2, 3] or args['epochs'] != 15
            or args['teacher_seed'] != 0 or args['n_boot'] != 1000):
        raise ValueError('source training settings differ from the frozen protocol')
    if set(metrics['runs']) != FOLDS or set(metrics['config']['arms']) != {'erm', 'hardneg'}:
        raise ValueError('both fresh arms and all frozen folds required')
    for block in metrics['runs'].values():
        for arm in ('erm', 'hardneg'):
            if set(block['arms'][arm]) != {'1', '2', '3'}:
                raise ValueError('three exact training seeds required')
        for seed in ('1', '2', '3'):
            base, aug = (block['arms'][arm][seed] for arm in ('erm', 'hardneg'))
            audit = aug['training_augmentation']
            original = base['training']['rows']['rows']
            expected = min(12000, int(.25 * audit['original_binary_negative_n']))
            if (audit['original_rows'] != original or audit['synthetic_rows'] != expected
                    or audit['teacher_synthetic_rows'] != 0 or audit['generator_seed'] != 20261101
                    or audit['template_sha256'] != fingerprint()
                    or aug['training']['rows']['rows'] != original + expected
                    or aug['training']['rows']['with_teacher'] != original):
                raise ValueError('hard-negative training provenance differs from predeclaration')
    return {'folder': folder, 'tree': tree, 'metrics_sha256': digest}



def validated_v1_reference(root, manifest_digest, metrics_digest):
    matches = []
    for path in Path(root).rglob('fusion_v1.json'):
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest_digest:
            continue
        manifest = json.loads(path.read_text())
        receipt = json.loads(path.with_name('results.json').read_text())
        metrics_path = path.with_name('metrics.json')
        if hashlib.sha256(metrics_path.read_bytes()).hexdigest() != metrics_digest:
            raise ValueError('frozen v1 reporting receipt changed')
        metrics = json.loads(metrics_path.read_text())
        if (manifest['schema'] != 'rakshak-fusion-v1' or receipt['status'] != 'ok'
                or receipt.get('failures') or metrics['artifact_gate'] != 'go'
                or metrics['manifest_sha256'] != manifest_digest
                or receipt['tree_sha256'] != manifest['fitting']['tree_sha256']):
            raise ValueError('completed frozen v1 reference required')
        for entry in manifest['artifacts'].values():
            artifact = path.parent/entry['file']
            if (Path(entry['file']).name != entry['file'] or artifact.is_symlink()
                    or hashlib.sha256(artifact.read_bytes()).hexdigest() != entry['sha256']):
                raise ValueError('v1 reference expert changed')
        matches.append({'manifest_sha256': manifest_digest, 'metrics_sha256': metrics_digest,
                        'description': 'newly fitted (not previously shipped)',
                        'fitting': manifest['fitting'], 'external': metrics['external'],
                        'safety_gate': metrics['safety_gate']})
    if len(matches) != 1:
        raise ValueError('one exact immutable v1 reference required')
    return matches[0]

def run(ctx):
    import fusion_transactional as comparison

    p = argparse.ArgumentParser()
    p.add_argument('--source-tree', required=True)
    p.add_argument('--metrics-sha256', required=True)
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--v1-manifest-sha256', required=True)
    p.add_argument('--v1-metrics-sha256', required=True)
    args = p.parse_args(ctx.argv)
    source = validated_source(ctx.input_root, args.source_tree, args.metrics_sha256)
    source['frozen_v1_reference'] = validated_v1_reference(
        ctx.input_root, args.v1_manifest_sha256, args.v1_metrics_sha256)
    ctx.argv = ['--n-boot', str(args.n_boot)]
    return comparison.run(ctx, source_info=source,
                          model_arms={'baseline': 'erm', 'transactional': 'hardneg'},
                          feature_roles={'baseline': False, 'transactional': False},
                          gate=hardneg_gate)
