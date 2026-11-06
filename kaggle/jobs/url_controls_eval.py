"""Frozen six-control URL evaluation; failed teacher never replaced or retrained."""

import argparse
import json

import numpy as np
import pandas as pd
from _common import write_metrics

from rakshak.eval.metrics import binary_metrics, threshold_for_fpr, threshold_max_f1
from rakshak.eval.url_protocol_bootstrap import paired_protocol


def restore_cal(ds, seed, entry, folder, cfg):
    import torch
    import url_student as base
    from url_controls_cal import ids_hash, sha

    metadata = entry['metadata']
    if (metadata['seed'] != seed or metadata['teacher_enabled'] is not False
            or metadata['spec'] != {'kind': 'charcnn', 'key': 'charcnn-wide',
                                    'side': True, 'kd': False}):
        raise ValueError('exact original hard-label control required')
    for part in ('fit', 'cal', 'test'):
        if ids_hash(ds.row_ids[getattr(ds, part)]) != metadata[f'{part}_ids_sha256']:
            raise ValueError('original ordered memberships changed')
    scaler = metadata['scaler']
    if (scaler['columns'] != ds.side_cols
            or not np.array_equal(scaler['mean'], ds.side_stats[0])
            or not np.array_equal(scaler['std'], ds.side_stats[1])):
        raise ValueError('original fit-only side scaler changed')
    file = folder/entry['checkpoint']['file']
    if sha(file) != entry['checkpoint']['sha256']:
        raise ValueError('checkpoint hash changed')
    saved = torch.load(file, map_location='cpu', weights_only=True)
    if saved['metadata'] != metadata:
        raise ValueError('checkpoint metadata changed')
    model = base.make_model(base.Spec(**metadata['spec']), cfg, ds.side.shape[1])
    model.load_state_dict(saved['state_dict'], strict=True)
    model.cuda()
    cache_file = folder/entry['cal_logits']['file']
    if sha(cache_file) != entry['cal_logits']['sha256']:
        raise ValueError('cached calibration hash changed')
    cached = pd.read_parquet(cache_file)
    if (cached.id.astype(str).tolist() != list(map(str, ds.row_ids[ds.cal]))
            or set(cached['split']) != {'cal'} or not np.isfinite(cached.logit).all()):
        raise ValueError('complete ordered finite original calibration scores required')
    logits = base.predict_logits(model, 'charcnn', ds.ids, ds.side, ds.cal, 'cuda', True)
    error = float(np.max(np.abs(logits-cached.logit.to_numpy())))
    if not np.isfinite(error) or error > 1e-5:
        raise ValueError(f'unchanged control full-cal logit parity exceeds1e-5: {error}')
    scores = base.sigmoid(cached.logit.to_numpy())
    thresholds = {'max_f1': threshold_max_f1(ds.labels[ds.cal], scores)[0],
                  'cal_fpr_1pct': threshold_for_fpr(ds.labels[ds.cal], scores, .01)['threshold']}
    return model, scores, {'dataset': ds.name, 'seed': seed, 'thresholds': thresholds,
                           'cal_max_abs_logit_error': error, 'parent': entry}


def run(ctx):
    import torch
    import url_student as base
    from url_controls_cal import sha
    from url_teacher_kd import controls

    parser = argparse.ArgumentParser()
    parser.add_argument('--controls-manifest-sha256', required=True)
    args = parser.parse_args(ctx.argv)
    if not torch.cuda.is_available() or not str(ctx.work).startswith('/kaggle/working'):
        raise ValueError('private Kaggle CUDA evaluation only')
    torch.set_num_threads(2)
    folder, manifest = controls(ctx.input_root, args.controls_manifest_sha256, 'phreshphish')
    other_folder, other = controls(ctx.input_root, args.controls_manifest_sha256, 'url_domain')
    if folder != other_folder or manifest != other:
        raise ValueError('one identical six-control source required')
    cfg = base.parse_args(['--models', 'charcnn', '--seeds', '1', '2', '3', '--device', 'cuda'])
    root = base.D.resolve_data_root(marker=base.PHRESH_FILE)
    files = manifest['controls']['phreshphish']['1']['metadata']['input_files_sha256']
    if any(sha(root/name) != value for name, value in files.items()):
        raise ValueError('frozen input data/split files changed')
    datasets = [base.build_phreshphish(
        root, cfg, lambda urls: base.encode_batch(urls, cfg.max_len))]
    datasets += base.build_run_datasets(root, cfg,
                                        lambda urls: base.encode_batch(urls, cfg.max_len), ctx)
    if {ds.name for ds in datasets} != {'phreshphish', 'url_domain'}:
        raise ValueError('both measurable frozen URL datasets required')
    frozen, models, calibration = {}, {}, {}
    # All six calibration checks/thresholds freeze BEFORE any test prediction.
    for ds in datasets:
        frozen[ds.name], models[ds.name], calibration[ds.name] = {}, {}, {}
        for seed in (1, 2, 3):
            model, cal, record = restore_cal(ds, seed, manifest['controls'][ds.name][str(seed)],
                                             folder, cfg)
            frozen[ds.name][str(seed)] = record
            models[ds.name][seed] = model.cpu()
            calibration[ds.name][seed] = cal
    audit = {'schema': 'rakshak-url-e0-evaluation-v1', 'parent_sha256': sha(
        folder/'cal_controls_manifest.json'), 'controls': frozen,
        'input_files_sha256': files, 'git_sha': ctx.results['git_sha'],
        'tree_sha256': ctx.results['tree_sha256'], 'shipment_approved': False,
        'teacher_status': 'UNMEASURABLE: fixed p1 nonfinite loss; no teacher or KD targets',
        'scope': 'in-sample intervention selection; no SOTA or untouched benchmark claim'}
    with (ctx.work/'evaluation_manifest.json').open('x') as stream:
        json.dump(audit, stream, indent=2)
    frozen_hash = sha(ctx.work/'evaluation_manifest.json')
    result = {}
    for ds in datasets:
        result[ds.name] = {'seeds': {}}
        scores_all = {}
        for seed in (1, 2, 3):
            model = models[ds.name].pop(seed).cuda()
            logits = base.predict_logits(model, 'charcnn', ds.ids, ds.side, ds.test, 'cuda', True)
            scores = base.sigmoid(logits)
            scores_all[f's{seed}'] = scores
            pd.DataFrame({'id': ds.row_ids[ds.test], 'split': 'test', 'logit': logits}).to_parquet(
                ctx.work/f'url_e0_{ds.name}_s{seed}_test.parquet')
            del model
            torch.cuda.empty_cache()
        result[ds.name] = cached_report(ds, scores_all, calibration[ds.name], frozen)
    temporal = temporal_status(root)
    if sha(ctx.work/'evaluation_manifest.json') != frozen_hash:
        raise ValueError('threshold manifest changed after scoring')
    write_metrics('metrics.json', {'controls': result, 'manifest_sha256': frozen_hash,
        'teacher_gate': 'UNMEASURABLE', 'kd_gate': 'UNMEASURABLE',
        'reason': 'fixed Phresh teacher failed nonfinite loss; no selected teacher/targets/KD',
        'url_temporal': temporal,
        'hf_publish': 'HF publish pending', 'shipment_approved': False})
    ctx.results['summary'] = 'six frozen URL controls evaluated; teacher/KD unmeasurable'
    return True


def cached_report(ds, scores_all, calibration, frozen):
    """Original frozen endpoint/subset report; consumes cached scores, no inference."""
    from url_controls_cal import ids_hash

    from rakshak.datasets.common import cached_registered_domain
    from rakshak.eval.phreshphish import protocol_section
    from rakshak.eval.protocol import evaluate_scores

    result = {ds.name: {'seeds': {}}}
    for seed in (1, 2, 3):
        scores = scores_all[f's{seed}']
        thresholds = frozen[ds.name][str(seed)]['thresholds']
        row = {'thresholds': thresholds, 'n_test': len(ds.test),
               'fixed_operating_points': {key: binary_metrics(ds.labels[ds.test], scores, value)
                                          for key, value in thresholds.items()}}
        if ds.protocol is None:
            row['evaluation'] = evaluate_scores(
                ds.labels[ds.cal], calibration[seed], ds.labels[ds.test], scores,
                groups=ds.groups, n_boot=1000, seed=13, cluster_ids=ds.clusters)
        else:
            all_scores = np.full(len(ds.labels), np.nan)
            all_scores[ds.test] = scores
            row['protocol'] = protocol_section(ds.labels, all_scores, ds.test,
                                                 ds.protocol['bench_full'],
                                                 ds.protocol['subsets'])
        result[ds.name]['seeds'][str(seed)] = row
    if ds.protocol is not None:
        # Convert unchanged global positions to exactly scored test positions.
        lookup = np.full(len(ds.labels), -1, dtype=np.int64)
        lookup[ds.test] = np.arange(len(ds.test))
        subsets = [(rate, lookup[rows]) for rate, rows in ds.protocol['subsets']]
        if any((rows < 0).any() for _, rows in subsets):
            raise ValueError('original benchmark subset outside official test IDs')
        # All subsets must retain identical original benign pool; module checks each.
        labels = ds.labels[ds.test]
        benign = subsets[0][1][labels[subsets[0][1]] == 0]
        domains = [cached_registered_domain(str(ds.urls[row])) or str(ds.urls[row])
                   for row in ds.test]
        boot = paired_protocol(labels, scores_all, benign, subsets, domains)
        for rate, block in boot['rates'].items():
            for seed in (1, 2, 3):
                original = result[ds.name]['seeds'][str(seed)]['protocol']['per_rate'][rate]
                actual = block['models'][f's{seed}']
                if (block['n_subsets'] != original['n_subsets']
                        or abs(actual['ap']['point']-original['ap_mean']) > 1e-12
                        or abs(actual['p_at_r90']['point']-original['p_at_r90_mean']) > 1e-12):
                    raise ValueError('optimized subset metrics differ from original protocol')
        result[ds.name]['paired_domain_bootstrap'] = boot
        for name, global_rows in [('full_benchmark', ds.protocol['bench_full']),
                                  ('full_test', ds.test)]:
            rows = lookup[global_rows]
            if (rows < 0).any():
                raise ValueError('full benchmark outside original scored test IDs')
            result[ds.name][name+'_bootstrap'] = paired_protocol(
                labels, scores_all, rows[labels[rows] == 0], [(name, rows)], domains)
        result[ds.name]['test_ids_sha256'] = ids_hash(ds.row_ids[ds.test])
    return result[ds.name]


def temporal_status(root):
    """Explicit unavailable status; never fabricate a missing frozen temporal split."""
    from url_controls_cal import ids_hash

    from rakshak.eval import data as D

    if not D.list_split_files(root, ['url_temporal']):
        return {'status': 'UNMEASURABLE: url_temporal file absent from frozen inputs',
                'n': None, 'positives': None}
    urls = D.load_urls(root)
    runs = D.load_runs(root, urls, patterns=['url_temporal'],
                       labels=urls.label.to_numpy(np.int64), mask=None)
    if len(runs) != 1:
        raise ValueError('exact original url_temporal diagnostic required')
    temporal = runs[0]
    labels = urls.label.to_numpy(np.int64)[temporal.test]
    if len(np.unique(labels)) != 1:
        raise ValueError('historical single-class temporal status changed; report requires review')
    return {'status': 'NA: original test is single-class', 'n': len(labels),
            'positives': int(labels.sum()),
            'test_ids_sha256': ids_hash(urls.id.to_numpy()[temporal.test])}
