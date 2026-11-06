"""Three-seed URL control artifacts/cal-only logits; no test scoring or model selection."""

import hashlib
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from _common import env_report, write_metrics


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ids_hash(ids):
    raw = json.dumps(list(map(str, ids)), separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def train_control(ctx, ds, seed, args):
    import torch
    import url_student as base

    if len(np.unique(ds.fit)) != len(ds.fit) or len(np.unique(ds.cal)) != len(ds.cal):
        raise ValueError('unique fit/cal IDs required')
    if set(ds.fit) & set(ds.cal) or set(ds.fit) & set(ds.test) or set(ds.cal) & set(ds.test):
        raise ValueError('original fit/cal/test row memberships must remain disjoint')
    partitions = [set(map(str, ds.row_ids[rows])) for rows in (ds.fit, ds.cal, ds.test)]
    if any(partitions[a] & partitions[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('frozen fit/cal/test IDs must remain disjoint')
    spec = base.unit_specs(args)[0]
    if len(base.unit_specs(args)) != 1 or spec.kind != 'charcnn' or spec.kd:
        raise ValueError('one non-KD charCNN control required')
    torch.manual_seed(seed)  # construction must not inherit the preceding unit's RNG state
    n_side = ds.side.shape[1]
    model = base.make_model(spec, args, n_side)
    cfg = base.TrainCfg(epochs=8, batch_size=512, lr=.002, weight_decay=.0001,
                        patience=3, seed=seed, kd_alpha=.5, kd_temp=2., amp=True)
    history = base.train_model(model, 'charcnn', ds, cfg, torch.device('cuda'), True,
                               None, None, ctx.log)
    if (history['best_epoch'] < 0 or history['best_cal_pr_auc'] is None
            or not np.isfinite(history['best_cal_pr_auc'])):
        raise ValueError('finite calibration-selected control checkpoint required')
    cal_logits = base.predict_logits(model, 'charcnn', ds.ids, ds.side, ds.cal, 'cuda', True)
    if cal_logits.shape != (len(ds.cal),) or not np.isfinite(cal_logits).all():
        raise ValueError('complete finite original cal logits required')
    stem = f'{ds.name.replace(":", "_")}_s{seed}'
    checkpoint = ctx.work/f'url_control_{stem}.pt'
    state = base._cpu_state(model)
    metadata = {'dataset': ds.name, 'seed': seed, 'config': asdict(cfg),
                'spec': asdict(spec), 'git_sha': ctx.results['git_sha'],
                'tree_sha256': ctx.results['tree_sha256'],
                'input_files_sha256': ctx.results.get('input_files_sha256', {}),
                'fit_ids_sha256': ids_hash(ds.row_ids[ds.fit]),
                'cal_ids_sha256': ids_hash(ds.row_ids[ds.cal]),
                'test_ids_sha256': ids_hash(ds.row_ids[ds.test]),
                'scaler': {'columns': ds.side_cols, 'mean': ds.side_stats[0].tolist(),
                           'std': ds.side_stats[1].tolist(), 'clip': 10., 'missing_as': 0.},
                'selection': 'cal PR-AUC only; no test predictions', 'teacher_enabled': False}
    torch.save({'state_dict': state, 'metadata': metadata}, checkpoint)
    restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if any(not torch.equal(value, restored['state_dict'][key]) for key, value in state.items()):
        raise ValueError('control CPU checkpoint round-trip failed')
    cal_path = ctx.work/f'url_cal_{stem}.parquet'
    pd.DataFrame({'id': ds.row_ids[ds.cal], 'split': 'cal',
                  'logit': cal_logits}).to_parquet(cal_path)
    del model
    torch.cuda.empty_cache()
    return {'checkpoint': {'file': checkpoint.name, 'sha256': sha(checkpoint),
                           'bytes': checkpoint.stat().st_size},
            'cal_logits': {'file': cal_path.name, 'sha256': sha(cal_path), 'n': len(ds.cal)},
            'metadata': metadata, 'training': history}


def run(ctx):
    import torch
    import url_student as base

    if ctx.argv or not torch.cuda.is_available() or not str(ctx.work).startswith('/kaggle/working'):
        raise ValueError('fixed private Kaggle CUDA controls only; no protocol overrides')
    torch.set_num_threads(2)
    args = base.parse_args(['--models', 'charcnn', '--seeds', '1', '2', '3', '--device', 'cuda'])
    root = base.D.resolve_data_root(marker=base.PHRESH_FILE)
    input_paths = [root/base.PHRESH_FILE, root/base.D.URL_FILE]
    input_paths += sorted((root/'splits').glob('url*.json'))
    ctx.results['input_files_sha256'] = {
        path.relative_to(root).as_posix(): sha(path) for path in input_paths}
    datasets = [base.build_phreshphish(root, args,
                                     lambda urls: base.encode_batch(urls, args.max_len))]
    datasets += base.build_run_datasets(root, args,
                                       lambda urls: base.encode_batch(urls, args.max_len), ctx)
    if {dataset.name for dataset in datasets} != {'phreshphish', 'url_domain'}:
        raise ValueError('both original measurable URL datasets required')
    archive = ctx.work/'inference_source.zip'
    files = [(path, path.relative_to(ctx.src).as_posix())
             for path in sorted(Path(ctx.src).rglob('*.py'))]
    files += [(Path(base.__file__), 'url_student.py'), (Path(__file__), 'url_controls_cal.py')]
    if any(path.is_symlink() for path, _ in files):
        raise ValueError('source symlink refused')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path, name in files:
            entry = zipfile.ZipInfo(name)
            entry.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(entry, path.read_bytes())
    results = {}
    for ds in datasets:
        results[ds.name] = {}
        for seed in (1, 2, 3):
            results[ds.name][str(seed)] = train_control(ctx, ds, seed, args)
            write_metrics('cal_controls_manifest.json', {
                'schema': 'rakshak-url-controls-e0', 'complete': False,
                'controls': results, 'no_test_predictions': True, 'shipment_approved': False})
    manifest = {'schema': 'rakshak-url-controls-e0', 'complete': True, 'controls': results,
                'no_test_predictions': True, 'shipment_approved': False,
                'source_archive': {'file': archive.name, 'sha256': sha(archive)},
                'git_sha': ctx.results['git_sha'], 'tree_sha256': ctx.results['tree_sha256'],
                'environment': env_report(), 'hf_publish': 'HF publish pending',
                'url_temporal_status':
                    'historically single-class; not scored in E0; final evaluation must confirm'}
    write_metrics('cal_controls_manifest.json', manifest)
    write_metrics('metrics.json', {'artifact_gate': 'go', 'no_test_outcomes': True,
                                   'manifest_sha256': sha(ctx.work/'cal_controls_manifest.json')})
    ctx.results['summary'] = 'six URL controls frozen; cal logits only, no test outcomes'
    return True
