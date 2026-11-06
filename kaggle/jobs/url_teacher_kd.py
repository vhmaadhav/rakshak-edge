"""Fixed ByT5 teacher, cal-only eligibility, then original-fit-ID KD. No test scoring."""

import argparse
import json
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from _common import write_metrics

from rakshak.eval.url_teacher_cal import eligibility, paired_cal_ap
from rakshak.models.url_byt5 import CARD_SHA, CONFIG_SHA, MODEL, REVISION, WEIGHT_SHA

CONTROL_TREE = 'ebcc1bb75c6e9668a789928946c7247e66b560305d5e62e38598fe7d675f715b'


def controls(root, expected_manifest, dataset):
    from url_controls_cal import sha

    found = []
    for file in Path(root).rglob('cal_controls_manifest.json'):
        if sha(file) != expected_manifest:
            continue
        manifest = json.loads(file.read_text())
        receipt = json.loads(file.with_name('results.json').read_text())
        metrics = json.loads(file.with_name('metrics.json').read_text())
        if (receipt['status'] != 'ok' or receipt.get('failures')
                or receipt['tree_sha256'] != CONTROL_TREE
                or receipt['job'] != 'url_controls_cal' or manifest['complete'] is not True
                or manifest['no_test_predictions'] is not True
                or metrics['no_test_outcomes'] is not True
                or metrics['artifact_gate'] != 'go' or metrics['manifest_sha256'] != sha(file)
                or set(manifest['controls']) != {'phreshphish', 'url_domain'}
                or any(set(seeds) != {'1', '2', '3'} for seeds in manifest['controls'].values())):
            raise ValueError('exact completed six-control cal-only source required')
        for block in manifest['controls'][dataset].values():
            for key in ('checkpoint', 'cal_logits'):
                entry = block[key]
                name = entry['file']
                if (Path(name).name != name or (file.parent/name).is_symlink()
                        or sha(file.parent/name) != entry['sha256']):
                    raise ValueError('control artifact hash/path changed')
        found.append((file.parent, manifest))
    if len(found) != 1:
        raise ValueError('one exact original control manifest required')
    return found[0]


def load_pretrained(n_side, work):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import T5Config
    from url_controls_cal import sha

    from rakshak.models.url_byt5 import URLByT5

    folder = Path(snapshot_download(
        MODEL, revision=REVISION,
        allow_patterns=['pytorch_model.bin', 'config.json', 'README.md']))
    for file, expected in [('README.md', CARD_SHA), ('config.json', CONFIG_SHA),
                            ('pytorch_model.bin', WEIGHT_SHA)]:
        if sha(folder/file) != expected:
            raise ValueError('pinned Apache-2.0 pretrained source hash changed')
    config = json.loads((folder/'config.json').read_text())
    torch.manual_seed(0)
    model = URLByT5(T5Config.from_dict(config), n_side)
    state = torch.load(folder/'pytorch_model.bin', map_location='cpu', weights_only=True)
    model.load_encoder(state)
    del state
    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False})
    source = {'model': MODEL, 'revision': REVISION, 'licence': 'Apache-2.0',
              'README_sha256': CARD_SHA, 'config_sha256': CONFIG_SHA, 'weights_sha256': WEIGHT_SHA,
              'decoder_executed': False, 'upstream_pretraining_overlap': 'unknown'}
    (work/'PRETRAINED_README.md').write_bytes((folder/'README.md').read_bytes())
    return model, config, source


def train_teacher(model, ds, teacher_ids, deadline, log):
    import torch
    from url_student import _cpu_state

    from rakshak.eval.metrics import pr_auc
    from rakshak.models.url_byt5 import predict

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda')
    weight = torch.tensor(float((ds.labels[ds.fit] == 0).sum()/(ds.labels[ds.fit] == 1).sum()),
                          dtype=torch.float32, device='cuda')
    best, state, stale, history = -np.inf, None, 0, []
    for epoch in range(8):
        model.train()
        order = rng.permutation(ds.fit)
        total_loss, updates, skipped = 0., 0, 0
        for update_start in range(0, len(order), 128):
            update_rows = order[update_start:update_start+128]
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(update_rows), 16):
                if time.monotonic() >= deadline:
                    raise RuntimeError('fixed 11-hour budget exhausted; incomplete experiment')
                rows = update_rows[start:start+16]
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(torch.as_tensor(teacher_ids[rows], device='cuda'),
                                   torch.as_tensor(ds.side[rows], device='cuda'))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.float(),
                    torch.as_tensor(ds.labels[rows], dtype=torch.float32, device='cuda'),
                    pos_weight=weight, reduction='sum')
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite FP32 teacher loss; stop fixed arm')
                scaler.scale(loss/len(update_rows)).backward()
                total_loss += float(loss.detach())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                skipped += 1
            else:
                updates += 1
        optimizer.zero_grad(set_to_none=True)
        scores = predict(model, teacher_ids, ds.side, ds.cal)
        ap = float(pr_auc(ds.labels[ds.cal], scores))
        history.append({'epoch': epoch, 'cal_ap': ap, 'train_loss': total_loss/len(ds.fit),
                        'optimizer_updates': updates, 'skipped_amp_updates': skipped})
        log(f'teacher epoch{epoch}: cal AP{ap:.6f}')
        if np.isfinite(ap) and ap > best:
            best, state, stale = ap, _cpu_state(model), 0
        else:
            stale += 1
            if stale >= 3:
                break
    if state is None or not sum(item['optimizer_updates'] for item in history):
        raise ValueError('no finite cal-selected teacher checkpoint')
    model.load_state_dict(state)
    return model.cpu(), {'history': history, 'best_cal_ap': best,
                         'selection': 'natural-cal AP', 'seed': 0}


def save_model(model, file, metadata):
    import torch
    from url_controls_cal import sha
    from url_student import _cpu_state

    state = _cpu_state(model)
    torch.save({'state_dict': state, 'metadata': metadata}, file)
    reloaded = torch.load(file, map_location='cpu', weights_only=True)
    if any(not torch.equal(value, reloaded['state_dict'][key]) for key, value in state.items()):
        raise ValueError('exact CPU serialization failed')
    return {'file': file.name, 'sha256': sha(file), 'bytes': file.stat().st_size}


def run(ctx):
    import torch
    import url_student as base
    from url_controls_cal import ids_hash, sha

    from rakshak.datasets.common import cached_registered_domain
    from rakshak.models.url_byt5 import encode_urls, predict

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['phreshphish', 'url_domain'], required=True)
    parser.add_argument('--controls-manifest-sha256', required=True)
    args = parser.parse_args(ctx.argv)
    if not torch.cuda.is_available() or not str(ctx.work).startswith('/kaggle/working'):
        raise ValueError('Kaggle CUDA only')
    torch.set_num_threads(2)
    deadline = time.monotonic()+11*3600
    folder, control = controls(ctx.input_root, args.controls_manifest_sha256, args.dataset)
    cfg = base.parse_args(['--models', 'charcnn', '--seeds', '1', '2', '3', '--device', 'cuda'])
    root = base.D.resolve_data_root(marker=base.PHRESH_FILE)
    entries = control['controls'][args.dataset]
    source_hashes = entries['1']['metadata']['input_files_sha256']
    if any(sha(root/name) != expected for name, expected in source_hashes.items()):
        raise ValueError('original input data/split hashes changed')
    ds = (base.build_phreshphish(root, cfg, lambda urls: base.encode_batch(urls, cfg.max_len))
          if args.dataset == 'phreshphish' else base.build_run_datasets(
              root, cfg, lambda urls: base.encode_batch(urls, cfg.max_len), ctx)[0])
    if ds.name != args.dataset:
        raise ValueError('original dataset identity changed')
    archive = ctx.work/'inference_source.zip'
    files = [(path, path.relative_to(ctx.src).as_posix())
             for path in sorted(Path(ctx.src).rglob('*.py'))]
    files += [(Path(base.__file__), 'url_student.py'), (Path(__file__), 'url_teacher_kd.py'),
              (Path(__file__).with_name('url_controls_cal.py'), 'url_controls_cal.py')]
    if any(path.is_symlink() for path, _ in files):
        raise ValueError('source symlink refused')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path, name in files:
            entry = zipfile.ZipInfo(name)
            entry.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(entry, path.read_bytes())
    students = []
    for seed in ('1', '2', '3'):
        metadata = entries[seed]['metadata']
        if any(ids_hash(ds.row_ids[getattr(ds, part)]) != metadata[f'{part}_ids_sha256']
               for part in ('fit', 'cal', 'test')):
            raise ValueError('original fit/cal/test IDs changed')
        cached = pd.read_parquet(folder/entries[seed]['cal_logits']['file'])
        if (cached.id.astype(str).tolist() != list(map(str, ds.row_ids[ds.cal]))
                or set(cached['split']) != {'cal'} or not np.isfinite(cached.logit).all()):
            raise ValueError('complete ordered original cal scores required')
        students.append(cached.logit.to_numpy())
    teacher_ids = encode_urls(ds.urls)
    model, architecture, source = load_pretrained(ds.side.shape[1], ctx.work)
    model, history = train_teacher(model, ds, teacher_ids, deadline, ctx.log)
    common = {key: entries['1']['metadata'][key] for key in (
        'dataset', 'scaler', 'fit_ids_sha256', 'cal_ids_sha256', 'test_ids_sha256')}
    common.update({'parent_controls_sha256': args.controls_manifest_sha256,
              'dataset': args.dataset, 'input_files_sha256': source_hashes,
              'git_sha': ctx.results['git_sha'], 'tree_sha256': ctx.results['tree_sha256'],
              'inference_source_sha256': sha(archive)})
    metadata = {**common, 'role': 'teacher', 'seed': 0, 'source': source,
                'architecture': architecture, 'training': history,
                'config': {'seed': 0, 'lr': 5e-5, 'weight_decay': .01, 'batch_size': 16,
                           'accumulation': 8, 'epochs': 8, 'patience': 3, 'clip': 1.,
                           'amp': 'fp16', 'gradient_checkpointing': True,
                           'pooling': 'masked_mean', 'head_hidden': 128, 'max_bytes': 200},
                'teacher_enabled': False, 'no_test_predictions': True}
    teacher = save_model(model, ctx.work/'url_teacher.pt', metadata)
    model.cuda()
    cal = predict(model, teacher_ids, ds.side, ds.cal)
    pd.DataFrame({'id': ds.row_ids[ds.cal], 'split': 'cal', 'logit': cal}).to_parquet(
        ctx.work/'teacher_cal.parquet')
    domains = [cached_registered_domain(str(ds.urls[row])) or str(ds.urls[row]) for row in ds.cal]
    rates = (.0005, .001) if args.dataset == 'phreshphish' else (None,)
    blocks = {str(rate) if rate is not None else 'natural': paired_cal_ap(
        ds.labels[ds.cal], cal, students, domains, prevalence=rate) for rate in rates}
    gate = eligibility(args.dataset, blocks)
    frozen = {'schema': 'rakshak-url-teacher-cal-eligibility-v1', 'dataset': args.dataset,
              'teacher': teacher, 'teacher_metadata': metadata, 'cal_comparison': blocks,
              'eligibility': gate, 'parent_controls_sha256': args.controls_manifest_sha256,
              'cal_ids_sha256': ids_hash(ds.row_ids[ds.cal]),
              'cal_scores_sha256': sha(ctx.work/'teacher_cal.parquet'),
              'no_test_predictions': True, 'shipment_approved': False}
    with (ctx.work/'eligibility.json').open('x') as stream:
        json.dump(frozen, stream, indent=2)
    frozen_hash = sha(ctx.work/'eligibility.json')
    students_kd = {}
    if gate['decision'] == 'go':
        targets = predict(model, teacher_ids, ds.side, ds.fit)
        target_file = ctx.work/'train_teacher_logits.parquet'
        pd.DataFrame({'id': ds.row_ids[ds.fit], 'split': 'train', 'logit': targets}).to_parquet(
            target_file)
        model.cpu()
        torch.cuda.empty_cache()
        for seed in (1, 2, 3):
            torch.manual_seed(seed)
            spec = base.Spec('charcnn', 'charcnn-wide-kd', True, True)
            student = base.make_model(spec, cfg, ds.side.shape[1])
            train_cfg = base.TrainCfg(epochs=8, batch_size=512, lr=.002, weight_decay=.0001,
                                      patience=3, seed=seed, kd_alpha=.5, kd_temp=2., amp=True)
            training = base.train_model(student, 'charcnn', ds, train_cfg, torch.device('cuda'),
                                        True, deadline, targets, ctx.log)
            if training['stopped'] == 'budget' or training['best_epoch'] < 0:
                raise RuntimeError('incomplete fixed KD fit; no outcome scoring')
            artifact = save_model(student, ctx.work/f'url_kd_s{seed}.pt', {
                **common, 'role': 'student_kd', 'seed': seed, 'training': training,
                'config': asdict(train_cfg), 'spec': asdict(spec),
                'eligibility_sha256': frozen_hash, 'train_targets_sha256': sha(target_file),
                'teacher_enabled': True, 'teacher_scope': 'single teacher, fit IDs only, not OOF'})
            logits = base.predict_logits(student, 'charcnn', ds.ids, ds.side, ds.cal, 'cuda', True)
            cal_file = ctx.work/f'kd_cal_s{seed}.parquet'
            pd.DataFrame({'id': ds.row_ids[ds.cal], 'split': 'cal', 'logit': logits}).to_parquet(
                cal_file)
            students_kd[str(seed)] = {'checkpoint': artifact, 'cal_logits_sha256': sha(cal_file)}
            del student
            torch.cuda.empty_cache()
    if sha(ctx.work/'eligibility.json') != frozen_hash:
        raise ValueError('eligibility changed after target creation')
    write_metrics('metrics.json', {'eligibility': gate, 'cal_comparison': blocks,
                                   'teacher': teacher, 'kd': students_kd,
                                   'eligibility_sha256': frozen_hash, 'no_test_outcomes': True,
                                   'scientific_performance_gate': 'not evaluated',
                                   'hf_publish': 'HF publish pending'})
    ctx.results['summary'] = f'{args.dataset} teacher cal eligibility {gate["decision"]}; no tests'
    return True
