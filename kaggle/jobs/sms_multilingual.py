"""Predeclared byte / compact multilingual / multilingual-KD, train IDs only."""

import argparse
import hashlib
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from _common import env_report, write_metrics

from rakshak.canonicalize import canonicalize
from rakshak.datasets.common import cached_registered_domain
from rakshak.datasets.dedup import assign_campaigns, mark_exact_duplicates
from rakshak.eval import data as D
from rakshak.eval.clean_kd import FOLDS
from rakshak.eval.multilingual_gate import multilingual_gate
from rakshak.eval.protocol import evaluate_scores
from rakshak.eval.rule_audit import describe_rate
from rakshak.eval.source_comparison import paired_seed_f1
from rakshak.models.distill import (
    build_training_pool,
    find_transfer_file,
    load_transfer,
    verify_manifest,
)
from rakshak.models.features import featurize
from rakshak.models.multilingual_vocab import (
    CARD_SHA256,
    CONFIG_SHA256,
    MAX_LEN,
    MODEL,
    REVISION,
    encode_tokens,
    project_embeddings,
    select_vocabulary,
)

ARMS = ('byte_nokd', 'multilingual_nokd', 'multilingual_kd')
WEIGHT_SHA256 = 'be7c9ce2af947fb5bb17a3bc919552f898c5ef4164bf0a7190cf25a0f98f85b6'
EMBEDDING_SHAPE = (256000, 384)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ids_hash(ids):
    raw = json.dumps(list(map(str, ids)), separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def pretrained(work):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    folder = Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=[
        'pytorch_model.bin', '*.json', 'README.md', 'LICENSE*', '*.model', '*.txt']))
    if sha(folder/'README.md') != CARD_SHA256 or sha(folder/'config.json') != CONFIG_SHA256:
        raise ValueError('pinned MIT model card/config changed')
    tokenizer = AutoTokenizer.from_pretrained(
        folder, local_files_only=True, trust_remote_code=False)
    weight_path = folder/'pytorch_model.bin'
    if sha(weight_path) != WEIGHT_SHA256:
        raise ValueError('pinned pretrained weight hash changed')
    state = torch.load(weight_path, map_location='cpu', weights_only=True)
    matrices = [tensor.float().numpy() for key, tensor in state.items()
                if key.endswith('embeddings.tok_embeddings.weight')]
    if (len(matrices) != 1 or matrices[0].shape != EMBEDDING_SHAPE
            or not np.isfinite(matrices[0]).all()):
        raise ValueError('one exact pretrained input embedding tensor required')
    del state
    source = {'model': MODEL, 'revision': REVISION, 'licence': 'MIT',
              'files': {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
              'upstream_pretraining_overlap': 'unknown', 'transformer_layers_deployed': 0}
    (work/'pretrained_source.json').write_text(json.dumps(source, indent=2)+'\n')
    # Save tokenizer/config/licence separately; classifier-weight limit excludes this footprint.
    tokenizer.save_pretrained(work/'tokenizer')
    (work/'tokenizer'/'SOURCE_README.md').write_bytes((folder/'README.md').read_bytes())
    for file in folder.glob('LICENSE*'):
        (work/'tokenizer'/file.name).write_bytes(file.read_bytes())
    return tokenizer, matrices[0], source


def tokenize(tokenizer, texts, *, full=False):
    return tokenizer([canonicalize(str(t)).normalized for t in texts], add_special_tokens=False,
                     truncation=not full, **({} if full else {'max_length': MAX_LEN}))['input_ids']


def external_frames(root):
    frames = {}
    for name in ('groundlm', 'smishx'):
        frame = pd.read_parquet(root/'external'/f'external_{name}.parquet')
        frame = frame[frame.text_available.fillna(False) & D.eval_mask(frame)].copy()
        frame['reg_domains'] = [sorted({cached_registered_domain(u)
                                       for u in canonicalize(t).urls}) for t in frame.text]
        frame, _ = assign_campaigns(mark_exact_duplicates(frame.reset_index(drop=True)))
        frames[name] = frame
    return frames



def write_source_archive(source_root, job_path, output):
    source_root, job_path = Path(source_root), Path(job_path)
    files = [(file, file.relative_to(source_root).as_posix())
             for file in sorted(source_root.rglob('*.py'))]
    files.append((job_path, 'sms_multilingual.py'))
    if any(file.is_symlink() for file, _ in files):
        raise ValueError('source symlink refused')
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for file, name in files:
            entry = zipfile.ZipInfo(name)
            entry.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(entry, file.read_bytes())


def run(ctx):
    import torch
    from threadpoolctl import threadpool_limits

    from rakshak.models.multilingual import predict, train
    from rakshak.models.student import TrainConfig, state_bytes

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-boot', type=int, default=1000)
    args = parser.parse_args(ctx.argv)
    if args.n_boot != 1000 or not torch.cuda.is_available() or not str(ctx.work).startswith(
            '/kaggle/working'):
        raise ValueError('fixed 1000-resample Kaggle CUDA experiment required')
    torch.set_num_threads(2)
    threadpool_limits(2)
    tokenizer, embeddings, source = pretrained(ctx.work)
    root = D.resolve_data_root()
    sms = D.load_sms(root)
    ids, labels = sms.id.astype(str).to_numpy(), sms.label_binary.to_numpy(dtype=np.int64)
    runs = D.load_runs(root, sms, patterns=['campaign', 'source_holdout', 'lang_shift', 'temporal'],
                       labels=labels, mask=D.eval_mask(sms))
    if {run.name for run in runs} != FOLDS or len(runs) != 6:
        raise ValueError('all six original frozen folds required')
    features = featurize(sms.text.fillna('').tolist())
    folders = {p.parent for p in Path(ctx.input_root).rglob('transfer_*.parquet')}
    if len(folders) != 1:
        raise ValueError('one private cached teacher folder required')
    teacher_folder = folders.pop()
    archive = ctx.work/'inference_source.zip'
    write_source_archive(ctx.src, __file__, archive)
    config = {'inference_source_sha256': sha(archive), 'arms': ARMS, 'seeds': [1, 2, 3],
              'train': asdict(TrainConfig()),
              'pretrained': source, 'env': env_report(), 'n_boot': args.n_boot,
              'teacher': 'single teacher0, exact train IDs only, not OOF/ensemble',
              'selection': 'cal PR-AUC stopping and cal-only thresholds',
              'design_exposure': 'in-sample intervention selection; Indian validation OPEN'}
    results, comparisons, artifacts = {}, {}, {}
    frames = None
    for run_ in runs:
        path = find_transfer_file(teacher_folder, run_.name, 0)
        if verify_manifest(teacher_folder, run_.name, path) is None:
            raise ValueError('explicit teacher source manifest required')
        pool = build_training_pool(run_, ids, labels, load_transfer(path))
        if (not np.array_equal(pool.positions, run_.train) or pool.n_unlabeled
                or not np.isfinite(pool.teacher).all()):
            raise ValueError('complete teacher targets on exactly original train IDs required')
        positions = pool.positions
        if not np.array_equal(pool.labels, labels[positions]):
            raise ValueError('pool labels must equal original ordered training labels')
        train_tokens = tokenize(tokenizer, sms.text.iloc[positions].fillna('').tolist(), full=True)
        vocabulary = select_vocabulary(train_tokens, ids[positions], ids[run_.train])
        table, mean, components = project_embeddings(embeddings[vocabulary])
        stem = run_.name.replace(':', '_')
        vocab_file = ctx.work/f'vocabulary_{stem}.npz'
        np.savez(vocab_file, original_token_ids=vocabulary, table=table, mean=mean,
                 components=components)
        frozen = {'fold': run_.name, 'train_ids_sha256': ids_hash(ids[positions]),
                  'train_ids': ids[positions].tolist(), 'rows': len(positions),
                  'original_synthetic_train_rows': int(sms.synthetic.iloc[positions].sum()),
                  'cal_ids_sha256': ids_hash(ids[run_.cal]),
                  'test_ids_sha256': ids_hash(ids[run_.test_all]),
                  'inference_source_sha256': sha(archive),
                  'vocabulary_sha256': sha(vocab_file), 'source_revision': REVISION,
                  'teacher_sha256': sha(path), 'git_sha': ctx.results['git_sha'],
                  'tree_sha256': ctx.results['tree_sha256']}
        # This exclusive manifest exists BEFORE any cal/test token transformation.
        manifest = ctx.work/f'vocabulary_{stem}.json'
        with manifest.open('x') as stream:
            json.dump(frozen, stream, indent=2)
        tokens = encode_tokens(tokenize(tokenizer, sms.text.fillna('').tolist()), vocabulary)
        results[run_.name] = {'vocabulary': {'file': vocab_file.name, 'sha256': sha(vocab_file),
                                           'selected_tokens': len(vocabulary)}, 'arms': {}}
        predictions = {arm: [] for arm in ARMS}
        for arm in ARMS:
            results[run_.name]['arms'][arm] = {}
            for seed in (1, 2, 3):
                cfg = TrainConfig(seed=seed)
                model, history = train(features, tokens, positions, labels,
                                       pool.teacher if arm == 'multilingual_kd' else None,
                                       run_.cal, cfg, None if arm == 'byte_nokd' else table,
                                       ctx.log)
                model.cuda()
                sc = predict(model, features, tokens, run_.cal)
                # Cal thresholds and selected weights are serialized BEFORE test/external scores.
                from rakshak.eval.metrics import threshold_for_fpr, threshold_max_f1

                threshold, cal_f1 = threshold_max_f1(labels[run_.cal], sc)
                thresholds = {'max_f1': {'threshold': threshold, 'cal_f1': cal_f1},
                              'fpr_1pct': threshold_for_fpr(labels[run_.cal], sc, .01)}
                checkpoint = ctx.work/f'student_{stem}_{arm}_s{seed}.pt'
                model.cpu()
                torch.save({'state_dict': model.state_dict(), 'arm': arm, 'seed': seed,
                            'thresholds': thresholds, 'vocabulary_sha256': sha(vocab_file),
                            'training': frozen, 'config': asdict(cfg)}, checkpoint)
                reloaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if any(not torch.equal(value, reloaded['state_dict'][key])
                       for key, value in model.state_dict().items()):
                    raise ValueError('selected CPU checkpoint serialization changed weights')
                model.load_state_dict(reloaded['state_dict'])
                del reloaded
                entry = {'file': checkpoint.name, 'sha256': sha(checkpoint),
                         'bytes': state_bytes(model), 'checkpoint_bytes': checkpoint.stat().st_size}
                if arm == 'multilingual_kd':
                    artifacts[(run_.name, seed)] = entry
                model.cuda()
                st = predict(model, features, tokens, run_.test)
                metrics = evaluate_scores(labels[run_.cal], sc, labels[run_.test], st,
                                          cluster_ids=sms.campaign_id.iloc[run_.test].astype(str),
                                          groups=D.group_values(sms, run_.test), n_boot=args.n_boot,
                                          seed=13)
                predictions[arm].append(st >= metrics['thresholds']['max_f1']['threshold'])
                name = f'{stem}_{arm}_s{seed}'
                pd.DataFrame({'id': ids[np.r_[run_.cal, run_.test]],
                              'split': ['cal']*len(sc)+['test']*len(st),
                              'score': np.r_[sc, st]}).to_parquet(ctx.work/f'scores_{name}.parquet')
                external = {}
                if run_.name == 'campaign':
                    if frames is None:
                        frames = external_frames(root)
                    for population, frame in frames.items():
                        ext_feats = featurize(frame.text.tolist())
                        ext_tokens = encode_tokens(
                            tokenize(tokenizer, frame.text.tolist()), vocabulary)
                        scores = predict(model, ext_feats, ext_tokens, np.arange(len(frame)))
                        external[population] = external_metrics(frame, scores, metrics, args.n_boot)
                        pd.DataFrame({'id': frame.id, 'score': scores}).to_parquet(
                            ctx.work/f'external_{population}_{arm}_s{seed}.parquet')
                results[run_.name]['arms'][arm][str(seed)] = {
                    'metrics': metrics, 'training': history, 'checkpoint': entry,
                    'external': external}
                del model
                torch.cuda.empty_cache()
                write_metrics('metrics.json', {'config': config, 'runs': results,
                                               'comparisons': comparisons})
        comparisons[run_.name] = {
            arm: paired_seed_f1(labels[run_.test], predictions[arm], predictions['byte_nokd'],
                                sms.campaign_id.iloc[run_.test].astype(str), n_boot=args.n_boot)
            for arm in ARMS[1:]}
    gate = multilingual_gate(comparisons, artifacts)
    write_metrics('metrics.json', {'config': config, 'runs': results, 'comparisons': comparisons,
                                   'multilingual_gate': gate})
    ctx.results['summary'] = f'multilingual candidate {gate["decision"]}; 54 fits, single teacher'
    return 'ok'


def external_metrics(frame, scores, metrics, n_boot):
    output = {}
    for op in ('max_f1', 'fpr_1pct'):
        threshold = metrics['thresholds'][op]['threshold']
        labels = {}
        for label in ('label_binary', 'label_scam'):
            values = frame[label].fillna(-1).to_numpy(dtype=int)
            labels[label] = {}
            for value, endpoint in ((0, 'fpr'), (1, 'recall')):
                mask = values == value
                labels[label][endpoint] = describe_rate((scores[mask] >= threshold).astype(int),
                                                        frame.campaign_id[mask].astype(str),
                                                        n_boot=n_boot)
        output[op] = {'threshold': threshold, 'labels': labels}
    return output
