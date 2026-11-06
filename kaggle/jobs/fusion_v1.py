"""Newly fitted cal-only reference with actual portable experts and immutable manifest."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from _common import env_report, seed_everything, write_metrics
from scipy.special import expit
from threadpoolctl import threadpool_limits

from rakshak.canonicalize import canonicalize
from rakshak.datasets.common import cached_registered_domain
from rakshak.datasets.dedup import assign_campaigns, mark_exact_duplicates
from rakshak.eval import data as D
from rakshak.eval.metrics import threshold_for_fpr
from rakshak.eval.protocol import evaluate_scores
from rakshak.eval.report import sanitize
from rakshak.fusion.audit import baseline_gate, triage_report
from rakshak.fusion.cal_only import fit_cal_only, predict_frozen
from rakshak.fusion.frozen_bundle import (
    artifact,
    load_manifest,
    probability_threshold,
    sha256,
    source_archive,
    write_manifest,
)
from rakshak.fusion.fuse import feature_matrix
from rakshak.fusion.selective import SelectiveThresholds
from rakshak.fusion.url_expert import URL_TREE, FrozenURLExpert, frozen_url_folder
from rakshak.models.distill import (
    build_training_pool,
    find_transfer_file,
    load_transfer,
    verify_manifest,
)
from rakshak.models.features import featurize


def onnx_scores(session, features):
    names = [item.name for item in session.get_inputs()]
    if names != ['raw', 'canon', 'side']:
        raise ValueError('frozen dual/rules ONNX input contract changed')
    scores = []
    for start in range(0, len(features), 256):
        part = features.take(np.arange(start, min(start+256, len(features))))
        feed = {view: part.bytes_view(view).astype(np.int64) for view in ('raw', 'canon')}
        feed['side'] = np.ascontiguousarray(part.side('canon'), dtype=np.float32)
        scores.extend(expit(session.run(None, feed)[0]).reshape(-1).tolist())
    values = np.asarray(scores)
    if values.shape != (len(features),) or not np.isfinite(values).all():
        raise ValueError('invalid frozen student scores')
    return values


def matrix(frame, scores, url):
    texts = frame.text.fillna('').astype(str).tolist()
    url_score, present = url.score(texts)
    return feature_matrix(texts, scores, url_score, has_url=present)


def reports(frame, probability, student_score, threshold, student_threshold, n_boot):
    out = {}
    for name in ('label_scam', 'label_binary'):
        labels = frame[name].fillna(-1).to_numpy(dtype=int)
        valid = np.isin(labels, [0, 1])
        out[name] = triage_report(
            labels[valid], probability[valid], threshold,
            frame.campaign_id[valid].astype(str).tolist(),
            student_pred=student_score[valid] >= student_threshold, n_boot=n_boot)
        out[name]['excluded_unknown_n'] = int((~valid).sum())
    return out


def fit_and_freeze(ctx, args, sms, labels, ids, features, split, references):
    import onnxruntime as ort
    import torch

    from rakshak.models.student import (
        ModelSpec,
        TrainConfig,
        export_onnx,
        onnx_parity,
        save_checkpoint,
        train_student,
    )

    folders = {path.parent for path in Path(ctx.input_root).rglob('transfer_*.parquet')}
    if len(folders) != 1:
        raise ValueError('one private hash-manifested teacher folder required')
    teacher_folder = folders.pop()
    path = find_transfer_file(teacher_folder, split.name, 0)
    if verify_manifest(teacher_folder, split.name, path) is None:
        raise ValueError('teacher manifest required')
    pool = build_training_pool(split, ids, labels, load_transfer(path))
    if not np.array_equal(pool.positions, split.train) or pool.n_unlabeled:
        raise ValueError('exact ordered original train IDs only')
    if not np.isfinite(pool.teacher).all():
        raise ValueError('baseline requires complete finite original-train teacher targets')
    spec, config = ModelSpec('dual', 'base', 'dual'), TrainConfig(seed=1, epochs=15)
    model, history = train_student(
        spec, features.take(pool.positions), pool.labels, features.take(split.cal),
        labels[split.cal], config, device='cuda', teacher=pool.teacher, log=ctx.log)
    onnx = ctx.work/'student.onnx'
    exported = export_onnx(model, onnx)
    parity_parts = [onnx_parity(model, onnx, features.take(split.cal[start:start+256]), n=256)
                    for start in range(0, len(split.cal), 256)]
    parity = {'n': sum(item.get('n', 0) for item in parity_parts),
              'max_abs_logit_diff': max(item.get('max_abs_logit_diff', np.inf)
                                        for item in parity_parts), 'batch_size': 256}
    if (parity.get('n') != len(split.cal)
            or not 0 <= parity.get('max_abs_logit_diff', np.inf) <= 1e-5):
        raise ValueError('cal-only ONNX logit parity gate failed')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    cal = sms.iloc[split.cal]
    cal_scores = onnx_scores(session, features.take(split.cal))
    student_threshold = probability_threshold(
        threshold_for_fpr(labels[split.cal], cal_scores, .01)['threshold'])
    save_checkpoint(ctx.work/'student.pt', model, spec, student_threshold,
                    extra={'run': split.name, 'seed': 1, 'training': history,
                           'train_ids_sha256': hashlib.sha256(
                               json.dumps(ids[pool.positions].tolist()).encode()).hexdigest()})
    del model
    torch.cuda.empty_cache()
    url_folder = frozen_url_folder(ctx.input_root)
    url_receipt = json.loads((url_folder/'results.json').read_text())
    if url_receipt['tree_sha256'] != URL_TREE:
        raise ValueError('URL source receipt changed')
    for original, target in (('best_url_student.onnx', 'url.onnx'),
                              ('url_side_scaler.json', 'url_side_scaler.json')):
        shutil.copyfile(url_folder/original, ctx.work/target)
    url = FrozenURLExpert(url_folder)
    x_cal = matrix(cal, cal_scores, url)
    known = cal.label_scam.isin([0, 1]).to_numpy()
    with threadpool_limits(limits=1):
        state = fit_cal_only(
            x_cal[known], cal.label_scam.to_numpy()[known].astype(int), cal.id.to_numpy()[known],
            cal.campaign_id.astype(str).to_numpy()[known], train_ids=ids[split.train].tolist(),
            test_ids=ids[split.test_all].tolist())
    if state['status'] != 'ok':
        write_metrics('freeze_failure.json', sanitize(state))
        raise ValueError('fixed cal-only baseline infeasible; no external tuning')
    source_archive(ctx.src, ctx.work/'inference_source.zip')
    artifacts = {key: artifact(ctx.work/name) for key, name in {
        'student_checkpoint': 'student.pt', 'student_onnx': 'student.onnx',
        'url_onnx': 'url.onnx', 'url_scaler': 'url_side_scaler.json',
        'inference_source': 'inference_source.zip'}.items()}
    own = {'git_sha': ctx.results['git_sha'], 'tree_sha256': ctx.results['tree_sha256'],
           'dirty': ctx.results['dirty'], 'kernel': args.kernel,
           'kernel_version': args.kernel_version}
    previous = {**own, 'git_sha': url_receipt['git_sha'],
                'tree_sha256': url_receipt['tree_sha256'], 'dirty': url_receipt['dirty'],
                'kernel': 'maadhavvh/rakshak-url-student', 'kernel_version': 1}
    manifest = {
        'schema': 'rakshak-fusion-v1', 'description': 'newly fitted (not previously shipped)',
        'shipment_approved': False, 'alpha': .01, 'beta': .05, 'fusion': state,
        'fitting': own, 'artifacts': artifacts, 'validation_references': references,
        'fold': split.name, 'seed': 1,
        'experts': {'student': {**own, 'artifact_sha256': artifacts['student_onnx']['sha256'],
                                'spec': asdict(spec), 'training': asdict(config)},
                    'url': {**previous, 'artifact_sha256': artifacts['url_onnx']['sha256']},
                    'rules': {**own, 'artifact_sha256': artifacts['inference_source']['sha256'],
                              'kind': 'deterministic source; no learned weights'}},
        'training': {'pool': pool.stats(), 'teacher_sha256': sha256(path),
                     'ids_sha256': hashlib.sha256(
                         json.dumps(ids[pool.positions].tolist()).encode()).hexdigest()},
        'student_cal_1pct_threshold': student_threshold, 'onnx_export': exported,
        'cal_parity': parity, 'environment': env_report(),
        'inference_runtime': {name: importlib.metadata.version(name)
                              for name in ('onnx', 'onnxruntime', 'torch', 'numpy')},
        'policy': {'url': 'max extracted URL probability; NaN absent with explicit presence',
                   'ties': 'SAFE p<=t_safe; SCAM p>=t_scam; UNCERTAIN otherwise',
                   'crc': 'diagnostic only: underlying checkpoint selection reused cal',
                   'external_scope': 'GroundLM rules design and GroundLM/SmishX aggregate '
                                     'feedback in-sample for intervention selection'}}
    manifest_path = ctx.work/'fusion_v1.json'
    frozen_hash = write_manifest(manifest_path, sanitize(manifest))
    loaded = load_manifest(manifest_path, source_root=ctx.src)
    if not np.array_equal(predict_frozen(state, x_cal), predict_frozen(loaded['fusion'], x_cal)):
        raise ValueError('frozen fusion JSON round-trip mismatch')
    ctx.log(f'froze fusion_v1.json SHA256 {frozen_hash} before test/external scoring')
    return {'manifest': loaded, 'sha256': frozen_hash, 'cal_scores': cal_scores,
            'session': session, 'url': url, 'history': history, 'work': ctx.work}


def run(ctx, *, freeze=None, scorer=None):
    from types import SimpleNamespace

    freeze = fit_and_freeze if freeze is None else freeze
    scorer = onnx_scores if scorer is None else scorer

    import torch

    p = argparse.ArgumentParser()
    p.add_argument('--kernel', required=True)
    p.add_argument('--kernel-version', required=True, type=int)
    p.add_argument('--n-boot', type=int, default=1000)
    args = p.parse_args(ctx.argv)
    if args.n_boot != 1000 or args.kernel_version != 1:
        raise ValueError('frozen v1 requires 1000 bootstrap draws and a fresh version1 kernel')
    if not torch.cuda.is_available():
        raise RuntimeError('private Kaggle CUDA training required')
    seed_everything(13)
    root = D.resolve_data_root()
    sms = D.load_sms(root)
    labels = sms.label_binary.to_numpy(dtype=int)
    ids = sms.id.astype(str).to_numpy()
    runs = D.load_runs(root, sms, patterns=['campaign', 'source_holdout', 'lang_shift', 'temporal'],
                       labels=labels, mask=D.eval_mask(sms))
    FOLDS = {'campaign', 'lang_shift', 'temporal', 'source_holdout:holdout_scamshield',
             'source_holdout:holdout_uci_sms_spam',
             'source_holdout:holdout_mishra_soni_smsphishing'}

    if {item.name for item in runs} != FOLDS:
        raise ValueError('all six frozen validation folds required')
    features = featurize(sms.text.fillna('').tolist())
    fitted, references = {}, {}
    for split in sorted(runs, key=lambda item: (item.name == 'campaign', item.name)):
        folder = ctx.work if split.name == 'campaign' else ctx.work/split.name.replace(':', '_')
        folder.mkdir(exist_ok=True)
        child = SimpleNamespace(**vars(ctx))
        child.log = ctx.log
        child.work = folder
        fitted[split.name] = freeze(
            child, args, sms, labels, ids, features, split,
            dict(references) if split.name == 'campaign' else {})
        if split.name != 'campaign':
            references[split.name] = {
                'manifest': str((folder/'fusion_v1.json').relative_to(ctx.work)),
                'sha256': fitted[split.name]['sha256']}
    # All six experts/cal states now frozen; evaluate only the matching fold's model.
    internal, controls, histories = {}, {}, {}
    for split in runs:
        item = fitted[split.name]
        loaded = item['manifest']
        session, url, cal_scores = item['session'], item['url'], item['cal_scores']
        threshold = SelectiveThresholds.from_dict(loaded['fusion']['thresholds'])
        student_threshold = loaded['student_cal_1pct_threshold']
        test = sms.iloc[split.test]
        scores = scorer(session, features.take(split.test))
        fused = predict_frozen(loaded['fusion'], matrix(test, scores, url))
        internal[split.name] = reports(test, fused, scores, threshold,
                                       student_threshold, args.n_boot)
        controls[split.name] = evaluate_scores(
            labels[split.cal], cal_scores, labels[split.test], scores,
            cluster_ids=test.campaign_id.astype(str).tolist(),
            groups=D.group_values(sms, split.test), n_boot=args.n_boot)
        histories[split.name] = item['history']
        pd.DataFrame({'id': test.id, 'student': scores, 'fused': fused,
                      'state': threshold.transform(fused)}).to_parquet(
                          item['work']/'test_v1.parquet')
    primary = fitted['campaign']
    loaded, frozen_hash = primary['manifest'], primary['sha256']
    session, url = primary['session'], primary['url']
    threshold = SelectiveThresholds.from_dict(loaded['fusion']['thresholds'])
    student_threshold = loaded['student_cal_1pct_threshold']
    manifest_path = ctx.work/'fusion_v1.json'
    manifest = loaded
    external = {}
    for name in ('groundlm', 'smishx'):
        frame = pd.read_parquet(root/'external'/f'external_{name}.parquet')
        frame = frame[frame.text_available.fillna(False) & D.eval_mask(frame)].copy()
        frame['reg_domains'] = [sorted({cached_registered_domain(u) for u in canonicalize(t).urls})
                                for t in frame.text]
        frame, _ = assign_campaigns(mark_exact_duplicates(frame.reset_index(drop=True)))
        scores = scorer(session, featurize(frame.text.fillna('').tolist()))
        fused = predict_frozen(loaded['fusion'], matrix(frame, scores, url))
        external[name] = reports(frame, fused, scores, threshold, student_threshold, args.n_boot)
        pd.DataFrame({'id': frame.id, 'student': scores, 'fused': fused,
                      'state': threshold.transform(fused)}).to_parquet(
                          ctx.work/f'external_{name}_v1.parquet')
    if sha256(manifest_path) != frozen_hash:
        raise ValueError('manifest changed after outcomes')
    for item in fitted.values():
        if sha256(item['work']/'fusion_v1.json') != item['sha256']:
            raise ValueError('fold-validation manifest changed after outcomes')
        load_manifest(item['work']/'fusion_v1.json', source_root=ctx.src)
    decision = baseline_gate(external['groundlm']['label_scam'], external['smishx']['label_scam'])
    write_metrics('metrics.json', sanitize({
        'manifest_sha256': frozen_hash, 'artifact_gate': 'go', 'safety_gate': decision,
        'scope': manifest['description'], 'internal': internal,
        'external': external, 'student': controls, 'training': histories,
        'validation_manifests': references,
        'no_external_fit_or_selection': True}))
    ctx.results['summary'] = f'frozen v1 artifact valid; safety {decision["decision"]}; not shipped'
    return True
