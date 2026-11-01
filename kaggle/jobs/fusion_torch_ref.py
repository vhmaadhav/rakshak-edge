"""Exact-checkpoint measurement reference preparation; no ONNX/Arm portability claim."""

import hashlib
import importlib.metadata
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
from _common import env_report
from threadpoolctl import threadpool_limits

from rakshak.fusion.cal_only import fit_cal_only, predict_frozen
from rakshak.fusion.frozen_bundle import (
    artifact,
    load_manifest,
    probability_threshold,
    sha256,
    source_archive,
    write_manifest,
)
from rakshak.fusion.url_expert import URL_TREE, FrozenURLExpert, frozen_url_folder
from rakshak.models.checkpoint_reference import checkpoint_scores, reference_logits
from rakshak.models.distill import (
    build_training_pool,
    find_transfer_file,
    load_transfer,
    verify_manifest,
)


def freeze_checkpoint(ctx, args, sms, labels, ids, features, split, references):
    import torch
    from fusion_v1 import matrix

    from rakshak.eval.metrics import threshold_for_fpr
    from rakshak.models.student import (
        ModelSpec,
        TrainConfig,
        load_checkpoint,
        save_checkpoint,
        train_student,
    )

    torch.set_num_threads(2)
    torch.backends.mkldnn.enabled = True
    folders = {path.parent for path in Path(ctx.input_root).rglob('transfer_*.parquet')}
    if len(folders) != 1:
        raise ValueError('one hash-manifested teacher input required')
    teacher_folder = folders.pop()
    teacher_path = find_transfer_file(teacher_folder, split.name, 0)
    if verify_manifest(teacher_folder, split.name, teacher_path) is None:
        raise ValueError('teacher manifest required')
    pool = build_training_pool(split, ids, labels, load_transfer(teacher_path))
    if (not np.array_equal(pool.positions, split.train) or pool.n_unlabeled
            or not np.isfinite(pool.teacher).all()):
        raise ValueError('exact original train IDs and finite matched teacher targets required')
    spec, config = ModelSpec('dual', 'base', 'dual'), TrainConfig(seed=1, epochs=15)
    model, history = train_student(
        spec, features.take(pool.positions), pool.labels, features.take(split.cal),
        labels[split.cal], config, device='cuda', teacher=pool.teacher, log=ctx.log)
    cal_features = features.take(split.cal)
    before = reference_logits(model, cal_features)
    from scipy.special import expit

    student_threshold = probability_threshold(
        threshold_for_fpr(labels[split.cal], expit(before), .01)['threshold'])
    train_digest = hashlib.sha256(json.dumps(ids[pool.positions].tolist()).encode()).hexdigest()
    target = ctx.work/'student.pt'
    save_checkpoint(target, model.cpu().eval(), spec, student_threshold, extra={
        'run': split.name, 'seed': 1, 'training': history, 'train_ids_sha256': train_digest})
    selected, selected_spec, selected_threshold = load_checkpoint(target)
    if selected_spec != spec or selected_threshold != student_threshold:
        raise ValueError('checkpoint schema/spec/threshold changed after serialization')
    original = model.state_dict()
    restored = selected.state_dict()
    if set(original) != set(restored) or any(
        not torch.equal(value, restored[key]) for key, value in original.items()):
        raise ValueError('checkpoint changed a selected parameter')
    after = reference_logits(selected, cal_features)
    parity = {'n': len(after), 'max_abs_logit_diff': float(np.max(np.abs(before-after))),
              'scope': 'same CPU PyTorch framework checkpoint round-trip, not ONNX equivalence'}
    if parity['n'] != len(split.cal) or parity['max_abs_logit_diff'] > 1e-5:
        raise ValueError('checkpoint fullcal round-trip logit parity gate failed')
    cal_scores = expit(after)
    del model
    torch.cuda.empty_cache()
    url_folder = frozen_url_folder(ctx.input_root)
    url_receipt = json.loads((url_folder/'results.json').read_text())
    if url_receipt['tree_sha256'] != URL_TREE:
        raise ValueError('pinned URL expert changed')
    for source, destination in (('best_url_student.onnx', 'url.onnx'),
                                 ('url_side_scaler.json', 'url_side_scaler.json')):
        shutil.copyfile(url_folder/source, ctx.work/destination)
    url = FrozenURLExpert(url_folder)
    cal = sms.iloc[split.cal]
    x_cal = matrix(cal, cal_scores, url)
    known = cal.label_scam.isin([0, 1]).to_numpy()
    with threadpool_limits(limits=1):
        state = fit_cal_only(
            x_cal[known], cal.label_scam.to_numpy()[known].astype(int), cal.id.to_numpy()[known],
            cal.campaign_id.astype(str).to_numpy()[known], train_ids=ids[split.train].tolist(),
            test_ids=ids[split.test_all].tolist())
    if state['status'] != 'ok':
        raise ValueError('fixed cal-only checkpoint reference infeasible')
    source_archive(ctx.src, ctx.work/'inference_source.zip')
    artifacts = {key: artifact(ctx.work/name) for key, name in {
        'student_checkpoint': 'student.pt', 'url_onnx': 'url.onnx',
        'url_scaler': 'url_side_scaler.json', 'inference_source': 'inference_source.zip'}.items()}
    own = {'git_sha': ctx.results['git_sha'], 'tree_sha256': ctx.results['tree_sha256'],
           'dirty': ctx.results['dirty'], 'kernel': args.kernel,
           'kernel_version': args.kernel_version}
    manifest = {
        'schema': 'rakshak-fusion-v1', 'description': 'newly fitted (not previously shipped)',
        'student_format': 'pytorch_checkpoint', 'measurement_reference_only': True,
        'arm_android_portability': 'UNVERIFIED', 'shipment_approved': False,
        'alpha': .01, 'beta': .05, 'fusion': state, 'fold': split.name, 'seed': 1,
        'fitting': own, 'artifacts': artifacts, 'validation_references': references,
        'experts': {'student': {**own,
                                'artifact_sha256': artifacts['student_checkpoint']['sha256'],
                                'spec': asdict(spec), 'training': asdict(config)},
                    'url': {**own, 'git_sha': url_receipt['git_sha'],
                             'tree_sha256': url_receipt['tree_sha256'],
                             'dirty': url_receipt['dirty'],
                             'kernel': 'maadhavvh/rakshak-url-student', 'kernel_version': 1,
                             'artifact_sha256': artifacts['url_onnx']['sha256']},
                    'rules': {**own, 'artifact_sha256': artifacts['inference_source']['sha256'],
                              'kind': 'deterministic source; no learned weights'}},
        'training': {'pool': pool.stats(), 'teacher_sha256': sha256(teacher_path),
                     'ids_sha256': train_digest},
        'student_cal_1pct_threshold': student_threshold, 'cal_parity': parity,
        'environment': env_report(),
        'inference_runtime': {'torch': importlib.metadata.version('torch'), 'torch_threads': 2,
                              'torch_mkldnn': True, 'device': 'cpu',
                              'onnxruntime_url_only': importlib.metadata.version('onnxruntime')},
        'policy': {'url': 'max extracted URL probability; NaN absent with explicit presence',
                   'ties': 'SAFE p<=t_safe; SCAM p>=t_scam; UNCERTAIN otherwise',
                   'crc': 'diagnostic only: checkpoint selection reused cal',
                   'exposure': 'GroundLM rules design and GroundLM/SmishX feedback in-sample',
                   'hf_publish': 'HF publish pending'}}
    path = ctx.work/'fusion_v1.json'
    digest = write_manifest(path, manifest)
    loaded = load_manifest(path, source_root=ctx.src)
    if not np.array_equal(predict_frozen(state, x_cal), predict_frozen(loaded['fusion'], x_cal)):
        raise ValueError('frozen checkpoint fusion state round-trip mismatch')
    ctx.log(f'froze {split.name} checkpoint measurement reference {digest} before outcomes')
    return {'manifest': loaded, 'sha256': digest, 'cal_scores': cal_scores,
            'session': selected, 'url': url, 'history': history, 'work': ctx.work}


def run(ctx):
    import fusion_v1 as common

    return common.run(ctx, freeze=freeze_checkpoint, scorer=checkpoint_scores)
