"""Last P1 cal-only family arm on all three frozen clean no-KD checkpoint seeds."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from _common import write_metrics
from fusion_v1 import matrix
from threadpoolctl import threadpool_limits

from rakshak.canonicalize import canonicalize
from rakshak.datasets.common import cached_registered_domain
from rakshak.datasets.dedup import assign_campaigns, mark_exact_duplicates
from rakshak.eval import data as D
from rakshak.eval.audit import paired_rate_test
from rakshak.eval.report import sanitize
from rakshak.fusion.cal_only import fit_cal_only, predict_frozen
from rakshak.fusion.family_audit import family_gate, family_triage
from rakshak.fusion.family_cal import fit_family_calibration, predict_family_calibration
from rakshak.fusion.frozen_bundle import artifact, load_manifest, sha256, source_archive
from rakshak.fusion.hardneg_gate import FOLDS
from rakshak.fusion.selective import SelectiveThresholds
from rakshak.fusion.url_expert import FrozenURLExpert, frozen_url_folder
from rakshak.models.checkpoint_reference import checkpoint_scores
from rakshak.models.features import featurize

SOURCE_TREE = '4d8d927781cdfc90d92b1aad05413ba91b2676160e640e3b1f9b42b648052cb7'
SOURCE_METRICS = '708b0fdc6e7b1646717c0bc572a56105100884c1eaaab6e44b47e8693cc8dcde'


def clean_source(root):
    found = []
    for path in Path(root).rglob('results.json'):
        receipt = json.loads(path.read_text())
        if receipt.get('tree_sha256') != SOURCE_TREE:
            continue
        if (receipt['status'] != 'ok' or receipt.get('failures')
                or receipt['job'] != 'sms_clean_kd' or sha256(path.with_name('metrics.json'))
                != SOURCE_METRICS):
            raise ValueError('complete exact clean no-KD source required')
        metrics = json.loads(path.with_name('metrics.json').read_text())
        if (set(metrics['runs']) != FOLDS or metrics['skipped']
                or metrics['config']['teacher_arms'] != {'erm': False, 'kd': True}
                or metrics['config']['args'] != {'epochs': 15, 'n_boot': 1000,
                                                 'seeds': [1, 2, 3], 'teacher_seed': 0}):
            raise ValueError('predeclared three-seed/six-fold clean cohort required')
        for fold, block in metrics['runs'].items():
            if set(block['arms']['erm']) != {'1', '2', '3'}:
                raise ValueError('all three no-KD checkpoints required')
            for seed, info in block['arms']['erm'].items():
                entry = info['checkpoint']
                name = f'student_{fold.replace(":", "_")}_erm_s{seed}.pt'
                if (entry['file'] != name or (path.parent/name).is_symlink()
                        or sha256(path.parent/name) != entry['sha256']
                        or info['teacher_enabled'] is not False
                        or info['training']['rows']['with_teacher'] != 0
                        or info['training']['rows']['unlabeled'] != 0):
                    raise ValueError('clean no-KD checkpoint hash/teacher/training scope changed')
        found.append((path.parent, metrics, receipt))
    if len(found) != 1:
        raise ValueError('one exact completed clean cohort required')
    return found[0]


def parent_v1(root, digest, metrics_hash, source_root):
    found = []
    for path in Path(root).rglob('fusion_v1.json'):
        if sha256(path) != digest:
            continue
        manifest = load_manifest(path, source_root=source_root)
        if sha256(path.with_name('metrics.json')) != metrics_hash:
            raise ValueError('immutable v1 reference metrics changed')
        receipt = json.loads(path.with_name('results.json').read_text())
        metrics = json.loads(path.with_name('metrics.json').read_text())
        if (receipt['status'] != 'ok' or receipt.get('failures')
                or metrics['artifact_gate'] != 'go' or metrics['manifest_sha256'] != digest):
            raise ValueError('complete artifact-valid v1 reference required')
        found.append({'manifest_sha256': digest, 'metrics_sha256': metrics_hash,
                      'fitting': manifest['fitting'], 'external': metrics['external'],
                      'safety_gate': metrics['safety_gate']})
    if len(found) != 1:
        raise ValueError('one exact immutable v1 reference required')
    return found[0]


def aligned_cal(path, ids):
    frame = pd.read_parquet(path, filters=[('split', '==', 'cal')])
    if (not frame.id.is_unique or set(frame.id.astype(str)) != set(ids)
            or set(frame['split']) != {'cal'}):
        raise ValueError('exact original cached cal IDs only required')
    frame.id = frame.id.astype(str)
    scores = frame.set_index('id').reindex(ids).score.to_numpy()
    if not np.isfinite(scores).all():
        raise ValueError('finite original cal scores required')
    return scores


def endpoint_reports(frame, probabilities, decisions, student, policy_hash, n_boot):
    reports = {}
    for label in ('label_binary', 'label_scam'):
        values = frame[label].fillna(-1).to_numpy(dtype=int)
        known = np.isin(values, [0, 1])
        reports[label] = family_triage(
            values[known], probabilities[known], decisions[known],
            frame.campaign_id.to_numpy()[known], student[known], policy_hash, n_boot=n_boot)
        reports[label]['excluded_unknown_n'] = int((~known).sum())
    return reports


def comparisons(frame, baseline, candidate, n_boot):
    result = {}
    for label in ('label_binary', 'label_scam'):
        values = frame[label].fillna(-1).to_numpy(dtype=int)
        result[label] = {}
        for name, target, positive in (('fpr_scam', 'SCAM', 0), ('recall_scam', 'SCAM', 1),
                                         ('fpr_alert', 'alert', 0), ('recall_alert', 'alert', 1)):
            mask = values == positive
            a = candidate == 'SCAM' if target == 'SCAM' else candidate != 'SAFE'
            b = baseline == 'SCAM' if target == 'SCAM' else baseline != 'SAFE'
            result[label][name] = paired_rate_test(
                a[mask].astype(int), b[mask].astype(int),
                frame.campaign_id.to_numpy()[mask].astype(str), n_boot=n_boot)
    return result


def run(ctx):
    import torch

    from rakshak.models.student import load_checkpoint

    p = argparse.ArgumentParser()
    p.add_argument('--v1-manifest-sha256', required=True)
    p.add_argument('--v1-metrics-sha256', required=True)
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--kernel', required=True)
    args = p.parse_args(ctx.argv)
    if args.n_boot != 1000 or not torch.cuda.is_available():
        raise ValueError('Kaggle T4 and fixed 1000 campaign resamples required')
    torch.set_num_threads(2)
    source, meta, receipt = clean_source(ctx.input_root)
    parent = parent_v1(ctx.input_root, args.v1_manifest_sha256,
                       args.v1_metrics_sha256, ctx.src)
    root = D.resolve_data_root()
    sms = D.load_sms(root)
    ids = sms.id.astype(str).to_numpy()
    labels = sms.label_binary.to_numpy(dtype=int)
    runs = D.load_runs(root, sms, patterns=['campaign', 'source_holdout', 'lang_shift', 'temporal'],
                       labels=labels, mask=D.eval_mask(sms))
    if {split.name for split in runs} != FOLDS:
        raise ValueError('all six original folds required')
    features = featurize(sms.text.fillna('').tolist())
    url_folder = frozen_url_folder(ctx.input_root)
    url = FrozenURLExpert(url_folder)
    states, models, student_thresholds, provenance = {}, {}, {}, {}
    # Every selected model and all18 cal states freeze before any new outcome decision.
    for split in runs:
        states[split.name], models[split.name] = {}, {}
        student_thresholds[split.name], provenance[split.name] = {}, {}
        cal = sms.iloc[split.cal]
        for seed in ('1', '2', '3'):
            info = meta['runs'][split.name]['arms']['erm'][seed]
            original = source/info['checkpoint']['file']
            blob = torch.load(original, map_location='cpu', weights_only=False)
            extra = blob['extra']
            if (extra['run'] != split.name or extra['arm'] != 'erm' or extra['seed'] != int(seed)
                    or extra['teacher_enabled'] is not False
                    or extra['git_sha'] != receipt['git_sha']
                    or extra['tree_sha256'] != SOURCE_TREE):
                raise ValueError('checkpoint internal source/fold/arm/seed provenance changed')
            model, spec, threshold = load_checkpoint(original)
            if ((spec.kind, spec.size, spec.variant) != ('dual', 'base', 'dual')
                    or not spec.use_rules):
                raise ValueError('predeclared dual/base student required')
            scores = checkpoint_scores(model, features.take(split.cal))
            cached_path = source/f"scores_{split.name.replace(':', '_')}_erm_s{seed}.parquet"
            cached = aligned_cal(cached_path, ids[split.cal])
            difference = float(np.max(np.abs(scores-cached)))
            if not 0 <= difference <= 1e-5:
                raise ValueError('full cal checkpoint/source probability parity failed')
            expected_threshold = info['metrics']['thresholds']['fpr_1pct']['threshold']
            if threshold != expected_threshold:
                raise ValueError('student threshold differs from original frozen cal choice')
            x = matrix(cal, scores, url)
            known = cal.label_scam.isin([0, 1]).to_numpy()
            with threadpool_limits(limits=1):
                base = fit_cal_only(x[known], cal.label_scam.to_numpy()[known].astype(int),
                                    ids[split.cal][known], cal.campaign_id.to_numpy()[known],
                                    train_ids=ids[split.train], test_ids=ids[split.test_all])
                if base['status'] != 'ok':
                    write_metrics('freeze_failure.json', sanitize({
                        'fold': split.name, 'seed': seed, 'state': base}))
                    raise ValueError('global cal baseline infeasible; no outcome evaluation')
                family = fit_family_calibration(
                    base, x[known], cal.label_scam.to_numpy()[known].astype(int),
                    ids[split.cal][known], cal.text.fillna('').to_numpy()[known])
            states[split.name][seed] = {'baseline': base, 'family': {'status': 'ok', **family}}
            models[split.name][seed] = model
            student_thresholds[split.name][seed] = threshold
            target = ctx.work/original.name
            shutil.copyfile(original, target)
            provenance[split.name][seed] = {
                'checkpoint': artifact(target), 'source': extra,
                'max_cal_probability_diff': difference,
                'student_cal_1pct_threshold': threshold, 'cal_rows': len(cal),
                'train_ids_sha256': hashlib.sha256(json.dumps(ids[split.train].tolist()).encode())
                .hexdigest()}
    source_archive(ctx.src, ctx.work/'inference_source.zip')
    for src, dst in (('best_url_student.onnx', 'url.onnx'),
                       ('url_side_scaler.json', 'url_side_scaler.json')):
        shutil.copyfile(url_folder/src, ctx.work/dst)
    manifest = {'schema': 'rakshak-template-cal-c1',
                'description': 'newly fitted (not previously shipped)',
                'shipment_approved': False, 'arm_android_portability': 'UNVERIFIED',
                'source_tree': SOURCE_TREE, 'source_metrics_sha256': SOURCE_METRICS,
                'source_kernel': 'maadhavvh/rakshak-sms-clean-kd-kd1-20261101',
                'source_kernel_version': 1, 'fitting_kernel': args.kernel,
                'fitting_kernel_version': 1,
                'git_sha': ctx.results['git_sha'], 'tree_sha256': ctx.results['tree_sha256'],
                'parent_v1': parent, 'states': states, 'student_provenance': provenance,
                'other_artifacts': {name: artifact(ctx.work/name) for name in (
                    'url.onnx', 'url_side_scaler.json', 'inference_source.zip')},
                'scope': 'in-sample intervention selection; no independent Indian validation',
                'hf_publish': 'HF publish pending'}
    frozen = ctx.work/'frozen_template_cal.json'
    with frozen.open('x') as stream:
        json.dump(sanitize(manifest), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    frozen_hash = sha256(frozen)
    ctx.log(f'froze all18 model/role/family states {frozen_hash} before outcomes')
    internal, external, paired_internal, paired_external = {}, {}, {}, {}

    def evaluate(frame, fold, population):
        roles, pairs = {}, {}
        feats = featurize(frame.text.fillna('').tolist())
        for seed in ('1', '2', '3'):
            scores = checkpoint_scores(models[fold][seed], feats)
            x = matrix(frame, scores, url)
            student = scores >= student_thresholds[fold][seed]
            base = states[fold][seed]['baseline']
            probability = predict_frozen(base, x)
            baseline = SelectiveThresholds.from_dict(base['thresholds']).transform(probability)
            candidate_prob, candidate, families = predict_family_calibration(
                states[fold][seed]['family'], x, frame.text.fillna('').tolist())
            roles[seed] = {'baseline': endpoint_reports(
                frame, probability, baseline, student, frozen_hash, args.n_boot),
                'family': endpoint_reports(frame, candidate_prob, candidate, student,
                                            frozen_hash, args.n_boot)}
            pairs[seed] = comparisons(frame, baseline, candidate, args.n_boot)
            pd.DataFrame({'id': frame.id.to_numpy(), 'baseline_probability': probability,
                          'family_probability': candidate_prob, 'baseline_state': baseline,
                          'family_state': candidate, 'family': families}).to_parquet(
                              ctx.work/f"scores_{population.replace(':', '_')}_s{seed}.parquet")
        return roles, pairs
    for split in runs:
        internal[split.name], paired_internal[split.name] = evaluate(
            sms.iloc[split.test], split.name, split.name)
    for name in ('groundlm', 'smishx'):
        frame = pd.read_parquet(root/'external'/f'external_{name}.parquet')
        frame = frame[frame.text_available.fillna(False) & D.eval_mask(frame)].copy()
        frame['reg_domains'] = [sorted({cached_registered_domain(u) for u in canonicalize(t).urls})
                                for t in frame.text]
        frame, _ = assign_campaigns(mark_exact_duplicates(frame.reset_index(drop=True)))
        external[name], paired_external[name] = evaluate(frame, 'campaign', 'external_'+name)
    if sha256(frozen) != frozen_hash:
        raise ValueError('family policy changed after outcomes')
    decision = family_gate(internal, external, states)
    write_metrics('metrics.json', sanitize({'cohort': 'template-cal-c1', 'decision': decision,
                                           'manifest_sha256': frozen_hash, 'parent_v1': parent,
                                           'internal': internal, 'external': external,
                                           'paired_internal': paired_internal,
                                           'paired_external': paired_external}))
    ctx.results['summary'] = 'template-cal-c1 '+decision['decision']
    return True
