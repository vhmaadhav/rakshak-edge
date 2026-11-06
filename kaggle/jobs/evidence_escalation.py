"""Decision-only audit of a frozen-v1 evidence-corrobation policy, Kaggle only."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from _common import write_metrics

from rakshak.canonicalize import canonicalize
from rakshak.datasets.common import cached_registered_domain
from rakshak.datasets.dedup import assign_campaigns, mark_exact_duplicates
from rakshak.eval import data as D
from rakshak.eval.metrics import threshold_for_fpr
from rakshak.eval.report import sanitize
from rakshak.fusion.escalation import apply_escalation, corroboration
from rakshak.fusion.escalation_audit import FOLDS, escalation_gate, state_report
from rakshak.fusion.frozen_bundle import load_manifest, probability_threshold, sha256
from rakshak.fusion.selective import SelectiveThresholds
from rakshak.fusion.url_expert import URL_HASH, FrozenURLExpert, frozen_url_folder


def v1_inputs(ctx, tree, digest):
    matches = []
    for path in Path(ctx.input_root).rglob('results.json'):
        receipt = json.loads(path.read_text())
        if receipt.get('tree_sha256') != tree:
            continue
        manifest_path = path.with_name('fusion_v1.json')
        if receipt['status'] != 'ok' or receipt.get('failures') or sha256(manifest_path) != digest:
            raise ValueError('complete exact frozen v1 kernel required')
        primary = load_manifest(manifest_path, source_root=ctx.src)
        if primary['experts']['url']['artifact_sha256'] != URL_HASH:
            raise ValueError('URL expert differs from v1')
        references = primary['validation_references']
        if set(references) != FOLDS - {'campaign'}:
            raise ValueError('all frozen fold-validation experts required')
        models = {'campaign': (path.parent, primary, digest)}
        for fold, reference in references.items():
            relative = Path(reference['manifest'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('unsafe fold-validation manifest path')
            child = path.parent/relative
            if sha256(child) != reference['sha256']:
                raise ValueError('validation manifest changed')
            manifest = load_manifest(child, source_root=ctx.src)
            if manifest['fold'] != fold or manifest['fitting'] != primary['fitting']:
                raise ValueError('fold expert provenance differs from v1')
            models[fold] = (child.parent, manifest, reference['sha256'])
        matches.append(models)
    if len(matches) != 1:
        raise ValueError('one completed immutable v1 source required')
    return matches[0]


def bound_scores(path, frame):
    saved = pd.read_parquet(path)
    saved.id = saved.id.astype(str)
    if not saved.id.is_unique or set(saved.id) != set(frame.id.astype(str)):
        raise ValueError('saved v1 decisions must exactly match the frozen eligible population')
    aligned = saved.set_index('id').reindex(frame.id.astype(str))
    probabilities = aligned[['student', 'fused']].to_numpy()
    if (not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
            or not aligned['state'].isin(['SAFE', 'UNCERTAIN', 'SCAM']).all()):
        raise ValueError('valid frozen expert probabilities and states required')
    return aligned


def audit(frame, scores, threshold, student_threshold, url, url_threshold, n_boot):
    url_scores, _ = url.score(frame.text.fillna('').tolist())
    supported, reasons = corroboration(frame.text.fillna('').tolist(), url_scores,
                                      url_threshold=url_threshold)
    base = threshold.transform(scores.fused.to_numpy())
    if not np.array_equal(base, scores['state'].to_numpy()):
        raise ValueError('saved baseline decisions differ from frozen v1 thresholds')
    states = apply_escalation(scores.fused.to_numpy(), threshold, supported)
    if not np.all((base != 'SAFE') <= (states != 'SAFE')):
        raise ValueError('policy must not silently suppress a pre-existing alert')
    report = {}
    for label in ('label_binary', 'label_scam'):
        y = frame[label].fillna(-1).to_numpy(dtype=int)
        mask = np.isin(y, [0, 1])
        report[label] = state_report(
            y[mask], states[mask], frame.campaign_id.to_numpy()[mask],
            scores.student.to_numpy()[mask] >= student_threshold, base[mask],
            scores.fused.to_numpy()[mask], n_boot=n_boot)
        report[label]['excluded_unknown_n'] = int((~mask).sum())
    return report, pd.DataFrame({'id': frame.id.to_numpy(), 'state': states,
                                 'corroboration': supported,
                                 'reasons': [json.dumps(value) for value in reasons]})


def run(ctx):
    p = argparse.ArgumentParser()
    p.add_argument('--v1-tree', required=True)
    p.add_argument('--v1-manifest-sha256', required=True)
    p.add_argument('--n-boot', type=int, default=1000)
    args = p.parse_args(ctx.argv)
    if args.n_boot != 1000:
        raise ValueError('predeclared 1000 campaign draws required')
    models = v1_inputs(ctx, args.v1_tree, args.v1_manifest_sha256)
    root = D.resolve_data_root()
    sms = D.load_sms(root)
    runs = D.load_runs(root, sms, patterns=['campaign', 'source_holdout', 'lang_shift', 'temporal'],
                       labels=sms.label_binary.to_numpy(dtype=int), mask=D.eval_mask(sms))
    if {run.name for run in runs} != FOLDS:
        raise ValueError('every unchanged frozen fold required')
    url = FrozenURLExpert(frozen_url_folder(ctx.input_root))
    thresholds = {}
    # URL corroboration thresholds are fixed on cal labels ONLY, before any outcome file.
    for split in runs:
        cal = sms.iloc[split.cal]
        values, present = url.score(cal.text.fillna('').tolist())
        labels = cal.label_scam.fillna(-1).to_numpy(dtype=int)
        mask = present & np.isin(labels, [0, 1])
        if set(labels[mask]) != {0, 1}:
            raise ValueError('required URL cal threshold unmeasurable; never fit externally')
        selected = threshold_for_fpr(labels[mask], values[mask], .01)
        selected['threshold'] = probability_threshold(selected['threshold'])
        thresholds[split.name] = selected
    policy = {'version': 'evidence-escalation-candidate-20261101',
              'scope': 'in-sample policy selection; no independent Indian validation',
              'parent_v1_sha256': args.v1_manifest_sha256,
              'parent_fold_sha256': {fold: value[2] for fold, value in models.items()},
              'url_cal_thresholds': sanitize(thresholds),
              'policy_source_sha256': sha256(ctx.src/'rakshak/fusion/escalation.py'),
              'audit_source_sha256': sha256(ctx.src/'rakshak/fusion/escalation_audit.py'),
              'git_sha': ctx.results['git_sha'], 'tree_sha256': ctx.results['tree_sha256'],
              'shipment_approved': False, 'hf_publish': 'HF publish pending'}
    path = ctx.work/'frozen_candidate_policy.json'
    with path.open('x') as stream:
        json.dump(policy, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    policy_hash = sha256(path)
    internal, external = {}, {}
    for split in runs:
        folder, manifest, _ = models[split.name]
        test = sms.iloc[split.test]
        saved = bound_scores(folder/'test_v1.parquet', test)
        internal[split.name], decisions = audit(
            test, saved, SelectiveThresholds.from_dict(manifest['fusion']['thresholds']),
            manifest['student_cal_1pct_threshold'], url,
            thresholds[split.name]['threshold'], args.n_boot)
        decisions.to_parquet(ctx.work/f"policy_{split.name.replace(':', '_')}.parquet")
    folder, manifest, _ = models['campaign']
    for name in ('groundlm', 'smishx'):
        frame = pd.read_parquet(root/'external'/f'external_{name}.parquet')
        frame = frame[frame.text_available.fillna(False) & D.eval_mask(frame)].copy()
        frame['reg_domains'] = [sorted({cached_registered_domain(u) for u in canonicalize(t).urls})
                                for t in frame.text]
        frame, _ = assign_campaigns(mark_exact_duplicates(frame.reset_index(drop=True)))
        saved = bound_scores(folder/f'external_{name}_v1.parquet', frame)
        external[name], decisions = audit(
            frame, saved, SelectiveThresholds.from_dict(manifest['fusion']['thresholds']),
            manifest['student_cal_1pct_threshold'], url,
            thresholds['campaign']['threshold'], args.n_boot)
        decisions.to_parquet(ctx.work/f'policy_external_{name}.parquet')
    if sha256(path) != policy_hash:
        raise ValueError('policy changed after outcomes')
    v1_inputs(ctx, args.v1_tree, args.v1_manifest_sha256)
    decision = escalation_gate(internal, external)
    write_metrics('metrics.json', sanitize({'policy_sha256': policy_hash, 'policy': policy,
                                           'internal': internal, 'external': external,
                                           'decision': decision}))
    ctx.results['summary'] = str(decision)
    return True
