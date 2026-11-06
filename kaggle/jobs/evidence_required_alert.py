"""Owner-requested evidence-required alert, with native URL cal only; Kaggle evaluation."""

import argparse
import json

import numpy as np
import pandas as pd
from _common import write_metrics
from evidence_escalation import bound_scores, v1_inputs

from rakshak.canonicalize import canonicalize
from rakshak.datasets.common import cached_registered_domain
from rakshak.datasets.dedup import assign_campaigns, mark_exact_duplicates
from rakshak.eval import data as D
from rakshak.eval.report import sanitize
from rakshak.fusion.escalation import corroboration
from rakshak.fusion.escalation_audit import FOLDS
from rakshak.fusion.frozen_bundle import sha256
from rakshak.fusion.native_url_cal import calibrate_native_url
from rakshak.fusion.required_alert import (
    apply_required_alert,
    required_alert_gate,
    required_report,
    rules_only_states,
)
from rakshak.fusion.selective import SelectiveThresholds
from rakshak.fusion.url_expert import FrozenURLExpert, frozen_url_folder


def audit(frame, saved, manifest, url, url_threshold, n_boot):
    text = frame.text.fillna('').astype(str).tolist()
    url_scores, _ = url.score(text)
    supported, reasons = corroboration(text, url_scores, url_threshold=url_threshold)
    threshold = SelectiveThresholds.from_dict(manifest['fusion']['thresholds'])
    if not np.array_equal(threshold.transform(saved.fused.to_numpy()), saved['state'].to_numpy()):
        raise ValueError('saved baseline decisions differ from frozen v1')
    states = apply_required_alert(saved.fused.to_numpy(), threshold, supported)
    if not np.array_equal(states != 'SAFE', supported):
        raise ValueError('qualifying evidence must be necessary and sufficient for an alert')
    rules = rules_only_states(text)
    student = saved.student.to_numpy() >= manifest['student_cal_1pct_threshold']
    reports = {}
    for label in ('label_binary', 'label_scam'):
        labels = frame[label].fillna(-1).to_numpy(dtype=int)
        valid = np.isin(labels, [0, 1])
        reports[label] = required_report(
            labels[valid], states[valid], frame.campaign_id.to_numpy()[valid],
            student[valid], rules[valid], saved.fused.to_numpy()[valid], n_boot=n_boot)
        reports[label]['excluded_unknown_n'] = int((~valid).sum())
    decisions = pd.DataFrame({'id': frame.id.to_numpy(), 'state': states,
                              'rules_state': rules, 'qualifying_evidence': supported,
                              'reasons': [json.dumps(value) for value in reasons]})
    return reports, decisions


def run(ctx):
    p = argparse.ArgumentParser()
    p.add_argument('--v1-tree', required=True)
    p.add_argument('--v1-manifest-sha256', required=True)
    p.add_argument('--n-boot', type=int, default=1000)
    args = p.parse_args(ctx.argv)
    if args.n_boot != 1000:
        raise ValueError('predeclared 1000 campaign bootstrap draws required')
    models = v1_inputs(ctx, args.v1_tree, args.v1_manifest_sha256)
    root = D.resolve_data_root()
    url_folder = frozen_url_folder(ctx.input_root)
    url = FrozenURLExpert(url_folder)
    native_cal = calibrate_native_url(root, url_folder, url, ctx.src)
    policy = {'schema': 'rakshak-evidence-required-alert-candidate-v1',
              'scope': 'in-sample policy selection; independent Indian validation OPEN',
              'shipment_approved': False, 'measurement_reference_only': True,
              'arm_android_portability': 'UNVERIFIED',
              'parent_v1_sha256': args.v1_manifest_sha256,
              'parent_fold_sha256': {fold: value[2] for fold, value in models.items()},
              'native_url_calibration': sanitize(native_cal),
              'url_expert': models['campaign'][1]['experts']['url'],
              'policy': 'no qualifying evidence SAFE; qualifying and fused>=t_scam SCAM; '
                        'qualifying and lower risk UNCERTAIN; no metric-exempt info alerts',
              'rules_comparator': 'run_rules/build_package(scam_state=benign): '
                                  'HIGH SCAM, MEDIUM/UNCERTAIN alert, LOW SAFE; no URL model',
              'source_sha256': {name: sha256(ctx.src/'rakshak/fusion'/name) for name in (
                  'required_alert.py', 'escalation.py', 'escalation_audit.py',
                  'native_url_cal.py')},
              'git_sha': ctx.results['git_sha'], 'tree_sha256': ctx.results['tree_sha256'],
              'hf_publish': 'HF publish pending'}
    policy_path = ctx.work/'frozen_required_alert_policy.json'
    with policy_path.open('x') as stream:
        json.dump(policy, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    policy_hash = sha256(policy_path)
    ctx.log(f'froze native URL cal split/hash/threshold and policy {policy_hash} '
            'before SMS scoring')
    # Not even SMS corpus loading can precede native-cal selection and policy freeze.
    sms = D.load_sms(root)
    runs = D.load_runs(root, sms, patterns=['campaign', 'source_holdout', 'lang_shift', 'temporal'],
                       labels=sms.label_binary.to_numpy(dtype=int), mask=D.eval_mask(sms))
    if {split.name for split in runs} != FOLDS:
        raise ValueError('all six unchanged frozen folds required')
    internal, external = {}, {}
    threshold = native_cal['threshold_selection']['threshold']
    for split in runs:
        folder, manifest, _ = models[split.name]
        test = sms.iloc[split.test]
        saved = bound_scores(folder/'test_v1.parquet', test)
        internal[split.name], decisions = audit(test, saved, manifest, url, threshold, args.n_boot)
        decisions.to_parquet(ctx.work/f"required_{split.name.replace(':', '_')}.parquet")
    folder, manifest, _ = models['campaign']
    for name in ('groundlm', 'smishx'):
        frame = pd.read_parquet(root/'external'/f'external_{name}.parquet')
        frame = frame[frame.text_available.fillna(False) & D.eval_mask(frame)].copy()
        frame['reg_domains'] = [sorted({cached_registered_domain(u) for u in canonicalize(t).urls})
                                for t in frame.text]
        frame, _ = assign_campaigns(mark_exact_duplicates(frame.reset_index(drop=True)))
        saved = bound_scores(folder/f'external_{name}_v1.parquet', frame)
        external[name], decisions = audit(frame, saved, manifest, url, threshold, args.n_boot)
        decisions.to_parquet(ctx.work/f'required_external_{name}.parquet')
    if sha256(policy_path) != policy_hash:
        raise ValueError('policy changed after outcomes')
    v1_inputs(ctx, args.v1_tree, args.v1_manifest_sha256)
    decision = required_alert_gate(internal, external)
    write_metrics('metrics.json', sanitize({'policy_sha256': policy_hash, 'policy': policy,
                                           'internal': internal, 'external': external,
                                           'decision': decision}))
    ctx.results['summary'] = 'evidence-required alert '+decision['decision']
    return True
