"""New cal-only fusion baseline/candidate, never labelled the shipped configuration."""

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from _common import write_metrics
from threadpoolctl import threadpool_limits

from rakshak.datasets.common import cached_registered_domain
from rakshak.datasets.dedup import assign_campaigns, mark_exact_duplicates
from rakshak.eval import data as D
from rakshak.eval.audit import paired_rate_test
from rakshak.eval.report import sanitize
from rakshak.fusion.audit import baseline_gate, triage_report
from rakshak.fusion.cal_only import fit_cal_only, predict_frozen
from rakshak.fusion.fuse import FEATURE_NAMES, feature_matrix
from rakshak.fusion.selective import SelectiveThresholds
from rakshak.fusion.source_inputs import (
    METRICS_HASH,
    SOURCE_TREE,
    aligned_scores,
    source_folder,
)
from rakshak.fusion.url_expert import (
    SCALER_HASH,
    URL_HASH,
    URL_TREE,
    FrozenURLExpert,
    frozen_url_folder,
)

ARMS = ('baseline', 'transactional')
REQUIRED_RUNS = {'campaign', 'lang_shift', 'temporal',
                 'source_holdout:holdout_scamshield', 'source_holdout:holdout_uci_sms_spam',
                 'source_holdout:holdout_mishra_soni_smsphishing'}


def matrices(frame, expert):
    texts = frame.text.fillna('').astype(str).tolist()
    url, present = expert.score(texts)
    extended = feature_matrix(texts, np.full(len(frame), .5), url,
                              has_url=present, transactional=True)
    return {'baseline': extended[:, :len(FEATURE_NAMES)], 'transactional': extended}


def endpoints(frame, scores, threshold, student_pred, n_boot):
    result = {}
    for label in ('label_scam', 'label_binary'):
        if label not in frame:
            continue
        y = frame[label].fillna(-1).to_numpy(dtype=int)
        valid = np.isin(y, [0, 1])
        result[label] = triage_report(y[valid], scores[valid], threshold,
                                      frame.campaign_id[valid].astype(str).tolist(),
                                      student_pred=student_pred[valid], n_boot=n_boot)
        result[label]['excluded_unknown_n'] = int((~valid).sum())
    return result


def compare_recalls(frame, pred_a, pred_b, *, n_boot):
    y = frame.label_scam.fillna(-1).to_numpy(dtype=int)
    pos = y == 1
    return paired_rate_test(pred_a[pos].astype(int), pred_b[pos].astype(int),
                            frame.campaign_id[pos].astype(str).tolist(), n_boot=n_boot)


def candidate_decision(internal, external, states):
    if (set(internal) != REQUIRED_RUNS or set(states) != REQUIRED_RUNS
            or any(set(block) != {'1', '2', '3'} for block in internal.values())):
        return {'decision': 'unmeasurable', 'reason': 'all six folds and three seeds required'}
    checks = {}
    for seed in ('1', '2', '3'):
        ground = external.get('groundlm', {}).get(seed, {})
        smish = external.get('smishx', {}).get(seed, {})
        if not all(arm in ground and arm in smish for arm in ARMS):
            return {'decision': 'unmeasurable', 'reason': 'missing external seed/arm'}
        b, a = ground['baseline']['label_scam'], ground['transactional']['label_scam']
        gate = baseline_gate(a, smish['transactional']['label_scam'])
        checks[f'{seed}/absolute_gate'] = gate['decision'] == 'go'
        for metric in ('fpr_scam', 'fpr_at_least_uncertain'):
            aa, bb = a[metric]['rate'], b[metric]['rate']
            if aa is None or bb is None:
                return {'decision': 'unmeasurable', 'reason': 'undefined external FPR'}
            checks[f'{seed}/groundlm/{metric}'] = aa <= bb
        for run, block in internal.items():
            if any(states[run][seed][arm]['status'] != 'ok' for arm in ARMS):
                return {'decision': 'unmeasurable', 'reason': 'infeasible required cal state'}
            base = block[seed]['baseline']['label_scam']['recall_scam']
            tuned = block[seed]['transactional']['label_scam']['recall_scam']
            if base['rate'] is None or tuned['rate'] is None:
                return {'decision': 'unmeasurable', 'reason': 'undefined required fold recall'}
            half = (base['ci95'][1] - base['ci95'][0]) / 2
            checks[f'{seed}/{run}/recall'] = tuned['rate'] - base['rate'] >= -half
    return {'decision': 'go' if all(checks.values()) else 'kill', 'checks': checks,
            'scope': 'all three seeds must pass; no independent held-out improvement claim'}


def run(ctx):
    p = argparse.ArgumentParser()
    p.add_argument('--n-boot', type=int, default=1000)
    args = p.parse_args(ctx.argv)
    source = source_folder(ctx.input_root)
    metadata = json.loads((source / 'metrics.json').read_text())
    expert = FrozenURLExpert(frozen_url_folder(ctx.input_root))
    root = D.resolve_data_root()
    sms = D.load_sms(root)
    runs = D.load_runs(root, sms,
                       patterns=['campaign', 'source_holdout', 'lang_shift', 'temporal'],
                       labels=sms.label_binary.to_numpy(), mask=D.eval_mask(sms))
    if {r.name for r in runs} != set(metadata['runs']) or set(metadata['runs']) != REQUIRED_RUNS:
        raise ValueError('every frozen source-control split required')
    features = matrices(sms, expert)
    states, aligned = {}, {}
    # Freeze ALL states before reading any test/external decision outcome.
    for run_ in runs:
        positions = np.r_[run_.cal, run_.test]
        frame = sms.iloc[positions]
        known = np.isin(frame.label_scam.fillna(-1).to_numpy(dtype=int), [0, 1])
        cal_mask = known & np.isin(positions, run_.cal)
        cal_pos = positions[cal_mask]
        y_cal = sms.label_scam.iloc[cal_pos].to_numpy(dtype=int)
        states[run_.name], aligned[run_.name] = {}, {}
        for seed in (1, 2, 3):
            name = run_.name.replace(':', '_') + f'_erm_s{seed}'
            path = source / f'scores_{name}.parquet'
            scores = aligned_scores(path, frame)
            saved = pd.read_parquet(path).set_index('id')
            if (set(saved.reindex(sms.id.iloc[run_.cal])['split']) != {'cal'}
                    or set(saved.reindex(sms.id.iloc[run_.test])['split']) != {'test'}):
                raise ValueError('saved score split roles disagree with frozen IDs')
            aligned[run_.name][str(seed)] = (positions, scores)
            states[run_.name][str(seed)] = {}
            for arm in ARMS:
                x = features[arm][cal_pos].copy()
                x[:, 0] = scores[cal_mask]
                with threadpool_limits(limits=1):
                    state = fit_cal_only(x, y_cal, sms.id.iloc[cal_pos].to_numpy(),
                                         sms.campaign_id.iloc[cal_pos].astype(str).to_numpy(),
                                         train_ids=sms.id.iloc[run_.train].tolist(),
                                         test_ids=sms.id.iloc[run_.test_all].tolist(),
                                         transactional=arm == 'transactional')
                states[run_.name][str(seed)][arm] = state
    frozen_path = write_metrics('frozen_fusion_states.json', sanitize(states))
    frozen_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    internal, external, paired, external_paired = {}, {}, {}, {}
    for run_ in runs:
        internal[run_.name], paired[run_.name] = {}, {}
        test = sms.iloc[run_.test]
        for seed in (1, 2, 3):
            key = str(seed)
            positions, message = aligned[run_.name][key]
            student_thr = metadata['runs'][run_.name]['arms']['erm'][key][
                'metrics']['thresholds']['fpr_1pct']['threshold']
            message = pd.Series(message, index=positions).reindex(run_.test).to_numpy()
            predictions = {}
            internal[run_.name][key] = {}
            for arm in ARMS:
                state = states[run_.name][key][arm]
                if state['status'] != 'ok':
                    continue
                x = features[arm][run_.test].copy()
                x[:, 0] = message
                prob = predict_frozen(state, x)
                threshold = SelectiveThresholds.from_dict(state['thresholds'])
                internal[run_.name][key][arm] = endpoints(
                    test, prob, threshold, message >= student_thr, args.n_boot)
                predictions[arm] = threshold.transform(prob) == 'SCAM'
                pd.DataFrame({'id': test.id, 'prob': prob, 'scam': predictions[arm]}).to_parquet(
                    ctx.work / f"fused_{run_.name.replace(':', '_')}_{arm}_s{seed}.parquet")
            if set(predictions) == set(ARMS):
                paired[run_.name][key] = compare_recalls(
                    test, predictions['transactional'], predictions['baseline'], n_boot=args.n_boot)
    # External labels/decisions are used for reporting only after the artifacts are saved.
    from rakshak.canonicalize import canonicalize

    for name in ('groundlm', 'smishx'):
        frame = pd.read_parquet(root / 'external' / f'external_{name}.parquet')
        frame = frame[frame.text_available.fillna(False) & D.eval_mask(frame)].copy()
        frame['reg_domains'] = [sorted({cached_registered_domain(u) for u in canonicalize(t).urls})
                                for t in frame.text]
        frame, _ = assign_campaigns(mark_exact_duplicates(frame.reset_index(drop=True)))
        matrices_ = matrices(frame, expert)
        external[name], external_paired[name] = {}, {}
        for seed in (1, 2, 3):
            key = str(seed)
            message = aligned_scores(source / f'external_{name}_erm_s{seed}.parquet', frame)
            student_thr = metadata['runs']['campaign']['arms']['erm'][key][
                'metrics']['thresholds']['fpr_1pct']['threshold']
            external[name][key], decisions = {}, {}
            for arm in ARMS:
                state = states['campaign'][key][arm]
                if state['status'] != 'ok':
                    continue
                x = matrices_[arm].copy()
                x[:, 0] = message
                prob = predict_frozen(state, x)
                threshold = SelectiveThresholds.from_dict(state['thresholds'])
                decisions[arm] = threshold.transform(prob)
                external[name][key][arm] = endpoints(
                    frame, prob, threshold, message >= student_thr, args.n_boot)
                pd.DataFrame({'id': frame.id, 'prob': prob,
                              'state': threshold.transform(prob)}).to_parquet(
                    ctx.work / f'fused_external_{name}_{arm}_s{seed}.parquet')
            if set(decisions) == set(ARMS):
                labels = frame.label_scam.fillna(-1).to_numpy(dtype=int)
                groups = frame.campaign_id.astype(str).to_numpy()
                pairs = {}
                for endpoint, mask, alert in (('fpr_scam', labels == 0, False),
                                               ('fpr_alert', labels == 0, True),
                                               ('recall_scam', labels == 1, False)):
                    values = {arm: ((value != 'SAFE') if alert else (value == 'SCAM'))
                              for arm, value in decisions.items()}
                    pairs[endpoint] = paired_rate_test(
                        values['transactional'][mask].astype(int),
                        values['baseline'][mask].astype(int), groups[mask], n_boot=args.n_boot)
                external_paired[name][key] = pairs
    if hashlib.sha256(frozen_path.read_bytes()).hexdigest() != frozen_hash:
        raise ValueError('fitted fusion states changed after evaluation')
    report = {'scope': 'new cal-only baseline/candidate, not identified shipped state; '
                      'GroundLM/SmishX feedback in-sample for intervention selection',
              'internal': internal, 'external': external, 'paired_recall': paired,
              'external_paired': external_paired,
              'decision': candidate_decision(internal, external, states),
              'frozen_states_sha256': frozen_hash, 'source_tree': SOURCE_TREE,
              'source_metrics_sha256': METRICS_HASH, 'url_tree': URL_TREE,
              'url_sha256': URL_HASH, 'url_scaler_sha256': SCALER_HASH,
              'n_boot': args.n_boot, 'fitted_objects_frozen_before_external_outcomes': True}
    write_metrics('metrics.json', sanitize(report))
    ctx.results['summary'] = str(report['decision'])
    return True
