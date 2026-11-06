"""Policy reports on explicit states, preserving all-negative denominators and paired CIs."""

import numpy as np

from rakshak.eval.audit import paired_rate_test
from rakshak.eval.metrics import coverage_error_curve
from rakshak.eval.rule_audit import describe_rate

FOLDS = {'campaign', 'lang_shift', 'temporal', 'source_holdout:holdout_scamshield',
         'source_holdout:holdout_uci_sms_spam', 'source_holdout:holdout_mishra_soni_smsphishing'}


def state_report(labels, states, groups, student, baseline, probability, *, n_boot=1000):
    labels, states = np.asarray(labels), np.asarray(states)
    groups, student, baseline = np.asarray(groups), np.asarray(student), np.asarray(baseline)
    probability = np.asarray(probability)
    if (not np.isin(labels, [0, 1]).all()
            or not np.isin(states, ['SAFE', 'UNCERTAIN', 'SCAM']).all()
            or any(value.shape != labels.shape
                   for value in (states, groups, student, baseline, probability))):
        raise ValueError('aligned binary labels, states and groups required')
    neg, pos = labels == 0, labels == 1
    masks = {'fpr_scam': (states == 'SCAM', neg),
             'fpr_at_least_uncertain': (states != 'SAFE', neg),
             'recall_scam': (states == 'SCAM', pos),
             'recall_alert': (states != 'SAFE', pos),
             'coverage': (states != 'UNCERTAIN', np.ones(len(labels), dtype=bool)),
             'uncertain_share': (states == 'UNCERTAIN', np.ones(len(labels), dtype=bool)),
             'student_recall_scam': (student, pos)}
    result = {key: describe_rate(values[mask].astype(int), groups[mask].astype(str).tolist(),
                                 n_boot=n_boot) for key, (values, mask) in masks.items()}
    decided = states != 'UNCERTAIN'
    # Curves rank already-fixed policy outputs; abstentions are not silently counted as errors.
    confidence = np.abs(probability-.5)
    result['risk_coverage_auto_decided'] = coverage_error_curve(
        labels[decided], states[decided] == 'SCAM', confidence[decided])
    result['risk_coverage_alert_endpoint'] = coverage_error_curve(
        labels, states != 'SAFE', confidence)
    result['paired_alert_vs_base'] = paired_rate_test(
        (states[pos] != 'SAFE').astype(int), (baseline[pos] != 'SAFE').astype(int),
        groups[pos].astype(str).tolist(), n_boot=n_boot)
    result['baseline_alert_recall'] = describe_rate(
        (baseline[pos] != 'SAFE').astype(int), groups[pos].astype(str).tolist(), n_boot=n_boot)
    result['conditional_decided_error'] = describe_rate(
        ((states[decided] == 'SCAM') != labels[decided]).astype(int),
        groups[decided].astype(str).tolist(), n_boot=n_boot)
    result['n'] = len(labels)
    result['n_fraud'] = int(pos.sum())
    return result


def escalation_gate(internal, external):
    checks, missing, inapplicable = {}, [], []
    try:
        ground, smish = external['groundlm']['label_scam'], external['smishx']['label_scam']
        for name, value, limit in (
            ('groundlm_scam', ground['fpr_scam']['rate'], .01),
            ('groundlm_alert', ground['fpr_at_least_uncertain']['rate'], .15)):
            if value is None or not np.isfinite(value):
                missing.append(name)
            else:
                checks[name] = value <= limit
        recall, control = smish['recall_alert']['rate'], smish['student_recall_scam']['rate']
        if recall is None or control is None or not np.isfinite([recall, control]).all():
            missing.append('smishx_recall')
        else:
            checks['smishx_alert_recall_ge_student_scam'] = recall >= control
    except KeyError:
        missing.append('external')
    for fold in sorted(FOLDS):
        for label in ('label_binary', 'label_scam'):
            block = internal.get(fold, {}).get(label)
            if block is None:
                missing.append(fold+'/'+label)
                continue
            if (label == 'label_scam' and fold == 'source_holdout:holdout_uci_sms_spam'
                    and block['n_fraud'] == 0):
                inapplicable.append(fold+'/'+label)
                continue
            base, tuned = block['baseline_alert_recall'], block['recall_alert']
            values = [base['rate'], tuned['rate'], *base['ci95']]
            if any(value is None for value in values) or not np.isfinite(values).all():
                missing.append(fold+'/'+label)
                continue
            half = (base['ci95'][1]-base['ci95'][0])/2
            checks[fold+'/'+label] = tuned['rate']-base['rate'] >= -half
    decision = ('kill' if any(not value for value in checks.values()) else
                'unmeasurable' if missing or set(internal) != FOLDS else 'go')
    return {'decision': decision, 'checks': checks, 'missing': missing,
            'inapplicable': inapplicable,
            'scope': 'in-sample policy selection, not independent validation'}
