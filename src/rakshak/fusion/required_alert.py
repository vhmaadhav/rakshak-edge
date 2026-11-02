"""Evidence-required alerts: the student alone cannot issue a warning."""

import numpy as np

from rakshak.canonicalize import canonicalize
from rakshak.eval.audit import paired_rate_test
from rakshak.eval.rule_audit import describe_rate
from rakshak.evidence import Risk, build_package
from rakshak.fusion.calibrate import scores_array
from rakshak.fusion.escalation_audit import FOLDS, state_report
from rakshak.rules import run_rules


def apply_required_alert(probability, thresholds, supported):
    probability = scores_array(probability, probability=True)
    supported = np.asarray(supported)
    if supported.shape != probability.shape or not np.isin(supported, [False, True]).all():
        raise ValueError('one qualifying-evidence boolean per decision required')
    return np.where(~supported.astype(bool), 'SAFE',
                    np.where(probability >= thresholds.t_scam, 'SCAM', 'UNCERTAIN'))


def rules_only_states(texts):
    states = []
    for text in texts:
        canon = canonicalize(text)
        risk = build_package(canon, run_rules(canon), scam_state='benign').risk
        states.append('SCAM' if risk == Risk.HIGH else 'SAFE' if risk == Risk.LOW else 'UNCERTAIN')
    return np.asarray(states)


def required_report(labels, states, groups, student, rules, probability, *, n_boot=1000):
    result = state_report(labels, states, groups, student, rules, probability, n_boot=n_boot)
    result['rules_recall_alert'] = result.pop('baseline_alert_recall')
    result['paired_alert_vs_rules'] = result.pop('paired_alert_vs_base')
    labels, rules, groups = np.asarray(labels), np.asarray(rules), np.asarray(groups)
    student, states = np.asarray(student), np.asarray(states)
    pos, neg = labels == 1, labels == 0
    for name, values, mask in (
            ('rules_recall_scam', rules == 'SCAM', pos),
            ('rules_fpr_scam', rules == 'SCAM', neg),
            ('rules_fpr_alert', rules != 'SAFE', neg),
            ('student_fpr_scam', student, neg)):
        result[name] = describe_rate(values[mask].astype(int), groups[mask].astype(str).tolist(),
                                     n_boot=n_boot)
    result['paired_alert_vs_student'] = paired_rate_test(
        (states[pos] != 'SAFE').astype(int), student[pos].astype(int),
        groups[pos].astype(str).tolist(), n_boot=n_boot)
    return result


def required_alert_gate(internal, external):
    checks, missing, inapplicable = {}, [], []
    ground = external.get('groundlm', {}).get('label_scam', {})
    for name, key, limit in (('groundlm_scam', 'fpr_scam', .01),
                              ('groundlm_alert', 'fpr_at_least_uncertain', .15)):
        value = ground.get(key, {}).get('rate')
        if value is None or not np.isfinite(value):
            missing.append(name)
        else:
            checks[name] = value <= limit
    populations = {**internal, 'smishx': external.get('smishx', {})}
    for name in sorted(FOLDS | {'smishx'}):
        for label in ('label_binary', 'label_scam'):
            block = populations.get(name, {}).get(label)
            key = name+'/'+label+'/recall_vs_rules'
            if block is None:
                missing.append(key)
                continue
            if (name == 'source_holdout:holdout_uci_sms_spam' and label == 'label_scam'
                    and block['n_fraud'] == 0):
                inapplicable.append(key)
                continue
            rules, tuned = block['rules_recall_alert'], block['recall_alert']
            values = [rules['rate'], tuned['rate'], *rules['ci95']]
            if any(value is None for value in values) or not np.isfinite(values).all():
                missing.append(key)
            else:
                checks[key] = tuned['rate']-rules['rate'] >= -(rules['ci95'][1]-rules['ci95'][0])/2
    decision = ('kill' if any(not value for value in checks.values()) else
                'unmeasurable' if missing or set(internal) != FOLDS else 'go')
    return {'decision': decision, 'checks': checks, 'missing': missing,
            'inapplicable': inapplicable,
            'scope': 'in-sample policy selection; recall guard rules-only, student comparison only'}
