"""Unconditional triage endpoints from frozen probabilities; never fits or selects."""

import numpy as np

from rakshak.eval.audit import paired_rate_test
from rakshak.eval.rule_audit import describe_rate
from rakshak.fusion.calibrate import scores_array
from rakshak.fusion.selective import selective_metrics


def triage_report(y, prob, thresholds, campaigns, *, student_pred=None, n_boot=1000):
    p = scores_array(prob, probability=True)
    y, groups = np.asarray(y), np.asarray(campaigns, dtype=str)
    if y.shape != p.shape or groups.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('binary endpoint labels and aligned campaigns required')
    if n_boot < 1 or np.any(groups == ''):
        raise ValueError('positive resampling count and nonempty campaign IDs required')
    pred = thresholds.transform(p)
    neg, pos = y == 0, y == 1
    scam, alert, uncertain = pred == 'SCAM', pred != 'SAFE', pred == 'UNCERTAIN'
    masks = {'fpr_scam': (scam, neg), 'fpr_at_least_uncertain': (alert, neg),
             'negative_uncertain_only': (uncertain, neg), 'recall_scam': (scam, pos),
             'recall_at_least_uncertain': (alert, pos),
             'coverage': (~uncertain, np.ones(len(y), dtype=bool)),
             'uncertain_fraction': (uncertain, np.ones(len(y), dtype=bool))}
    result = {name: describe_rate(values[mask].astype(int), groups[mask], n_boot=n_boot)
              for name, (values, mask) in masks.items()}
    result['conditional_auto_decision_metrics'] = selective_metrics(y, p, thresholds)
    result['n'] = len(y)
    result['thresholds'] = thresholds.to_dict()
    result['fpr_denominator'] = 'all eligible endpoint negatives, including UNCERTAIN'
    if student_pred is not None:
        student = np.asarray(student_pred)
        if student.shape != p.shape or not np.isin(student, [0, 1]).all():
            raise ValueError('aligned binary student decisions required')
        result['student_recall'] = describe_rate(student[pos], groups[pos], n_boot=n_boot)
        result['paired_recall_delta'] = paired_rate_test(
            scam[pos].astype(int), student[pos], groups[pos], n_boot=n_boot)
    return result


def baseline_gate(groundlm, smishx):
    """Exact recorded point-estimate gate; undefined endpoints cannot pass."""
    rates = [groundlm['fpr_scam']['rate'], groundlm['fpr_at_least_uncertain']['rate'],
             smishx['recall_scam']['rate'], smishx['student_recall']['rate']]
    if any(v is None or not np.isfinite(v) for v in rates):
        return {'decision': 'unmeasurable', 'reason': 'undefined required endpoint'}
    checks = {'groundlm_scam_le_1pct': rates[0] <= .01,
              'groundlm_at_least_uncertain_le_10pct': rates[1] <= .10,
              'smishx_recall_ge_student': rates[2] >= rates[3]}
    return {'decision': 'go' if all(checks.values()) else 'kill', 'checks': checks,
            'scope': 'point-estimate audit gate, not a population guarantee'}
