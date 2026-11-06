"""Abstention denominators and fixed-fold gates cannot hide a failed alert FPR."""

import copy

import numpy as np

from rakshak.fusion.escalation_audit import FOLDS, escalation_gate, state_report


def rate(value):
    return {'rate': value, 'ci95': [max(0, value-.02), min(1, value+.02)]}


def fixture():
    internal = {fold: {label: {'n_fraud': 10, 'baseline_alert_recall': rate(.8),
                              'recall_alert': rate(.8)}
                       for label in ('label_binary', 'label_scam')} for fold in FOLDS}
    internal['source_holdout:holdout_uci_sms_spam']['label_scam']['n_fraud'] = 0
    external = {'groundlm': {'label_scam': {'fpr_scam': rate(.005),
                                           'fpr_at_least_uncertain': rate(.1)}},
                'smishx': {'label_scam': {'recall_alert': rate(.8),
                                         'student_recall_scam': rate(.7)}}}
    return internal, external


def test_demoting_scam_does_not_hide_alert_false_alarms():
    report = state_report([0, 0, 1, 1], ['UNCERTAIN', 'SAFE', 'SCAM', 'UNCERTAIN'],
                           ['a', 'b', 'c', 'd'], [0, 0, 1, 0],
                           ['SCAM', 'SAFE', 'SCAM', 'SCAM'], [.9, .1, .9, .9], n_boot=20)
    assert report['fpr_scam']['rate'] == 0
    assert report['fpr_at_least_uncertain']['rate'] == .5
    assert report['coverage']['rate'] == .5
    assert report['uncertain_share']['rate'] == .5
    assert report['recall_scam']['rate'] == .5
    assert report['recall_alert']['rate'] == 1
    assert report['paired_alert_vs_base']['rate'] == 0
    assert np.isfinite(report['risk_coverage_auto_decided']['aurc'])


def test_every_fold_and_both_false_alarm_gates_are_mandatory():
    internal, external = fixture()
    assert escalation_gate(internal, external)['decision'] == 'go'
    ext = copy.deepcopy(external)
    ext['groundlm']['label_scam']['fpr_at_least_uncertain'] = rate(.16)
    assert escalation_gate(internal, ext)['decision'] == 'kill'
    internal['source_holdout:holdout_uci_sms_spam']['label_binary']['recall_alert'] = rate(.6)
    assert escalation_gate(internal, external)['decision'] == 'kill'
    del internal['campaign']
    assert escalation_gate(internal, external)['decision'] == 'kill'
    internal, external = fixture()
    del internal['campaign']
    assert escalation_gate(internal, external)['decision'] == 'unmeasurable'
