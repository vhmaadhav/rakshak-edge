"""Synthetic policy fixtures only; the student cannot issue any alert without evidence."""

import copy

import numpy as np

from rakshak.fusion.escalation_audit import FOLDS
from rakshak.fusion.required_alert import (
    apply_required_alert,
    required_alert_gate,
    required_report,
    rules_only_states,
)
from rakshak.fusion.selective import SelectiveThresholds


def test_no_evidence_always_safe_and_qualifying_low_risk_still_uncertain():
    values = [0, .5, 1, .1, .8, 1]
    decisions = apply_required_alert(values, SelectiveThresholds(.2, .8),
                                     [False, False, False, True, True, True])
    assert decisions.tolist() == ['SAFE', 'SAFE', 'SAFE', 'UNCERTAIN', 'SCAM', 'SCAM']
    assert rules_only_states(['Plain synthetic reminder']).tolist() == ['SAFE']


def gate_fixture():
    report = {'n_fraud': 20, 'rules_recall_alert': {'rate': .6, 'ci95': [.5, .7]},
              'recall_alert': {'rate': .6}, 'fpr_scam': {'rate': .01},
              'fpr_at_least_uncertain': {'rate': .15}, 'student_recall_scam': {'rate': .99}}
    internal = {fold: {label: copy.deepcopy(report) for label in ('label_binary', 'label_scam')}
                for fold in FOLDS}
    internal['source_holdout:holdout_uci_sms_spam']['label_scam']['n_fraud'] = 0
    external = {name: {label: copy.deepcopy(report) for label in ('label_binary', 'label_scam')}
                for name in ('groundlm', 'smishx')}
    return internal, external


def test_rules_guard_is_mandatory_and_student_comparison_cannot_override_it():
    internal, external = gate_fixture()
    assert required_alert_gate(internal, external)['decision'] == 'go'
    external['smishx']['label_scam']['recall_alert']['rate'] = .49
    assert required_alert_gate(internal, external)['decision'] == 'kill'
    internal, external = gate_fixture()
    internal['lang_shift']['label_binary']['recall_alert']['rate'] = .49
    assert required_alert_gate(internal, external)['decision'] == 'kill'
    internal, external = gate_fixture()
    del internal['campaign']
    assert required_alert_gate(internal, external)['decision'] == 'unmeasurable'
    external['groundlm']['label_scam']['fpr_scam']['rate'] = .011
    assert required_alert_gate(internal, external)['decision'] == 'kill'


def test_reports_keep_full_negative_denominators_and_paired_comparators():
    result = required_report(np.array([0, 0, 1, 1]),
                             np.array(['SAFE', 'UNCERTAIN', 'SCAM', 'SAFE']),
                             np.array(['a', 'b', 'c', 'd']),
                             np.array([True, True, True, True]),
                             np.array(['SAFE', 'UNCERTAIN', 'SCAM', 'SAFE']),
                             np.array([.1, .2, .9, .99]), n_boot=100)
    assert result['fpr_scam']['n'] == 2
    assert result['fpr_at_least_uncertain']['rate'] == .5
    assert result['rules_recall_alert']['rate'] == .5
    assert result['paired_alert_vs_rules']['rate'] == 0
    assert result['paired_alert_vs_student']['rate'] == -.5
    assert 'baseline_alert_recall' not in result
