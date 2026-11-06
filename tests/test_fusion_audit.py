"""Synthetic abstention and pairing cases, never benchmark data."""

import pytest

from rakshak.fusion.audit import baseline_gate, triage_report
from rakshak.fusion.selective import SelectiveThresholds


def test_unconditional_fpr_counts_abstained_negatives_and_threshold_ties():
    result = triage_report([0, 0, 0, 1, 1], [.2, .5, .8, .8, .2],
                           SelectiveThresholds(.2, .8), ['a', 'b', 'c', 'd', 'e'],
                           student_pred=[0, 1, 1, 1, 1], n_boot=20)
    assert result['fpr_scam']['events'] == 1 and result['fpr_scam']['n'] == 3
    assert result['fpr_at_least_uncertain']['events'] == 2
    assert result['conditional_auto_decision_metrics']['fpr'] == .5
    assert result['recall_scam']['rate'] == .5
    assert result['paired_recall_delta']['rate'] == -.5
    assert result['coverage']['rate'] == .8
    assert baseline_gate(result, result)['decision'] == 'kill'


def test_zero_negatives_does_not_certify_zero_fpr():
    result = triage_report([1], [.9], SelectiveThresholds(.2, .8), ['a'],
                           student_pred=[1], n_boot=20)
    assert result['fpr_scam']['rate'] is None
    assert baseline_gate(result, result)['decision'] == 'unmeasurable'
    with pytest.raises(ValueError, match='binary'):
        triage_report([None], [.9], SelectiveThresholds(.2, .8), ['a'], n_boot=20)
