"""A known failure remains a rejection even when another required endpoint is NA."""

import copy

from rakshak.fusion.decision_audit import available_checks


def fixture():
    endpoint = {'fpr_scam': {'rate': .005}, 'fpr_at_least_uncertain': {'rate': .05},
                'recall_scam': {'rate': .8, 'ci95': [.7, .9]}, 'student_recall': {'rate': .7}}
    row = {arm: {'label_scam': copy.deepcopy(endpoint)} for arm in ('baseline', 'transactional')}
    return {'decision': {'decision': 'unmeasurable', 'reason': 'undefined recall'},
            'external': {name: {str(s): copy.deepcopy(row) for s in (1, 2, 3)}
                         for name in ('groundlm', 'smishx')},
            'internal': {'uci': {str(s): {arm: {'label_scam': {
                'recall_scam': {'rate': None, 'ci95': [None, None]}}}
                for arm in ('baseline', 'transactional')} for s in (1, 2, 3)}}}


def test_does_not_promote_incomplete_gate_or_replace_fraud_labels():
    report = fixture()
    result = available_checks(report)
    assert result['operational_decision'] == 'unmeasurable'
    assert len(result['unavailable_endpoints']) == 3
    assert result['original_runner_decision'] == report['decision']
    report['external']['groundlm']['1']['transactional']['label_scam']['fpr_scam']['rate'] = .02
    result = available_checks(report)
    assert result['operational_decision'] == 'kill'
    assert result['original_runner_decision']['decision'] == 'unmeasurable'
    assert '1/groundlm_scam_le_1pct' in result['observable_failures']
