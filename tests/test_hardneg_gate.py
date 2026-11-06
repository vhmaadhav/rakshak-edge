"""Prospective applicability cannot hide a regression or invent UCI fraud positives."""

import copy

from rakshak.fusion.hardneg_gate import FOLDS, hardneg_gate


def fixture():
    endpoint = {'fpr_scam': {'rate': .005}, 'fpr_at_least_uncertain': {'rate': .05},
                'recall_scam': {'rate': .8, 'ci95': [.7, .9], 'n': 10},
                'student_recall': {'rate': .7}}
    row = {arm: {label: copy.deepcopy(endpoint) for label in ('label_binary', 'label_scam')}
           for arm in ('baseline', 'transactional')}
    internal = {fold: {str(s): copy.deepcopy(row) for s in (1, 2, 3)} for fold in FOLDS}
    for block in internal['source_holdout:holdout_uci_sms_spam'].values():
        for arm in block.values():
            arm['label_scam']['recall_scam'] = {'n': 0, 'rate': None, 'ci95': [None, None]}
    external = {name: {str(s): copy.deepcopy(row) for s in (1, 2, 3)}
                for name in ('groundlm', 'smishx')}
    states = {fold: {str(s): {arm: {'status': 'ok'} for arm in row} for s in (1, 2, 3)}
              for fold in FOLDS}
    return internal, external, states


def test_explicit_uci_na_requires_legacy_guard_and_all_other_folds():
    internal, external, states = fixture()
    result = hardneg_gate(internal, external, states)
    assert result['decision'] == 'go' and len(result['fraud_recall_inapplicable']) == 3
    internal['source_holdout:holdout_uci_sms_spam']['1']['transactional'][
        'label_binary']['recall_scam']['rate'] = .5
    assert hardneg_gate(internal, external, states)['decision'] == 'kill'
    internal, external, states = fixture()
    del internal['lang_shift']['2']
    assert hardneg_gate(internal, external, states)['decision'] == 'unmeasurable'


def test_external_failure_rejects_even_if_another_endpoint_is_missing():
    internal, external, states = fixture()
    external['groundlm']['1']['transactional']['label_scam']['fpr_scam']['rate'] = .02
    internal['temporal']['2']['transactional']['label_scam']['recall_scam']['rate'] = None
    result = hardneg_gate(internal, external, states)
    assert result['decision'] == 'kill' and result['unavailable'] and result['failed']
