"""Explicit state reports preserve unconditional denominators and the original P1 gate."""

import copy

import numpy as np
import pytest

from rakshak.fusion.family_audit import FrozenDecisions, family_gate, family_triage
from rakshak.fusion.hardneg_gate import FOLDS


def test_family_report_uses_frozen_states_and_refuses_different_rows():
    probability = np.array([.9, .9, .9, .9])
    states = np.array(['SAFE', 'UNCERTAIN', 'SCAM', 'SAFE'])
    result = family_triage(np.array([0, 0, 1, 1]), probability, states,
                           ['a', 'b', 'c', 'd'], np.array([True]*4), 'synthetic-hash', n_boot=20)
    assert result['fpr_scam']['rate'] == 0
    assert result['fpr_at_least_uncertain']['rate'] == .5
    assert result['recall_scam']['rate'] == .5
    assert result['thresholds']['kind'].startswith('explicit frozen per-family')
    adapter = FrozenDecisions(probability, states, 'synthetic-hash')
    with pytest.raises(ValueError, match='other rows'):
        adapter.transform([.9, .9])


def test_family_role_mapping_keeps_absolute_gate_and_explicit_uci_na():
    report = {'fpr_scam': {'rate': .005}, 'fpr_at_least_uncertain': {'rate': .05},
              'recall_scam': {'rate': .8, 'ci95': [.75, .85], 'n': 20},
              'student_recall': {'rate': .8}}
    block = {'baseline': {'label_binary': copy.deepcopy(report),
                          'label_scam': copy.deepcopy(report)},
             'family': {'label_binary': copy.deepcopy(report), 'label_scam': copy.deepcopy(report)}}
    internal = {fold: {seed: copy.deepcopy(block) for seed in ('1', '2', '3')} for fold in FOLDS}
    external = {name: {seed: copy.deepcopy(block) for seed in ('1', '2', '3')}
                for name in ('groundlm', 'smishx')}
    states = {fold: {seed: {arm: {'status': 'ok'} for arm in ('baseline', 'family')}
                     for seed in ('1', '2', '3')} for fold in FOLDS}
    for seed in ('1', '2', '3'):
        for arm in ('baseline', 'family'):
            internal['source_holdout:holdout_uci_sms_spam'][seed][arm][
                'label_scam']['recall_scam'] = {'rate': None, 'ci95': [None, None], 'n': 0}
    assert family_gate(internal, external, states)['decision'] == 'go'
    external['groundlm']['2']['family']['label_scam']['fpr_scam']['rate'] = .02
    assert family_gate(internal, external, states)['decision'] == 'kill'
    external['groundlm']['2']['family']['label_scam']['fpr_scam']['rate'] = .005
    internal['lang_shift']['1']['family']['label_binary']['recall_scam']['rate'] = .7
    assert family_gate(internal, external, states)['decision'] == 'kill'
