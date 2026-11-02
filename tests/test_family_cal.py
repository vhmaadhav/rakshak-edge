"""Synthetic calibration-only fixtures, including sparse exact-global fallbacks."""

import copy

import numpy as np
import pytest

from rakshak.fusion.cal_only import fit_cal_only, predict_frozen
from rakshak.fusion.family_cal import (
    assign_families,
    fit_family_calibration,
    predict_family_calibration,
)
from rakshak.fusion.fuse import FEATURE_NAMES
from rakshak.fusion.selective import SelectiveThresholds


def fixture():
    # Each campaign has10negatives/1positive, so every cal stage has250neg/25pos.
    labels = np.tile([0]*10+[1], 100)
    x = np.zeros((len(labels), len(FEATURE_NAMES)))
    x[:, 0] = np.where(labels, .9, .1)
    ids = np.asarray([f'cal{i}' for i in range(len(labels))])
    groups = np.asarray([f'campaign{i//11}' for i in range(len(labels))])
    base = fit_cal_only(x, labels, ids, groups, train_ids=['train-only'], test_ids=['test-only'])
    assert base['status'] == 'ok'
    return base, x, labels, ids


def test_fixed_priority_and_sparse_family_exact_fallback():
    assert assign_families(['bank UPI OTP', 'bank UPI', 'bank parcel', 'parcel bill',
                            'bill due', 'ordinary synthetic note']).tolist() == [
                                'otp', 'upi', 'bank', 'delivery', 'telecom_utility', 'other']
    base, x, y, ids = fixture()
    texts = ['ordinary synthetic fixture']*len(y)
    texts[0] = 'bank synthetic fixture'
    state = fit_family_calibration(base, x, y, ids, texts)
    assert all(entry['mode'] == 'global' for entry in state['families'].values())
    scores, states, _ = predict_family_calibration(state, x, texts)
    assert np.array_equal(scores, predict_frozen(base, x))
    assert np.array_equal(
        states, SelectiveThresholds.from_dict(base['thresholds']).transform(scores))
    with pytest.raises(ValueError, match='four disjoint'):
        fit_family_calibration(
            base, x, y, np.concatenate([np.asarray(['test-only']), ids[1:]]), texts)


def test_eligible_fit_uses_its_own_map_and_never_changes_other_family():
    base, x, y, ids = fixture()
    state = fit_family_calibration(base, x, y, ids, ['bank synthetic fixture']*len(y))
    assert state['families']['bank']['mode'] == 'family'
    assert state['families']['bank']['counts']['probability'] == {'negative': 250, 'positive': 25}
    altered = copy.deepcopy(state)
    altered['families']['bank']['calibrator'] = {'kind': 'platt', 'weight': 0., 'intercept': 2.}
    altered['families']['bank']['thresholds'] = {'t_safe': .2, 't_scam': .8}
    values, states, groups = predict_family_calibration(
        altered, x[:2], ['bank synthetic fixture', 'ordinary synthetic fixture'])
    assert states.tolist() == ['SCAM', 'SAFE']
    assert groups.tolist() == ['bank', 'other']
    assert values[1] == predict_frozen(base, x[:2])[1]
    assert state['base'] == base
