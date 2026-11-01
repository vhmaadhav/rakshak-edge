"""Synthetic cal-stage isolation and frozen serialization only."""

import json

import numpy as np
import pytest

from rakshak.fusion.cal_only import campaign_parts, fit_cal_only, predict_frozen
from rakshak.fusion.fuse import FEATURE_NAMES


def test_four_campaign_stages_are_disjoint_and_all_ids_guarded():
    n = 80
    labels = np.arange(n) % 2
    campaigns = np.arange(n) // 2
    ids = np.asarray([f'fixture{i}' for i in range(n)])
    x = np.zeros((n, len(FEATURE_NAMES)))
    x[:, 0] = np.where(labels, .9, .1)
    parts = campaign_parts(campaigns)
    assert sorted(np.concatenate(parts).tolist()) == list(range(n))
    sets = [set(campaigns[part]) for part in parts]
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i+1:])
    state = fit_cal_only(x, labels, ids, campaigns, train_ids=['train'], test_ids=['test'])
    assert state['status'] == 'ok'
    restored = json.loads(json.dumps(state))
    np.testing.assert_allclose(predict_frozen(restored, x), predict_frozen(state, x))
    # An ID in a non-fusion calibration stage must still be rejected.
    bad = ids[parts[1][0]]
    with pytest.raises(ValueError, match='overlap'):
        fit_cal_only(x, labels, ids, campaigns, train_ids=[bad], test_ids=[])


def test_single_class_cal_stage_is_infeasible_not_silent_fallback():
    x = np.zeros((8, len(FEATURE_NAMES)))
    x[:, 0] = .1
    state = fit_cal_only(x, np.zeros(8), list('abcdefgh'), np.arange(8),
                         train_ids=[], test_ids=[])
    assert state['status'] == 'infeasible' and 'model' not in state
