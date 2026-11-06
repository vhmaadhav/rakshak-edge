"""Opt-in features round-trip and preserve calibration-ID isolation."""

import json

import numpy as np
import pytest

from rakshak.fusion.fuse import FEATURE_NAMES, TRANSACTIONAL_NAMES, FusionModel, feature_matrix


def test_opt_in_context_changes_only_extended_schema_and_fits_cal_ids():
    texts = ['OTP 123456. Do not share.', 'Share OTP for bank verification',
             'Your parcel has arrived.', 'ordinary invented fixture'] * 8
    message = np.tile([.8, .9, .4, .1], 8)
    base = feature_matrix(texts, message, np.full(32, np.nan), has_url=np.zeros(32))
    x = feature_matrix(texts, message, np.full(32, np.nan), has_url=np.zeros(32),
                       transactional=True)
    np.testing.assert_equal(x[:, :len(FEATURE_NAMES)], base)
    np.testing.assert_equal(x[0, -3:], [1, 1, 0])
    assert x[1, -2] == 0
    ids = np.asarray([f'fixture{i}' for i in range(32)])
    y = np.tile([0, 1, 0, 0], 8)
    kwargs = dict(ids=ids, cal_ids=ids[:24], train_ids=[], test_ids=ids[24:])
    model = FusionModel(transactional=True).fit(x, y, **kwargs)
    before = model.transform(x)
    altered, labels = x.copy(), y.copy()
    altered[24:] = np.nan
    labels[24:] = -99
    other = FusionModel(transactional=True).fit(altered, labels, **kwargs)
    np.testing.assert_allclose(other.transform(x), before)
    restored = FusionModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.feature_names == FEATURE_NAMES + TRANSACTIONAL_NAMES
    np.testing.assert_allclose(restored.transform(x), before)
    with pytest.raises(ValueError, match='matrix'):
        FusionModel().fit(x, y, **kwargs)
