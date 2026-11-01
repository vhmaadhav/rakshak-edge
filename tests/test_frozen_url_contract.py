"""Synthetic session stub tests URL aggregation without loading/running any model."""

import numpy as np

from rakshak.fusion.url_expert import FrozenURLExpert


def test_url_free_missing_and_cached_multiple_url_max():
    expert = object.__new__(FrozenURLExpert)
    expert.cache = {'https://example.com/a': .2, 'https://example.org/b': .8}
    scores, present = expert.score(['ordinary fixture',
                                   'https://example.com/a and https://example.org/b'])
    assert np.isnan(scores[0]) and scores[1] == .8
    assert present.tolist() == [False, True]
