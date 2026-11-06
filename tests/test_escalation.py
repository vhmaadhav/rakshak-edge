"""Synthetic decision truth table and non-corroborating urgency/link-only cases."""

import numpy as np
import pytest

from rakshak.fusion.escalation import apply_escalation, corroboration
from rakshak.fusion.selective import SelectiveThresholds


def test_escalation_preserves_alert_set_and_requires_corroboration():
    thresholds = SelectiveThresholds(.2, .8)
    scores = np.asarray([.1, .2, .5, .8, .99, .1])
    evidence = [False, False, False, False, True, True]
    result = apply_escalation(scores, thresholds, evidence)
    assert result.tolist() == ['SAFE', 'SAFE', 'UNCERTAIN', 'UNCERTAIN', 'SCAM', 'UNCERTAIN']
    assert np.all((thresholds.transform(scores) != 'SAFE') <= (result != 'SAFE'))
    with pytest.raises(ValueError):
        apply_escalation(scores, thresholds, [False])


def test_urgency_and_lexical_url_flag_cannot_substitute_for_expert():
    texts = ['Do this urgently today', 'KYC is pending today',
             'KYC is pending today https://example.com',
             'Ordinary note https://example.com', 'Ordinary note https://example.com']
    values = np.asarray([np.nan, np.nan, .1, .79, .8])
    supported, reasons = corroboration(texts, values, url_threshold=.8)
    assert supported.tolist() == [False, False, True, False, True]
    assert reasons[2] == ['kyc_block_urgency']
    assert reasons[4] == ['url_expert']
    with pytest.raises(ValueError, match='absence'):
        corroboration(['hello'], [.2], url_threshold=.8)
