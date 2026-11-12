"""Synthetic sender IDs only; checks the five DLT side features."""

import numpy as np
import pytest

from rakshak.models.sender_features import SENDER_FEATURE_NAMES, sender_features, sender_matrix
from tools.inbox_export.scrub import scrub_sender


def test_feature_names_are_fixed():
    assert SENDER_FEATURE_NAMES == ("sender_alphanumeric", "dlt_suffix_P", "dlt_suffix_S",
                                    "dlt_suffix_T", "dlt_suffix_G")


@pytest.mark.parametrize('sender, expected', [
    ('AX-HDFCBK', [1, 0, 0, 0, 0]),
    ('AX-HDFCBK-S', [1, 0, 1, 0, 0]),
    ('VM-AIRTEL-T', [1, 0, 0, 1, 0]),
    ('JD-ICICIT-G', [1, 0, 0, 0, 1]),
    ('AX-HDFCBK-P', [1, 1, 0, 0, 0]),
    ('  ax-hdfcbk-s \n', [1, 0, 1, 0, 0]),
    ('AX-HDFC<NUMBER>-S', [1, 0, 1, 0, 0]),
    (scrub_sender('AX-HDFC5-S'), [1, 0, 1, 0, 0]),
])
def test_dlt_parsing(sender, expected):
    out = sender_features(sender)
    assert out.dtype == np.float32 and out.shape == (5,)
    assert out.tolist() == expected


@pytest.mark.parametrize('sender', [
    None, '', '   ', '9876543210', '+919876543210', '<PHONE>', '<NAME>', '<SENDER>',
    'SBI', 'hdfcbk', 'AX-AB', 'AX-HDFCBK-X', 'AX-HDFCBK-S-T', 'अमित', 'AX HDFCBK',
    float('nan'),
])
def test_unknown_or_numeric_senders_are_all_zero(sender):
    assert sender_features(sender).tolist() == [0, 0, 0, 0, 0]


def test_matrix_shape_and_order():
    matrix = sender_matrix(['AX-HDFCBK-S', None, '9876543210'])
    assert matrix.dtype == np.float32 and matrix.shape == (3, 5)
    assert matrix[0].tolist() == [1, 0, 1, 0, 0]
    assert not matrix[1:].any()
    empty = sender_matrix([])
    assert empty.shape == (0, 5) and empty.dtype == np.float32
    assert np.array_equal(sender_matrix(['VM-AIRTEL-T']), sender_features('VM-AIRTEL-T')[None])


def test_raw_and_scrubbed_forms_agree():
    for raw in ('AX-HDFCBK-S', 'VM-AIRTEL-T', 'JD-ICICIT-G', 'AX-HDFC5-P'):
        assert np.array_equal(sender_features(raw), sender_features(scrub_sender(raw)))
