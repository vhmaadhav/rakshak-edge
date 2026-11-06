"""Proxy diagnostics use invented messages and never return raw content."""

import json

from rakshak.fusion.transactional_profile import aggregate_profile, categories


def test_protective_composition_and_missing_sender_are_explicit():
    found = categories('OTP 123456. Do not share with anyone.')
    assert {'otp', 'do_not_share', 'otp_protective_no_url', 'sender:missing'} <= found
    assert 'otp_protective_no_url' not in categories('OTP 123456. Do not share. https://example.com')


def test_aggregate_negative_denominators_and_no_raw_text():
    text = 'OTP 123456. Do not share with anyone.'
    result = aggregate_profile([text, 'Parcel arrived', 'fraud fixture'], [0, 0, 1],
                               [1, 0, 1], ['a', 'b', 'c'], n_boot=20)
    assert result['negative_n'] == 2 and result['false_alarm_n'] == 1
    assert result['categories']['otp']['events'] == 1
    assert result['categories']['otp']['n'] == 1
    assert text not in json.dumps(result) and '123456' not in json.dumps(result)
