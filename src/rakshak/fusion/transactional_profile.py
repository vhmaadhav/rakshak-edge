"""Aggregate diagnostic proxies fixed from task categories, never learned from test text."""

import re

import numpy as np

from rakshak.canonicalize import canonicalize
from rakshak.eval.rule_audit import describe_rate

PATTERNS = {
    'bank': r'\b(?:bank|a/c|acct|account|debit(?:ed)?|credit(?:ed)?)\b',
    'upi': r'\b(?:upi|vpa|imps|neft|rtgs)\b',
    'otp': r'\b(?:otp|one[ -]time password|verification code)\b',
    'delivery': r'\b(?:parcel|courier|delivery|delivered|shipment|tracking)\b',
    'telecom_utility': r'\b(?:recharge|telecom|data pack|electricity|utility|bill|broadband)\b',
    'do_not_share': r'\b(?:do not share|don.t share|never share|share mat|mat share)\b',
}


def categories(text, sender=None):
    canon = canonicalize(str(text))
    raw = canon.normalized
    found = {name for name, pattern in PATTERNS.items() if re.search(pattern, raw, re.I)}
    found.add('length:<80' if len(raw) < 80 else 'length:80-159' if len(raw) < 160
              else 'length:>=160')
    digits = sum(c.isdigit() for c in raw)
    found.add('digits:none' if not digits else 'digits:>=10pct' if digits / max(1, len(raw)) >= .1
              else 'digits:<10pct')
    found.add('urls:present' if canon.urls else 'urls:absent')
    if sender is None or str(sender).strip() in {'', 'nan', 'None'}:
        found.add('sender:missing')
    elif re.fullmatch(r'[A-Za-z]{2}-[A-Za-z0-9]{3,9}', str(sender)):
        found.add('sender:alphanumeric_id')
    elif re.fullmatch(r'[+\d -]+', str(sender)):
        found.add('sender:numeric')
    else:
        found.add('sender:other')
    if {'otp', 'do_not_share', 'urls:absent'} <= found:
        found.add('otp_protective_no_url')
    return found


def aggregate_profile(texts, y, decisions, campaigns, *, senders=None, n_boot=1000):
    """Only counts/rates leave this function. Categories overlap, not geography or truth."""
    y, decisions = np.asarray(y), np.asarray(decisions)
    groups = np.asarray(campaigns, dtype=str)
    if (y.shape != decisions.shape or y.shape != groups.shape or len(texts) != len(y)
            or not np.isin(y, [0, 1]).all() or not np.isin(decisions, [0, 1]).all()):
        raise ValueError('aligned binary labels/decisions/campaigns required')
    if senders is not None and len(senders) != len(texts):
        raise ValueError('sender metadata must align')
    rows = [categories(text, None if senders is None else senders[i])
            for i, text in enumerate(texts)]
    result = {}
    for category in sorted(set().union(*rows)):
        mask = np.asarray([category in row for row in rows]) & (y == 0)
        result[category] = describe_rate(decisions[mask], groups[mask], n_boot=n_boot)
    return {'negative_n': int((y == 0).sum()),
            'false_alarm_n': int(((y == 0) & (decisions == 1)).sum()),
            'categories': result, 'scope': 'overlapping lexical/structure proxies, '
            'not human transaction labels or evidence of Indian provenance'}
