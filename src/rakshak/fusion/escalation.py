"""Frozen decision-only corroboration policy; no model or external-set fitting."""

import numpy as np

from rakshak.canonicalize import canonicalize
from rakshak.fusion.calibrate import scores_array
from rakshak.rules import run_rules

HIGH = {'credential_request', 'upi_collect_or_pay', 'brand_domain_mismatch',
        'apk_install', 'remote_access_app'}
LINKED = {'kyc_block_urgency', 'account_suspension'}


def corroboration(texts, url_scores, *, url_threshold):
    values = np.asarray(url_scores, dtype=float)
    if values.shape != (len(texts),) or not 0 <= url_threshold <= np.nextafter(1., np.inf):
        raise ValueError('aligned URL scores and finite cal-only threshold required')
    present = np.asarray([bool(canonicalize(text).urls) for text in texts])
    scores_array(values[present], probability=True)
    if not np.isnan(values[~present]).all():
        raise ValueError('URL absence must be represented by NaN')
    reasons = []
    for text, probability, has_url in zip(texts, values, present, strict=True):
        hits = run_rules(canonicalize(text))
        eligible = {hit.type for hit in hits if (hit.type in HIGH and hit.weight >= .85)
                    or (hit.type in LINKED and hit.weight >= .75 and has_url)}
        if has_url and probability >= url_threshold:
            eligible.add('url_expert')
        reasons.append(sorted(eligible))
    return np.asarray([bool(row) for row in reasons]), reasons


def apply_escalation(probability, thresholds, supported):
    probability = scores_array(probability, probability=True)
    supported = np.asarray(supported)
    if supported.shape != probability.shape or not np.isin(supported, [False, True]).all():
        raise ValueError('one corroboration boolean per decision required')
    base = thresholds.transform(probability)
    return np.where((base == 'SCAM') & supported, 'SCAM',
                    np.where((base == 'SAFE') & ~supported, 'SAFE', 'UNCERTAIN'))
