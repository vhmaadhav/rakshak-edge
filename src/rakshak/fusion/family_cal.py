"""Fixed train-independent family routing; sigmoid/threshold fits use cal stages only."""

import re

import numpy as np

from rakshak.canonicalize import canonicalize
from rakshak.fusion.cal_only import predict_frozen
from rakshak.fusion.calibrate import PlattScaling, logits
from rakshak.fusion.fuse import FusionModel
from rakshak.fusion.selective import SelectiveThresholds, conformal_fpr, search_thresholds
from rakshak.fusion.transactional_profile import PATTERNS

FAMILIES = ('otp', 'upi', 'bank', 'delivery', 'telecom_utility')
MIN_NEGATIVE, MIN_POSITIVE = 200, 20


def assign_families(texts):
    result = []
    for text in texts:
        normalized = canonicalize(str(text)).normalized
        result.append(next((name for name in FAMILIES
                            if re.search(PATTERNS[name], normalized, re.I)), 'other'))
    return np.asarray(result)


def fit_family_calibration(base, x, y, ids, texts):
    x, y, ids = np.asarray(x), np.asarray(y), np.asarray(ids, dtype=str)
    if base['status'] != 'ok' or len(texts) != len(ids) or y.shape != ids.shape:
        raise ValueError('feasible global state and aligned calibration rows required')
    if x.shape[0] != len(ids) or not np.isin(y, [0, 1]).all():
        raise ValueError('known fraud cal labels required')
    stages = base['cal_stage_ids']
    flattened = [item for values in stages.values() for item in values]
    if (set(stages) != {'fusion', 'probability', 'threshold', 'crc'}
            or len(flattened) != len(set(flattened)) or len(ids) != len(set(ids))
            or set(flattened) != set(ids)):
        raise ValueError('family fits must use exactly the four disjoint original cal stages')
    if set(base['model']['fit_ids']) != set(stages['fusion']):
        raise ValueError('stacker fitting IDs differ from declared cal stage')
    parts = {name: np.isin(ids, values) for name, values in stages.items()}
    raw = logits(FusionModel.from_dict(base['model']).transform(x))
    family = assign_families(texts)
    fitted = {}
    for name in FAMILIES:
        masks = {stage: mask & (family == name) for stage, mask in parts.items()}
        counts = {stage: {'negative': int((y[mask] == 0).sum()),
                           'positive': int((y[mask] == 1).sum())} for stage, mask in masks.items()}
        if any(counts[stage]['negative'] < MIN_NEGATIVE
               or counts[stage]['positive'] < MIN_POSITIVE
               for stage in ('probability', 'threshold')):
            fitted[name] = {'mode': 'global', 'reason': 'sparse_fixed_cal_stages', 'counts': counts}
            continue
        cal = PlattScaling().fit(raw[masks['probability']], y[masks['probability']])
        scores = cal.transform(raw)
        selected = search_thresholds(y[masks['threshold']], scores[masks['threshold']],
                                      alpha=.01, beta=.05)
        if selected['status'] != 'ok':
            fitted[name] = {'mode': 'global', 'reason': 'infeasible_family_thresholds',
                            'counts': counts, 'search': selected}
            continue
        fitted[name] = {'mode': 'family', 'counts': counts, 'calibrator': cal.to_dict(),
                        'thresholds': selected['thresholds'], 'threshold_selection': selected,
                        'crc_diagnostic': conformal_fpr(
                            y[masks['crc']], scores[masks['crc']], alpha=.01,
                            t_safe=selected['thresholds']['t_safe'])}
    return {'schema': 'rakshak-template-cal-c1', 'base': base, 'families': fitted,
            'other': 'global', 'priority': list(FAMILIES),
            'minimum_negative': MIN_NEGATIVE, 'minimum_positive': MIN_POSITIVE}


def predict_family_calibration(state, x, texts):
    if (state['schema'] != 'rakshak-template-cal-c1' or state['priority'] != list(FAMILIES)
            or set(state['families']) != set(FAMILIES) or state['other'] != 'global'):
        raise ValueError('fixed family policy required')
    x = np.asarray(x)
    if len(x) != len(texts):
        raise ValueError('aligned feature/text rows required')
    probabilities = predict_frozen(state['base'], x)
    states = SelectiveThresholds.from_dict(state['base']['thresholds']).transform(probabilities)
    family = assign_families(texts)
    raw = logits(FusionModel.from_dict(state['base']['model']).transform(x))
    for name, entry in state['families'].items():
        mask = family == name
        if entry['mode'] == 'family' and mask.any():
            probabilities[mask] = PlattScaling.from_dict(entry['calibrator']).transform(raw[mask])
            states[mask] = SelectiveThresholds.from_dict(entry['thresholds']).transform(
                probabilities[mask])
        elif entry['mode'] != 'global' and entry['mode'] != 'family':
            raise ValueError('unknown family calibration mode')
    return probabilities, states, family
