"""Predeclared multilingual candidate gate; the no-KD hybrid is diagnostic only."""

import numpy as np

from rakshak.eval.clean_kd import FOLDS, clean_kd_gate


def multilingual_gate(comparisons, artifacts):
    mapped = {fold: {'kd': arms['multilingual_kd']} for fold, arms in comparisons.items()
              if 'multilingual_kd' in arms}
    result = clean_kd_gate(mapped)
    missing, checks = result['missing'], result['checks']
    language = mapped.get('lang_shift', {}).get('kd')
    if language is not None:
        delta = language.get('delta_ci95', [])
        if len(delta) != 2 or not np.isfinite(delta).all() or delta[0] > delta[1]:
            missing.append('lang_shift/paired_delta_ci95')
        else:
            checks['lang_shift/paired_positive'] = delta[0] > 0
    expected = {(fold, seed) for fold in FOLDS for seed in (1, 2, 3)}
    if set(artifacts) != expected:
        missing.append('all_candidate_artifacts')
    for key, info in artifacts.items():
        size = info.get('bytes')
        if not isinstance(size, int) or size <= 0 or not info.get('sha256'):
            missing.append(str(key)+'/weights')
        else:
            checks[str(key)+'/size_le_8MiB'] = size <= 8*1024*1024
    decision = ('kill' if any(not ok for ok in checks.values()) else
                'unmeasurable' if missing or set(comparisons) != FOLDS else 'go')
    return {'decision': decision, 'checks': checks, 'missing': sorted(set(missing)),
            'scope': result['scope'], 'candidate': 'multilingual_kd',
            'ablation_promotable': False}
