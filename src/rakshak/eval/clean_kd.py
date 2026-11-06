"""Recorded mean-F1 gate for the clean KD versus hard-label-only experiment."""

import numpy as np

FOLDS = {'campaign', 'lang_shift', 'temporal', 'source_holdout:holdout_scamshield',
         'source_holdout:holdout_uci_sms_spam', 'source_holdout:holdout_mishra_soni_smsphishing'}


def clean_kd_gate(comparisons):
    checks, missing = {}, sorted(FOLDS - set(comparisons))
    for fold in sorted(FOLDS & set(comparisons)):
        block = comparisons[fold].get('kd')
        if block is None or block.get('training_seeds') != 3:
            missing.append(fold)
            continue
        values = [block['a'], block['b'], *block['a_ci95'], *block['b_ci95']]
        if (not np.isfinite(values).all() or np.any((np.asarray(values) < 0)
                                                    | (np.asarray(values) > 1))
                or any(block[key][0] > block[key][1] for key in ('a_ci95', 'b_ci95'))):
            missing.append(fold)
            continue
        half = (block['b_ci95'][1] - block['b_ci95'][0])/2
        checks[fold+'/no_fold_loss'] = block['a'] - block['b'] >= -half
        if fold == 'lang_shift':
            checks['lang_shift/kd_lower_above_control'] = block['a_ci95'][0] > block['b']
    if any(not value for value in checks.values()):
        decision = 'kill'
    elif missing or set(comparisons) != FOLDS:
        decision = 'unmeasurable'
    else:
        decision = 'go'
    return {'decision': decision, 'checks': checks, 'missing': missing,
            'scope': 'legacy spam-inclusive mean F1, conditional on three fixed training seeds'}
