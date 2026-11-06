"""Arm(a)'s prospectively declared legacy/fraud applicability and safety gates."""

import math

from rakshak.fusion.audit import baseline_gate

FOLDS = {'campaign', 'lang_shift', 'temporal', 'source_holdout:holdout_scamshield',
         'source_holdout:holdout_uci_sms_spam', 'source_holdout:holdout_mishra_soni_smsphishing'}


def hardneg_gate(internal, external, states):
    if set(internal) != FOLDS or set(states) != FOLDS:
        return {'decision': 'unmeasurable', 'reason': 'all six frozen folds required'}
    checks, missing, inapplicable = {}, [], []
    for seed in ('1', '2', '3'):
        try:
            ground = external['groundlm'][seed]
            smish = external['smishx'][seed]
            base, candidate = (ground[arm]['label_scam'] for arm in ('baseline', 'transactional'))
            absolute = baseline_gate(candidate, smish['transactional']['label_scam'])
            if absolute['decision'] == 'unmeasurable':
                missing.append(f'{seed}/absolute_external_gate')
            else:
                checks.update({f'{seed}/{name}': value
                               for name, value in absolute['checks'].items()})
            for metric in ('fpr_scam', 'fpr_at_least_uncertain'):
                a, b = candidate[metric]['rate'], base[metric]['rate']
                if a is None or b is None:
                    missing.append(f'{seed}/groundlm/{metric}')
                else:
                    checks[f'{seed}/groundlm/{metric}/no_increase'] = a <= b
            for fold, block in internal.items():
                if set(block) != {'1', '2', '3'} or set(states[fold]) != {'1', '2', '3'}:
                    raise KeyError('missing training seed')
                if any(states[fold][seed][arm]['status'] != 'ok'
                       for arm in ('baseline', 'transactional')):
                    missing.append(f'{seed}/{fold}/cal_state')
                    continue
                for label in ('label_binary', 'label_scam'):
                    b = block[seed]['baseline'][label]['recall_scam']
                    a = block[seed]['transactional'][label]['recall_scam']
                    key = f'{seed}/{fold}/{label}/recall'
                    if (label == 'label_scam' and fold == 'source_holdout:holdout_uci_sms_spam'
                            and b['n'] == a['n'] == 0):
                        inapplicable.append(key)
                        continue
                    values = [a['rate'], b['rate'], *b['ci95']]
                    if (a['n'] != b['n'] or not a['n'] or any(v is None for v in values)
                            or not all(math.isfinite(v) for v in values)):
                        missing.append(key)
                        continue
                    half = (b['ci95'][1] - b['ci95'][0]) / 2
                    if half < 0:
                        missing.append(key)
                        continue
                    checks[key] = a['rate'] - b['rate'] >= -half
        except KeyError as exc:
            missing.append(f'{seed}/missing endpoint or arm: {exc}')
    failed = [name for name, passed in checks.items() if not passed]
    return {'decision': 'kill' if failed else 'unmeasurable' if missing else 'go',
            'checks': checks, 'failed': failed, 'unavailable': missing,
            'fraud_recall_inapplicable': inapplicable,
            'scope': 'prospective arm(a) applicability; legacy guard every fold plus fraud '
                     'where positives exist; explicit UCI NA, no label substitution; all3seeds'}
