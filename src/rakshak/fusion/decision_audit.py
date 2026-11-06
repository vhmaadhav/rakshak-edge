"""Report observable rejection grounds without overriding an incomplete original gate."""


def available_checks(report):
    original = report['decision']
    external = report['external']
    checks, unavailable = {}, []
    for seed in ('1', '2', '3'):
        ground = external['groundlm'][seed]
        smish = external['smishx'][seed]
        base, arm = ground['baseline']['label_scam'], ground['transactional']['label_scam']
        required = {'groundlm_scam_le_1pct': (arm['fpr_scam']['rate'], .01, '<='),
                    'groundlm_alert_le_10pct': (arm['fpr_at_least_uncertain']['rate'], .10, '<='),
                    'groundlm_scam_no_increase': (arm['fpr_scam']['rate'],
                                                 base['fpr_scam']['rate'], '<='),
                    'groundlm_alert_no_increase': (arm['fpr_at_least_uncertain']['rate'],
                                                  base['fpr_at_least_uncertain']['rate'], '<='),
                    'smishx_recall_ge_student': (
                        smish['transactional']['label_scam']['recall_scam']['rate'],
                        smish['transactional']['label_scam']['student_recall']['rate'], '>=')}
        for name, (a, b, sign) in required.items():
            key = f'{seed}/{name}'
            if a is None or b is None:
                unavailable.append(key)
            else:
                checks[key] = {'passes': a <= b if sign == '<=' else a >= b,
                               'candidate': a, 'reference': b, 'comparison': sign}
    for fold, block in report['internal'].items():
        for seed in ('1', '2', '3'):
            key = f'{seed}/{fold}/fraud_recall'
            base = block[seed]['baseline']['label_scam']['recall_scam']
            arm = block[seed]['transactional']['label_scam']['recall_scam']
            if base['rate'] is None or arm['rate'] is None:
                unavailable.append(key)
            else:
                half = (base['ci95'][1] - base['ci95'][0]) / 2
                checks[key] = {'passes': arm['rate'] - base['rate'] >= -half,
                               'candidate': arm['rate'], 'reference': base['rate'],
                               'allowed_loss': half}
    failures = [key for key, check in checks.items() if not check['passes']]
    return {'original_runner_decision': original, 'observable_checks': checks,
            'observable_failures': failures, 'unavailable_endpoints': unavailable,
            'operational_decision': 'kill' if failures else 'unmeasurable' if unavailable
                                    else original['decision'],
            'scope': 'saved-output reporting supplement; original decision retained; '
                     'undefined fraud recall is never substituted with spam recall'}
