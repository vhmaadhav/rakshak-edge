"""Matched no-KD/clean-KD student controls, no extra transfer-pool training rows."""

import json

from _common import write_metrics

from rakshak.eval.clean_kd import clean_kd_gate


def run(ctx):
    import sms_source_balance as control

    status = control.run(
        ctx, arms=('erm', 'kd'), teacher_arms={'erm': False, 'kd': True},
        criterion='KD lang_shift lower campaign mean-F1 CI above no-KD point mean; '
                  'no frozen-fold mean-F1 loss greater than no-KD CI half-width')
    path = ctx.work/'metrics.json'
    metrics = json.loads(path.read_text())
    metrics['clean_kd_gate'] = clean_kd_gate(metrics['comparisons'])
    metrics['teacher_target_scope'] = 'cached single teacher, train IDs only, not OOF targets'
    metrics['control_arm'] = 'erm = hard-label-only no-KD in this experiment'
    write_metrics('metrics.json', metrics)
    ctx.results['summary'] = f'clean KD gate {metrics["clean_kd_gate"]["decision"]}; {status}'
    return status
