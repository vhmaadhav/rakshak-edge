"""Smoke job: the bundle imports, the environment reports, and canonicalize runs on three messages.

Launch: kaggle/launch.sh smoke --internet   (CPU; no GPU needed)
Returns True when canonicalize gives non-empty normalized text for every sample.
Writes metrics.json.
"""

from dataclasses import asdict

from _common import env_report, seed_everything, timed, write_metrics

import rakshak
from rakshak.canonicalize import canonicalize

SAMPLES = [
    'Dear customer, your SBI KYC expires today. Verify at http://sbi-kyc-update.xyz/now',
    'Ｈｅｌｌｏ, your OTP is 4​8​1​2 do not share it',
    'Lunch at 1pm tomorrow?',
]


def run(ctx):
    seed_everything(0)
    env = env_report()
    ctx.log(f'env: {env}')
    timings = {}
    rows = []
    with timed('canonicalize_s', timings):
        for text in SAMPLES:
            try:
                out = canonicalize(text)
                ok = isinstance(out.normalized, str) and len(out.normalized) > 0
                error = None
            except Exception as exc:  # recorded per sample, not hidden
                out, ok, error = None, False, f'{type(exc).__name__}: {exc}'
            output = asdict(out) if out is not None else None
            rows.append({'input': text, 'output': output, 'ok': ok, 'error': error})
    passed = all(row['ok'] for row in rows)
    metrics = {
        'job': 'smoke',
        'rakshak_version': getattr(rakshak, '__version__', None),
        'env': env,
        'timings': timings,
        'canonicalize': rows,
        'passed': passed,
    }
    write_metrics('metrics.json', metrics)
    ok_count = sum(row['ok'] for row in rows)
    device = 'gpu' if env['gpu'] else 'cpu'
    ctx.results['summary'] = f'canonicalize {ok_count}/{len(rows)} py{env["python"]} {device}'
    ctx.log(f'wrote metrics.json: {ok_count}/{len(rows)} samples ok')
    return passed
