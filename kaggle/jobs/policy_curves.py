"""Format only saved policy risk-coverage curves, no models, corpus or resampling."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

from _common import write_metrics

FOLDS = {'campaign', 'lang_shift', 'temporal', 'source_holdout:holdout_scamshield',
         'source_holdout:holdout_uci_sms_spam', 'source_holdout:holdout_mishra_soni_smsphishing'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def saved_policy(root, tree, metrics_hash, policy_hash):
    found = []
    for path in Path(root).rglob('results.json'):
        receipt = json.loads(path.read_text())
        if receipt.get('tree_sha256') != tree:
            continue
        metrics, policy = path.with_name('metrics.json'), path.with_name(
            'frozen_required_alert_policy.json')
        if (receipt['status'] != 'ok' or receipt.get('failures')
                or sha(metrics) != metrics_hash or sha(policy) != policy_hash):
            raise ValueError('complete exact policy outcome required')
        data = json.loads(metrics.read_text())
        if (data['policy_sha256'] != policy_hash or set(data['internal']) != FOLDS
                or set(data['external']) != {'groundlm', 'smishx'}
                or data['policy'] != json.loads(policy.read_text())):
            raise ValueError('policy manifest and reporting populations differ')
        found.append((path.parent, data))
    if len(found) != 1:
        raise ValueError('one exact completed policy source required')
    return found[0]


def run(ctx):
    p = argparse.ArgumentParser()
    p.add_argument('--source-tree', required=True)
    p.add_argument('--metrics-sha256', required=True)
    p.add_argument('--policy-sha256', required=True)
    args = p.parse_args(ctx.argv)
    folder, metrics = saved_policy(ctx.input_root, args.source_tree,
                                   args.metrics_sha256, args.policy_sha256)
    # Verified exact release on official PyPI; installation occurs only on Kaggle.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',
                    '--disable-pip-version-check', 'matplotlib==3.10.8'], check=True)
    import matplotlib

    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    populations = {**metrics['internal'], **metrics['external']}
    names = sorted(populations)
    plots = {}
    for label in ('label_binary', 'label_scam'):
        for endpoint in ('auto_decided', 'alert_endpoint'):
            fig, axes = plt.subplots(2, 4, figsize=(17, 7), sharex=True, sharey=True)
            for axis, name in zip(axes.flat, names, strict=True):
                block = populations[name][label]['risk_coverage_'+endpoint]
                curve = block['curve']
                axis.plot(curve['coverage'], curve['risk'], color='#174ea6', linewidth=1.6)
                axis.set(xlim=(0, 1), ylim=(0, 1), xlabel='Ranked coverage', ylabel='Error rate')
                axis.set_title(name.replace('source_holdout:holdout_', ''), fontsize=9)
                axis.grid(alpha=.2)
                axis.text(.97, .95, f'Saved AURC={block["aurc"]:.4f}',
                          ha='right', va='top', transform=axis.transAxes, fontsize=8)
            scope = ('within auto-decided rows only; excludes UNCERTAIN'
                     if endpoint == 'auto_decided'
                     else 'all eligible rows; positive = SCAM or UNCERTAIN')
            fig.suptitle(f'Evidence-required alert (KILL): {label}; {scope}', fontsize=12)
            fig.text(.5, .005, 'Saved confidence-ranked curves only; not alternative thresholds. '
                     'In-sample policy selection; no independent Indian validation.',
                     ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .035, 1, .94))
            filename = f'policy_curves_{label}_{endpoint}.png'
            path = ctx.work/filename
            fig.savefig(path, dpi=160)
            plt.close(fig)
            plots[filename] = {'sha256': sha(path), 'bytes': path.stat().st_size,
                              'endpoint': endpoint, 'label': label}
    for original, target in (('metrics.json', 'policy_metrics.json'),
                              ('results.json', 'policy_results.json'),
                              ('frozen_required_alert_policy.json',
                               'frozen_required_alert_policy.json')):
        shutil.copyfile(folder/original, ctx.work/target)
    report = {'scope': 'saved aggregate formatting only, no inference/fit/resampling',
              'source_tree': args.source_tree, 'source_metrics_sha256': args.metrics_sha256,
              'source_policy_sha256': args.policy_sha256, 'plots': plots,
              'original_decision_unchanged': metrics['decision'],
              'matplotlib': importlib.metadata.version('matplotlib'),
              'shipment_approved': False, 'hf_publish': 'HF publish pending'}
    write_metrics('metrics.json', report)
    ctx.results['summary'] = ('4 saved-curve figures; original policy '
                              +metrics['decision']['decision'])
    return True
