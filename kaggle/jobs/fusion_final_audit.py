"""Saved-output audit of known failures and NA endpoints. No inference, fitting or resampling."""

import argparse
import hashlib
import json
from pathlib import Path

from _common import write_metrics

from rakshak.fusion.decision_audit import available_checks

SOURCE_TREE = '67b2545f3d290a11a565003c15b15639ed70ecaf87861b7c22b44ef32b8c6d84'
METRICS_HASH = '1ccaf29eb90e4542dc93881282255d6f4670651d888bd50abc5e2d3d68378984'
STATES_HASH = '87b5da994a3c7d579362243fc8cfcf8bfa827cd08621842478cff25f02218ce6'


def run(ctx):
    argparse.ArgumentParser().parse_args(ctx.argv)
    matches = []
    for path in Path(ctx.input_root).rglob('results.json'):
        result = json.loads(path.read_text())
        if result.get('tree_sha256') == SOURCE_TREE:
            if result.get('status') != 'ok':
                raise ValueError('successful source required')
            matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError('one completed exact source tree required')
    root = matches[0]
    for name, expected in (('metrics.json', METRICS_HASH),
                           ('frozen_fusion_states.json', STATES_HASH)):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('saved source metrics/states hash mismatch')
    report = json.loads((root / 'metrics.json').read_text())
    if report['frozen_states_sha256'] != STATES_HASH:
        raise ValueError('source frozen-state reference differs')
    decision = available_checks(report)
    write_metrics('metrics.json', {'source_tree': SOURCE_TREE,
                                   'source_metrics_sha256': METRICS_HASH,
                                   'source_states_sha256': STATES_HASH, **decision})
    ctx.results['summary'] = (f"original {report['decision']['decision']}; "
                              f"observable decision {decision['operational_decision']}; "
                              f"{len(decision['observable_failures'])} failed checks, "
                              f"{len(decision['unavailable_endpoints'])} NA endpoints")
    return True
