"""Synthetic curve receipts cannot silently replace a failed or changed policy run."""

import importlib.util
import json
from pathlib import Path

import pytest


def test_curve_report_requires_exact_complete_policy_and_all_populations(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('policy_curve_fixture',
                                                repo/'kaggle/jobs/policy_curves.py')
    job = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(repo/'kaggle/jobs'))
    try:
        spec.loader.exec_module(job)
    finally:
        sys.path.pop(0)
    policy = {'synthetic': True}
    path = tmp_path/'frozen_required_alert_policy.json'
    path.write_text(json.dumps(policy))
    metrics = {'policy_sha256': job.sha(path), 'policy': policy,
               'internal': dict.fromkeys(job.FOLDS, {}),
               'external': {'groundlm': {}, 'smishx': {}}}
    (tmp_path/'metrics.json').write_text(json.dumps(metrics))
    receipt = tmp_path/'results.json'
    receipt.write_text(json.dumps({'tree_sha256': '1'*64, 'status': 'ok'}))
    args = (tmp_path, '1'*64, job.sha(tmp_path/'metrics.json'), job.sha(path))
    assert job.saved_policy(*args)[1] == metrics
    receipt.write_text(json.dumps({'tree_sha256': '1'*64, 'status': 'error'}))
    with pytest.raises(ValueError, match='complete exact'):
        job.saved_policy(*args)
    receipt.write_text(json.dumps({'tree_sha256': '1'*64, 'status': 'ok'}))
    path.write_text('{}')
    with pytest.raises(ValueError, match='complete exact'):
        job.saved_policy(*args)
