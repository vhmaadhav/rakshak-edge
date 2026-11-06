"""The intervention cannot pass by omitting a failed fold or training seed."""

import copy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def job(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'kaggle/jobs'))
    spec = importlib.util.spec_from_file_location('fusion_candidate_fixture',
                                                root / 'kaggle/jobs/fusion_transactional.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_fold_seed_and_recall_guard_required(job):
    endpoint = {'fpr_scam': {'rate': .005}, 'fpr_at_least_uncertain': {'rate': .05},
                'recall_scam': {'rate': .8, 'ci95': [.7, .9]}, 'student_recall': {'rate': .7}}
    row = {arm: {'label_scam': copy.deepcopy(endpoint)} for arm in job.ARMS}
    internal = {run: {str(seed): copy.deepcopy(row) for seed in (1, 2, 3)}
                for run in job.REQUIRED_RUNS}
    states = {run: {str(seed): {arm: {'status': 'ok'} for arm in job.ARMS}
                         for seed in (1, 2, 3)} for run in job.REQUIRED_RUNS}
    external = {name: {str(seed): copy.deepcopy(row) for seed in (1, 2, 3)}
                for name in ('groundlm', 'smishx')}
    assert job.candidate_decision(internal, external, states)['decision'] == 'go'
    broken = copy.deepcopy(internal)
    broken['lang_shift']['2']['transactional']['label_scam']['recall_scam']['rate'] = .6
    assert job.candidate_decision(broken, external, states)['decision'] == 'kill'
    del broken['lang_shift']['2']
    assert job.candidate_decision(broken, external, states)['decision'] == 'unmeasurable'
    assert job.candidate_decision({}, external, {})['decision'] == 'unmeasurable'
    external['groundlm']['1']['transactional']['label_scam']['fpr_scam']['rate'] = .006
    assert job.candidate_decision(internal, external, states)['decision'] == 'kill'
