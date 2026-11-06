"""Missing arms cannot pass and a favourable paired delta cannot bypass the recorded gate."""

import copy
import importlib.util
from pathlib import Path

from rakshak.eval.clean_kd import FOLDS, clean_kd_gate


def comparisons():
    return {name: {'kd': {'a': .85, 'b': .8, 'a_ci95': [.82, .88],
                          'b_ci95': [.78, .82], 'training_seeds': 3}}
            for name in FOLDS}


def test_clean_kd_lower_bound_and_every_fold_guard():
    values = comparisons()
    assert clean_kd_gate(values)['decision'] == 'go'
    # A positive point gain is insufficient if the KD lower bound crosses the control.
    values['lang_shift']['kd']['a_ci95'] = [.79, .9]
    assert clean_kd_gate(values)['decision'] == 'kill'
    values = comparisons()
    values['temporal']['kd']['a'] = .75
    assert clean_kd_gate(values)['decision'] == 'kill'
    partial = copy.deepcopy(values)
    del partial['campaign']
    assert clean_kd_gate(partial)['decision'] == 'kill'
    del values['temporal']
    assert clean_kd_gate(values)['decision'] == 'unmeasurable'


def test_clean_kd_bundle_includes_the_actual_shared_guarded_runner():
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('clean_kd_builder', repo/'kaggle/build_kernel.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert 'kaggle/jobs/sms_source_balance.py' in module.bundle_names('sms_clean_kd')
