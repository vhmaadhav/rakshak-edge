from copy import deepcopy

from rakshak.eval.clean_kd import FOLDS
from rakshak.eval.multilingual_gate import multilingual_gate


def fixture():
    comparisons = {fold: {'multilingual_kd': {'training_seeds': 3, 'a': .7, 'b': .5,
                                             'a_ci95': [.65, .75], 'b_ci95': [.45, .55],
                                             'delta_ci95': [.12, .25]}} for fold in FOLDS}
    artifacts = {(fold, seed): {'bytes': 5000000, 'sha256': 'abc'} for fold in FOLDS
                 for seed in (1, 2, 3)}
    return comparisons, artifacts


def test_gate_go_and_each_predeclared_failure():
    comparisons, artifacts = fixture()
    assert multilingual_gate(comparisons, artifacts)['decision'] == 'go'
    for key, value in [('a_ci95', [.5, .8]), ('delta_ci95', [0, .2])]:
        changed = deepcopy(comparisons)
        changed['lang_shift']['multilingual_kd'][key] = value
        assert multilingual_gate(changed, artifacts)['decision'] == 'kill'
    changed = deepcopy(comparisons)
    changed['campaign']['multilingual_kd']['a'] = .4
    assert multilingual_gate(changed, artifacts)['decision'] == 'kill'
    oversized = deepcopy(artifacts)
    oversized[('campaign', 1)]['bytes'] = 8*1024*1024+1
    assert multilingual_gate(comparisons, oversized)['decision'] == 'kill'


def test_missing_candidate_does_not_promote_ablation():
    comparisons, artifacts = fixture()
    comparisons['lang_shift'] = {'multilingual_nokd': comparisons['lang_shift']['multilingual_kd']}
    result = multilingual_gate(comparisons, artifacts)
    assert result['decision'] == 'unmeasurable' and not result['ablation_promotable']
    comparisons, artifacts = fixture()
    artifacts.pop(('temporal', 3))
    assert multilingual_gate(comparisons, artifacts)['decision'] == 'unmeasurable'
