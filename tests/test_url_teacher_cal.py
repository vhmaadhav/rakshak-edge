from copy import deepcopy

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from rakshak.eval.url_teacher_cal import RankedAP, eligibility, paired_cal_ap, prevalence_weights


def test_ranked_weighted_ap_matches_primary_api_with_ties_and_zero_weights():
    y = np.asarray([0, 1, 1, 0, 1, 0])
    scores = np.asarray([.7, .7, .4, .2, .2, .2])
    for weights in (np.ones(6), np.asarray([0, 3, .5, 2, 1, 0])):
        actual = RankedAP(y, scores).score(weights)
        assert actual == pytest.approx(average_precision_score(y, scores, sample_weight=weights))
    weights = prevalence_weights(y, np.asarray([1, 0, 2, 3, 1, 4]), .0005)
    assert weights[y == 1].sum() == pytest.approx(.0005)
    assert weights[y == 0].sum() == pytest.approx(.9995)
    with pytest.raises(ValueError, match='both classes'):
        RankedAP(y, scores).score(y)


def test_cal_pairing_fixed_seed_mean_and_eligibility_guards():
    y = np.asarray([0, 1]*20)
    domains = [f'domain{i//2}' for i in range(40)]
    teacher, students = y*.8+.1, np.zeros((3, 40))
    blocks = {str(prevalence): paired_cal_ap(y, teacher, students, domains,
                                            prevalence=prevalence)
              for prevalence in (.0005, .001)}
    assert eligibility('phreshphish', blocks)['decision'] == 'go'
    assert blocks['0.0005']['a'] == pytest.approx(1.)
    assert blocks['0.0005']['b'] == pytest.approx(.0005)
    assert blocks['0.0005']['valid_draws'] == 1000
    assert blocks['0.001']['delta_ci95'][0] > 0
    changed = deepcopy(blocks)
    changed['0.001']['delta_ci95'][0] = 0
    assert eligibility('phreshphish', changed)['decision'] == 'kill'
    changed = deepcopy(blocks)
    changed['0.001']['valid_draws'] = 949
    assert eligibility('phreshphish', changed)['decision'] == 'unmeasurable'
    assert eligibility('phreshphish', {'0.0005': blocks['0.0005']})[
        'decision'] == 'unmeasurable'
    with pytest.raises(ValueError, match='three aligned'):
        paired_cal_ap(y, teacher, students[:2], domains)
