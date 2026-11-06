import numpy as np
import pytest
from sklearn.metrics import average_precision_score, precision_recall_curve

from rakshak.eval.url_protocol_bootstrap import SharedBenignRanking, paired_protocol


def fixture():
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 1])
    scores = np.asarray([.8, .5, .5, .1, .9, .5, .5, .2, .05])
    benign = np.arange(4)
    subsets = [('rare', np.asarray([0, 1, 2, 3, 4, 5])),
               ('rare', np.asarray([0, 1, 2, 3, 6, 7])),
               ('less_rare', np.arange(9))]
    return y, scores, benign, subsets


def test_shared_benign_exact_sklearn_weighted_and_replicated_rows():
    y, scores, benign, subsets = fixture()
    ranking = SharedBenignRanking(y, scores, benign, subsets)
    for mass in [np.ones(9), np.asarray([2, 0, 3, 1, 0, 2, 1, 4, 0]), np.zeros(9)]:
        actual = ranking.score(mass)
        for index, (_, rows) in enumerate(subsets):
            repeated = np.repeat(rows, mass[rows].astype(int))
            if len(np.unique(y[repeated])) != 2:
                assert np.isnan(actual[index]).all()
                continue
            expected = average_precision_score(y[repeated], scores[repeated])
            precision, recall, _ = precision_recall_curve(y[repeated], scores[repeated])
            np.testing.assert_allclose(actual[index], [expected, precision[recall >= .9].max()],
                                       rtol=0, atol=1e-14)
    with pytest.raises(ValueError, match='shared benign'):
        SharedBenignRanking(y, scores, benign, [('bad', np.asarray([0, 1, 4]))])


def test_paired_protocol_matches_explicit_domain_resampling_and_fixed_seed_mean():
    y, scores, benign, subsets = fixture()
    domains = np.asarray(['a', 'b', 'c', 'd', 'a', 'b', 'e', 'f', 'g'])
    models = {'s1': scores, 's2': -scores, 's3': scores*.5, 'teacher': scores+.1}
    out = paired_protocol(y, models, benign, subsets, domains,
                          comparisons={'teacher': ['teacher']}, n_boot=37, seed=13)
    reference = []
    _, codes = np.unique(domains, return_inverse=True)
    rng = np.random.default_rng(13)
    for _ in range(37):
        mass = np.bincount(rng.integers(7, size=7), minlength=7)[codes]
        values = []
        for values_model in models.values():
            aps = []
            for _, rows in subsets[:2]:
                sampled = np.repeat(rows, mass[rows])
                aps.append(average_precision_score(y[sampled], values_model[sampled])
                           if len(np.unique(y[sampled])) == 2 else np.nan)
            values.append(np.mean(aps))
        reference.append(values[3]-np.mean(values[:3]))
    valid = np.asarray(reference)[np.isfinite(reference)]
    entry = out['rates']['rare']['comparisons']['teacher']['ap']
    assert entry['valid_draws'] == len(valid)
    np.testing.assert_allclose(entry['ci95'], np.quantile(valid, [.025, .975]), atol=1e-14)
    point = [np.mean([average_precision_score(y[rows], values[rows])
                      for _, rows in subsets[:2]]) for values in models.values()]
    assert entry['point'] == pytest.approx(point[3]-np.mean(point[:3]))
    same = paired_protocol(y, models, benign, subsets, domains,
                           comparisons={'teacher': ['teacher']}, n_boot=37, seed=13)
    assert same['draw_sha256'] == out['draw_sha256']
