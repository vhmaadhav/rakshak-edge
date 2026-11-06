"""Cal-only teacher eligibility: paired registered-domain bootstrap and AP proxies."""

import numpy as np


class RankedAP:
    """Fixed score order/ties, matching sklearn's non-interpolated weighted AP."""

    def __init__(self, labels, scores):
        labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
        if (labels.shape != scores.shape or labels.ndim != 1 or not len(labels)
                or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all()):
            raise ValueError('finite binary aligned labels/scores required')
        self.order = np.argsort(-scores, kind='stable')
        self.labels = labels[self.order].astype(float)
        ordered = scores[self.order]
        self.ends = np.r_[np.nonzero(ordered[:-1] != ordered[1:])[0], len(ordered)-1]

    def score(self, weights):
        weights = np.asarray(weights, dtype=float)
        if (weights.shape != self.labels.shape or not np.isfinite(weights).all()
                or (weights < 0).any()):
            raise ValueError('finite nonnegative aligned weights required')
        ordered = weights[self.order]
        tp = np.cumsum(ordered*self.labels)[self.ends]
        total = np.cumsum(ordered)[self.ends]
        if tp[-1] <= 0 or total[-1] <= tp[-1]:
            raise ValueError('weighted AP requires positive mass on both classes')
        precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
        return float(np.sum(np.diff(np.r_[0., tp])/tp[-1]*precision))


def prevalence_weights(labels, multiplicities, prevalence=None):
    labels, counts = np.asarray(labels), np.asarray(multiplicities, dtype=float)
    if (labels.shape != counts.shape or not np.isin(labels, [0, 1]).all()
            or not np.isfinite(counts).all() or (counts < 0).any()):
        raise ValueError('aligned binary labels and finite multiplicities required')
    pos, neg = counts[labels == 1].sum(), counts[labels == 0].sum()
    if min(pos, neg) <= 0:
        return None
    if prevalence is None:
        return counts
    if not 0 < prevalence < 1:
        raise ValueError('prevalence must be between zero and one')
    return counts*np.where(labels == 1, prevalence/pos, (1-prevalence)/neg)


def paired_cal_ap(labels, teacher, students, domains, *, prevalence=None, n_boot=1000, seed=13):
    y, teacher = np.asarray(labels), np.asarray(teacher, dtype=float)
    students, domains = np.asarray(students, dtype=float), np.asarray(domains, dtype=str)
    if (students.shape != (3, len(y)) or teacher.shape != y.shape or domains.shape != y.shape
            or (domains == '').any() or n_boot < 1):
        raise ValueError('three aligned controls and nonempty original cal domains required')
    rankings = [RankedAP(y, values) for values in [teacher, *students]]
    _, codes = np.unique(domains, return_inverse=True)
    groups = int(codes.max())+1
    weights = prevalence_weights(y, np.ones(len(y)), prevalence)
    if weights is None:
        raise ValueError('original cal needs both classes')
    point = [ranking.score(weights) for ranking in rankings]
    a, b = point[0], float(np.mean(point[1:]))
    draws, rng = [], np.random.default_rng(seed)
    for _ in range(n_boot):
        chosen = rng.integers(groups, size=groups)
        multiplicities = np.bincount(chosen, minlength=groups)[codes]
        weights = prevalence_weights(y, multiplicities, prevalence)
        if weights is not None:
            values = [ranking.score(weights) for ranking in rankings]
            draws.append((values[0], float(np.mean(values[1:]))))
    draws = np.asarray(draws, dtype=float).reshape(-1, 2)
    delta = draws[:, 0]-draws[:, 1]
    return {'a': a, 'b': b, 'delta': a-b,
            'a_ci95': np.quantile(draws[:, 0], [.025, .975]).tolist() if len(draws) else None,
            'b_ci95': np.quantile(draws[:, 1], [.025, .975]).tolist() if len(draws) else None,
            'delta_ci95': np.quantile(delta, [.025, .975]).tolist() if len(draws) else None,
            'p_value': (float((1+(np.abs(delta-(a-b)) >= abs(a-b)-1e-12).sum())
                              /(len(draws)+1)) if len(draws) else None),
            'n': len(y), 'n_domains': groups, 'n_boot': n_boot, 'valid_draws': len(draws),
            'prevalence_proxy': prevalence, 'seed': seed, 'student_seeds': 3, 'teacher_seeds': 1,
            'scope': 'in-sample cal selection, not the Phresh test-subset protocol',
            'method': 'paired registered-domain bootstrap; mean AP of three fixed controls'}


def eligibility(dataset, blocks):
    expected = {'0.0005', '0.001'} if dataset == 'phreshphish' else {'natural'}
    if dataset not in {'phreshphish', 'url_domain'} or set(blocks) != expected:
        return {'decision': 'unmeasurable', 'reason': 'all declared cal endpoints required'}
    checks, missing = {}, []
    for name, block in blocks.items():
        ci = block.get('delta_ci95')
        if (block.get('n_boot') != 1000 or block.get('valid_draws', 0) < 950
                or ci is None or len(ci) != 2 or not np.isfinite(ci).all() or ci[0] > ci[1]):
            missing.append(name)
        else:
            checks[name] = ci[0] > 0
    return {'decision': 'kill' if any(not ok for ok in checks.values()) else
            'unmeasurable' if missing else 'go', 'checks': checks, 'missing': missing,
            'scope': 'cal-only KD eligibility, no independent teacher-win claim'}
