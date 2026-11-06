"""Exact shared-benign subset metrics with paired registered-domain bootstrap.

Only positive score knots can increase precision or AP. Compute benign cumulative
mass once per model/draw, then reuse it at every original subset's positive knots.
This avoids sorting/scanning the shared benign pool separately for 975 subsets.
No pooling of subsets, prevalence reweighting, or seed resampling is performed.
"""

import hashlib

import numpy as np


class SharedBenignRanking:
    def __init__(self, labels, scores, benign, subsets):
        y, scores = np.asarray(labels), np.asarray(scores, dtype=float)
        benign = np.asarray(benign, dtype=np.int64)
        if (y.ndim != 1 or scores.shape != y.shape or not np.isin(y, [0, 1]).all()
                or not np.isfinite(scores).all() or not len(benign)
                or (benign < 0).any() or (benign >= len(y)).any()
                or len(np.unique(benign)) != len(benign) or y[benign].any()):
            raise ValueError('finite aligned binary scores and unique benign rows required')
        self.n = len(y)
        self.negative = benign[np.argsort(-scores[benign], kind='stable')]
        self.names, positives, ends, sizes, knots = [], [], [], [], []
        benign_set = set(benign.tolist())
        offset = 0
        for name, rows in subsets:
            rows = np.asarray(rows, dtype=np.int64)
            if (rows.ndim != 1 or (rows < 0).any() or (rows >= len(y)).any()
                    or len(np.unique(rows)) != len(rows)
                    or set(rows[y[rows] == 0].tolist()) != benign_set):
                raise ValueError('each original subset must use exactly the shared benign rows')
            pos = rows[y[rows] == 1]
            if not len(pos):
                raise ValueError('each original subset requires positive rows')
            pos = pos[np.argsort(-scores[pos], kind='stable')]
            local_ends = np.r_[np.flatnonzero(scores[pos][:-1] != scores[pos][1:]), len(pos)-1]
            positives.append(pos)
            ends.append(offset+local_ends)
            knots.append(len(local_ends))
            sizes.append(len(pos))
            self.names.append(str(name))
            offset += len(pos)
        if not positives:
            raise ValueError('at least one original subset required')
        self.positive = np.concatenate(positives)
        self.ends = np.concatenate(ends)
        self.sizes = np.asarray(sizes)
        self.starts = np.r_[0, np.cumsum(self.sizes)[:-1]]
        self.knot_sizes = np.asarray(knots)
        self.knot_starts = np.r_[0, np.cumsum(self.knot_sizes)[:-1]]
        # side=right includes all benign scores tied with each positive threshold.
        self.negative_prefix = np.searchsorted(-scores[self.negative],
                                               -scores[self.positive[self.ends]], side='right')

    def score(self, multiplicities):
        mass = np.asarray(multiplicities, dtype=float)
        if mass.shape != (self.n,) or not np.isfinite(mass).all() or (mass < 0).any():
            raise ValueError('finite nonnegative multiplicity per original row required')
        pos_prefix = np.r_[0., np.cumsum(mass[self.positive])]
        before = pos_prefix[self.starts]
        tp = pos_prefix[1:][self.ends]-np.repeat(before, self.knot_sizes)
        pos = pos_prefix[self.starts+self.sizes]-before
        total_pos = np.repeat(pos, self.knot_sizes)
        negative_prefix = np.r_[0., np.cumsum(mass[self.negative])]
        fp = negative_prefix[self.negative_prefix]
        precision = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=(tp+fp) > 0)
        delta = np.diff(np.r_[0., tp])
        delta[self.knot_starts] = tp[self.knot_starts]
        recall = np.divide(tp, total_pos, out=np.zeros_like(tp), where=total_pos > 0)
        increments = np.divide(delta, total_pos, out=np.zeros_like(tp), where=total_pos > 0)
        ap = np.add.reduceat(increments*precision, self.knot_starts)
        pr90 = np.maximum.reduceat(np.where(recall >= .9, precision, 0.), self.knot_starts)
        valid = (pos > 0) & (negative_prefix[-1] > 0)
        ap[~valid], pr90[~valid] = np.nan, np.nan
        return np.column_stack([ap, pr90])


def _summary(point, draws):
    valid = np.isfinite(draws)
    values = draws[valid]
    return {'point': float(point), 'ci95': np.quantile(values, [.025, .975]).tolist()
            if len(values) else None, 'valid_draws': int(valid.sum())}


def paired_protocol(labels, model_scores, benign, subsets, domains, *,
                    controls=('s1', 's2', 's3'), comparisons=None, n_boot=1000, seed=13):
    """Average original subset metrics by rate, then fixed student seeds.

    Same global domain multiplicities, regenerated from one seed per model. A
    rate/draw is invalid if ANY of its original subsets loses either class;
    invalid draws never silently change the subset or seed denominator.
    """
    y, domains = np.asarray(labels), np.asarray(domains, dtype=str)
    subsets = [(str(rate), np.asarray(rows, dtype=np.int64)) for rate, rows in subsets]
    if (domains.shape != y.shape or (domains == '').any() or n_boot < 1
            or len(controls) != 3 or not set(controls) <= set(model_scores)):
        raise ValueError('aligned domains and all three fixed controls required')
    comparisons = {} if comparisons is None else comparisons
    for names in comparisons.values():
        if not names or not set(names) <= set(model_scores):
            raise ValueError('all declared comparison models required')
    # Sampling population is exactly the union of original scored subsets.
    universe = np.unique(np.concatenate([rows for _, rows in subsets]))
    _, codes = np.unique(domains[universe], return_inverse=True)
    groups = int(codes.max())+1
    rates = list(dict.fromkeys(rate for rate, _ in subsets))
    by_rate = {rate: np.asarray([i for i, (name, _) in enumerate(subsets) if name == rate])
               for rate in rates}
    points, draws = {}, {}
    for name, scores in model_scores.items():
        ranking = SharedBenignRanking(y, scores, benign, subsets)
        point = ranking.score(np.ones(len(y)))
        points[name] = np.asarray([point[indices].mean(0) for indices in by_rate.values()])
        values = np.full((n_boot, len(rates), 2), np.nan)
        rng = np.random.default_rng(seed)
        mass = np.zeros(len(y))
        for draw in range(n_boot):
            chosen = rng.integers(groups, size=groups)
            mass[universe] = np.bincount(chosen, minlength=groups)[codes]
            block = ranking.score(mass)
            # np.mean deliberately propagates NaN: retain all original subsets.
            values[draw] = [block[indices].mean(0) for indices in by_rate.values()]
        draws[name] = values
    control_point = np.mean([points[name] for name in controls], axis=0)
    control_draws = np.mean([draws[name] for name in controls], axis=0)
    out = {}
    for ri, rate in enumerate(rates):
        row = {'n_subsets': len(by_rate[rate]), 'models': {}, 'comparisons': {}}
        for name in model_scores:
            row['models'][name] = {metric: _summary(points[name][ri, mi], draws[name][:, ri, mi])
                                   for mi, metric in enumerate(('ap', 'p_at_r90'))}
        row['control_mean'] = {metric: _summary(control_point[ri, mi], control_draws[:, ri, mi])
                               for mi, metric in enumerate(('ap', 'p_at_r90'))}
        for name, members in comparisons.items():
            a_point = np.mean([points[key] for key in members], axis=0)
            a_draws = np.mean([draws[key] for key in members], axis=0)
            row['comparisons'][name] = {}
            for mi, metric in enumerate(('ap', 'p_at_r90')):
                delta = a_draws[:, ri, mi]-control_draws[:, ri, mi]
                point = a_point[ri, mi]-control_point[ri, mi]
                entry = _summary(point, delta)
                valid = delta[np.isfinite(delta)]
                entry['p_value'] = (float((1+(np.abs(valid-point) >= abs(point)-1e-12).sum())
                                          /(len(valid)+1)) if len(valid) else None)
                row['comparisons'][name][metric] = entry
        out[rate] = row
    draw_hash = hashlib.sha256()
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        draw_hash.update(rng.integers(groups, size=groups).astype('<i8').tobytes())
    return {'rates': out, 'n_boot': n_boot, 'seed': seed, 'domains': groups,
            'draw_sha256': draw_hash.hexdigest(), 'population': 'union of original subsets',
            'method': ('paired domain multiplicities within each original subset; '
                       'mean subset metrics'),
            'seed_scope': 'conditional on fixed three students and one teacher; no seed resampling'}
