"""Four cal-campaign stages; returned state is frozen before any test/external scoring."""

import numpy as np

from rakshak.fusion.calibrate import PlattScaling, logits
from rakshak.fusion.fuse import FusionModel
from rakshak.fusion.selective import conformal_fpr, search_thresholds


def campaign_parts(campaigns, *, seed=0):
    groups = np.asarray(campaigns, dtype=str)
    unique = np.unique(groups)
    if len(unique) < 4 or np.any(groups == ''):
        raise ValueError('four nonempty calibration campaigns required')
    np.random.default_rng(seed).shuffle(unique)
    return [np.flatnonzero(np.isin(groups, chunk)) for chunk in np.array_split(unique, 4)]


def fit_cal_only(x, y, ids, campaigns, *, train_ids, test_ids, transactional=False):
    ids, y = np.asarray(ids, dtype=str), np.asarray(y)
    x = np.asarray(x, dtype=float)
    if x.shape[0] != len(ids) or y.shape != ids.shape or len(campaigns) != len(ids):
        raise ValueError('calibration inputs must align')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('known binary fraud calibration labels required')
    # Guard every calibration row, not only fusion-fit rows, against train/test IDs.
    if set(ids) & (set(train_ids) | set(test_ids)) or len(set(ids)) != len(ids):
        raise ValueError('calibration IDs overlap train/test or are duplicated')
    parts = campaign_parts(campaigns)
    if any(set(y[part]) != {0, 1} for part in parts):
        return {'status': 'infeasible', 'reason': 'each fixed cal stage needs both classes'}
    fit, probability, threshold, crc = parts
    model = FusionModel('logistic', transactional=transactional).fit(
        x, y, ids=ids, cal_ids=ids[fit], train_ids=train_ids, test_ids=test_ids)
    raw = model.transform(x)
    calibrator = PlattScaling().fit(logits(raw[probability]), y[probability])
    scores = calibrator.transform(logits(raw))
    selected = search_thresholds(y[threshold], scores[threshold], alpha=.01, beta=.05)
    result = {'status': selected['status'], 'model': model.to_dict(),
              'calibrator': calibrator.to_dict(), 'threshold_selection': selected,
              'cal_stage_ids': {name: ids[part].tolist() for name, part in zip(
                  ('fusion', 'probability', 'threshold', 'crc'), parts, strict=True)}}
    if selected['status'] == 'ok':
        result['thresholds'] = selected['thresholds']
        result['crc'] = conformal_fpr(y[crc], scores[crc],
                                      t_safe=selected['thresholds']['t_safe'], alpha=.01)
    return result


def predict_frozen(state, x):
    if state['status'] != 'ok':
        raise ValueError('infeasible calibration state cannot produce decisions')
    model = FusionModel.from_dict(state['model'])
    calibrator = PlattScaling.from_dict(state['calibrator'])
    return calibrator.transform(logits(model.transform(x)))
