"""Reconstruct only the pinned URL expert's original grouped calibration IDs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rakshak.eval import data as D
from rakshak.eval.phreshphish import grouped_cal
from rakshak.fusion.frozen_bundle import probability_threshold, sha256
from rakshak.models.encoding import apply_standardizer, encode_batch
from rakshak.url_features import url_features

URL_METRICS = '70b34a58853d1ac2d29ed7a1dadaf506b871c8999a8ba712d1021a7ff4ddec7f'
CAL_SOURCE = '839fa8fda0a9b0135c87299e67f39c9170ecbde9f0da833b38902dd35536210c'


def ids_hash(values):
    return hashlib.sha256(json.dumps(list(values), ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def native_cal_rows(root):
    """Read URL/label columns only for official train IDs; never load benchmark columns."""
    partition = Path(root)/D.SPLIT_DIR/'url_phreshphish.json'
    train_ids = D.read_partitions(partition)['official']['train']
    if not train_ids or len(train_ids) != len(set(train_ids)):
        raise ValueError('unique original official URL train IDs required')
    columns = ['id', 'url', 'label', 'reg_domain', 'url_norm']
    frame = pd.read_parquet(Path(root)/'urls_phreshphish.parquet', columns=columns,
                            filters=[('id', 'in', train_ids)])
    if not frame.id.is_unique or set(frame.id) != set(train_ids):
        raise ValueError('native URL train-only parquet filter changed membership')
    # Preserve original physical row order and partition order separately. grouped_cal
    # returns sorted physical positions, exactly like the original full-table builder.
    rows = pd.Index(frame.id).get_indexer(train_ids)
    labels = frame.label.to_numpy(dtype=int)
    reg = frame.reg_domain.fillna('').astype(str).to_numpy()
    norm = frame.url_norm.astype(str).to_numpy()
    groups = np.where(reg != '', reg, norm)
    _, cal = grouped_cal(rows, labels, groups[rows], .05, 0)
    selected = frame.iloc[cal].copy()
    if set(selected.label) != {0, 1} or not set(selected.id) <= set(train_ids):
        raise ValueError('original native URL cal requires both classes and train membership')
    ids = selected.id.astype(str).tolist()
    return selected, {'split_id': 'url_student/phreshphish/official-train/grouped-cal-seed0-f05',
                      'partition_file_sha256': sha256(partition),
                      'official_train_ids_sha256': ids_hash(train_ids),
                      'cal_ids': ids, 'cal_ids_sha256': ids_hash(ids),
                      'cal_rows': len(ids), 'cal_positive_rows': int(selected.label.sum()),
                      'cal_negative_rows': int((selected.label == 0).sum()),
                      'cal_fraction': .05, 'group_seed': 0,
                      'grouping': 'registered domain; url_norm fallback',
                      'benchmark_columns_loaded': False,
                      'rows_scored': 'original native URL cal only, never URL test or SMS'}


def native_url_probabilities(url, values):
    """Score each native URL directly, as in the original URL job, without SMS extraction."""
    from scipy.special import expit

    scores = []
    for start in range(0, len(values), 256):
        part = values[start:start+256]
        features = [url_features(value) for value in part]
        matrix = np.asarray([[row[name] for name in url.scaler['columns']]
                             for row in features], dtype=np.float32)
        side = apply_standardizer(matrix, np.asarray(url.scaler['mean']),
                                  np.asarray(url.scaler['std']))
        logits = url.session.run(None, {'raw': encode_batch(part, 200).astype(np.int64),
                                        'side': side})[0].reshape(-1)
        if len(logits) != len(part) or not np.isfinite(logits).all():
            raise ValueError('native URL calibration logits missing or nonfinite')
        scores.extend(expit(logits.astype(np.float64)).tolist())
    result = np.asarray(scores)
    if result.shape != (len(values),):
        raise ValueError('one original native cal URL score required per row')
    return result


def calibrate_native_url(root, source_folder, url, source_root):
    from rakshak.eval.metrics import threshold_for_fpr

    path = Path(source_folder)/'metrics.json'
    if sha256(path) != URL_METRICS:
        raise ValueError('original URL metadata changed')
    meta = json.loads(path.read_text())['meta']
    if (meta['best_student'] != 'charcnn-wide' or meta['cal_fraction'] != .05
            or meta['seeds'] != [0] or meta['max_train'] is not None
            or sha256(Path(source_root)/'rakshak/eval/phreshphish.py') != CAL_SOURCE):
        raise ValueError('native URL calibration recipe/source differs from exported expert')
    frame, provenance = native_cal_rows(root)
    scores = native_url_probabilities(url, frame.url.astype(str).tolist())
    selection = threshold_for_fpr(frame.label.to_numpy(dtype=int), scores, .01)
    selection['threshold'] = probability_threshold(selection['threshold'])
    provenance.update({'threshold_selection': selection, 'url_metrics_sha256': URL_METRICS,
                       'original_cal_source_sha256': CAL_SOURCE,
                       'inference': 'original raw URL core model, same byte encoder/side scaler; '
                                    'no SMS URL extraction or aggregation during native cal',
                       'label_endpoint': 'native URL phishing label, not SMS fraud policy'})
    return provenance
