"""Aggregate-only transactional student diagnostics on Kaggle; no training or tuning."""

import argparse
import json

import numpy as np
import pandas as pd
from _common import write_metrics

from rakshak.canonicalize import canonicalize
from rakshak.datasets.common import cached_registered_domain
from rakshak.datasets.dedup import assign_campaigns, mark_exact_duplicates
from rakshak.eval import data as D
from rakshak.fusion.source_inputs import (
    METRICS_HASH,
    SOURCE_TREE,
    aligned_scores,
    source_folder,
)
from rakshak.fusion.transactional_profile import aggregate_profile, categories


def training_profile(frame):
    labels = frame.label_scam.fillna(-1).to_numpy(dtype=int)
    negative = frame[labels == 0]
    counts = {}
    senders = negative.sender.tolist() if 'sender' in negative else [None] * len(negative)
    for text, sender in zip(negative.text, senders, strict=True):
        for category in categories(text, sender):
            counts[category] = counts.get(category, 0) + 1
    return {'train_rows': len(frame), 'fraud_negative_n': len(negative),
            'unknown_policy_n': int((labels == -1).sum()),
            'source_counts': {str(k): int(v) for k, v in
                              negative.source.fillna('missing').value_counts().items()},
            'category_counts': counts,
            'scope': 'lexical proxy counts, not verified real Indian SMS'}


def run(ctx):
    p = argparse.ArgumentParser()
    p.add_argument('--n-boot', type=int, default=1000)
    args = p.parse_args(ctx.argv)
    source = source_folder(ctx.input_root)
    metadata = json.loads((source / 'metrics.json').read_text())
    root = D.resolve_data_root()
    sms = D.load_sms(root)
    runs = D.load_runs(root, sms, patterns=['campaign'], labels=sms.label_binary.to_numpy(),
                       mask=D.eval_mask(sms))
    if len(runs) != 1:
        raise ValueError('one frozen campaign partition required')
    training = training_profile(sms.iloc[runs[0].train])
    results = {}
    for name in ('groundlm', 'smishx'):
        frame = pd.read_parquet(root / 'external' / f'external_{name}.parquet')
        frame = frame[frame.text_available.fillna(False) & D.eval_mask(frame)].copy()
        frame['reg_domains'] = [sorted({cached_registered_domain(u) for u in canonicalize(t).urls})
                                for t in frame.text]
        frame, _ = assign_campaigns(mark_exact_duplicates(frame.reset_index(drop=True)))
        labels = frame.label_scam.fillna(-1).to_numpy(dtype=int)
        eligible = np.isin(labels, [0, 1])
        senders = frame.sender.to_numpy() if 'sender' in frame else None
        results[name] = {}
        for seed in (1, 2, 3):
            threshold = metadata['runs']['campaign']['arms']['erm'][str(seed)][
                'metrics']['thresholds']['fpr_1pct']['threshold']
            scores = aligned_scores(source / f'external_{name}_erm_s{seed}.parquet', frame)
            results[name][str(seed)] = {
                'threshold': threshold,
                'profile': aggregate_profile(frame.text[eligible].tolist(), labels[eligible],
                                             scores[eligible] >= threshold,
                                             frame.campaign_id[eligible].astype(str).tolist(),
                                             senders=None if senders is None else senders[eligible],
                                             n_boot=args.n_boot)}
    write_metrics('metrics.json', {'training': training, 'external': results,
                                   'source_tree': SOURCE_TREE,
                                   'source_metrics_sha256': METRICS_HASH,
                                   'n_boot': args.n_boot,
                                   'scope': 'aggregate student-only diagnostic; no fused result; '
                                   'GroundLM in-sample for rule-feature design; '
                                   'not geography truth'})
    ctx.results['summary'] = 'aggregate student diagnostics complete; no model/threshold changes'
    return True
