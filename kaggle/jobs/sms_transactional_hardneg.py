"""Fresh paired ERM/independent synthetic transaction negatives; Kaggle T4 only."""

import json
from collections import Counter

import numpy as np
import pandas as pd
from _common import write_metrics

from rakshak.eval import data as D
from rakshak.models.transactional_negatives import (
    append_training,
    fingerprint,
    generate,
    key,
)


def run(ctx):
    import sms_source_balance as controls
    import torch
    import torch.nn.functional as F  # noqa: N812

    from rakshak.models.student import distillation_loss

    # Synthetic loss/gradient invariant before corpus access: no teacher means full hard BCE.
    logits = torch.tensor([.2, -.3, .1], requires_grad=True)
    labels = torch.tensor([0., 1., 0.])
    teacher = torch.tensor([[2., 0.], [0., 2.], [float('nan'), float('nan')]])
    losses = distillation_loss(logits, labels, teacher, pos_weight=torch.tensor(2.),
                               alpha=.7, temp=2., reduction='none')
    expected = F.binary_cross_entropy_with_logits(logits[2:], labels[2:])
    if not torch.allclose(losses[2], expected):
        raise ValueError('synthetic rows must receive full BCE without teacher targets')
    losses.mean().backward()
    if not torch.isfinite(logits.grad).all():
        raise ValueError('synthetic missing-teacher gradient is not finite')
    ctx.log('synthetic full-BCE/no-teacher Torch invariants passed')
    forbidden, cached, all_rows = None, {}, None

    def transform(run_, train, labels, teacher, sources, arm, seed):
        nonlocal forbidden, all_rows
        if arm == 'erm':
            return train, labels, teacher, sources, {
                'synthetic_rows': 0, 'original_rows': len(train),
                'positive_weight': float((np.asarray(labels) == 0).sum()
                                         / (np.asarray(labels) == 1).sum())}
        if arm != 'hardneg':
            raise ValueError('only the two predeclared arms are permitted')
        if run_.name not in cached:
            if forbidden is None:
                root = D.resolve_data_root()
                sms = D.load_sms(root)
                forbidden = {key(t) for t in sms.text.fillna('')}
                for name in ('groundlm', 'smishx'):
                    frame = pd.read_parquet(root/'external'/f'external_{name}.parquet')
                    forbidden.update(key(t) for t in frame.text.fillna(''))
                all_rows = generate(12000)
                pd.DataFrame([row.record() for row in all_rows]).to_parquet(
                    ctx.work/'synthetic_transactional_v1.parquet', index=False)
                pd.DataFrame([row.record() for row in all_rows[:200]]).to_parquet(
                    ctx.work/'synthetic_review_sample_200.parquet', index=False)
            n = min(12000, int(.25 * int((np.asarray(labels) == 0).sum())))
            rows = all_rows[:n]
            combined, y, soft = append_training(train, labels, teacher, rows,
                                                forbidden_keys=forbidden)
            groups = np.r_[sources, np.full(n, 'synthetic_transactional_v1')]
            info = {'synthetic_rows': n, 'original_rows': len(train), 'teacher_synthetic_rows': 0,
                    'generator_seed': 20261101, 'template_sha256': fingerprint(),
                    'synthetic_ids': [row.id for row in rows],
                    'family_counts': dict(Counter(row.family for row in rows)),
                    'language_counts': dict(Counter(row.lang for row in rows)),
                    'original_binary_negative_n': int((np.asarray(labels) == 0).sum()),
                    'augmented_positive_weight': float((y == 0).sum() / (y == 1).sum()),
                    'source': 'original deterministic code; no corpus/model targets; '
                              'synthetic detector-only training rows'}
            cached[run_.name] = combined, y, soft, groups, info
            write_metrics('synthetic_training_manifest.json', {
                name: values[4] for name, values in cached.items()})
        return cached[run_.name]

    outcome = controls.run(ctx, arms=('erm', 'hardneg'), training_transform=transform,
                           method_reference='docs/TRANSACTIONAL_HARDNEG.md; original '
                                            'templates/task brief/public security semantics')
    # The shared controls runner enforces the original frozen train pool before transformation.
    if all_rows is None:
        raise ValueError('no complete augmentation unit was reached')
    ctx.results['summary'] += '; synthetic train-only arm, no real messages in generated targets'
    (ctx.work/'synthetic_provenance.json').write_text(json.dumps({
        'generator_seed': 20261101, 'template_sha256': fingerprint(),
        'synthetic': True, 'partition': 'train', 'used_for_advisory': False,
        'human_review_sample_n': 200, 'human_review_completed': False,
        'source_register': 'docs/TRANSACTIONAL_HARDNEG.md',
        'external_feedback_scope': 'in-sample for intervention selection'}))
    return outcome
