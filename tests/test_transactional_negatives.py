"""Invented training notifications only; no real or benchmark messages."""

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from rakshak.models.features import featurize
from rakshak.models.transactional_negatives import (
    FAMILIES,
    append_training,
    generate,
    key,
)


def test_fixed_prefix_balanced_families_languages_and_train_only_tags():
    rows = generate(120)
    assert rows[:24] == generate(24)
    assert len({key(r.text) for r in rows}) == 120
    assert set(Counter(r.family for r in rows).values()) == {20}
    assert Counter(r.lang for r in rows) == {'en': 60, 'hinglish': 60}
    assert set(r.family for r in rows) == set(FAMILIES)
    assert all(r.synthetic and r.partition == 'train' and r.label_binary == r.label_scam == 0
               and r.sender.endswith(('SBISYN', 'HDFSYN', 'ICISYN', 'PRCSYN', 'UTLSYN'))
               and len(r.text.encode()) <= 256 for r in rows)


def test_append_preserves_original_teacher_and_refuses_eval_tags_or_text_collision():
    train = featurize(['invented original positive', 'invented original negative'])
    labels, teacher = np.array([1, 0]), np.array([[0., 2.], [2., 0.]])
    rows = generate(12)
    combined, out_labels, out_teacher = append_training(
        train, labels, teacher, rows, forbidden_keys={'unrelated fixture'})
    np.testing.assert_equal(out_labels[:2], labels)
    np.testing.assert_equal(out_teacher[:2], teacher)
    assert np.isnan(out_teacher[2:]).all() and (out_labels[2:] == 0).all()
    assert len(combined) == 14 and combined.texts[:2] == train.texts
    with pytest.raises(ValueError, match='collision'):
        append_training(train, labels, teacher, rows, forbidden_keys={key(rows[0].text)})
    with pytest.raises(ValueError, match='training-only'):
        append_training(train, labels, teacher, [replace(rows[0], partition='test')],
                        forbidden_keys=set())
