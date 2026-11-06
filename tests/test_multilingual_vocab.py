import numpy as np
import pytest

from rakshak.models.multilingual_vocab import (
    encode_tokens,
    project_embeddings,
    select_vocabulary,
)


def test_train_vocabulary_order_unknown_and_padding():
    rows = [[7, 9, 7], [4, 9]]
    vocabulary = select_vocabulary(rows, ['a', 'b'], ['a', 'b'], limit=2)
    assert vocabulary.tolist() == [7, 9]
    assert encode_tokens([[7, 99, 9], []], vocabulary, max_len=4).tolist() == [
        [2, 1, 3, 0], [0, 0, 0, 0]]
    with pytest.raises(ValueError, match='frozen train'):
        select_vocabulary(rows, ['a', 'test-row'], ['a', 'b'])
    with pytest.raises(ValueError, match='frozen train'):
        select_vocabulary(rows, ['b', 'a'], ['a', 'b'])


def test_projected_table_reconstructs_train_pca_and_rejects_insufficient_rank_shape():
    rows = np.random.default_rng(3).normal(size=(80, 70))
    table, mean, components = project_embeddings(rows)
    assert table.shape == (82, 64)
    assert not table[0].any()
    np.testing.assert_allclose(table[2:], (rows-mean)@components.T, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(table[1], table[2:].mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(components@components.T, np.eye(64), atol=1e-10)
    assert (components[np.arange(64), np.abs(components).argmax(axis=1)] > 0).all()
    for result, repeated in zip((table, mean, components), project_embeddings(rows), strict=True):
        np.testing.assert_array_equal(result, repeated)
    with pytest.raises(ValueError, match='insufficient'):
        project_embeddings(rows[:64])
