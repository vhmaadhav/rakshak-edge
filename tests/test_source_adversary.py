"""Metadata checks locally; tiny torch invariants run on Kaggle before corpus access."""

import numpy as np
import pytest

from rakshak.models.source_adversary import conditional_layout, verify_torch_invariants


def test_conditional_weighting_gives_equal_training_cell_mass():
    sources = ["a", "a", "b", "b", "a", "b", "b", "b"]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    supports, codes, weights = conditional_layout(sources, labels)
    assert supports == {0: ["a", "b"], 1: ["a", "b"]}
    assert codes.tolist() == [0, 0, 1, 1, 0, 1, 1, 1]
    for y in (0, 1):
        for source in ("a", "b"):
            mask = (np.asarray(labels) == y) & (np.asarray(sources) == source)
            assert weights[mask].sum() == pytest.approx(1)


def test_single_source_class_is_not_forced_to_align_and_unlabelled_rows_are_rejected():
    supports, codes, weights = conditional_layout(["a", "a", "a", "b"], [0, 0, 1, 1])
    assert supports == {1: ["a", "b"]}
    assert (codes[:2] == -1).all() and (weights[:2] == 0).all()
    with pytest.raises(ValueError, match="labelled frozen train"):
        conditional_layout(["a", "b"], [0, np.nan])
    with pytest.raises(ValueError, match="two training sources"):
        conditional_layout(["a", "b"], [0, 1])


def test_gradient_reversal_rng_zero_coefficient_and_inference_architecture():
    pytest.importorskip("torch")
    verify_torch_invariants()
