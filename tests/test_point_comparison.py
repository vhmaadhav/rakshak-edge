"""Small synthetic paired-ratio fixtures, without model runtime or corpus data."""

import pytest

from rakshak.eval.point_comparison import paired_point_pass


def test_paired_points_preserves_pooling_and_signed_changes():
    # Pooled gain 1/3, not an unweighted mean of package pass fractions (1/2).
    result = paired_point_pass([0, 1], [1, 1], [1, 2], ["a", "b"], n_boot=30)
    assert result["delta"] == pytest.approx(1 / 3)
    reverse = paired_point_pass([1, 1], [0, 1], [1, 2], ["a", "b"], n_boot=30)
    assert reverse["delta"] == pytest.approx(-1 / 3)
    assert reverse["ci95"] == pytest.approx([-result["ci95"][1], -result["ci95"][0]])


def test_same_campaign_moves_together_and_identical_models_have_zero_difference():
    same_campaign = paired_point_pass([0, 0], [1, 0], [1, 1], ["same", "same"], n_boot=30)
    assert same_campaign["ci95"] == [0.5, 0.5]
    result = paired_point_pass([1, 0], [1, 0], [1, 1], ["a", "b"], n_boot=30)
    assert result["ci95"] == [0, 0] and result["p_value"] == 1
    assert paired_point_pass([0], [0], [0], ["empty"])["delta"] is None


def test_invalid_or_misaligned_point_counts_are_rejected():
    with pytest.raises(ValueError, match="invalid point counts"):
        paired_point_pass([2], [0], [1], ["a"])
    with pytest.raises(ValueError, match="per campaign row"):
        paired_point_pass([1], [1], [1], ["a", "b"])
