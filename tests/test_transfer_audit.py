"""Synthetic split-membership fixtures; no corpus/model compute."""

import pytest

from rakshak.eval.transfer_audit import audit_extra_rows


def test_broad_transfer_cannot_hide_test_copies_campaigns_or_external_overlap():
    keys = ["train", "cal", "test", "test", "neighbour", "external", "fresh"]
    campaigns = ["a", "b", "test", "test", "test", "e", "f"]
    sources = ["uci", "uci", "held", "held", "held", "uci", "held"]
    result = audit_extra_rows([0, 3, 4, 5, 6], [0], [1], [2], keys, campaigns, sources,
                              external_keys={"external"}, source_holdout=True)
    assert result["extra_unlabeled_rows"] == 4
    assert result["extra_exact_test_copies"] == 1
    assert result["extra_in_test_campaigns"] == 2
    assert result["extra_external_exact_overlap"] == 1
    assert result["extra_heldout_source"] == 3
    clean = audit_extra_rows([0], [0], [1], [2], keys, campaigns, sources)
    assert clean["extra_unlabeled_rows"] == 0
    assert clean["extra_exact_test_copies"] == 0


def test_missing_campaign_does_not_join_all_missing_rows():
    result = audit_extra_rows([0, 2], [0], [], [1], ["a", "b", "c"], ["", "", ""],
                              ["x"] * 3)
    assert result["extra_in_test_campaigns"] == 0
    with pytest.raises(ValueError, match="one key"):
        audit_extra_rows([0], [0], [], [1], ["a", "b"], ["a"], ["x", "x"])
