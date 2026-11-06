"""Synthetic gates for a frozen base-versus-adapter comparison."""

import pytest

from rakshak.llm.ela_comparison import compare


def rows(passes=1, unsupported=0, harmful=False):
    result = []
    for arm in ("slot", "slot_hinglish", "rank", "freeform"):
        for injected in (False, True):
            if arm == "slot_hinglish" and injected:
                continue
            result.append({
                "arm": arm, "sample_id": "fixture", "campaign_id": "c1", "side": "scam",
                "injected": injected, "family": "family" if injected else None,
                "points_n": 1, "points_pass_n": passes, "unsupported_n": unsupported,
                "safe_claim": harmful and injected, "rank_items": 1,
            })
    return result


def test_go_requires_point_gain_and_safe_matched_pairs():
    result = compare(rows(passes=0), rows())
    assert result["decision"] == "go"
    assert result["injection_vs_base"]["slot"]["pairs"] == 1
    assert result["injection_vs_base"]["slot"]["new"] == 0
    assert result["injection_vs_base"]["slot"]["mcnemar_exact_p"] == 1
    assert result["language"]["slot"]["wilson95_iid_reference"][0] > 0


def test_unsupported_claims_or_low_en_bound_kill():
    assert compare(rows(), rows(unsupported=1))["decision"] == "kill"
    assert compare(rows(), rows(passes=0))["decision"] == "kill"
    result = compare(rows(), rows(harmful=True))
    assert result["decision"] == "kill"
    assert result["reassuring_advisory_rows"] > 0
    assert result["slot_no_new_harm"] is False


def test_comparison_refuses_missing_or_changed_pairs():
    with pytest.raises(ValueError, match="match exactly"):
        compare(rows(), rows()[:-1])
    changed = rows()
    changed[0]["campaign_id"] = "wrong"
    with pytest.raises(ValueError, match="metadata differs"):
        compare(rows(), changed)
    with pytest.raises(ValueError, match="duplicate"):
        compare(rows(), rows() + rows()[:1])


def test_comparison_refuses_changed_evidence_denominator():
    changed = rows()
    changed[0]["points_n"] = 2
    with pytest.raises(ValueError, match="denominator differs"):
        compare(rows(), changed)


def test_unsupported_raw_and_rendered_on_one_row_count_once():
    changed = rows()
    changed[0].update(unsupported_n=1, rendered_unsupported_n=1)
    result = compare(rows(), changed)
    assert result["unsupported_advisory_rows"] == 1
    assert result["decision"] == "kill"


def test_family_tests_keep_empty_rank_families_unestimated():
    from rakshak.ela_eval import INJECTION_FAMILIES

    result = compare(rows(), rows(harmful=True))
    family = result["injection_vs_base"]["slot"]["per_family"]["family"]
    assert family["new"] == 1 and family["mcnemar_exact_p"] == 1
    assert family["tuned_wilson95"][0] > 0
    for name in INJECTION_FAMILIES:
        missing = result["injection_vs_base"]["rank"]["per_family"][name]
        assert missing["n"] == 0
        assert missing["fisher_p"] is None
        assert missing["tuned_wilson95"] == [None, None]


def test_action_only_new_harm_blocks_go_even_without_an_unsupported_entity():
    baseline, changed = rows(), rows()
    for row in baseline + changed:
        row["action"] = "VERIFY_OFFICIAL"
    changed[1]["action"] = "IGNORE"
    result = compare(baseline, changed)
    assert result["unsupported_advisory_rows"] == 0
    assert result["reassuring_advisory_rows"] == 0
    assert result["decision"] == "inconclusive"
    assert result["slot_no_new_harm"] is False
