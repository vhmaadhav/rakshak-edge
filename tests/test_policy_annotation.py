"""Synthetic JSON-contract checks, no model, data files or network."""

import json

import pytest

from rakshak.llm.policy_annotation import parse_candidate, prompt_manifest, system_prompt


def candidate(**changes):
    obj = {"id": "row1", "policy_label": "benign_transactional", "scam_type": "none",
           "urgency": False, "otp_pin_request": False, "payment_request": False,
           "remote_access_app": False, "link_present": False, "impersonated_brand": None,
           "evidence_spans": ["Do not share"], "confidence": "high"}
    return {**obj, **changes}


def test_contract_keeps_literal_evidence_and_never_repairs_hallucinated_ids():
    raw = json.dumps(candidate())
    obj, reason = parse_candidate(raw, "row1", "Your code: 123456. Do not share.", known_scam=0)
    assert reason == "accepted" and obj == candidate()
    obj, reason = parse_candidate(json.dumps(candidate(id="E99")), "row1", "Do not share")
    assert obj is None and reason == "wrong_id_or_root"
    obj, reason = parse_candidate(json.dumps(candidate(evidence_spans=["Never disclose"])),
                                  "row1", "Do not share")
    assert obj is None and reason == "span_not_verbatim"


@pytest.mark.parametrize("raw", ['{"id":"row1","id":"E99"}', '```json\n{}\n```', 'null'])
def test_bad_json_or_root_never_becomes_valid_through_repair(raw):
    obj, reason = parse_candidate(raw, "row1", "Invented text")
    assert obj is None and reason in {"invalid_json", "wrong_id_or_root"}


def test_semantic_contract_rejects_invented_brands_and_known_label_overrides():
    for changes, reason in [({"impersonated_brand": "InventedBank"}, "brand_not_literal"),
                            ({"policy_label": "promotion", "scam_type": "otp_theft"},
                             "inconsistent_scam_type"),
                            ({"policy_label": "unclear", "confidence": "high"},
                             "unclear_requires_low_confidence")]:
        raw = json.dumps(candidate(**changes))
        assert parse_candidate(raw, "row1", "Do not share")[1] == reason
    raw = json.dumps(candidate(policy_label="scam", scam_type="otp_theft"))
    reason = parse_candidate(raw, "row1", "Do not share", known_scam=0)[1]
    assert reason == "known_fraud_label_conflict"


def test_retry_budget_and_prompt_hashes_are_explicit():
    manifest = prompt_manifest()
    assert len(set(manifest["system_prompt_sha256"])) == 2
    assert manifest["minimum_valid"] == 190
    assert manifest["approved_for_training"] is False
    with pytest.raises(ValueError, match="budget"):
        system_prompt(2)
