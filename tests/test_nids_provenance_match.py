"""Synthetic label-free matching regressions, no actual network flows."""

import pandas as pd
import pytest

from rakshak.nids.provenance import MetadataAuditError
from rakshak.nids.provenance_match import CandidatePairs, numeric_keys


def test_matching_uses_frozen_precision_and_all_fields_with_normalized_zero():
    dtypes = {"IN_BYTES": "uint32", "L7_PROTO": "float32", "L4_SRC_PORT": "uint16"}
    a = pd.DataFrame({"IN_BYTES": [42], "L7_PROTO": [-0.0], "L4_SRC_PORT": [123]})
    b = pd.DataFrame({"IN_BYTES": ["42"], "L7_PROTO": ["0.0"], "L4_SRC_PORT": ["123"]})
    assert numeric_keys(a, dtypes) == numeric_keys(b, dtypes)
    b["L4_SRC_PORT"] = "124"
    assert numeric_keys(a, dtypes) != numeric_keys(b, dtypes)
    a["L7_PROTO"], b["L7_PROTO"] = 1.2345678901, 1.23456789
    b["L4_SRC_PORT"] = "123"
    assert numeric_keys(a, dtypes) == numeric_keys(b, dtypes)


def test_matching_rejects_label_features_and_lossy_integer_casts():
    for column in ["Label", "Attack", "IPV4_SRC_ADDR"]:
        with pytest.raises(MetadataAuditError, match="forbidden"):
            numeric_keys(pd.DataFrame({column: [1]}), {column: "int64"})
    frame = pd.DataFrame({"IN_BYTES": ["NaN", "inf", "-1", "256", "1.5", "1"]})
    assert numeric_keys(frame, {"IN_BYTES": "uint8"})[:5] == [None] * 5
    assert numeric_keys(frame, {"IN_BYTES": "uint8"})[-1] is not None
    assert numeric_keys(pd.DataFrame({"IN_BYTES": [2**53 + 1]}),
                        {"IN_BYTES": "uint64"}) == [None]


def test_ambiguity_and_missing_rows_cannot_be_solved_by_order_or_duplicates():
    candidates = CandidatePairs([b"a", b"b"])
    candidates.observe([b"a", b"a"], ["192.0.2.1"] * 2, ["192.0.2.2"] * 2)
    assert candidates.summary()["unmapped_keys"] == 1
    assert candidates.summary()["ambiguous_keys"] == 0
    candidates.observe([b"b"], ["192.0.2.3"], ["192.0.2.4"])
    assert candidates.summary()["complete_unambiguous"]
    assert candidates.pairs([b"b", b"a"])[0] == ("192.0.2.3", "192.0.2.4")
    candidates.observe([b"a"], ["192.0.2.5"], ["192.0.2.6"])
    assert not candidates.summary()["complete_unambiguous"]
    assert candidates.summary()["ambiguous_keys"] == 1
    with pytest.raises(MetadataAuditError, match="ambiguous"):
        candidates.pairs([b"a", b"b"])


def test_duplicate_or_invalid_frozen_keys_are_not_deduped_to_make_gate_pass():
    for wanted in [[], [b"a", b"a"], [None]]:
        with pytest.raises(MetadataAuditError, match="unique"):
            CandidatePairs(wanted)
