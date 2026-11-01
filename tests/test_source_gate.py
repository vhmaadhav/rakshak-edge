"""Acceptance-boundary fixtures, not model training or benchmark evaluation."""

from copy import deepcopy

import pytest

from rakshak.eval.source_gate import source_gate


def fixture():
    names = ["campaign", "source_holdout:holdout_scamshield", "source_holdout:holdout_uci_sms_spam",
             "source_holdout:holdout_mishra_soni_smsphishing"]
    paired = {"a": 0.8, "b": 0.7, "delta": 0.1, "delta_ci95": [0.01, 0.2],
              "b_ci95": [0.6, 0.8]}
    return {"runs": {name: {"arms": {arm: {str(s): {} for s in (1, 2, 3)}
                                     for arm in ("erm", "balanced", "dro")}} for name in names},
            "comparisons": {name: {arm: deepcopy(paired) for arm in ("balanced", "dro")}
                            for name in names}, "skipped": {}}


def test_go_needs_positive_scamshield_ci_and_no_excess_other_fold_loss():
    data = fixture()
    assert source_gate(data)["arms"]["balanced"]["decision"] == "go"
    data["comparisons"]["source_holdout:holdout_scamshield"]["balanced"]["delta_ci95"] = [0, 0.2]
    assert source_gate(data)["arms"]["balanced"]["decision"] == "kill"
    assert source_gate(data)["arms"]["dro"]["decision"] == "go"
    data["comparisons"]["source_holdout:holdout_uci_sms_spam"]["dro"]["delta"] = -0.1001
    assert source_gate(data)["arms"]["dro"]["decision"] == "kill"


def test_incomplete_or_nonfinite_result_cannot_receive_a_decision():
    data = fixture()
    del data["runs"]["source_holdout:holdout_mishra_soni_smsphishing"]
    with pytest.raises(ValueError, match="three frozen source folds"):
        source_gate(data)
    data = fixture()
    del data["runs"]["campaign"]["arms"]["erm"]["3"]
    with pytest.raises(ValueError, match="three frozen seeds"):
        source_gate(data)
    data = fixture()
    data["comparisons"]["source_holdout:holdout_scamshield"]["dro"]["delta_ci95"][0] = None
    with pytest.raises(ValueError, match="finite paired intervals"):
        source_gate(data)
