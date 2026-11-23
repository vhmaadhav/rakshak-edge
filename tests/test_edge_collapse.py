"""Fix D (edge pipeline): one explainer call and one advisory point per evidence type.

The shapes are the two cases from the 40-row Pi eval: row 4 (six prize_lottery items) and row 28
(two shortened_url items). ``run_rules`` is patched to return those items, so the rule text is not
needed. The explainer is a mock runner; no model is loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1] / "deploy" / "pi" / "service"
REPO = Path(__file__).resolve().parents[1]
for p in (SERVICE, REPO / "src", REPO / "deploy" / "pi" / "bench", REPO / "deploy" / "pi"):
    sys.path.insert(0, str(p))

import edge_pipeline  # noqa: E402
from edge_pipeline import EdgeAnalyzer, collapse_by_type, ids_by_type  # noqa: E402

from rakshak.canonicalize import canonicalize  # noqa: E402
from rakshak.evidence import build_package  # noqa: E402
from rakshak.rules import Evidence  # noqa: E402
from rakshak.verifier import check_points, template_explanation  # noqa: E402

THRESHOLD = 0.35
TEXT = "Congrats! You have won the 1st Prize. Visit the link to claim it."  # not used by the rules


class FakeDetector:
    def __init__(self, p: float):
        self.p = p

    def score(self, text: str) -> float:
        return self.p


class CountingLLM:
    """One scripted sentence per call (the same sentence every time, as in the Pi runs)."""

    def __init__(self, sentence: str):
        self.sentence = sentence
        self.calls = 0

    def generate(self, prompt, max_new_tokens=64, stop_when=None):
        self.calls += 1
        if stop_when is not None:
            stop_when(self.sentence)
        return self.sentence, {
            "wall_ms": 100.0,
            "new_tokens": 8,
            "prompt_tokens": 90,
            "ttft_ms": 30.0,
            "stop_hit": True,
        }


def _items(etype: str, n: int, text: str, weight: float) -> list[Evidence]:
    return [
        Evidence(id=f"{etype}:{i}", type=etype, span=(i, i + 1), text=text, weight=weight)
        for i in range(n)
    ]


def _row4_items() -> list[Evidence]:
    # Row 4 (Prize Patrol SMS, label 1): six prize_lottery items.
    return _items("prize_lottery", 6, "prize patrol", 0.6)


def _row28_items() -> list[Evidence]:
    # Row 28 (bit.ly SMS, label 1): two shortened_url items.
    return _items("shortened_url", 2, "bit.ly/example", 0.7)


def _run(monkeypatch, items, llm):
    monkeypatch.setattr(edge_pipeline, "run_rules", lambda canon: list(items))
    analyser = EdgeAnalyzer(FakeDetector(0.9), llm, THRESHOLD)
    return analyser.analyse(TEXT)


# --- helpers -----------------------------------------------------------------------------------


def test_collapse_keeps_first_item_per_type_in_order():
    canon = canonicalize(TEXT)
    items = _row4_items() + _row28_items()
    pkg = build_package(canon, items, "scam")
    collapsed = collapse_by_type(pkg)
    assert [e.type for e in collapsed.evidence] == ["prize_lottery", "shortened_url"]
    assert [e.id for e in collapsed.evidence] == ["prize_lottery:0", "shortened_url:0"]
    # Risk, actions and the domain fields come from the full package and are unchanged.
    assert collapsed.risk == pkg.risk
    assert collapsed.allowed_actions == pkg.allowed_actions
    assert collapsed.domains == pkg.domains


def test_collapse_is_identity_when_types_are_unique():
    canon = canonicalize(TEXT)
    items = [
        Evidence(id="a", type="prize_lottery", span=(0, 1), text="x", weight=0.5),
        Evidence(id="b", type="shortened_url", span=(2, 3), text="y", weight=0.5),
    ]
    pkg = build_package(canon, items, "scam")
    assert collapse_by_type(pkg).evidence == pkg.evidence


def test_ids_by_type_groups_every_item():
    canon = canonicalize(TEXT)
    pkg = build_package(canon, _row4_items() + _row28_items(), "scam")
    groups = ids_by_type(pkg)
    assert len(groups["prize_lottery"]) == 6
    assert groups["shortened_url"] == ["shortened_url:0", "shortened_url:1"]


# --- row 4 shape: six prize_lottery items --------------------------------------------------------


def test_row4_shape_one_call_one_point_contract_holds(monkeypatch):
    llm = CountingLLM(" is asking the reader to do something.")
    out = _run(monkeypatch, _row4_items(), llm)
    assert out["decision"]["evidence_count"] == 6
    assert len(out["evidence"]) == 6  # full list kept in the response
    assert llm.calls == 1
    assert len(out["llm_calls"]) == 1
    adv = out["advisory"]
    assert len(adv["points"]) == 1
    assert adv["evidence_items"] == 6
    pt = adv["points"][0]
    assert pt["type"] == "prize_lottery"
    assert pt["count"] == 6
    assert pt["evidence_ids"] == [f"prize_lottery:{i}" for i in range(6)]
    assert pt["source"] == "llm"
    assert pt["check"]["ok"] is True
    assert out["verifier"]["final_contract_ok"] is True
    assert out["advisory"]["mode"] == "llm"


def test_row4_shape_points_pass_the_verifier_against_collapsed_package(monkeypatch):
    llm = CountingLLM(" is asking the reader to do something.")
    out = _run(monkeypatch, _row4_items(), llm)
    canon = canonicalize(TEXT)
    collapsed = collapse_by_type(build_package(canon, _row4_items(), "scam"))
    import json

    final = {
        "lang": "en",
        "summary": out["advisory"]["summary"]["text"],
        "points": [p["text"] for p in out["advisory"]["points"]],
    }
    assert check_points(json.dumps(final), collapsed, "en").ok


def test_row4_summary_is_still_the_template_of_the_full_package(monkeypatch):
    llm = CountingLLM(" is asking the reader to do something.")
    out = _run(monkeypatch, _row4_items(), llm)
    canon = canonicalize(TEXT)
    full = build_package(canon, _row4_items(), "scam")
    assert out["advisory"]["summary"]["text"] == template_explanation(full)["summary"]


# --- row 28 shape: two shortened_url items -------------------------------------------------------


def test_row28_shape_one_call_one_point(monkeypatch):
    llm = CountingLLM(" contains a shortened link and is likely to be a link to a website.")
    out = _run(monkeypatch, _row28_items(), llm)
    assert llm.calls == 1
    adv = out["advisory"]
    assert len(adv["points"]) == 1
    pt = adv["points"][0]
    assert pt["type"] == "shortened_url"
    assert pt["count"] == 2
    assert pt["evidence_ids"] == ["shortened_url:0", "shortened_url:1"]
    assert pt["check"]["ok"] is True
    assert out["verifier"]["final_contract_ok"] is True
    assert len(out["evidence"]) == 2


# --- mixed: types keep their first-occurrence order -----------------------------------------------


def test_mixed_types_one_point_per_type_in_order(monkeypatch):
    items = _row4_items() + _row28_items()
    llm = CountingLLM(" is asking the reader to do something.")
    out = _run(monkeypatch, items, llm)
    assert llm.calls == 2
    types = [p["type"] for p in out["advisory"]["points"]]
    assert types == ["prize_lottery", "shortened_url"]
    assert [p["count"] for p in out["advisory"]["points"]] == [6, 2]
    assert len(out["advisory"]["points"]) == len({e["type"] for e in out["evidence"]})


def test_template_only_path_still_one_point_per_type(monkeypatch):
    # Forced LOW risk (G1): no explainer call, template points, one per type.
    import dataclasses

    import edge_decision

    from rakshak.evidence import Risk

    real_build = edge_decision.build_package

    def low_build(canon, evidence, state):
        return dataclasses.replace(real_build(canon, evidence, state), risk=Risk.LOW)

    monkeypatch.setattr(edge_decision, "build_package", low_build)
    llm = CountingLLM("unused")
    out = _run(monkeypatch, _row4_items(), llm)
    assert out["decision"]["risk"] == "LOW"
    assert llm.calls == 0
    assert out["advisory"]["llm_skipped_low_risk"] is True
    assert len(out["advisory"]["points"]) == 1
    assert out["advisory"]["points"][0]["count"] == 6
    assert out["verifier"]["final_contract_ok"] is True


def test_g2_rejected_uses_collapsed_evidence_id_not_raw_index(monkeypatch):
    """When earlier duplicate types are collapsed, g2_rejected must still report the true ID."""

    class SecondCallScamLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, max_new_tokens=64, stop_when=None):
            self.calls += 1
            if self.calls == 1:
                sentence = " is asking the reader to claim a prize."
            else:
                sentence = " is a scam trying to steal your money."
            if stop_when is not None:
                stop_when(sentence)
            return sentence, {
                "wall_ms": 10.0,
                "new_tokens": 8,
                "prompt_tokens": 50,
                "stop_hit": True,
            }

    items = _row4_items() + _row28_items()  # 6 prize_lottery items, then 2 shortened_url items
    # Low detector score + rule evidence gives UNCERTAIN risk so G2 attribution guard applies.
    monkeypatch.setattr(edge_pipeline, "run_rules", lambda canon: list(items))
    analyser = EdgeAnalyzer(FakeDetector(0.1), SecondCallScamLLM(), THRESHOLD)
    out = analyser.analyse(TEXT)
    g2 = out["verifier"]["g2_rejected"]
    assert len(g2) == 1
    assert g2[0]["evidence_id"] == "shortened_url:0"
