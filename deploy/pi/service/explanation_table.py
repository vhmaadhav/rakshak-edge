"""Precomputed explanation table (verdict-first, step 2).

What the explainer conditions on, for one point: the evidence type (its TYPE_PHRASE, fixed), the
language, the prompt variant, and the slot (``rakshak.prompts.fact_slot``: the matched text, or
the registered domain for a URL). The type set is finite. The slot is message text, so it is not.
The table therefore stores each sentence with the placeholder ``[[SLOT]]`` where the slot goes,
and the runtime fills it with the message's own slot. Key: (lang, variant, type, has_slot).

Build time (deploy/pi/bench/build_explanation_table.py) runs the same runner with the same prompt
shape, with the placeholder in the fact, and cleans each sentence the way the live path does. A
sentence is kept for a risk only if the filled sentence passes ``check_points`` at that risk, G2
included, on every sample message of its type. Anything else is left out, and a point with no
entry falls back to the template.

Runtime (``table_gen``): each point is looked up, filled, and then run through ``render_advisory``,
the same assembly and per-point check as the live explainer. So a table sentence that does not
pass the verifier for the actual message is replaced by the template there too. The table can
only make an answer faster, never let an unchecked sentence through.

The table is a versioned JSON document. A version other than TABLE_VERSION is refused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rbench.slot import duplicate_points

from rakshak.evidence import EvidencePackage, Risk
from rakshak.prompts import URL_TYPES, fact_slot
from rakshak.url_model import URL_MODEL_TYPE
from rakshak.verifier import _template_summary, check_points, render_advisory

TABLE_VERSION = 1
# Table sentences are slot-free: Llama refers to the evidence generically and never quotes it. The
# runtime appends the real slot as a deterministic clause (see ``slot_clause``), so the verifier
# checks the full text. A stored sentence that still holds PLACEHOLDER is refused.
PLACEHOLDER = "[[SLOT]]"
# The risks an explainer sentence can be used at. LOW never reaches the explainer (G1).
TABLE_RISKS = (Risk.MEDIUM.value, Risk.UNCERTAIN.value, Risk.HIGH.value)


def key(lang: str, variant: str, etype: str, has_slot: bool) -> str:
    return f"{lang}|{variant}|{etype}|{int(bool(has_slot))}"


def slot_label(etype: str) -> str:
    """The label of the slot clause for one evidence type: the registered domain for a URL, the
    matched text otherwise."""
    return "domain" if etype in URL_TYPES or etype == URL_MODEL_TYPE else "matched text"


def slot_clause(etype: str, slot: str) -> str:
    """The deterministic clause that carries the slot value, or "" when there is no slot."""
    return f" ({slot_label(etype)}: {slot})" if slot else ""


def fill(sentence: str, slot: str, etype: str = "") -> str:
    """The table sentence followed by its slot clause, whitespace normalised the way
    ``clean_sentence`` normalises a live sentence. With no slot the sentence is returned as is."""
    text = sentence + slot_clause(etype, slot)
    return " ".join(text.split())


class ExplanationTable:
    """Read-only lookup over a built table document."""

    def __init__(self, doc: dict[str, Any], source: str = "<memory>") -> None:
        if not isinstance(doc, dict) or doc.get("version") != TABLE_VERSION:
            raise ValueError(
                f"explanation table version must be {TABLE_VERSION}, "
                f"got {doc.get('version') if isinstance(doc, dict) else None!r}"
            )
        entries = doc.get("entries")
        if not isinstance(entries, dict):
            raise ValueError("explanation table has no entries object")
        self.doc = doc
        self.source = source
        self.entries: dict[str, dict[str, Any]] = entries
        self.hits = 0
        self.misses = 0

    @classmethod
    def load(cls, path: Path | str) -> ExplanationTable:
        path = Path(path)
        return cls(json.loads(path.read_text(encoding="utf-8")), source=str(path))

    def sentence(self, etype: str, lang: str, variant: str, slot: str, risk: Risk) -> str | None:
        """The stored sentence for one point, or None when there is no entry for its key, or the
        entry was not kept for this risk, or the stored text holds a placeholder (never valid)."""
        entry = self.entries.get(key(lang, variant, etype, bool(slot)))
        if entry is None or risk.value not in entry.get("risks", []):
            self.misses += 1
            return None
        sentence = entry["sentence"]
        if PLACEHOLDER in sentence:
            self.misses += 1
            return None
        self.hits += 1
        return sentence

    def stats(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "version": TABLE_VERSION,
            "entries": len(self.entries),
            "hits": self.hits,
            "misses": self.misses,
            "model": self.doc.get("model"),
            "built": self.doc.get("built"),
        }


def table_gen(
    table: ExplanationTable, package: EvidencePackage, lang: str, variant: str
) -> dict[str, Any] | None:
    """The explainer output (same shape as ``generate_slot_advisory``) built from the table, or
    None when any point has no usable entry. ``package`` is the one-item-per-type package.

    A None answer means the caller must run the explainer (or wait for it): a partly filled
    advisory would mix table sentences with template sentences the explainer would have written.
    """
    if not package.evidence:
        return None  # nothing to explain: the caller uses the template path
    risk = package.risk
    slots = [fact_slot(e) for e in package.evidence]
    sentences: list[str] = []
    for e, slot in zip(package.evidence, slots, strict=False):
        stored = table.sentence(e.type, lang, variant, slot, risk)
        if stored is None:
            return None
        sentences.append(fill(stored, slot, e.type))
    dup_of = duplicate_points(sentences)
    for i, j in enumerate(dup_of):
        if j is not None:
            sentences[i] = ""
    # The summary is the template, as in the live path (rbench.slot.assemble).
    raw_output = json.dumps(
        {"lang": lang, "summary": _template_summary(package, lang), "points": sentences},
        ensure_ascii=False,
    )
    verdict = check_points(raw_output, package, lang, risk)
    rendered = render_advisory(raw_output, package, lang, risk)
    final = {
        "lang": lang,
        "summary": rendered["summary"],
        "points": [p["text"] for p in rendered["points"]],
    }
    final_output = json.dumps(final, ensure_ascii=False)
    sources = [
        "table" if (sent and p["source"] == "llm") else "template"
        for sent, p in zip(sentences, rendered["points"], strict=False)
    ]
    g2_rejected = [
        {"index": c.index, "type": c.type, "sentence": sentences[c.index], "problem": p}
        for c in verdict.points
        for p in c.problems
        if p.startswith("attribution of scam intent")
    ]
    return {
        "raw_output": raw_output,
        "output": final_output,
        "calls": [],
        "skipped": False,
        "point_checks": [[c.type, c.ok] for c in verdict.points],
        "point_problems": [list(c.problems) for c in verdict.points],
        "point_status": ["table" if s else "empty" for s in sentences],
        "point_sources": sources,
        "g2_rejected": g2_rejected,
        "cleanup": [],
        "final_verifier_ok": check_points(final_output, package, lang).ok,
    }


__all__ = [
    "PLACEHOLDER",
    "TABLE_RISKS",
    "TABLE_VERSION",
    "ExplanationTable",
    "fill",
    "key",
    "slot_clause",
    "slot_label",
    "table_gen",
]
