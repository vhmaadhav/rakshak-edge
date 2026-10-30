"""Label taxonomy for SMS rows: ``label_policy`` and ``label_scam``.

``label_binary`` (spam or scam = 1) is the legacy label and is never changed here. The new columns
separate fraud from marketing and from legitimate traffic (docs/CONTEXT.md, "Spam != scam"):

  label_policy  benign_transactional | benign_personal | promotion | scam | unclear | None
  label_scam    1 = scam, 0 = benign or promotion, None = unknown

A value is filled only where the source's own label determines it:
  * smishing / scam from a source that only holds fraud (Mishra & Soni smishing, IMC25,
    SmishX smishing)                                              -> scam, 1
  * ScamShield non-spam scam intents (heuristic intent, see DATACARD) -> scam, 1
  * spam from Mishra & Soni and SmishX                               -> promotion, 0
  * legitimate / ham from any source                                 -> None (the benign subtype
    is not in the source label), 0
  * spam from UCI, ScamShield, anmol and the bolewara positives      -> None, None (needs a
    relabel: a generic "spam" may be fraud or marketing)
  * a text whose duplicate group carries conflicting labels          -> unclear, None

benign_transactional and benign_personal are reserved for a relabel pass and are never set here.
"""

from __future__ import annotations

from typing import NamedTuple

from rakshak.datasets.loaders import (
    SOURCE_ANMOL,
    SOURCE_BOLEWARA,
    SOURCE_GROUNDLM,
    SOURCE_IMC25,
    SOURCE_MISHRA,
    SOURCE_SCAMSHIELD,
    SOURCE_SMISHX,
    SOURCE_UCI,
)

LABEL_POLICY_VALUES: tuple[str, ...] = (
    "benign_transactional",
    "benign_personal",
    "promotion",
    "scam",
    "unclear",
)


class Policy(NamedTuple):
    label_policy: str | None
    label_scam: int | None


_SCAM = Policy("scam", 1)
_PROMO = Policy("promotion", 0)
_BENIGN_UNSPECIFIED = Policy(None, 0)
_NEEDS_RELABEL = Policy(None, None)

_TABLE: dict[tuple[str, str], Policy] = {
    (SOURCE_MISHRA, "smishing"): _SCAM,
    (SOURCE_MISHRA, "spam"): _PROMO,
    (SOURCE_MISHRA, "ham"): _BENIGN_UNSPECIFIED,
    (SOURCE_IMC25, "smishing"): _SCAM,
    (SOURCE_SCAMSHIELD, "scam"): _SCAM,
    (SOURCE_SCAMSHIELD, "spam"): _NEEDS_RELABEL,
    (SOURCE_SCAMSHIELD, "ham"): _BENIGN_UNSPECIFIED,
    (SOURCE_UCI, "spam"): _NEEDS_RELABEL,
    (SOURCE_UCI, "ham"): _BENIGN_UNSPECIFIED,
    (SOURCE_ANMOL, "spam"): _NEEDS_RELABEL,
    (SOURCE_ANMOL, "ham"): _BENIGN_UNSPECIFIED,
    (SOURCE_BOLEWARA, "scam"): _NEEDS_RELABEL,
    (SOURCE_BOLEWARA, "ham"): _BENIGN_UNSPECIFIED,
    (SOURCE_SMISHX, "smishing"): _SCAM,
    (SOURCE_SMISHX, "spam"): _PROMO,
    (SOURCE_SMISHX, "legitimate"): _BENIGN_UNSPECIFIED,
    # GroundLM labels are binary fraud / safe; the marketing vs transactional split is not
    # released, so "safe" carries no label_policy.
    (SOURCE_GROUNDLM, "fraud"): _SCAM,
    (SOURCE_GROUNDLM, "safe"): _BENIGN_UNSPECIFIED,
}


def policy_for(source: str, label_fine: str | None, label_conflict: bool = False) -> Policy:
    """label_policy and label_scam for one row. Unknown (source, label) pairs get no value."""
    if label_conflict:
        return Policy("unclear", None)
    if label_fine is None:
        return _NEEDS_RELABEL
    return _TABLE.get((source, str(label_fine).lower()), _NEEDS_RELABEL)
