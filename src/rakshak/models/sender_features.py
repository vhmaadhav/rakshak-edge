"""Five side features from the sender ID's DLT structure (numpy only).

Accepts raw sender IDs such as ``AX-HDFCBK`` or ``VM-AIRTEL-T`` and the scrubbed form produced by
``tools/inbox_export/scrub.py::scrub_sender`` (digits replaced by ``<NUMBER>``). Phone numbers,
placeholders and anything unrecognised map to all zeros.
"""

import re
from collections.abc import Sequence

import numpy as np

SENDER_FEATURE_NAMES = (
    "sender_alphanumeric",
    "dlt_suffix_P",
    "dlt_suffix_S",
    "dlt_suffix_T",
    "dlt_suffix_G",
)

# Same structure as tools/inbox_export/scrub.py::DLT_SENDER (kept separate: no tools import here).
_DLT = re.compile(r"([A-Z]{2})-([A-Z0-9]{3,9})(?:-([PSTG]))?")


def sender_features(sender: str | None) -> np.ndarray:
    """Return float32[5]: alphanumeric flag, then one-hot DLT suffix P/S/T/G."""
    out = np.zeros(len(SENDER_FEATURE_NAMES), dtype=np.float32)
    if sender is None:
        return out
    text = str(sender).strip().upper().replace("<NUMBER>", "0")
    match = _DLT.fullmatch(text)
    if match is None:
        return out
    out[0] = 1.0
    if match.group(3):
        out[SENDER_FEATURE_NAMES.index(f"dlt_suffix_{match.group(3)}")] = 1.0
    return out


def sender_matrix(senders: Sequence[str | None]) -> np.ndarray:
    """Return float32[n, 5] with one row per sender."""
    if not len(senders):
        return np.zeros((0, len(SENDER_FEATURE_NAMES)), dtype=np.float32)
    return np.stack([sender_features(s) for s in senders]).astype(np.float32)
