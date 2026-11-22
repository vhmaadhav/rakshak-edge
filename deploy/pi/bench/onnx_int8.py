"""INT8 dynamic quantisation of the SMS student, and the verdict-flip count used to decide it.

INT8 is used only when ``verdict_flips`` is 0 over every local SMS message at the registry
threshold. ``verdict_flips`` counts messages whose ``score >= threshold`` decision differs between
two detectors (both need ``score_many``). The decision is made per run; see docs/EDGE_PIPELINE.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def quantize_int8_copy(src: Path, dst: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return dst


def verdict_flips(fp32: Any, other: Any, texts: list[str], threshold: float) -> dict[str, Any]:
    """Decision flips at ``threshold`` between two detectors on the same texts."""
    a = fp32.score_many(texts)
    b = other.score_many(texts)
    flips = [
        i
        for i, (x, y) in enumerate(zip(a, b, strict=False))
        if (x >= threshold) != (y >= threshold)
    ]
    return {
        "n": len(texts),
        "flips": len(flips),
        "flip_indices": flips[:20],
        "max_abs_score_diff": max((abs(x - y) for x, y in zip(a, b, strict=False)), default=0.0),
        "threshold": threshold,
    }
