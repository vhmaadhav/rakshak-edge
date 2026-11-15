"""Portable calibration, late fusion and selective decisions.

Names are loaded on first use (PEP 562), so importing a submodule such as
``rakshak.fusion.url_expert`` does not pull in scipy and sklearn. The Pi
edge bench has numpy and onnxruntime but no scipy.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "rakshak.fusion.calibrate": (
        "TemperatureScaling", "PlattScaling", "IsotonicCalibration", "calibration_metrics",
        "reliability_diagram",
    ),
    "rakshak.fusion.fuse": ("FusionModel", "feature_matrix"),
    "rakshak.fusion.selective": (
        "SelectiveThresholds", "conformal_fpr", "group_coverage", "risk_coverage",
        "search_thresholds", "selective_metrics",
    ),
}
_OWNER = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = list(_OWNER)


def __getattr__(name: str):
    module = _OWNER.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
