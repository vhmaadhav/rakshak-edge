"""Portable calibration, late fusion and selective decisions."""

from rakshak.fusion.calibrate import (
    IsotonicCalibration,
    PlattScaling,
    TemperatureScaling,
    calibration_metrics,
    reliability_diagram,
)
from rakshak.fusion.fuse import FusionModel, feature_matrix
from rakshak.fusion.selective import (
    SelectiveThresholds,
    conformal_fpr,
    group_coverage,
    risk_coverage,
    search_thresholds,
    selective_metrics,
)

__all__ = [
    "TemperatureScaling", "PlattScaling", "IsotonicCalibration", "calibration_metrics",
    "reliability_diagram", "FusionModel", "feature_matrix", "SelectiveThresholds",
    "conformal_fpr", "group_coverage", "risk_coverage", "search_thresholds", "selective_metrics",
]
