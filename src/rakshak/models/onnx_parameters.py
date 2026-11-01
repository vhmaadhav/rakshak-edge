"""Strict parameter recovery from named ONNX initializers, without changing tensor values."""

import numpy as np


def initializer_parameters(expected, initializers):
    recovered = {}
    for name, value in initializers:
        if not name.startswith('model.'):
            continue
        key = name.removeprefix('model.')
        if key not in expected or key in recovered:
            raise ValueError('unexpected or duplicated model initializer')
        value, reference = np.asarray(value), np.asarray(expected[key])
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError('initializer shape or dtype changed')
        if not np.isfinite(value).all():
            raise ValueError('nonfinite model initializer')
        recovered[key] = value.copy()
    if set(recovered) != set(expected):
        raise ValueError('not every model parameter has an exact named ONNX initializer')
    return recovered
