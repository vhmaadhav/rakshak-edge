"""Synthetic initializer identity, missing/duplicate and dtype corruption guards."""

import numpy as np
import pytest

from rakshak.models.onnx_parameters import initializer_parameters


def test_parameter_reconstruction_is_exact_and_never_initialises_missing_weights():
    template = {'weight': np.zeros((2, 3), dtype=np.float32),
                'bias': np.zeros(2, dtype=np.float32)}
    actual = [('model.weight', np.arange(6, dtype=np.float32).reshape(2, 3)),
              ('model.bias', np.asarray([.1, -.2], dtype=np.float32))]
    recovered = initializer_parameters(template, actual)
    assert np.array_equal(recovered['weight'], actual[0][1])
    assert np.array_equal(recovered['bias'], actual[1][1])
    with pytest.raises(ValueError, match='every model parameter'):
        initializer_parameters(template, actual[:1])
    with pytest.raises(ValueError, match='duplicated'):
        initializer_parameters(template, actual+actual[:1])
    with pytest.raises(ValueError, match='dtype'):
        initializer_parameters(template, [('model.weight', actual[0][1].astype(np.float64)),
                                          actual[1]])
