"""On-device student models for the SMS detector.

``encoding`` and ``features`` need only numpy, the canonicaliser and the rules. ``charcnn``,
``fasttext`` and ``student`` import torch, so import them only where torch is available
(Kaggle GPU kernels, or the tests after ``pytest.importorskip("torch")``).
"""
