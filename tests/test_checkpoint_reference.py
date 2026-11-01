"""No actual Torch/model inference: synthetic CPU feed and typed-manifest contracts."""

import contextlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rakshak.models.checkpoint_reference import checkpoint_scores
from rakshak.models.features import featurize


def test_checkpoint_reference_reads_canonical_side_and_all_batches(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        inference_mode=contextlib.nullcontext, from_numpy=lambda value: value))
    feats = featurize(['synthetic fixture']*257)
    batches = []
    class Tensor:
        def __init__(self, value):
            self.value = value
        def detach(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return self.value
    class Model:
        views = ('raw', 'canon')
        def cpu(self):
            return self
        def eval(self):
            return self
        def __call__(self, views, side):
            start = sum(batches)
            n = len(side)
            assert np.array_equal(views['canon'], feats.canon_bytes[start:start+n])
            assert np.array_equal(side, feats.side('canon')[start:start+n])
            batches.append(n)
            return Tensor(np.zeros(n))
    assert np.array_equal(checkpoint_scores(Model(), feats), np.full(257, .5))
    assert batches == [256, 1]


def test_checkpoint_job_bundles_the_same_cal_only_freeze_and_report_runner():
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('checkpoint_ref_builder',
                                                repo/'kaggle/build_kernel.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert 'kaggle/jobs/fusion_v1.py' in module.bundle_names('fusion_torch_ref')
