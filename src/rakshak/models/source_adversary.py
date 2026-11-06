"""Training-source regularization conditioned on true training labels, without target data.

Derived gradient-reversal experiment, not a DANN/CDAN reproduction. Auxiliary heads never
become classifier children or inference parameters. Torch is lazy for local metadata tests.
"""

from collections import Counter
from functools import lru_cache
from weakref import ref

import numpy as np


def conditional_layout(sources, labels):
    sources, labels = np.asarray(sources, dtype=str), np.asarray(labels, dtype=float)
    if sources.shape != labels.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("aligned nonempty training sources and labels required")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("only labelled frozen train rows are permitted")
    supports = {y: sorted(set(sources[labels == y])) for y in (0, 1)}
    supports = {y: names for y, names in supports.items() if len(names) >= 2}
    counts = Counter(zip(labels.astype(int), sources, strict=True))
    codes, weights = np.full(len(labels), -1, dtype=int), np.zeros(len(labels))
    for y, names in supports.items():
        for code, source in enumerate(names):
            mask = (labels == y) & (sources == source)
            codes[mask] = code
            weights[mask] = 1 / counts[(y, source)]
    if not supports:
        raise ValueError("adversary needs at least one label with two training sources")
    return supports, codes, weights


@lru_cache(maxsize=1)
def _reversal():
    import torch

    class Reverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, coefficient):
            ctx.coefficient = coefficient
            return value.view_as(value)

        @staticmethod
        def backward(ctx, gradient):
            return -ctx.coefficient * gradient, None

    return Reverse


def reverse_features(value, coefficient):
    return _reversal().apply(value, coefficient)


def make_adversary(model, sources, labels, device):
    import torch
    import torch.nn.functional as F
    from torch import nn

    supports, codes, weights = conditional_layout(sources, labels)
    if not hasattr(model, "encoders") or not hasattr(model, "head"):
        raise ValueError("source adversary requires the byte CNN model")
    width = sum(encoder.out_dim for encoder in model.encoders.values())
    model_ref = ref(model)

    class Heads(nn.Module):
        def __init__(self):
            super().__init__()
            # Preserve backbone/dropout RNG in both zero-weight and adversarial controls.
            with torch.random.fork_rng(devices=[]):
                self.heads = nn.ModuleDict({str(y): nn.Sequential(
                    nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, len(names)))
                    for y, names in supports.items()})
            self.to(device)
            self.features = None
            self.handle = model_ref().head.register_forward_pre_hook(self.capture)

        def capture(self, module, inputs):
            if model_ref().training:
                # Rule side features are immutable inputs, not a learned encoder output.
                self.features = inputs[0][:, :width]

        def loss(self, indices, y, coefficient):
            if self.features is None:
                raise ValueError("a training forward pass must precede adversary loss")
            features, self.features = self.features, None
            w = torch.as_tensor(weights[indices], dtype=features.dtype, device=features.device)
            targets = torch.as_tensor(codes[indices], device=features.device)
            numerator = features.sum() * 0
            for label, head in self.heads.items():
                mask = y == int(label)
                if bool(mask.any()):
                    per_row = F.cross_entropy(
                        head(reverse_features(features[mask], coefficient)), targets[mask],
                        reduction="none")
                    numerator = numerator + (per_row * w[mask]).sum()
            return numerator / w.sum().clamp_min(1e-12)

        def close(self):
            self.features = None
            self.handle.remove()

        def info(self):
            return {"supports": supports, "feature_width": width, "hidden": 64,
                    "params": sum(p.numel() for p in self.parameters()),
                    "weighting": "equal training label/source-cell mass; batch normalized",
                    "scope": "learned encoder features only; no test or external rows"}

    return Heads()


def verify_torch_invariants(device="cpu"):
    """Tiny synthetic unit checks before corpus access, also runnable where torch is installed."""
    import copy

    import torch

    from rakshak.models.encoding import N_RULES
    from rakshak.models.student import ModelSpec, build_model

    value = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
    reversed_value = reverse_features(value, 0.05)
    assert torch.equal(value, reversed_value)
    reversed_value.sum().backward()
    torch.testing.assert_close(value.grad, torch.full_like(value, -0.05))
    torch.manual_seed(7)
    model = build_model(ModelSpec("dual", "base", "dual")).to(device).train()
    reference = copy.deepcopy(model)
    state_keys, rng = set(model.state_dict()), torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if str(device).startswith("cuda") else None
    y = torch.tensor([0.0, 1.0, 0.0, 1.0], device=device)
    heads = make_adversary(model, ["a", "a", "b", "b"], y.cpu().numpy(), device)
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert set(model.state_dict()) == state_keys
    views = {v: torch.randint(1, 200, (4, 12), device=device) for v in model.views}
    side = torch.zeros((4, N_RULES), device=device)
    torch.manual_seed(19)
    reference_logits = reference(views, side)
    reference_logits.square().mean().backward()
    torch.manual_seed(19)
    logits = model(views, side)
    torch.testing.assert_close(logits, reference_logits)
    (logits.square().mean() + heads.loss(np.arange(4), y, 0.0)).backward()
    for a, b in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4)
    assert any(p.grad is not None and bool((p.grad != 0).any()) for p in heads.parameters())
    model.zero_grad(set_to_none=True)
    model(views, side)
    heads.loss(np.arange(4), y, 0.05).backward()
    assert any(p.grad is not None and bool((p.grad != 0).any())
               for p in model.encoders.parameters())
    heads.close()
    assert not model.head._forward_pre_hooks
