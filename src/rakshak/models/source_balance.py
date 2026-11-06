"""Training-source interventions; never uses cal/test source labels to set weights.

Balanced replacement sampling gives each training source equal expected mass at the same
number of updates as ERM. DRO adds exponentiated-gradient group weights to that sampler.
Missing minibatch groups get zero update; loss weights renormalise over present groups.
"""

import numpy as np


def source_probabilities(sources):
    names, groups = np.unique(np.asarray(sources, dtype=str), return_inverse=True)
    count = np.bincount(groups)
    return names, groups, 1.0 / (len(count) * count[groups])


class SourceObjective:
    def __init__(self, sources, n, strategy="erm", eta=0.01, device="cpu"):
        import torch

        if strategy not in {"erm", "balanced", "dro"} or eta <= 0:
            raise ValueError("unknown source strategy or nonpositive DRO eta")
        if sources is None:
            if strategy != "erm":
                raise ValueError("source intervention requires aligned training sources")
            sources = ["all"] * n
        if len(sources) != n or not n:
            raise ValueError("sources must be nonempty and aligned")
        self.names, self.groups, self.prob = source_probabilities(sources)
        self.strategy, self.eta, self.n = strategy, eta, n
        self.q = torch.ones(len(self.names), device=device) / len(self.names)

    def order(self, rng):
        if self.strategy == "erm":
            return rng.permutation(self.n)
        return rng.choice(self.n, size=self.n, replace=True, p=self.prob)

    def reduce(self, losses, indices):
        import torch

        if self.strategy != "dro":
            return losses.mean()
        groups = torch.as_tensor(self.groups[indices], device=losses.device)
        sums = torch.zeros_like(self.q).scatter_add_(0, groups, losses)
        counts = torch.zeros_like(self.q).scatter_add_(0, groups, torch.ones_like(losses))
        means = sums / counts.clamp_min(1)
        present = counts > 0
        with torch.no_grad():
            logq = self.q.clamp_min(1e-12).log() + self.eta * means.detach()
            self.q = torch.softmax(logq, dim=0)
        weights = self.q * present
        return (weights * means).sum() / weights.sum()

    def weights(self):
        return dict(zip(self.names.tolist(), self.q.detach().cpu().tolist(), strict=True))
