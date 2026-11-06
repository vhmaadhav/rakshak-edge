"""Reuse triage metrics and prospective P1 gates for explicit family decisions."""

import hashlib

import numpy as np

from rakshak.fusion.audit import triage_report
from rakshak.fusion.hardneg_gate import hardneg_gate


class FrozenDecisions:
    """Reporting adapter only; never selects a threshold or substitutes other rows."""

    def __init__(self, probabilities, states, family_policy):
        self.probabilities = np.asarray(probabilities).copy()
        self.states = np.asarray(states).copy()
        if (self.states.shape != self.probabilities.shape
                or not np.isin(self.states, ['SAFE', 'UNCERTAIN', 'SCAM']).all()):
            raise ValueError('aligned frozen triage states required')
        self.family_policy = family_policy

    def transform(self, probabilities):
        if not np.array_equal(np.asarray(probabilities), self.probabilities):
            raise ValueError('frozen decision adapter cannot be applied to other rows/scores')
        return self.states.copy()

    def to_dict(self):
        return {'kind': 'explicit frozen per-family decisions, not one global threshold',
                'states_sha256': hashlib.sha256('\n'.join(self.states).encode()).hexdigest(),
                'family_policy_sha256': self.family_policy}


def family_triage(labels, probabilities, states, campaigns, student, policy_hash, *, n_boot=1000):
    return triage_report(labels, probabilities,
                         FrozenDecisions(probabilities, states, policy_hash), campaigns,
                         student_pred=student, n_boot=n_boot)


def family_gate(internal, external, states):
    """Same predeclared absolute/no-increase/recall gates as arm(a), no endpoint changes."""
    def renamed(block):
        return {target: block[source] for source, target in (
            ('baseline', 'baseline'), ('family', 'transactional')) if source in block}
    result = hardneg_gate(
        {fold: {seed: renamed(block) for seed, block in seeds.items()}
         for fold, seeds in internal.items()},
        {name: {seed: renamed(block) for seed, block in seeds.items()}
         for name, seeds in external.items()},
        {fold: {seed: renamed(block) for seed, block in seeds.items()}
         for fold, seeds in states.items()})
    result['scope'] = ('template-cal-c1: original P1 absolute/no-increase/SCAM recall gates, '
                       'all3seeds; legacy every fold, known fraud where positives, explicit UCI NA')
    return result
