"""Membership-only audit of the inherited broad KD pool. No text/label outcomes reported."""

import numpy as np


def audit_extra_rows(pool_positions, train, cal, test, keys, campaigns, sources, *,
                     external_keys=(), source_holdout=False):
    keys, campaigns, sources = (np.asarray(x, dtype=str) for x in (keys, campaigns, sources))
    if not (keys.shape == campaigns.shape == sources.shape):
        raise ValueError("one key, campaign and source per corpus row required")
    extra = np.setdiff1d(pool_positions, train)
    test_campaigns = set(campaigns[test]) - {"", "nan", "None", "<NA>"}
    held_sources = set(sources[test]) if source_holdout else set()
    return {
        "pool_rows": len(pool_positions), "frozen_train_rows": len(train),
        "extra_unlabeled_rows": len(extra),
        "extra_exact_test_copies": int(np.isin(keys[extra], keys[test]).sum()),
        "extra_exact_cal_copies": int(np.isin(keys[extra], keys[cal]).sum()),
        "extra_in_test_campaigns": int(np.isin(campaigns[extra], list(test_campaigns)).sum()),
        "extra_external_exact_overlap": int(np.isin(keys[extra], list(external_keys)).sum()),
        "extra_heldout_source": int(np.isin(sources[extra], list(held_sources)).sum()),
        "frozen_train_in_test_campaigns": int(
            np.isin(campaigns[train], list(test_campaigns)).sum()),
        "scope": "membership diagnostics, no model outcomes; original split IDs unchanged",
    }
