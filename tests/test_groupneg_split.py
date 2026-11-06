"""Group-disjoint fold assignment for the cross-fit GroundLM negatives (no torch needed)."""

from rakshak.eval.groupneg import assign_folds, negative_clusters


def test_identical_keys_share_a_cluster_and_unrelated_keys_do_not():
    keys = [
        "your sbi account xx <NUM> has been debited with <AMT>",
        "your sbi account xx <NUM> has been debited with <AMT>",
        "tata power: bill for october is <AMT> due: <NUM>",
    ]
    clusters = negative_clusters(keys)
    assert clusters[0] == clusters[1]
    assert clusters[2] != clusters[0]


def test_near_duplicate_template_joins_its_cluster():
    base = "jio: main balance inr egb/day. data balance <NUM> gb. active plans: <NUM> renewal"
    variant = base.replace("egb", "sgb")
    clusters = negative_clusters([base, variant, "airtel: your plan will renew in <NUM> days"])
    assert clusters[0] == clusters[1]
    assert clusters[2] != clusters[0]


def test_clusters_are_never_split_across_folds():
    keys = (
        ["a template alpha <NUM>"] * 5
        + ["beta bank notice <AMT> credited"] * 3
        + ["gamma upi pay <PHONE>", "delta telecom reminder <NUM> gb"]
    )
    clusters = negative_clusters(keys)
    folds = assign_folds(clusters, n_folds=2)
    for c in set(clusters):
        assert len({f for f, cc in zip(folds, clusters, strict=True) if cc == c}) == 1
    assert set(folds) == {0, 1}


def test_fold_sizes_stay_balanced_when_clusters_are_small():
    clusters = list(range(10))  # ten singletons
    folds = assign_folds(clusters, n_folds=2)
    assert folds.count(0) == folds.count(1) == 5
