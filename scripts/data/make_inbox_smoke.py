"""Synthetic smoke fixture for kaggle/jobs/sms_inbox_retrain.py. No real inbox data is read.

Draws legit rows from the synthetic, licensed FinEE set and scam rows from the ridham115 set, both
in data/processed/external/indian_legit, and writes them as train_donor.parquet and
test_donor.parquet in tools/inbox_export COLUMNS format, with fake DLT sender IDs.
Train and test senders are disjoint (40 and 20 IDs), and the two donor files use disjoint
source rows.

The rows are exported as if reviewed (human_reviewed=True, synthetic=False) because the job refuses
synthetic=True rows and this path must match a real export. The dataset metadata says "synthetic".

Usage: python scripts/data/make_inbox_smoke.py --output DIR [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.inbox_export.export import write_parquet  # noqa: E402
from tools.inbox_export.scrub import VERSION, sender_style  # noqa: E402

SOURCES = REPO / "data" / "processed" / "external" / "indian_legit"
DATASET_ID = "maadhavvh/rakshak-inbox-smoke"
# 1100 legit: 20 % of the senders must hold at least 200 legit rows after dedup (stop rule).
TRAIN = {"legit": 1100, "scam": 60, "senders": 40}
TEST = {"legit": 400, "scam": 30, "senders": 20}
SUFFIXES = ("P", "S", "T", "G", None)
LEGIT_POLICIES = ("benign_transactional", "promotion")


def fake_senders(count: int, offset: int) -> list[str]:
    """AX-FK<2 letters>-<suffix>, alternating AX and VM, suffixes cycled (some without suffix)."""
    out = []
    for i in range(count):
        n = i + offset
        code = chr(65 + n // 26) + chr(65 + n % 26)
        prefix = "AX" if n % 2 == 0 else "VM"
        suffix = SUFFIXES[n % len(SUFFIXES)]
        out.append(f"{prefix}-FK{code}" + (f"-{suffix}" if suffix else ""))
    return out


def plan_rows(texts: list[str], n_train: int, n_test: int, rng: np.random.Generator):
    """Pick one row per cap group (retrain_prep clustering) and give each split group to one donor
    role. Split groups never straddle the donors, so no near-duplicate crosses them. Returns
    (train texts, train split groups, test texts, test split groups)."""
    from rakshak.data import retrain_prep as RP

    order = rng.permutation(len(texts))[:40_000]
    sample = [texts[i] for i in order]
    split, cap = RP.two_level_labels([RP.normalize(t) for t in sample])
    df = pd.DataFrame({"text": sample, "split": split, "cap": cap})
    picks = df.sample(frac=1, random_state=int(rng.integers(2**31))).drop_duplicates("cap")
    sizes = picks.groupby("split").size()
    test_groups, acc = set(), 0
    for g in rng.permutation(sizes.index.to_numpy()).tolist():
        if acc >= n_test:
            break
        test_groups.add(g)
        acc += int(sizes[g])
    is_test = picks["split"].isin(test_groups).to_numpy()
    test_rows = picks[is_test].head(n_test)
    train_rows = picks[~is_test]
    if len(train_rows) < n_train or len(test_rows) < n_test:
        raise ValueError(
            f"not enough template groups: train {len(train_rows)}, test {len(test_rows)}")
    train_rows = train_rows.head(n_train)
    return (train_rows["text"].tolist(), train_rows["split"].tolist(),
            test_rows["text"].tolist(), test_rows["split"].tolist())


def balanced_senders(groups: list[int], senders: list[str]) -> list[str]:
    """Each split group gets one sender; the largest groups go first to the least loaded sender."""
    counts: dict[int, int] = {}
    for g in groups:
        counts[g] = counts.get(g, 0) + 1
    load = dict.fromkeys(senders, 0)
    owner: dict[int, str] = {}
    for g, size in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        sender = min(senders, key=lambda s: (load[s], s))
        owner[g] = sender
        load[sender] += size
    return [owner[g] for g in groups]


def build(seed: int) -> dict[str, list[dict]]:
    rng = np.random.default_rng(seed)
    finee = pd.read_parquet(SOURCES / "finee_synth.parquet")["text"].tolist()
    ridham = pd.read_parquet(SOURCES / "ridham115.parquet")
    scam = ridham[ridham["label"].astype(int) == 1]["text"].tolist()
    train_senders = fake_senders(TRAIN["senders"], 0)
    test_senders = fake_senders(TEST["senders"], TRAIN["senders"])
    assert not set(train_senders) & set(test_senders)

    legit_tr, legit_tr_g, legit_te, legit_te_g = plan_rows(
        finee, TRAIN["legit"], TEST["legit"], rng)
    scam_tr, scam_tr_g, scam_te, scam_te_g = plan_rows(scam, TRAIN["scam"], TEST["scam"], rng)

    def policy(label, i):
        return "scam" if label == 1 else LEGIT_POLICIES[i % 2]

    def donor(legit, legit_g, scam_t, scam_g, senders):
        texts = legit + scam_t
        labels = [0] * len(legit) + [1] * len(scam_t)
        groups = legit_g + scam_g
        addresses = balanced_senders(groups, senders)
        rows = []
        for i, (text, label, address) in enumerate(zip(texts, labels, addresses, strict=True)):
            rows.append({
                "id": uuid.UUID(int=int(rng.integers(0, 2**63)) * 2**64 + i).hex,
                "text": str(text), "sender": address, "sender_style": sender_style(address),
                "label_policy": policy(label, i), "synthetic": False, "human_reviewed": True,
                "scrubber_version": VERSION,
            })
        return rows

    return {
        "train_donor.parquet": donor(legit_tr, legit_tr_g, scam_tr, scam_tr_g, train_senders),
        "test_donor.parquet": donor(legit_te, legit_te_g, scam_te, scam_te_g, test_senders),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in build(args.seed).items():
        target = out / name
        target.unlink(missing_ok=True)
        write_parquet(rows, target)
        print(f"{name}: {len(rows)} rows")
    meta = {
        "id": DATASET_ID,
        "title": "rakshak-inbox-smoke",
        "licenses": [{"name": "other"}],
        "description": ("Synthetic smoke fixture for sms_inbox_retrain: FinEE and ridham115 rows "
                        "with fake DLT sender IDs. Not real inbox data."),
        "isPrivate": True,
    }
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"metadata: {DATASET_ID}")


if __name__ == "__main__":
    main()
