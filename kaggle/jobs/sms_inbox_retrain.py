"""SMS student retrain on consented donor inbox data (docs/INBOX_RETRAIN_PLAN.md, approved).

Frozen plan. One arm per kernel, seeds 1 2 3 per arm, same dual-base recipe as sms_retrain (v0):
- R0: control. Campaign train only (no inbox rows). Side width 16.
- R1: R0 + train-donor training rows (legit and scam). Side width 16.
- R2: R1 + 5 sender side features appended to the 16 rule features. Side width 21. Rows without a
  sender (campaign, GroundLM, SmishX) get zeros.

Surrogate-filled arms (docs/SURROGATE_RETRAIN_PLAN.md sections 5 to 7, Amendment 1):
- S0, S1, S2: the recipes of R0, R1 and R2 on surrogate-filled text. --arm S0 (S1, S2) implies
  --fill-surrogates. --fill-surrogates with an R name is an error. The fill runs once, right after
  loading: every text of sms.parquet (all splits, all rows) and every inbox row (train and test
  donor) is replaced in memory by rakshak.data.surrogates.fill_many, keyed by its id. GroundLM,
  SmishX and the shift files are not filled (plan rule 4). Plan rule 5: if any filled text still
  matches <[A-Z_]+>, the job stops with a count. The payload, metrics.json and report.md name the
  arm (S0, S1, S2) and carry "fill_surrogates": true and "surrogate_seed". S0 also carries
  "stop_rule_2": {"g2_failed_all_seeds": bool}. Reported without a gate: the shipped v0 ONNX's
  campaign F1 on the filled test split, test-donor scam recall with exact (Clopper-Pearson) 95 %
  intervals, tag counts before filling per source, and the source-identity probe
  (rakshak.datasets.confound) on the campaign training scam rows, scrubbed and filled. S2's paired
  comparison takes S1's probs.parquet, passed with --r1-probs-dir as R2 takes R1's.

Donors. The train donor (Maadhav) gives training rows and the real legit validation set: 20 % of
sender IDs, held out whole (data seed 0). The test donor (Sachin) is test-only: scored once per arm,
never used for training, threshold or tuning.

PRIVACY (binding). No inbox text and no sender string leaves memory. Logs, metrics.json,
probs.parquet, prep_report.json, thresholds.json and report.md carry only counts, labels, scores,
tau, index ids ("legit_val_<i>", "test_donor_<i>") and integer group codes.

Pipeline:
0. S arms only: fill the texts (see above) and run the rule 5 check, before anything else.
1. Campaign data, lang_shift, temporal, GroundLM and SmishX exactly as sms_retrain loads them.
2. Inbox files train_donor.parquet and test_donor.parquet, found by name under /kaggle/input
   (the rakshak-inbox dataset). Labels: benign_transactional and promotion -> 0, scam -> 1;
   benign_personal and unclear dropped; rows not human-reviewed dropped.
3. Train donor: two-level clustering (retrain_prep.two_level_labels), test-only dedup (whole split
   group removed when it near-duplicates campaign test, lang_shift, temporal, GroundLM, SmishX or
   any test-donor row), sender split (sender IDs and split groups are joined into units; 20 % of
   sender IDs are held out as whole units), 5-per-cap-group cap and total cap (retrain_prep).
4. Stop rule, before any training: train-donor legit rows after review and dedup < 500, or held-out
   legit rows after review and dedup < 200, writes status "stopped: insufficient data" with counts.
5. Threshold: retrain_prep.threshold_rule on the held-out legit scores and campaign cal. tau is
   written to thresholds.json before any test set is scored.
6. Scoring: campaign test, lang_shift, temporal, GroundLM, SmishX, validation scam rows, test donor
   (full, and template-novel: no near-dup Jaccard >= 0.8 in the training rows). Gates G1 to G5 (G5:
   test-donor legit FPR point estimate <= 1 %), median plus 2 of 3 seeds.
7. R2 vs R1 on G5: paired bootstrap over test-donor sender groups of the FPR difference, at the
   seed-median model of each arm. R1's probs.parquet must be attached (--r1-probs-dir).
8. The shipped v0 ONNX (sha 5341e14c, attached with --kernels maadhavvh/rakshak-sms-student) is
   scored on both donor sets for comparison. Skipped with a record when not attached.
9. Each seed's ONNX is exported and parity-checked against torch.

Launch (one kernel per arm, T4, two datasets; R2 needs the finished R1 kernel output attached;
S2 needs the finished S1 kernel output attached, with the same pattern as R2):
  kaggle/launch.sh sms_inbox_retrain --run-name r0 --gpu t4 --internet \
      --datasets maadhavvh/rakshak-data maadhavvh/rakshak-inbox \
      --kernels maadhavvh/rakshak-sms-student --job-args="--arm R0"
  kaggle/launch.sh sms_inbox_retrain --run-name r1 --gpu t4 --internet \
      --datasets maadhavvh/rakshak-data maadhavvh/rakshak-inbox \
      --kernels maadhavvh/rakshak-sms-student --job-args="--arm R1"
  kaggle/launch.sh sms_inbox_retrain --run-name r2 --gpu t4 --internet \
      --datasets maadhavvh/rakshak-data maadhavvh/rakshak-inbox \
      --kernels maadhavvh/rakshak-sms-student maadhavvh/rakshak-sms-inbox-retrain-r1 \
      --job-args="--arm R2 --r1-probs-dir sms-inbox-retrain-r1"
  kaggle/launch.sh sms_inbox_retrain --run-name s0 --gpu t4 --internet \
      --datasets maadhavvh/rakshak-data maadhavvh/rakshak-inbox \
      --kernels maadhavvh/rakshak-sms-student --job-args="--arm S0"
  kaggle/launch.sh sms_inbox_retrain --run-name s1 --gpu t4 --internet \
      --datasets maadhavvh/rakshak-data maadhavvh/rakshak-inbox \
      --kernels maadhavvh/rakshak-sms-student --job-args="--arm S1"
  kaggle/launch.sh sms_inbox_retrain --run-name s2 --gpu t4 --internet \
      --datasets maadhavvh/rakshak-data maadhavvh/rakshak-inbox \
      --kernels maadhavvh/rakshak-sms-student maadhavvh/rakshak-sms-inbox-retrain-s1 \
      --job-args="--arm S2 --r1-probs-dir sms-inbox-retrain-s1"

Flags (``--job-args="FLAGS"``):
  --arm {R0,R1,R2,S0,S1,S2}  required
  --fill-surrogates     implied by an S name; an error with an R name
  --seeds INT ...       default 1 2 3
  --epochs INT          max epochs (default 15)
  --patience INT        early-stop patience on cal PR-AUC (default 3)
  --batch-size INT      (default 128)
  --lr FLOAT            (default 2e-3)
  --weight-decay FLOAT  (default 1e-4)
  --n-boot INT          bootstrap resamples (default 1000)
  --no-export           skip ONNX export (gate G4 then cannot pass)
  --r1-probs-dir TEXT   substring of the attached previous-arm kernel output path (R1 for R2,
                        S1 for S2: the paired comparison)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sms_retrain as base
from _common import Units, env_report, seed_everything, timed
from baseline_v0 import (
    ARTIFACT_PINS,
    SMS_THRESHOLD,
    _onnx_session,
    groundlm_table,
    locate_artifact,
    rules_any,
    score_sms_texts,
    smishx_table,
    sms_run_table,
)

from rakshak.data import retrain_prep as RP
from rakshak.data import surrogates as S
from rakshak.eval import data as D
from rakshak.models.encoding import N_RULES
from rakshak.models.features import featurize, with_side_extra
from rakshak.models.sender_features import SENDER_FEATURE_NAMES, sender_matrix

ARMS = ("R0", "R1", "R2")
FILL_ARMS = ("S0", "S1", "S2")  # surrogate-filled twins of R0, R1, R2 (plan sections 5 to 7)
RECIPE = {"S0": "R0", "S1": "R1", "S2": "R2"}  # the R arm whose recipe an S arm runs
PAIRED_PREV = {"R2": "R1", "S2": "S1"}  # the arm each paired comparison is against
ALL_ARMS = (*ARMS, *FILL_ARMS)
DEFAULT_SEEDS = (1, 2, 3)
SIDE_EXTRA = {"R0": 0, "R1": 0, "R2": len(SENDER_FEATURE_NAMES)}
DATA_SEED = 0
SPLIT_SEED_OFFSET = 1  # held-out sender units use seed + 1, as build_pools does for FinEE groups
BOOT_SEED = 0
TRAIN_FILE = "train_donor.parquet"
TEST_FILE = "test_donor.parquet"
INBOX_COLUMNS = ("id", "text", "sender", "sender_style", "label_policy", "synthetic",
                 "human_reviewed", "scrubber_version")  # tools/inbox_export/export.py COLUMNS
LABEL_MAP = {"benign_transactional": 0, "promotion": 0, "scam": 1}
HOLDOUT_SENDER_FRAC = 0.2
MIN_DONOR_LEGIT = 500
MIN_VAL_LEGIT = 200
G5_MAX_FPR = 0.01
V0_SHA = ARTIFACT_PINS["best_student.onnx"][0]
V0_NAME = "best_student.onnx"
V0_ROOT = Path("/kaggle/input")
STOP_STATUS = "stopped: insufficient data"
# Read by kaggle/build_kernel.py: an error is logged and stored as its type and frames only,
# never its message (a message can carry inbox text or sender strings).
REDACT_ERRORS = True
SET_TEST_DONOR = "test_donor"
SET_LEGIT_VAL = base.SET_LEGIT_VAL
SET_CAL = base.SET_CAL
SET_CAMPAIGN_TEST = base.SET_CAMPAIGN_TEST
SET_GROUNDLM = base.SET_GROUNDLM
SET_SMISHX = base.SET_SMISHX
SET_VAL_SCAM = base.SET_VAL_SCAM

# base.GATES (G1 to G4) plus G5, the test-donor legit FPR point estimate.
INBOX_GATES: tuple = (
    *base.GATES,
    ("G5_test_donor_legit_fpr", "donor_fpr", lambda v: v <= G5_MAX_FPR),
)


# ---- pure helpers (unit-tested in tests/test_sms_inbox_retrain.py) ----


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sms_inbox_retrain")
    p.add_argument("--arm", choices=ALL_ARMS, required=True)
    p.add_argument("--fill-surrogates", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--r1-probs-dir", default=None)
    args = p.parse_args(list(argv or []))
    if args.arm in FILL_ARMS:
        args.fill_surrogates = True
    elif args.fill_surrogates:
        p.error("--fill-surrogates needs an S arm name (S0, S1 or S2)")
    if len(set(args.seeds)) != len(args.seeds):
        p.error("--seeds must not repeat a value")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        p.error("--epochs, --patience and --batch-size must be positive")
    return args


def recipe_of(arm: str) -> str:
    """The R arm whose recipe ``arm`` runs: S0 to S2 map to R0 to R2; R names map to themselves."""
    return RECIPE.get(arm, arm)


def fill_fields(arm: str, fill: bool) -> dict:
    """Keys every S-arm payload carries. R arms get none, so their outputs do not change."""
    if not fill:
        return {}
    return {"fill_surrogates": True, "surrogate_seed": S.SURROGATE_SEED, "recipe": recipe_of(arm)}


def args_record(args: argparse.Namespace) -> dict:
    """The parsed arguments as recorded in the payload. The fill flag is kept only for S arms."""
    rec = dict(vars(args))
    if not rec.get("fill_surrogates"):
        rec.pop("fill_surrogates", None)
    return rec


def paired_key(arm: str) -> str:
    """Payload key of the paired comparison. Every R arm keeps the original key; the S arms share
    the S2 vs S1 key (S0 and S1 carry a skipped note there)."""
    return "paired_r2_vs_r1" if arm in ARMS else "paired_s2_vs_s1"


def side_width(arm: str) -> int:
    """Side input width of an arm: 16 for R0, R1 and their S twins, 21 for R2 and S2."""
    return N_RULES + SIDE_EXTRA[recipe_of(arm)]


def label_inbox(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep human-reviewed rows with a mapped label; return them with an int ``label`` column.
    Counts only leave this function. Raises when the file lacks a column or marks synthetic rows."""
    missing = [c for c in INBOX_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"inbox file lacks columns {missing}")
    if df["synthetic"].fillna(True).astype(bool).any():
        raise ValueError("inbox file marks rows as synthetic")
    counts: dict[str, Any] = {"rows": int(len(df))}
    reviewed = df["human_reviewed"].fillna(False).astype(bool).to_numpy()
    # Plan amendment 2026-11-07: the train donor's rows may be labelled by Codex on scrubbed text.
    # They stay human_reviewed=False and carry label_source="codex"; any other source is dropped.
    if "label_source" in df.columns:
        codex = (df["label_source"].astype(str) == "codex").to_numpy() & ~reviewed
        counts["codex_labelled"] = int(codex.sum())
        reviewed = reviewed | codex
    counts["not_human_reviewed"] = int((~reviewed).sum())
    d = df[reviewed].copy()
    policy = d["label_policy"].astype(str)
    counts["by_policy"] = {str(k): int(v) for k, v in policy.value_counts().items()}
    keep = policy.isin(list(LABEL_MAP)).to_numpy()
    counts["dropped_personal_or_unclear"] = int((~keep).sum())
    d = d[keep].copy()
    d["label"] = policy[keep].map(LABEL_MAP).astype(np.int64).to_numpy()
    d["text"] = [t if isinstance(t, str) else "" for t in d["text"]]
    d["sender"] = [("" if pd.isna(s) else str(s)).strip() for s in d["sender"]]
    counts["kept"] = int(len(d))
    counts["kept_legit"] = int((d["label"] == 0).sum())
    counts["kept_scam"] = int((d["label"] == 1).sum())
    return d.reset_index(drop=True), counts


def sender_codes(senders: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Integer sender group per row, and a has-sender flag. A row without a sender gets its own
    group, so it is never grouped with another row."""
    keys, has = [], []
    for i, s in enumerate(senders):
        text = str(s).strip()
        has.append(bool(text))
        keys.append(text if text else f"\x00row{i}")
    codes, _ = pd.factorize(pd.Series(keys, dtype=object))
    return codes.astype(np.int64), np.asarray(has, dtype=bool)


def sender_split(
    split_groups: np.ndarray, codes: np.ndarray, seed: int, frac: float
) -> tuple[np.ndarray, dict]:
    """Hold out whole units until about ``frac`` of the sender IDs are held. A unit is a connected
    component of the bipartite graph (split group <-> sender ID), so no sender ID and no split group
    straddles the held-out and training sides. Returns a boolean mask over rows and statistics."""
    n = len(codes)
    if n == 0:
        return np.zeros(0, dtype=bool), {"sender_ids": 0, "sender_ids_held_out": 0, "units": 0,
                                         "units_held_out": 0, "held_fraction": 0.0}
    n_send = int(codes.max()) + 1
    n_grp = int(split_groups.max()) + 1
    parent = list(range(n_send + n_grp))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, g in zip(codes.tolist(), split_groups.tolist(), strict=True):
        a, b = find(s), find(n_send + g)
        if a != b:
            parent[b] = a
    unit = np.array([find(int(s)) for s in codes], dtype=np.int64)
    _, unit_idx = np.unique(unit, return_inverse=True)
    n_units = int(unit_idx.max()) + 1
    pairs = pd.DataFrame({"u": unit_idx, "s": codes}).drop_duplicates()
    per_unit = pairs.groupby("u").size().reindex(range(n_units), fill_value=0).to_numpy()
    total = int(len(np.unique(codes)))
    target = int(round(frac * total))
    order = np.random.default_rng(seed).permutation(n_units)
    held_units: list[int] = []
    held = 0
    for u in order.tolist():
        if held >= target:
            break
        held_units.append(u)
        held += int(per_unit[u])
    mask = np.isin(unit_idx, held_units)
    return mask, {
        "sender_ids": total,
        "sender_ids_held_out": held,
        "units": n_units,
        "units_held_out": len(held_units),
        "largest_unit_sender_ids": int(per_unit.max()),
        "held_fraction": (held / total) if total else 0.0,
    }


def build_inbox_pools(
    train_df: pd.DataFrame,
    test_only_texts: Sequence[str],
    n_campaign_neg: int,
    seed: int = DATA_SEED,
) -> dict:
    """Train-donor training rows and held-out legit and scam validation rows (plan section 4).
    Same cap and total-cap helpers as retrain_prep.build_pools. ``train_df`` needs text, sender and
    label columns; the sender strings stay in memory and are dropped from every returned frame."""
    st: dict[str, Any] = {}
    report: dict[str, Any] = {"seed": seed, "split_jaccard": RP.SPLIT_JACCARD,
                              "cap_jaccard": RP.CAP_JACCARD, "stages": st}
    df = train_df.reset_index(drop=True).copy()
    norms = [RP.normalize(t) for t in df["text"]]
    lsh: dict[str, Any] = {}
    split, cap = RP.two_level_labels(norms, lsh)
    df["cluster"] = split
    df["cap_group"] = cap
    st["after_review_rows"] = int(len(df))
    st["after_review_legit"] = int((df["label"] == 0).sum())
    st["split_groups"] = int(lsh.get("split_groups", 0))
    st["cap_groups"] = int(lsh.get("cap_groups", 0))

    # 1. test-only dedup: whole split groups that near-duplicate a test-only text are removed
    test_norms = [RP.normalize(t) for t in test_only_texts]
    flagged = RP.test_contaminated(norms, test_norms, report.setdefault("lsh_test_stats", {}))
    bad = set(df.loc[flagged, "cluster"].tolist())
    st["rows_dropped_test_contaminated"] = int(df["cluster"].isin(bad).sum())
    st["split_groups_dropped_test_contaminated"] = len(bad)
    df = df[~df["cluster"].isin(bad)].reset_index(drop=True)
    st["after_dedup_rows"] = int(len(df))
    st["after_dedup_legit"] = int((df["label"] == 0).sum())
    st["after_dedup_scam"] = int((df["label"] == 1).sum())

    # 2. sender-disjoint hold-out of legit validation (scam rows of held senders report recall)
    codes, _has = sender_codes(df["sender"].tolist())
    df["sender_code"] = codes
    held, sstats = sender_split(df["cluster"].to_numpy(), codes, seed + SPLIT_SEED_OFFSET,
                                HOLDOUT_SENDER_FRAC)
    st["sender_split"] = sstats
    val_side = df[held].copy()
    train_side = df[~held].copy()
    st["val_legit_after_dedup"] = int((val_side["label"] == 0).sum())
    st["val_scam_after_dedup"] = int((val_side["label"] == 1).sum())
    st["train_legit_after_dedup"] = int((train_side["label"] == 0).sum())
    st["train_scam_after_dedup"] = int((train_side["label"] == 1).sum())
    st["sender_ids_disjoint"] = bool(
        set(train_side["sender_code"].tolist()).isdisjoint(val_side["sender_code"].tolist())
    )

    # 3. 5-per-cap-group cap on each side, seeded shuffle as in build_pools
    rng = np.random.default_rng(seed)

    def capped(side: pd.DataFrame) -> pd.DataFrame:
        if side.empty:
            return side
        shuffled = side.iloc[rng.permutation(len(side))]
        return RP._cap_per_group(shuffled, "cap_group", RP.PER_GROUP_CAP)

    train_capped = capped(train_side)
    val_capped = capped(val_side)
    st["train_side_rows_after_cap"] = int(len(train_capped))
    st["val_rows_after_cap"] = int(len(val_capped))

    # 4. total cap on training rows, round-robin over cap groups (scam rows get their own budget)
    budget = min(RP.TOTAL_CAP, int(RP.TOTAL_CAP_FRACTION * n_campaign_neg))
    st["total_cap_budget"] = budget
    legit_train = RP._round_robin_cap(train_capped[train_capped["label"] == 0], budget, seed + 2)
    scam_train = RP._round_robin_cap(train_capped[train_capped["label"] == 1], budget, seed + 3)
    train_extra = pd.concat([legit_train, scam_train], ignore_index=True)
    st["train_extra_legit"] = int(len(legit_train))
    st["train_extra_scam"] = int(len(scam_train))
    st["train_extra_total"] = int(len(train_extra))
    st["val_legit"] = int((val_capped["label"] == 0).sum())
    st["val_scam"] = int((val_capped["label"] == 1).sum())
    st["sender_ids_in_training"] = (
        int(train_extra["sender_code"].nunique()) if len(train_extra) else 0
    )

    keep = ["text", "label", "sender", "cluster", "cap_group", "sender_code"]
    return {
        "train_extra": train_extra[keep].reset_index(drop=True),
        "val_legit": val_capped[val_capped["label"] == 0][keep].reset_index(drop=True),
        "val_scam": val_capped[val_capped["label"] == 1][keep].reset_index(drop=True),
        "report": report,
    }


def stop_reason(report: dict) -> str | None:
    """None when the donor data is enough to train. Otherwise a reason with the counts only."""
    st = report["stages"]
    donor = st["after_dedup_legit"]
    val = st["val_legit_after_dedup"]
    if donor >= MIN_DONOR_LEGIT and val >= MIN_VAL_LEGIT:
        return None
    return (f"train donor legit after review and dedup {donor} (need {MIN_DONOR_LEGIT}); "
            f"held-out legit {val} (need {MIN_VAL_LEGIT})")


def novel_mask(test_texts: Sequence[str], train_texts: Sequence[str]) -> np.ndarray:
    """True for test-donor rows with no near-duplicate (Jaccard >= 0.8) among the training rows."""
    flagged = RP.test_contaminated(
        [RP.normalize(t) for t in test_texts], [RP.normalize(t) for t in train_texts], {}
    )
    return ~flagged


def rate_on(scores: np.ndarray, mask: np.ndarray, tau: float) -> dict:
    return base.rate_on(scores, mask, tau)


def donor_rates(labels: np.ndarray, scores: np.ndarray, tau: float,
                novel: np.ndarray | None = None) -> dict:
    """Legit FPR and scam recall of one donor set at tau, Wilson intervals. ``novel`` adds the
    template-novel subset."""
    labels = np.asarray(labels).astype(np.int64)
    s = np.asarray(scores, dtype=np.float64)
    legit = labels == 0
    scam = labels == 1
    out: dict[str, Any] = {
        "n_legit": int(legit.sum()),
        "n_scam": int(scam.sum()),
        "legit_fpr": rate_on(s, legit, tau),
        "scam_recall": rate_on(s, scam, tau),
    }
    if novel is not None:
        nov = np.asarray(novel, dtype=bool)
        out["n_template_novel_legit"] = int((legit & nov).sum())
        out["n_template_novel_scam"] = int((scam & nov).sum())
        out["legit_fpr_template_novel"] = rate_on(s, legit & nov, tau)
        out["scam_recall_template_novel"] = rate_on(s, scam & nov, tau)
    return out


def gate_eval_inbox(rows: Sequence[dict]) -> dict:
    """G1 to G5 over one row per requested seed (sms_retrain's median plus 2-of-3 rule)."""
    if not rows:
        raise ValueError("gate_eval_inbox needs at least one seed row")
    out: dict[str, Any] = {}
    for gate, key, ok in INBOX_GATES:
        out[gate] = base._gate([r.get(key) for r in rows], ok)
    out["arm_pass"] = all(out[gate]["passes"] for gate, _, _ in INBOX_GATES)
    return out


def median_seed(fpr_by_seed: dict[int, float]) -> int:
    """The seed with the median value (lower middle when the count is even); ties by seed."""
    ordered = sorted(fpr_by_seed.items(), key=lambda kv: (kv[1], kv[0]))
    return int(ordered[(len(ordered) - 1) // 2][0])


def donor_fpr_by_seed(probs: pd.DataFrame) -> dict[int, float]:
    """Test-donor legit FPR per seed, from a probs table (rows carry the seed's tau)."""
    t = probs[(probs["set"] == SET_TEST_DONOR) & (probs["label"] == 0)]
    out = {}
    for seed, g in t.groupby("seed"):
        out[int(seed)] = float((g["score"].to_numpy() >= float(g["tau"].iloc[0])).mean())
    return out


def _legit_test_rows(probs: pd.DataFrame, seed: int) -> pd.DataFrame:
    t = probs[(probs["set"] == SET_TEST_DONOR) & (probs["label"] == 0)
              & (probs["seed"] == seed)]
    return t.sort_values("id").reset_index(drop=True)


def paired_donor_fpr_diff(a: pd.DataFrame, b: pd.DataFrame, seed_a: int, seed_b: int,
                          n_boot: int, boot_seed: int = BOOT_SEED) -> dict:
    """FPR(b) - FPR(a) on the test-donor legit rows of two models, bootstrapped over sender groups.
    Both probs tables must hold the same test-donor rows (same ids and group codes)."""
    sa = _legit_test_rows(a, seed_a)
    sb = _legit_test_rows(b, seed_b)
    if list(sa["id"]) != list(sb["id"]) or not np.array_equal(
        sa["group"].to_numpy(), sb["group"].to_numpy()
    ):
        raise ValueError("the two arms' test-donor rows differ (ids or sender groups)")
    if sa.empty:
        raise ValueError("no test-donor legit rows to compare")
    flag_a = (sa["score"].to_numpy() >= float(sa["tau"].iloc[0])).astype(float)
    flag_b = (sb["score"].to_numpy() >= float(sb["tau"].iloc[0])).astype(float)
    _, inv = np.unique(sa["group"].to_numpy().astype(np.int64), return_inverse=True)
    n_groups = int(inv.max()) + 1
    n_g = np.bincount(inv, minlength=n_groups).astype(np.float64)
    d_g = np.bincount(inv, weights=flag_b - flag_a, minlength=n_groups)
    total = float(n_g.sum())
    rng = np.random.default_rng(boot_seed)
    idx = rng.integers(0, n_groups, size=(n_boot, n_groups))
    den = n_g[idx].sum(axis=1)
    diffs = d_g[idx].sum(axis=1)[den > 0] / den[den > 0]
    lo, hi = (float(x) for x in np.percentile(diffs, [2.5, 97.5]))
    return {
        "n_legit": int(total),
        "sender_groups": n_groups,
        "seed_a": int(seed_a),
        "seed_b": int(seed_b),
        "fpr_a": float(flag_a.mean()),
        "fpr_b": float(flag_b.mean()),
        "diff_b_minus_a": float((flag_b.sum() - flag_a.sum()) / total),
        "ci95": [lo, hi],
        "n_boot": int(n_boot),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def probs_frame(seed: int, tau: float, sets: dict[str, dict]) -> pd.DataFrame:
    """Per-row scores of every set for one seed. ``sets`` maps a set name to ids, labels, scores and
    optional extra columns (categories or group codes; never text)."""
    parts = []
    for name, entry in sets.items():
        n = len(entry["scores"])
        df = pd.DataFrame({
            "seed": np.full(n, seed, dtype=np.int64),
            "set": name,
            "id": np.asarray(entry["ids"]).astype(str),
            "label": np.asarray(entry["labels"], dtype=np.float64),
            "score": np.asarray(entry["scores"], dtype=np.float64),
            "tau": float(tau),
        })
        for key, col in entry.get("extra", {}).items():
            df[key] = col
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def write_thresholds(work: Path, thresholds: dict) -> None:
    (work / "thresholds.json").write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n")


def write_outputs(work: Path, payload: dict, probs: pd.DataFrame | None,
                  prep_report: dict) -> None:
    """prep_report.json, probs.parquet (when scored), metrics.json and report.md in ``work``."""
    work.mkdir(parents=True, exist_ok=True)
    (work / "prep_report.json").write_text(json.dumps(prep_report, indent=2, default=str) + "\n")
    if probs is not None:
        probs.to_parquet(work / "probs.parquet", index=False)
    tmp = work / "metrics.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(work / "metrics.json")
    (work / "report.md").write_text(render_report(payload))


def render_report(payload: dict) -> str:
    """Markdown summary of one arm (report.md). Counts and metrics only."""
    lines = [f"# sms_inbox_retrain arm {payload['arm']}", ""]
    if payload.get("status") == STOP_STATUS:
        lines += [f"Status: **{STOP_STATUS}**", "", f"Reason: {payload.get('reason', '')}", ""]
        return "\n".join(lines)
    gates = payload["gates"]
    lines += [
        f"Seeds {payload['seeds']}. Side width {payload['side_width']}. "
        f"Arm pass: **{gates['arm_pass']}**.",
        "",
        "| Gate | Per seed | Median | Seeds passing | Pass |",
        "|---|---|---|---|---|",
    ]
    for gate, _, _ in INBOX_GATES:
        g = gates[gate]
        vals = ", ".join("n/a" if v is None else f"{v:g}" for v in g["values"])
        med = "n/a" if g["median"] is None else f"{g['median']:g}"
        lines.append(f"| {gate} | {vals} | {med} | {g['seeds_pass']}/{g['seeds']} | "
                     f"{g['passes']} |")
    lines += [
        "",
        "| Seed | tau | fallback | GroundLM FP / 1133 | SmishX TP / 226 | Campaign F1 | "
        "Legit val FPR | Test donor legit FPR (Wilson 95 %) | Template-novel legit FPR | "
        "Test donor scam recall | McNemar p (GroundLM neg vs XLM-R) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for seed, block in payload["per_seed"].items():
        if block is None:
            lines.append(f"| {seed} | failed | | | | | | | | | |")
            continue
        d = block["test_donor"]
        fpr = d["legit_fpr"]
        mc = block["groundlm_paired"]["negatives"].get("mcnemar_exact_p", float("nan"))
        lines.append(
            f"| {seed} | {block['tau']:.6f} | {block['fallback_flag']} | "
            f"{block['gate_row']['groundlm_fp']} | {block['gate_row']['smishx_tp']} | "
            f"{block['gate_row']['campaign_f1']:.4f} | {block['legit_val']['rate']:.4f} | "
            f"{fpr['rate']:.4f} (upper {fpr['wilson95'][1]:.4f}) | "
            f"{d['legit_fpr_template_novel']['rate']:.4f} | "
            f"{d['scam_recall']['k']}/{d['scam_recall']['n']} | {mc:.3g} |"
        )
    v0 = payload.get("v0", {})
    if "skipped" in v0:
        lines += ["", f"v0: {v0['skipped']}"]
    elif v0:
        lines += ["", f"v0 (shipped, tau {v0['threshold']:.6f}): test donor legit FPR "
                  f"{v0['test_donor']['legit_fpr']['rate']:.4f}, train donor legit FPR "
                  f"{v0['train_donor']['legit_fpr']['rate']:.4f}."]
    # The paired block describes R2 vs R1 (R arms) or S2 vs S1 (S arms). S0 and S1 have none.
    pa, pb = ("S2", "S1") if payload["arm"] in FILL_ARMS else ("R2", "R1")
    paired = payload.get(paired_key(payload["arm"]), {})
    if paired and payload["arm"] not in ("S0", "S1"):
        if "skipped" in paired:
            lines += ["", f"{pa} vs {pb}: {paired['skipped']}"]
        else:
            lo, hi = paired["ci95"]
            minus = f"{pa} minus {pb}"
            beats = f"{pa} beats {pb}"
            beats_key = f"{pa.lower()}_beats_{pb.lower()}"
            lines += ["", f"{minus} test-donor legit FPR (seed medians {paired['seed_b']} "
                      f"and {paired['seed_a']}): {paired['diff_b_minus_a']:+.4f}, 95 % CI "
                      f"[{lo:+.4f}, {hi:+.4f}] over {paired['sender_groups']} sender groups. "
                      f"{beats}: {paired[beats_key]}."]
    lines += fill_lines(payload)
    lines.append("")
    return "\n".join(lines)


def fill_lines(payload: dict) -> list[str]:
    """Markdown lines for the fill-mode fields. Empty for R arms."""
    if not payload.get("fill_surrogates"):
        return []
    lines = ["", f"Surrogate-filled text: recipe {payload['recipe']}, surrogate seed "
             f"{payload['surrogate_seed']}."]
    v0 = payload.get("v0_campaign_filled") or {}
    if "skipped" in v0:
        lines.append(f"v0 campaign F1 on the filled test split: {v0['skipped']}")
    elif v0:
        lines.append(f"v0 (shipped, tau {v0['threshold']:.6f}) campaign F1 on the filled test "
                     f"split: {v0['f1']:.4f} (n {v0['n_test']}). Reported, not gated.")
    cp = payload.get("test_donor_scam_recall_cp95") or {}
    for seed, e in (cp.get("seeds") or {}).items():
        if e is None:
            lines.append(f"Test-donor scam recall, seed {seed}: failed seed, no value.")
        else:
            lines.append(f"Test-donor scam recall, seed {seed}: {e['k']}/{e['n']}, exact 95 % "
                         f"interval {e['clopper_pearson95']}.")
    v0cp = cp.get("v0") or {}
    if "skipped" in v0cp:
        lines.append(f"Test-donor scam recall, v0: {v0cp['skipped']}")
    elif v0cp:
        lines.append(f"Test-donor scam recall, v0: {v0cp['k']}/{v0cp['n']}, exact 95 % "
                     f"interval {v0cp['clopper_pearson95']}.")
    if "stop_rule_2" in payload:
        lines.append(f"Stop rule 2 (S0 fails G2 on all seeds): "
                     f"{payload['stop_rule_2']['g2_failed_all_seeds']}.")
    probe = payload.get("confound_probe") or {}
    if "skipped" in probe:
        lines.append(f"Source-identity probe: {probe['skipped']}")
    for version in ("scrubbed", "filled"):
        res = probe.get(version) or {}
        if "skipped" in res:
            lines.append(f"Source-identity probe, {version}: {res['skipped']}")
        elif res:
            lines.append(f"Source-identity probe, {version} text: accuracy {res['accuracy']:.4f} "
                         f"on {res['n_test']} held-out rows, majority baseline "
                         f"{res['majority_baseline']:.4f}.")
    return lines


# ---- surrogate fill and the S arms (plan sections 3 to 7; pure, unit-tested) ----


def tag_counts(texts: Sequence[str]) -> dict[str, int]:
    """Rows that hold a <TAG> placeholder before filling. Counts only."""
    k = sum(1 for t in texts if S.TAG_RE.search(t))
    return {"rows": int(len(texts)), "with_tag": int(k)}


def tag_counts_by(keys: Sequence[str], texts: Sequence[str]) -> dict[str, dict[str, int]]:
    """tag_counts per key (the sms source). Counts only."""
    groups: dict[str, list[str]] = {}
    for key, text in zip(keys, texts, strict=True):
        groups.setdefault(str(key), []).append(text)
    return {k: tag_counts(v) for k, v in sorted(groups.items())}


def fill_sms_frame(sms: pd.DataFrame) -> tuple[pd.DataFrame, dict, list[str]]:
    """sms.parquet with every text replaced by its surrogate fill, keyed by ``id`` (all splits, all
    rows). Returns the frame, the tag counts by source before filling, and the scrubbed texts. The
    scrubbed texts stay in memory and are used only by the confound probe."""
    scrubbed = base.str_texts(sms["text"])
    stats = tag_counts_by(sms["source"].astype(str).tolist(), scrubbed)
    out = sms.copy()
    out["text"] = S.fill_many(sms["id"].astype(str).tolist(), scrubbed)
    return out, stats, scrubbed


def fill_inbox_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """An inbox export with every row's text replaced by its fill, keyed by ``id``. Runs before
    label_inbox, so the rows that label_inbox drops are filled and checked too."""
    if not {"id", "text"} <= set(raw.columns):
        raise ValueError("inbox file lacks the id or text column")
    texts = [t if isinstance(t, str) else "" for t in raw["text"]]
    stats = tag_counts(texts)
    out = raw.copy()
    out["text"] = S.fill_many([str(i) for i in raw["id"]], texts)
    return out, stats


def assert_filled(texts: Sequence[str], name: str) -> None:
    """Plan rule 5: no filled text may still match <[A-Z_]+>. The error names the set and gives a
    count, never a text."""
    k = sum(1 for t in texts if S.TAG_RE.search(t))
    if k:
        raise ValueError(f"{name}: {k} row(s) still hold a <TAG> placeholder after filling")


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float] | None:
    """Exact two-sided Clopper-Pearson interval for k successes in n trials (None when n is 0).
    Bounds solve the binomial tail equations by bisection, so no SciPy is needed."""
    if n <= 0:
        return None
    if not 0 <= k <= n:
        raise ValueError("clopper_pearson needs 0 <= k <= n")

    def cdf(j: int, p: float) -> float:  # P(X <= j)
        return sum(math.comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(j + 1))

    def solve(fn: Callable[[float], float], target: float, increasing: bool) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if (fn(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: 1.0 - cdf(k - 1, p), alpha / 2, True)
    upper = 1.0 if k == n else solve(lambda p: cdf(k, p), alpha / 2, False)
    return [lower, upper]


def scam_recall_cp95(seed_out: dict, v0: dict) -> dict:
    """Test-donor scam recall per seed and for v0, each with an exact 95 % interval."""

    def entry(rate: dict) -> dict:
        ci = clopper_pearson(int(rate["k"]), int(rate["n"]))
        return {"k": int(rate["k"]), "n": int(rate["n"]),
                "clopper_pearson95": None if ci is None else [round(x, 4) for x in ci]}

    seeds = {str(s): (None if b is None else entry(b["test_donor"]["scam_recall"]))
             for s, b in seed_out.items()}
    v0_entry = ({"skipped": v0["skipped"]} if "skipped" in v0
                else entry(v0["test_donor"]["scam_recall"]))
    return {"seeds": seeds, "v0": v0_entry}


def g2_failed_all_seeds(rows: Sequence[dict]) -> bool:
    """Plan stop rule 2: True when every requested seed was scored and fails G2 (SmishX TP). A seed
    with no value (it failed) is not counted as a G2 failure."""
    ok = next(check for _name, key, check in INBOX_GATES if key == "smishx_tp")
    vals = [r.get("smishx_tp") for r in rows]
    return bool(vals) and all(v is not None and not ok(v) for v in vals)


def fill_result_fields(arm: str, rows: Sequence[dict], seed_out: dict, v0: dict,
                       v0_fill: dict | None, probe: dict | None) -> dict:
    """The fill-mode fields of a scored payload. stop_rule_2 is written for S0 only."""
    out: dict[str, Any] = {
        "v0_campaign_filled": v0_fill,
        "confound_probe": probe,
        "test_donor_scam_recall_cp95": scam_recall_cp95(seed_out, v0),
    }
    if arm == "S0":
        out["stop_rule_2"] = {"g2_failed_all_seeds": g2_failed_all_seeds(rows)}
    return out


def score_v0_campaign(texts: Sequence[str], y_test: np.ndarray, clusters: Any,
                      n_boot: int) -> dict:
    """The shipped v0 ONNX on the filled campaign test split at its shipped threshold. Reported, not
    gated (plan section 7). The file is found by its pinned sha256 under /kaggle/input."""
    path = RP.locate_pinned(V0_ROOT, V0_NAME, V0_SHA)
    scores = score_sms_texts(_onnx_session(path), list(texts))
    table = sms_run_table(np.array([], dtype=np.int64), np.array([], dtype=np.float64),
                          y_test, scores, clusters, threshold=SMS_THRESHOLD, n_boot=n_boot)
    return {"threshold": SMS_THRESHOLD, "n_test": int(len(y_test)),
            "f1": table["shipped_threshold"]["f1"]}


def confound_before_after(sources: Sequence[str], scrubbed: Sequence[str], filled: Sequence[str],
                          rows: Sequence[int]) -> dict:
    """rakshak.datasets.confound.source_identity_accuracy on the campaign training scam rows, once
    with the scrubbed text and once with the filled text, both canonicalised as the sms builder
    does. Source names and accuracies leave the function; no text does."""
    from rakshak.canonicalize import canonicalize
    from rakshak.datasets.confound import source_identity_accuracy

    src = [str(sources[i]) for i in rows]
    out: dict[str, Any] = {"scam_rows": len(src), "sources": sorted(set(src))}
    for version, texts in (("scrubbed", scrubbed), ("filled", filled)):
        if len(set(src)) < 2:
            out[version] = {"skipped": "fewer than two sources among the training scam rows"}
            continue
        kept = pd.DataFrame({"source": src,
                             "text_canon": [canonicalize(texts[i]).normalized for i in rows]})
        out[version] = source_identity_accuracy(kept)
    return out


# ---- kernel-side helpers (torch-free until run) ----


def find_one(name: str, roots: Sequence[Path]) -> Path:
    """The single copy of ``name`` under ``roots``. Raises when none or several are found."""
    hits = []
    for root in roots:
        if not root.exists():
            continue
        for dirpath, _dirs, files in os.walk(root):
            if name in files:
                hits.append(Path(dirpath) / name)
    if len(hits) != 1:
        raise FileNotFoundError(f"{name}: expected one copy under the inputs, found {len(hits)}")
    return hits[0]


def find_probs(substring: str, root: Path) -> Path:
    """The single probs.parquet whose path contains ``substring`` (the attached R1 kernel)."""
    hits = [Path(d) / "probs.parquet" for d, _dirs, files in os.walk(root)
            if "probs.parquet" in files and substring in str(Path(d))]
    if len(hits) != 1:
        raise FileNotFoundError(f"R1 probs matching {substring!r}: found {len(hits)}")
    return hits[0]


def score_v0_donors(roots: Sequence[Path], train_df: pd.DataFrame,
                    test_df: pd.DataFrame, novel: np.ndarray) -> dict:
    """The shipped v0 ONNX at its shipped threshold on both donor sets. Skipped if not attached."""
    try:
        path = locate_artifact(list(roots), "best_student.onnx", V0_SHA)
    except FileNotFoundError as exc:
        return {"skipped": f"skipped: v0 ONNX not attached ({exc})"}
    session = _onnx_session(path)
    tr_labels = train_df["label"].to_numpy().astype(np.int64)
    te_labels = test_df["label"].to_numpy().astype(np.int64)
    tr_scores = score_sms_texts(session, list(train_df["text"]))
    te_scores = score_sms_texts(session, list(test_df["text"]))
    return {
        "threshold": SMS_THRESHOLD,
        "train_donor": donor_rates(tr_labels, tr_scores, SMS_THRESHOLD),
        "test_donor": donor_rates(te_labels, te_scores, SMS_THRESHOLD, novel),
        "g5_pass": bool(donor_rates(te_labels, te_scores, SMS_THRESHOLD)["legit_fpr"]["rate"]
                        <= G5_MAX_FPR),
    }


# ---- the job ----


def redacted_error(exc: BaseException) -> str:
    return f"{type(exc).__name__} (message redacted)"


def redacted_frames(exc: BaseException) -> list[str]:
    """file:line in function for each frame. Source text and the message are left out."""
    return [f"{f.filename}:{f.lineno} in {f.name}" for f in traceback.extract_tb(exc.__traceback__)]


class RedactedUnits(Units):
    """Units whose failures keep the type and frames, never the exception message."""

    def run(self, unit: dict, fn: Callable[..., Any], *args: Any) -> Any:
        label = " ".join(f"{k}={v}" for k, v in unit.items())
        try:
            value = fn(*args)
        except Exception as exc:
            self.failed += 1
            error = redacted_error(exc)
            self.ctx.results["failures"].append({**unit, "error": error,
                                                 "frames": redacted_frames(exc)})
            self.ctx.log(f"FAILED {label}: {error}")
            for line in redacted_frames(exc):
                self.ctx.log(f"  at {line}")
            self.ctx.save()
            return None
        self.ok += 1
        self.ctx.save()
        return value


def run(ctx) -> bool:
    """Runs _run; any exception leaves here as RedactedError (type and frames logged)."""
    failure = None
    try:
        return _run(ctx)
    except Exception as exc:
        failure = exc
        for line in redacted_frames(exc):
            ctx.log(f"  at {line}")
        ctx.log(f"ERROR {redacted_error(exc)}")
    # Raised outside the except block so the original message is not chained onto the error.
    raise RedactedError(redacted_error(failure)) from None


class RedactedError(RuntimeError):
    """The job failed; the cause is in the log as type and frames, not as a message."""


def _run(ctx) -> bool:
    args = parse_args(ctx.argv)
    arm = args.arm
    recipe = recipe_of(arm)
    fill = bool(args.fill_surrogates)
    extra = SIDE_EXTRA[recipe]
    fill_stats: dict[str, Any] = {}
    # Imported here: torch is needed only on the kernel, not for the pure helpers.
    from rakshak.models.student import (
        ModelSpec,
        TrainConfig,
        export_onnx,
        onnx_parity,
        param_count,
        pick_device,
        predict_scores,
        train_student,
    )

    if N_RULES != 16:
        raise RuntimeError(f"expected 16 rule side features, found {N_RULES}")
    seed_everything(DATA_SEED)
    device = pick_device("auto")
    env = env_report()
    env["device"] = str(device)
    ctx.log(f"env: {env}")
    timings = ctx.results.setdefault("timings", {})
    work = ctx.work

    # 1. campaign and external sets, as sms_retrain loads them
    root = D.resolve_data_root(None)
    ctx.log(f"data root: {root}")
    sms = D.load_sms(root)
    scrubbed_texts: list[str] = []
    if fill:
        # Plan rule 4: every row of every split, in memory, before anything else reads the text.
        sms, fill_stats["sms_by_source_before_fill"], scrubbed_texts = fill_sms_frame(sms)
        assert_filled(base.str_texts(sms["text"]), "sms.parquet")
        ctx.log(f"sms.parquet: {len(sms)} texts replaced by surrogate fills")
    labels = sms["label_binary"].to_numpy().astype(np.int64)
    sms_texts = base.str_texts(sms["text"])
    sms_ids = sms["id"].astype(str).to_numpy()
    mask = D.eval_mask(sms)
    runs = D.load_runs(root, sms, patterns=["campaign", *base.SHIFT_PREFIXES], labels=labels,
                       mask=mask)
    campaign_runs = [r for r in runs if r.name.split(":")[0] == "campaign"]
    if not campaign_runs:
        raise RuntimeError("no campaign split run found")
    run_ = campaign_runs[0]
    shift_runs = [r for r in runs if r.name.split(":")[0] in base.SHIFT_PREFIXES]
    y_cal = labels[run_.cal]
    y_test = labels[run_.test]
    clusters_test = D.cluster_values(sms, run_.test, base.SMS_CLUSTER_COLUMN)
    shift_clusters = {r.name: D.cluster_values(sms, r.test, base.SMS_CLUSTER_COLUMN)
                      for r in shift_runs}

    gl = pd.read_parquet(root / "external" / "external_groundlm.parquet")
    gl_avail = gl["text_available"].fillna(False).astype(bool).to_numpy()
    gl_ok = gl["eval_ok"].fillna(False).astype(bool).to_numpy()
    gl_pol = gl["policy_negative"].fillna(False).astype(bool).to_numpy()
    gl_label = gl["label_binary"].to_numpy().astype(np.int64)
    gl_avail_idx = np.flatnonzero(gl_avail)
    neg_eval = gl_pol & gl_avail & gl_ok
    pos_eval = (gl_label == 1) & gl_avail & gl_ok
    neg_idx = np.flatnonzero(neg_eval)
    pos_idx = np.flatnonzero(pos_eval)
    if len(neg_idx) != base.GROUNDLM_SCOREABLE:
        raise RuntimeError(f"expected {base.GROUNDLM_SCOREABLE} scoreable negatives, "
                           f"got {len(neg_idx)}")
    gl_texts = base.str_texts(gl["text"].to_numpy()[gl_avail_idx])
    gl_cat = gl["category"].astype(str).to_numpy()

    sx = pd.read_parquet(root / "external" / "external_smishx.parquet")
    sx_avail = sx["text_available"].fillna(False).astype(bool).to_numpy()
    sx_avail_idx = np.flatnonzero(sx_avail)
    sx_texts = base.str_texts(sx["text"].to_numpy()[sx_avail_idx])
    sx_label = sx["label_binary"].to_numpy().astype(np.int64)
    sx_fine = sx["label_fine"].astype(str).to_numpy()

    xl_path = root / "external" / "indian_legit" / "groundlm_xlmr_probs.parquet"
    xl = pd.read_parquet(xl_path) if xl_path.is_file() else None

    # 2. inbox donors
    roots = [Path("/kaggle/input")]
    train_raw = pd.read_parquet(find_one(TRAIN_FILE, roots))
    test_raw = pd.read_parquet(find_one(TEST_FILE, roots))
    if fill:
        train_raw, fill_stats["train_donor_before_fill"] = fill_inbox_frame(train_raw)
        test_raw, fill_stats["test_donor_before_fill"] = fill_inbox_frame(test_raw)
        assert_filled(list(train_raw["text"]), "train donor")
        assert_filled(list(test_raw["text"]), "test donor")
        ctx.log("inbox train and test donor rows: texts replaced by surrogate fills")
    train_df, train_counts = label_inbox(train_raw)
    test_df, test_counts = label_inbox(test_raw)
    del train_raw, test_raw
    ctx.log(f"train donor counts: {train_counts}")
    ctx.log(f"test donor counts: {test_counts}")

    # 3. pools. The test-only texts are removed from every train and validation row.
    test_only = [sms_texts[i] for i in run_.test]
    for r in shift_runs:
        test_only += [sms_texts[i] for i in r.test]
    test_only += gl_texts + sx_texts
    test_only += [t for t in test_df["text"] if t]
    test_only = [t for t in test_only if t]
    with timed("prep_s", timings):
        pools = build_inbox_pools(train_df, test_only, int((labels[run_.train] == 0).sum()))
    prep_report: dict[str, Any] = {
        "arm": arm,
        "data_seed": DATA_SEED,
        "test_only_texts": len(test_only),
        "train_donor": train_counts,
        "test_donor": test_counts,
        **pools["report"],
    }
    if fill:
        prep_report["fill"] = fill_stats
    reason = stop_reason(pools["report"])
    if reason is not None:
        payload = {"job": "sms_inbox_retrain", "arm": arm, "status": STOP_STATUS,
                   "reason": reason, "counts": prep_report, **fill_fields(arm, fill)}
        write_outputs(work, payload, None, prep_report)
        ctx.results["summary"] = f"sms_inbox_retrain arm {arm}: {STOP_STATUS}: {reason}"
        ctx.log(STOP_STATUS + ": " + reason)
        return False
    ctx.log(f"prep: train_extra {pools['report']['stages']['train_extra_total']}, "
            f"legit_val {pools['report']['stages']['val_legit']}, "
            f"val_scam {pools['report']['stages']['val_scam']}")

    train_extra = pools["train_extra"]
    val_legit = pools["val_legit"]
    val_scam = pools["val_scam"]
    used = train_extra if recipe != "R0" else train_extra.iloc[0:0]
    y_used = used["label"].to_numpy().astype(np.int64)
    test_texts = list(test_df["text"])
    y_donor = test_df["label"].to_numpy().astype(np.int64)
    novel = novel_mask(test_texts, list(train_extra["text"]))
    prep_report["test_donor_template_novel"] = {
        "novel": int(novel.sum()), "near_duplicate": int((~novel).sum())}
    test_codes, _ = sender_codes(test_df["sender"].tolist())

    # 4. featurize. R2 appends the sender features; R0 and R1 keep the 16 rule features.
    def side(fz, senders=None):
        if not extra:
            return fz
        if senders is None:
            add = np.zeros((len(fz), extra), dtype=np.float32)
        else:
            add = sender_matrix(list(senders))
        return with_side_extra(fz, add)

    with timed("featurize_s", timings):
        fz_sms = featurize(sms_texts, buckets=None)
        fz_base = fz_sms.take(run_.train)
        if len(used):
            fz_used = featurize(list(used["text"]), buckets=None)
            train_feats = base.concat([fz_base, fz_used])
            train_side_senders = [None] * len(fz_base) + list(used["sender"])
            if extra:
                train_feats = with_side_extra(
                    train_feats, sender_matrix(train_side_senders))
            y_train = np.concatenate([labels[run_.train], y_used])
        else:
            train_feats = side(fz_base)
            y_train = labels[run_.train]
        ctx.log(f"train rows {len(y_train)} (base {len(fz_base)} + inbox {len(used)}), "
                f"positives {int(y_train.sum())}, side width {side_width(arm)}")
        feats: dict[str, Any] = {
            SET_LEGIT_VAL: side(featurize(list(val_legit["text"]), buckets=None),
                                val_legit["sender"]),
            SET_CAL: side(fz_sms.take(run_.cal)),
            SET_CAMPAIGN_TEST: side(fz_sms.take(run_.test)),
            SET_GROUNDLM: side(featurize(gl_texts, buckets=None)),
            SET_SMISHX: side(featurize(sx_texts, buckets=None)),
            SET_VAL_SCAM: side(featurize(list(val_scam["text"]), buckets=None),
                               val_scam["sender"]),
            SET_TEST_DONOR: side(featurize(test_texts, buckets=None), test_df["sender"]),
        }
        for r in shift_runs:
            feats[r.name] = side(fz_sms.take(r.test))
        gl_rules_av = rules_any(gl_texts)
        sx_rules_av = rules_any(sx_texts)
    legit_texts = list(val_legit["text"])
    scam_texts = list(val_scam["text"])
    post_sets = [SET_CAMPAIGN_TEST, *[r.name for r in shift_runs], SET_GROUNDLM, SET_SMISHX,
                 SET_VAL_SCAM, SET_TEST_DONOR]

    # 5. reference decision and the v0 comparator (before any seed is trained)
    if xl is not None:
        ref_gl, join_stats = base.xlmr_reference(gl, xl, neg_eval | pos_eval)
    else:
        ref_gl, join_stats = None, {"skipped": "skipped: not available"}
    ctx.log(f"XLM-R join: {join_stats}")
    u = RedactedUnits(ctx)
    v0 = u.run({"stage": "v0_donors"}, score_v0_donors, roots, train_df, test_df, novel) or {
        "skipped": "skipped: v0 scoring failed; see failures"}
    ctx.log(f"v0: {'skipped' if 'skipped' in v0 else 'scored'}")
    v0_fill: dict | None = None
    probe: dict | None = None
    if fill:
        v0_fill = u.run({"stage": "v0_campaign_filled"}, score_v0_campaign,
                        [sms_texts[i] for i in run_.test], y_test, clusters_test, args.n_boot)
        v0_fill = v0_fill or {"skipped": "skipped: v0 campaign scoring failed; see failures"}
        scam_rows = np.asarray(run_.train)[labels[run_.train] == 1]
        probe = u.run({"stage": "confound_probe"}, confound_before_after,
                      sms["source"].astype(str).tolist(), scrubbed_texts, sms_texts,
                      scam_rows.tolist())
        probe = probe or {"skipped": "skipped: source-identity probe failed; see failures"}

    spec = ModelSpec("dual", "base", "dual", True, side_extra=extra)
    cfg_common = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        adv_p=base.TRAIN_ADV_P,
        kd_alpha=base.KD_ALPHA,
        kd_temp=base.KD_TEMP,
    )
    thresholds: dict[str, Any] = {"arm": arm, "rule": "rakshak.data.retrain_prep.threshold_rule",
                                  "seeds": {}}
    seed_out: dict[int, Any] = {}
    frames: list[pd.DataFrame] = []

    def run_seed(seed: int) -> dict:
        seed_everything(seed)
        state: dict[str, Any] = {}

        def fit():
            cfg = TrainConfig(seed=seed, **cfg_common)
            with timed(f"train_seed{seed}_s", timings):
                model, info = train_student(
                    spec, train_feats, y_train, feats[SET_CAL], y_cal, cfg,
                    device=device, teacher=None, log=ctx.log,
                )
            state["info"] = info
            return model.to(device)

        def score(model, name: str) -> np.ndarray:
            return predict_scores(model, spec, feats[name], device)

        def on_threshold(tau: float, fallback: bool) -> None:
            thresholds["seeds"][str(seed)] = {
                "tau": tau,
                "fallback_flag": fallback,
                "n_legit_val": len(legit_texts),
                "n_scam_val": len(scam_texts),
                "n_campaign_cal": int(len(y_cal)),
            }
            write_thresholds(work, thresholds)
            ctx.log(f"seed {seed}: tau {tau:.6f} fallback {fallback} written before any test set")

        pipe = base.run_seed_pipeline(
            fit=fit,
            score=score,
            rule=RP.threshold_rule,
            y_cal=y_cal,
            post_sets=post_sets,
            on_threshold=on_threshold,
        )
        tau = pipe["tau"]
        model = pipe["model"]
        sc = pipe["scores"]
        ctx.log(f"seed {seed}: events {pipe['events']}")

        gl_full = base.expand(len(gl), gl_avail_idx, sc[SET_GROUNDLM], -1.0)
        gl_rules = base.expand(len(gl), gl_avail_idx, gl_rules_av, False)
        sx_full = base.expand(len(sx), sx_avail_idx, sc[SET_SMISHX], -1.0)
        sx_rules = base.expand(len(sx), sx_avail_idx, sx_rules_av, False)

        campaign = sms_run_table(y_cal, sc[SET_CAL], y_test, sc[SET_CAMPAIGN_TEST],
                                 clusters_test, threshold=tau, n_boot=args.n_boot)
        shifts = {}
        for r in shift_runs:
            shifts[r.name] = sms_run_table(
                np.array([], dtype=np.int64), np.array([], dtype=np.float64), labels[r.test],
                sc[r.name], shift_clusters[r.name], threshold=tau, n_boot=args.n_boot)
        gl_block = groundlm_table(gl, gl_full, gl_rules, tau)
        sx_block = smishx_table(sx, sx_full, sx_rules, tau)
        if ref_gl is not None:
            ours_neg = gl_full[neg_idx] >= tau
            neg_paired = base.paired_table(ours_neg, ref_gl[neg_idx])
            neg_paired["buckets"] = base.bucket_counts(gl_cat[neg_idx], ours_neg,
                                                       ref_gl[neg_idx])
            pos_paired = base.paired_table(gl_full[pos_idx] >= tau, ref_gl[pos_idx])
        else:
            neg_paired = pos_paired = {"skipped": "skipped: not available"}

        donor = donor_rates(y_donor, sc[SET_TEST_DONOR], tau, novel)
        legit_val = rate_on(sc[SET_LEGIT_VAL], np.ones(len(legit_texts), bool), tau)
        val_scam_recall = rate_on(sc[SET_VAL_SCAM], np.ones(len(scam_texts), bool), tau)
        block: dict[str, Any] = {
            "seed": seed,
            "tau": tau,
            "fallback_flag": pipe["fallback"],
            "side_width": int(model.n_side),
            "train": state.get("info"),
            "campaign": campaign,
            "shift": shifts,
            "groundlm": gl_block,
            "groundlm_paired": {"negatives": neg_paired, "positives": pos_paired},
            "smishx": sx_block,
            "legit_val": legit_val,
            "val_scam_recall": val_scam_recall,
            "test_donor": donor,
            "join": join_stats,
            "params": param_count(model.cpu()),
            "sets_scored": sorted(sc),
        }

        onnx_info = None
        if not args.no_export:
            with timed(f"export_seed{seed}_s", timings):
                path = work / f"best_student_seed{seed}.onnx"
                exported = export_onnx(model.cpu(), path)
                parity = onnx_parity(model.cpu(), path, feats[SET_CAMPAIGN_TEST])
                exported["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                onnx_info = {**exported, "parity": parity}
            ctx.log(f"seed {seed}: exported {exported['bytes']} bytes, parity {parity}")
        block["onnx"] = onnx_info
        block["gate_row"] = {
            "seed": seed,
            "groundlm_fp": gl_block["student_at_threshold"]["k"],
            "smishx_tp": sx_block["smishing_recall"]["student_at_threshold"]["k"],
            "campaign_f1": campaign["shipped_threshold"]["f1"],
            "onnx_bytes": onnx_info["bytes"] if onnx_info else None,
            "donor_fpr": donor["legit_fpr"]["rate"],
        }

        entries: dict[str, dict] = {
            SET_LEGIT_VAL: {"ids": [f"legit_val_{i}" for i in range(len(legit_texts))],
                            "labels": np.zeros(len(legit_texts)), "scores": sc[SET_LEGIT_VAL]},
            SET_CAL: {"ids": sms_ids[run_.cal], "labels": y_cal, "scores": sc[SET_CAL]},
            SET_CAMPAIGN_TEST: {"ids": sms_ids[run_.test], "labels": y_test,
                                "scores": sc[SET_CAMPAIGN_TEST]},
            SET_GROUNDLM: {"ids": gl["id"].astype(str).to_numpy()[gl_avail_idx],
                           "labels": gl_label[gl_avail_idx], "scores": sc[SET_GROUNDLM],
                           "extra": {"category": gl_cat[gl_avail_idx]}},
            SET_SMISHX: {"ids": [f"smishx_{i}" for i in range(len(sx_avail_idx))],
                         "labels": sx_label[sx_avail_idx], "scores": sc[SET_SMISHX],
                         "extra": {"label_fine": sx_fine[sx_avail_idx]}},
            SET_VAL_SCAM: {"ids": [f"val_scam_{i}" for i in range(len(scam_texts))],
                           "labels": np.ones(len(scam_texts)), "scores": sc[SET_VAL_SCAM]},
            SET_TEST_DONOR: {"ids": [f"test_donor_{i}" for i in range(len(test_texts))],
                             "labels": y_donor, "scores": sc[SET_TEST_DONOR],
                             "extra": {"group": test_codes, "novel": novel}},
        }
        for r in shift_runs:
            entries[r.name] = {"ids": sms_ids[r.test], "labels": labels[r.test],
                               "scores": sc[r.name]}
        frames.append(probs_frame(seed, tau, entries))
        return block

    for seed in args.seeds:
        seed_out[seed] = u.run({"seed": seed}, run_seed, seed)

    rows = [(seed_out.get(s) or {"gate_row": {"seed": s}})["gate_row"] for s in args.seeds]
    gates = gate_eval_inbox(rows)
    probs = pd.concat(frames, ignore_index=True) if frames else None

    paired: dict[str, Any]
    prev = PAIRED_PREV.get(arm)
    if prev is None:
        paired = {"skipped": ("only R2 is compared with R1" if arm in ARMS
                              else "only S2 is compared with S1")}
    elif not args.r1_probs_dir:
        paired = {"skipped": f"not attached: pass --r1-probs-dir with the {prev} kernel output"}
    elif probs is None:
        paired = {"skipped": "skipped: no R2 seeds scored"}
    else:
        try:
            r1 = pd.read_parquet(find_probs(args.r1_probs_dir, Path("/kaggle/input")))
            seed_r1 = median_seed(donor_fpr_by_seed(r1))
            seed_r2 = median_seed(donor_fpr_by_seed(probs))
            paired = paired_donor_fpr_diff(r1, probs, seed_r1, seed_r2, args.n_boot)
            paired[f"{arm.lower()}_beats_{prev.lower()}"] = bool(paired["ci95"][1] < 0)
            paired["sender_groups"] = int(paired["sender_groups"])
        except (FileNotFoundError, ValueError) as exc:
            paired = {"skipped": f"skipped: {type(exc).__name__}: {exc}"}

    payload = {
        "job": "sms_inbox_retrain",
        "arm": arm,
        "status": "scored",
        "seeds": list(args.seeds),
        "side_width": side_width(arm),
        "data_seed": DATA_SEED,
        "args": args_record(args),
        "env": env,
        "prep": prep_report,
        "gates": gates,
        "per_seed": {str(s): seed_out.get(s) for s in args.seeds},
        "v0": v0,
        paired_key(arm): paired,
        "failures": ctx.results.get("failures", []),
    }
    if "skipped" in v0:
        payload["v0_gate_g5"] = {"skipped": v0["skipped"]}
    else:
        payload["v0_gate_g5"] = {"donor_fpr": v0["test_donor"]["legit_fpr"]["rate"],
                                 "pass": v0["g5_pass"]}
    payload.update(fill_fields(arm, fill))
    if fill:
        payload.update(fill_result_fields(arm, rows, seed_out, v0, v0_fill, probe))
    write_outputs(work, payload, probs, prep_report)

    ctx.results["summary"] = (
        f"sms_inbox_retrain arm {arm}: seeds ok {u.ok}/{len(args.seeds)}, "
        f"arm_pass {gates['arm_pass']}, groundlm fp median {gates['G1_groundlm_fp']['median']}, "
        f"donor legit fpr median {gates['G5_test_donor_legit_fpr']['median']}"
    )
    status = u.status()
    return status if status != "fail" else False
