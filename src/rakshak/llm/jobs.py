"""Job definitions, input selection and seeded stratified sampling.

A job says which messages go to cloud labelling sessions and which instructions they receive.
Nothing here calls a model API. Jobs F, E and G are not runnable (see the error classes).
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rakshak.llm import prompts

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SEED = 13
SMS_COLUMNS = [
    "id",
    "text",
    "source",
    "lang",
    "label_fine",
    "label_binary",
    "campaign_id",
    "dup_of",
    "eval_ok",
    "label_conflict",
]
STRATA = ("source", "lang", "label_fine")
AUDIT_N = 2401
NOISE_N = 2500
ZERO_SHOT_N = 5000

Partitions = Mapping[str, set[str]]
Selector = Callable[[pd.DataFrame, Partitions], list[str]]


class JobDisabledError(RuntimeError):
    """Raised for a job that policy forbids. There is no override."""


class JobUnavailableError(RuntimeError):
    """Raised for a job that has no specification in code yet."""


@dataclass(frozen=True)
class JobSpec:
    name: str
    kind: str
    description: str
    labeller: str
    schema_name: str
    instructions: str
    select: Selector


def load_inputs(processed_dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """Load the message table and the campaign-disjoint partition id sets."""
    frame = pd.read_parquet(processed_dir / "sms.parquet", columns=SMS_COLUMNS)
    with (processed_dir / "splits" / "campaign.json").open(encoding="utf-8") as fh:
        raw = json.load(fh)["partitions"]
    return frame, {name: set(ids) for name, ids in raw.items()}


def _kept(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows that are not exact duplicates of another row."""
    return frame[frame["dup_of"].isna()]


def _allocate(sizes: Mapping[str, int], n: int) -> dict[str, int]:
    """Largest-remainder proportional allocation of n draws over strata, capped at stratum size."""
    total = sum(sizes.values())
    quota = {key: n * size / total for key, size in sizes.items()}
    alloc = {key: min(sizes[key], math.floor(quota[key])) for key in sizes}
    remaining = n - sum(alloc.values())
    while remaining > 0:
        candidates = sorted(
            (key for key in sizes if alloc[key] < sizes[key]),
            key=lambda key: (-(quota[key] - alloc[key]), key),
        )
        if not candidates:
            break
        for key in candidates:
            if remaining == 0:
                break
            alloc[key] += 1
            remaining -= 1
    return alloc


def _round_robin(by_campaign: dict[str, list[str]], k: int, rng: random.Random) -> list[str]:
    """Take k ids, one per campaign per round, so draws spread across campaigns first."""
    keys = sorted(by_campaign)
    rng.shuffle(keys)
    queues = []
    for key in keys:
        items = list(by_campaign[key])
        rng.shuffle(items)
        queues.append(items)
    out: list[str] = []
    while len(out) < k and any(queues):
        for queue in queues:
            if queue and len(out) < k:
                out.append(queue.pop())
    return out


def stratified_sample(
    frame: pd.DataFrame, n: int, seed: int = SEED, group: str = "campaign_id"
) -> list[str]:
    """Deterministic stratified sample of n ids over source x lang x label_fine.

    Within each stratum the draw is round-robin over campaigns, so a stratum with many campaigns
    is not dominated by one campaign. Same frame, n and seed always give the same sorted list.
    """
    if n <= 0 or frame.empty:
        return []
    df = frame[["id", *STRATA, group]].copy()
    df["id"] = df["id"].astype(str)
    if n >= len(df):
        return sorted(df["id"])
    df["_stratum"] = df[list(STRATA)].astype(str).agg("|".join, axis=1)
    df["_campaign"] = df[group].where(df[group].notna(), "id:" + df["id"])
    sizes = {str(key): int(size) for key, size in df.groupby("_stratum").size().items()}
    alloc = _allocate(sizes, n)
    chosen: list[str] = []
    for stratum in sorted(sizes):
        k = alloc[stratum]
        if k == 0:
            continue
        sub = df[df["_stratum"] == stratum]
        by_campaign: dict[str, list[str]] = {}
        for campaign, rid in sorted(zip(sub["_campaign"].astype(str), sub["id"], strict=True)):
            by_campaign.setdefault(campaign, []).append(rid)
        rng = random.Random(f"{seed}|{stratum}")
        chosen.extend(_round_robin(by_campaign, k, rng))
    return sorted(chosen)


def select_relabel(frame: pd.DataFrame, parts: Partitions) -> list[str]:
    """Job A: every kept message in train or cal. The test partition never goes to a labeller."""
    sub = _kept(frame)
    sub = sub[sub["id"].isin(parts["train"] | parts["cal"])]
    return sorted(sub["id"].astype(str))


def select_audit(frame: pd.DataFrame, parts: Partitions) -> list[str]:
    """Job B: stratified audit sample from cal, eval-eligible rows only."""
    sub = _kept(frame)
    sub = sub[sub["id"].isin(parts["cal"]) & sub["eval_ok"].astype(bool)]
    return stratified_sample(sub, AUDIT_N)


def select_noise(frame: pd.DataFrame, parts: Partitions) -> list[str]:
    """Job C: train messages whose label disagrees with their source's majority label first.

    The rest is filled with a stratified sample, so the set is never empty when train is not.
    """
    train = _kept(frame)
    train = train[train["id"].isin(parts["train"])]
    majority = train.groupby("source")["label_binary"].mean().ge(0.5).astype(int)
    expected = train["source"].map(majority).astype(int)
    disagree = train[train["label_binary"].astype(int) != expected]
    chosen = stratified_sample(disagree, NOISE_N)
    if len(chosen) < NOISE_N:
        rest = train[~train["id"].isin(chosen)]
        chosen = sorted(chosen + stratified_sample(rest, NOISE_N - len(chosen)))
    return chosen


def select_zero_shot(frame: pd.DataFrame, parts: Partitions) -> list[str]:
    """Job D: stratified sample of eval-eligible test rows, for a zero-shot comparison row."""
    sub = _kept(frame)
    sub = sub[sub["id"].isin(parts["test"]) & sub["eval_ok"].astype(bool)]
    return stratified_sample(sub, ZERO_SHOT_N)


JOBS: dict[str, JobSpec] = {
    "A": JobSpec(
        name="A",
        kind="relabel",
        description="Relabel and intent flags for kept train and cal messages",
        labeller="claude-haiku-5-5",
        schema_name="annotate",
        instructions=prompts.RELABEL_INSTRUCTIONS,
        select=select_relabel,
    ),
    "B": JobSpec(
        name="B",
        kind="audit",
        description=f"Blind second-reviewer audit of {AUDIT_N} stratified cal messages",
        labeller="claude-sonnet-5-5",
        schema_name="audit",
        instructions=prompts.AUDIT_INSTRUCTIONS,
        select=select_audit,
    ),
    "C": JobSpec(
        name="C",
        kind="noise_review",
        description=f"Noise review of up to {NOISE_N} train messages, disagreements first",
        labeller="claude-sonnet-5-5",
        schema_name="annotate",
        instructions=prompts.NOISE_INSTRUCTIONS,
        select=select_noise,
    ),
    "D": JobSpec(
        name="D",
        kind="zero_shot",
        description=f"Zero-shot comparison on {ZERO_SHOT_N} stratified test messages (optional)",
        labeller="claude-sonnet-5-5",
        schema_name="zero_shot",
        instructions=prompts.ZERO_SHOT_INSTRUCTIONS,
        select=select_zero_shot,
    ),
}

_DISABLED = {
    "F": (
        "Job F (Claude-written training data for the Llama explainer) is disabled by policy: "
        "Anthropic terms ambiguity. Claude outputs are used for labels, audit and evaluation "
        "only. See docs/BUDGET.md."
    ),
}

_UNAVAILABLE = {
    "E": "Job E (4k synthetic seeds) has no specification in code yet. See docs/BUDGET.md.",
    "G": "Job G (judge set) runs only if credit remains and has no specification yet.",
}


def get_job(name: str) -> JobSpec:
    """Return a runnable job or raise. Job F always raises JobDisabledError."""
    key = name.upper()
    if key in _DISABLED:
        raise JobDisabledError(_DISABLED[key])
    if key in JOBS:
        return JOBS[key]
    if key in _UNAVAILABLE:
        raise JobUnavailableError(_UNAVAILABLE[key])
    raise JobUnavailableError(f"unknown job {name!r}; known jobs: {', '.join(sorted(JOBS))}")
