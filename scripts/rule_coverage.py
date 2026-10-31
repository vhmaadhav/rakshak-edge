#!/usr/bin/env python
"""Measure rule coverage on the real ScamShield split.

The point of this script is to make recall/precision claims about ``src/rakshak/rules.py``
reproducible instead of anecdotal. It samples the real dataset with a fixed seed, slices it into
``fraud`` / ``promotion`` / ``benign``, and reports how often the deterministic rules stay silent.

Slices (from ``docs/RANK_ARM_AUDIT.md``):
  * ``fraud``      ``is_scam == 1`` and intent is NOT "General Spam / Telemarketing".
  * ``promotion``  ``is_scam == 1`` and intent IS "General Spam / Telemarketing".
  * ``benign``     ``is_scam == 0``.

A promotion or a benign SMS that fires no rule is correct behaviour. The actionable number is
the share of *genuine fraud* rows that produce no evidence at all.

Three metrics are reported per slice, because "a rule fired" and "the detector decided" are not
the same thing:

1. ``any_evidence``    - at least one ``Evidence`` item (rule coverage, the original metric).
2. ``>=UNCERTAIN``     - ``package.risk`` is UNCERTAIN, MEDIUM or HIGH. The package is built with
   ``scam_state=None`` so ``ela_eval.proxy_scam_state`` supplies the label-free fusion stand-in.
3. ``>=MEDIUM``        - ``package.risk`` is MEDIUM or HIGH.

``urgency``-only messages are LOW risk after the ``evidence._risk`` change, so metric 1 counts
them but metrics 2 and 3 do not.

Usage
-----
    .venv/bin/python scripts/rule_coverage.py                # default n=3000, seed 13
    .venv/bin/python scripts/rule_coverage.py --n 5000
    .venv/bin/python scripts/rule_coverage.py --json         # stable JSON blob to diff
    .venv/bin/python scripts/rule_coverage.py --compare HEAD:src/rakshak/rules.py ...
    .venv/bin/python scripts/rule_coverage.py --compare-git HEAD

``--compare-git REV`` copies the working tree to a temporary directory, replaces ``src/rakshak``
with the tree as of ``REV`` and runs this same script there with the current venv interpreter.
That gives a true before/after diff without touching the working tree.

Only the standard library plus the repo's own modules are used, so the script adds no
dependency and no network call.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

DATA_DIR = REPO_ROOT / "data" / "raw" / "hf_sidzzz07_scamshield"
SPLITS = ("train", "val", "test")
PROMOTION_INTENT = "General Spam / Telemarketing"
DEFAULT_SEED = 13
DEFAULT_N = 3000

SLICES = ("fraud", "promotion", "benign")


@dataclass
class SliceStats:
    n: int = 0
    empty: int = 0
    fired: Counter[str] = field(default_factory=Counter)
    at_least_uncertain: int = 0
    at_least_medium: int = 0

    @property
    def empty_rate(self) -> float:
        return self.empty / self.n if self.n else 0.0

    @property
    def uncertain_rate(self) -> float:
        return self.at_least_uncertain / self.n if self.n else 0.0

    @property
    def medium_rate(self) -> float:
        return self.at_least_medium / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "no_evidence": self.empty,
            "no_evidence_rate": round(self.empty_rate, 5),
            "ge_uncertain": self.at_least_uncertain,
            "ge_uncertain_rate": round(self.uncertain_rate, 5),
            "ge_medium": self.at_least_medium,
            "ge_medium_rate": round(self.medium_rate, 5),
            "evidence_types": dict(sorted(self.fired.items())),
        }


def load_rows(data_dir: Path | None = None) -> list[dict]:
    data_dir = data_dir or DATA_DIR
    """Every row of the three JSONL splits, in file order."""
    rows: list[dict] = []
    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing dataset file: {path}")
        with path.open(encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def slice_of(row: dict) -> str:
    if int(row.get("is_scam", 0)) != 1:
        return "benign"
    intent = row.get("head2_scam_intent") or ""
    return "promotion" if intent.strip() == PROMOTION_INTENT else "fraud"


def sample_rows(rows: list[dict], n: int, seed: int) -> list[dict]:
    if n >= len(rows):
        return list(rows)
    return random.Random(seed).sample(rows, n)


_RISK_ORDER = {"LOW": 0, "UNCERTAIN": 1, "MEDIUM": 2, "HIGH": 3}


def measure(rows: list[dict], *, examples: int = 0) -> dict[str, object]:
    """Run the rules over every row and aggregate into a stable, diffable structure."""
    from rakshak.canonicalize import canonicalize
    from rakshak.ela_eval import proxy_scam_state
    from rakshak.evidence import build_package
    from rakshak.rules import run_rules

    slices: dict[str, SliceStats] = {name: SliceStats() for name in SLICES}
    intent_stats: dict[str, SliceStats] = defaultdict(SliceStats)
    empty_examples: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        name = slice_of(row)
        text = row.get("text") or ""
        canon = canonicalize(text)
        evidence = run_rules(canon)
        types = {e.type for e in evidence}
        risk = build_package(canon, evidence, proxy_scam_state(evidence)).risk.value
        rank = _RISK_ORDER[risk]
        stats = slices[name]
        stats.n += 1
        stats.fired.update(types)
        if rank >= _RISK_ORDER["UNCERTAIN"]:
            stats.at_least_uncertain += 1
        if rank >= _RISK_ORDER["MEDIUM"]:
            stats.at_least_medium += 1
        if name == "fraud":
            key = row.get("head2_scam_intent") or "(none)"
            is_key = intent_stats[key]
            is_key.n += 1
            if rank >= _RISK_ORDER["UNCERTAIN"]:
                is_key.at_least_uncertain += 1
            if rank >= _RISK_ORDER["MEDIUM"]:
                is_key.at_least_medium += 1
        if not types:
            stats.empty += 1
            if name == "fraud":
                intent_stats[key].empty += 1
                if len(empty_examples[key]) < examples:
                    empty_examples[key].append(text)

    out: dict[str, object] = {
        "n": len(rows),
        "slices": {name: slices[name].as_dict() for name in SLICES},
        "fraud_by_intent": {
            intent: {
                "n": stats.n,
                "no_evidence": stats.empty,
                "no_evidence_rate": round(stats.empty / stats.n, 5) if stats.n else 0.0,
                "ge_uncertain_rate": round(stats.uncertain_rate, 5),
                "ge_medium_rate": round(stats.medium_rate, 5),
            }
            for intent, stats in sorted(
                intent_stats.items(), key=lambda kv: (-kv[1].n, kv[0])
            )
        },
    }
    if examples:
        out["empty_examples"] = {k: v for k, v in sorted(empty_examples.items())}
    return out


def _fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def render(result: dict[str, object], *, title: str = "Rule coverage") -> str:
    slices = result["slices"]  # type: ignore[index]
    lines = [
        f"### {title}",
        "",
        f"sample n = {result['n']}",
        "",
        "| slice | n | no evidence | empty rate | >=UNCERTAIN | rate | >=MEDIUM | rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in SLICES:
        s = slices[name]  # type: ignore[index]
        lines.append(
            f"| {name} | {s['n']} | {s['no_evidence']} | {_fmt_pct(s['no_evidence_rate'])} | "
            f"{s['ge_uncertain']} | {_fmt_pct(s['ge_uncertain_rate'])} | "
            f"{s['ge_medium']} | {_fmt_pct(s['ge_medium_rate'])} |"
        )
    lines += [
        "",
        "| fraud intent | n | no evidence | empty rate | >=UNCERTAIN | >=MEDIUM |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for intent, s in result["fraud_by_intent"].items():  # type: ignore[union-attr]
        lines.append(
            f"| {intent} | {s['n']} | {s['no_evidence']} | {_fmt_pct(s['no_evidence_rate'])} | "
            f"{_fmt_pct(s['ge_uncertain_rate'])} | {_fmt_pct(s['ge_medium_rate'])} |"
        )
    return "\n".join(lines)


def compare(before: dict[str, object], after: dict[str, object]) -> str:
    b_slices, a_slices = before["slices"], after["slices"]  # type: ignore[index]
    lines = [
        "| slice | n | metric | before | after | delta |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    keys = (
        ("no evidence", "no_evidence_rate"),
        (">=UNCERTAIN", "ge_uncertain_rate"),
        (">=MEDIUM", "ge_medium_rate"),
    )
    for name in SLICES:
        b, a = b_slices[name], a_slices[name]
        if b["n"] != a["n"]:
            lines.append(f"| {name} | {b['n']} -> {a['n']} | (different sample) | | | |")
            continue
        for label, key in keys:
            delta = a[key] - b[key]
            lines.append(
                f"| {name} | {b['n']} | {label} | {_fmt_pct(b[key])} | "
                f"{_fmt_pct(a[key])} | {100 * delta:+.2f} pp |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# --compare-git: run this same script against a git revision in a temp tree
# ---------------------------------------------------------------------------


_JUNK = {"__pycache__", ".git", ".venv", ".pytest_cache", ".ruff_cache"}


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    """Skip junk anywhere plus the repo-root ``data/`` (85k-row corpus; passed by --data-dir)."""
    skipped = {n for n in names if n in _JUNK or n.endswith(".pyc")}
    if Path(directory).resolve() == REPO_ROOT:
        skipped.add("data")
    return skipped


def _run_at_rev(rev: str, paths: list[str], argv: list[str]) -> dict[str, object]:
    """Measure the working tree with only ``paths`` reverted to ``rev``.

    Reverting the whole package would also revert unrelated in-flight edits to modules such as
    ``evidence.py``, which would make the before/after diff meaningless. Only the files named on
    ``--compare-path`` (default: ``src/rakshak/rules.py``) are swapped, so every other module,
    including ``evidence.py``, is identical in both runs.
    """
    with tempfile.TemporaryDirectory(prefix="rakshak-cov-") as tmp:
        tmp_path = Path(tmp)
        target = tmp_path / "repo"
        shutil.copytree(REPO_ROOT, target, ignore=_copy_ignore)
        tar_path = tmp_path / "rev.tar"
        with tar_path.open("wb") as fh:
            subprocess.run(
                ["git", "-C", str(REPO_ROOT), "archive", rev, "--", *paths],
                check=True,
                stdout=fh,
            )
        subprocess.run(["tar", "-xf", str(tar_path), "-C", str(target)], check=True)
        proc = subprocess.run(
            [
                sys.executable,
                str(target / "scripts" / "rule_coverage.py"),
                "--data-dir",
                str(DATA_DIR),
                *argv,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise SystemExit(f"compare-git run failed for {rev}:\n{proc.stderr}")
        return json.loads(proc.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="rows sampled (default 3000)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="sampling seed (13)")
    parser.add_argument("--json", action="store_true", help="print only the JSON blob")
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR, help="ScamShield JSONL directory"
    )
    parser.add_argument(
        "--examples", type=int, default=0, help="include N empty-evidence fraud texts per intent"
    )
    parser.add_argument(
        "--compare-git",
        metavar="REV",
        help="also measure REV (e.g. HEAD) with only --compare-path reverted, and diff",
    )
    parser.add_argument(
        "--compare-path",
        action="append",
        metavar="PATH",
        help="file reverted for --compare-git (repeatable; default src/rakshak/rules.py)",
    )
    parser.add_argument(
        "--compare",
        metavar="REV",
        help="compare the working tree against JSON printed by an earlier run (file or -)",
    )
    args = parser.parse_args(argv)

    rows = sample_rows(load_rows(args.data_dir), args.n, args.seed)
    after = measure(rows, examples=args.examples)

    if args.compare_git:
        child_argv = ["--json", "--n", str(args.n), "--seed", str(args.seed)]
        before = _run_at_rev(
            args.compare_git, args.compare_path or ["src/rakshak/rules.py"], child_argv
        )
        if args.json:
            print(json.dumps({"before": before, "after": after}, indent=2, sort_keys=True))
        else:
            print(render(before, title=f"BEFORE ({args.compare_git})"))
            print()
            print(render(after, title="AFTER (working tree)"))
            print()
            print(compare(before, after))
        return 0

    if args.compare:
        raw = sys.stdin.read() if args.compare == "-" else Path(args.compare).read_text("utf-8")
        print(compare(json.loads(raw), after))
        return 0

    if args.json:
        print(json.dumps(after, indent=2, sort_keys=True))
    else:
        print(render(after))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
