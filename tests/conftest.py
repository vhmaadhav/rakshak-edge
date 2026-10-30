"""Shared helpers for the rakshak test suite."""

from __future__ import annotations

import json
from pathlib import Path

from rakshak.canonicalize import canonicalize
from rakshak.evidence import EvidencePackage, build_package
from rakshak.rules import run_rules

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def package_for(text: str, state: str = "scam") -> EvidencePackage:
    canon = canonicalize(text)
    return build_package(canon, run_rules(canon), state)
