"""Artifact attribution from a hash-pinned upstream text; no model runtime."""

import hashlib

from rakshak.llama_source import UNSLOTH_REVISION
from rakshak.llm.llama_notice import LICENSE_SHA256, NOTICE, write_model_notice


def test_private_model_package_carries_exact_upstream_license_and_attribution(tmp_path):
    result = write_model_notice(tmp_path)
    assert hashlib.sha256((tmp_path / "LICENSE").read_bytes()).hexdigest() == LICENSE_SHA256
    assert (tmp_path / "NOTICE").read_text() == NOTICE
    card = (tmp_path / "README.md").read_text()
    assert "Built with Llama" in card and UNSLOTH_REVISION in card
    assert "not an approved deployment artifact" in card
    assert result["license_sha256"] == LICENSE_SHA256
