"""Pinned source licence and attribution for private Llama derivative artifacts."""

import hashlib
from importlib import resources
from pathlib import Path

from rakshak.llama_source import UNSLOTH_REPO, UNSLOTH_REVISION

LICENSE_REVISION = "8d29d93fa5700a60532e0061a02ffa89d0acd3fc"
LICENSE_URL = (f"https://github.com/meta-llama/llama-models/blob/{LICENSE_REVISION}/"
               "models/llama3_2/LICENSE")
LICENSE_SHA256 = "8cc15535a8a34b41888f644b339a1a9eb428af793a4f5e24df58a3e5b1487d74"
NOTICE = ("Llama 3.2 is licensed under the Llama 3.2 Community License, "
          "Copyright © Meta Platforms, Inc. All Rights Reserved.\n")


def write_model_notice(folder):
    """Package the exact upstream agreement; never synthesise or modify its text."""
    folder = Path(folder)
    content = resources.files("rakshak").joinpath("data/llama3_2_LICENSE.txt").read_bytes()
    if hashlib.sha256(content).hexdigest() != LICENSE_SHA256:
        raise ValueError("pinned Llama licence hash mismatch")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "LICENSE").write_bytes(content)
    (folder / "NOTICE").write_text(NOTICE)
    (folder / "README.md").write_text(
        "---\nlicense: llama3.2\nbase_model: " + UNSLOTH_REPO + "\n---\n\n"
        "# Llama-Rakshak-ELA-QLoRA\n\nBuilt with Llama.\n\n"
        "Private experimental adapter, not an approved deployment artifact. "
        "Read the completed frozen evaluation and safety decision before use.\n\n"
        f"Base: `{UNSLOTH_REPO}@{UNSLOTH_REVISION}`. "
        "Selected on synthetic-dev verifier pass; targets use existing deterministic templates "
        "and pinned Llama paraphrases only. No SMS corpus text is a training target.\n\n"
        f"[Llama 3.2 Community License]({LICENSE_URL}); exact copy in LICENSE, "
        f"SHA256 `{LICENSE_SHA256}`. Attribution in NOTICE. "
        "[Acceptable Use Policy](https://www.llama.com/llama3_2/use-policy).\n"
    )
    return {"license_revision": LICENSE_REVISION, "license_sha256": LICENSE_SHA256,
            "source": LICENSE_URL, "attribution": "Built with Llama"}
