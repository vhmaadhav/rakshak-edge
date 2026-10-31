"""Ungated Hugging Face sources for the Llama 3.2 1B Instruct Kaggle jobs.

Used by kaggle/jobs/llama_ela_eval.py and kaggle/jobs/llama_export.py. Nothing here imports torch,
transformers or huggingface_hub at import time. The downloader is passed in, so the source
resolution and the HF-to-Meta conversion are unit-tested on the laptop without network.

Pins were read from the Hub on 2026-10-31 (GET https://huggingface.co/api/models/<repo>, field
``sha``). Licence: Llama 3.2 Community License (see docs/EXECUTORCH_PLAN.md, section "Llama 3.2
licence obligations").
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

UNSLOTH_REPO = "unsloth/Llama-3.2-1B-Instruct"
UNSLOTH_REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"
UNSLOTH_ALLOW = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

# name -> (repo, pinned revision, .pte file name inside the repo)
PREBUILT_PTE = {
    "spinquant_int4_eo8": (
        "executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET",
        "911221e0f49d3354cbd4697a9c09feee25cadec7",
        "Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte",
    ),
    "qlora_int4_eo8": (
        "executorch-community/Llama-3.2-1B-Instruct-QLORA_INT4_EO8-ET",
        "9b418bfbb3391d77d8dc6ecaf936d5253ba0783c",
        "Llama-3.2-1B-Instruct-QLORA_INT4_EO8.pte",
    ),
}
PREBUILT_ALLOW = ("*.pte", "tokenizer.json", "params.json", "config.json")

# Meta-format params for Llama 3.2 1B (the same values the Meta release ships).
META_PARAMS_1B = {
    "dim": 2048,
    "n_layers": 16,
    "n_heads": 32,
    "n_kv_heads": 8,
    "vocab_size": 128256,
    "ffn_dim_multiplier": 1.5,
    "multiple_of": 256,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "use_scaled_rope": True,
}

_HF_LAYER_MAP = {
    "self_attn.q_proj.weight": "attention.wq.weight",
    "self_attn.k_proj.weight": "attention.wk.weight",
    "self_attn.v_proj.weight": "attention.wv.weight",
    "self_attn.o_proj.weight": "attention.wo.weight",
    "mlp.gate_proj.weight": "feed_forward.w1.weight",
    "mlp.down_proj.weight": "feed_forward.w2.weight",
    "mlp.up_proj.weight": "feed_forward.w3.weight",
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
}
_HF_TOP_MAP = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
}


def is_pinned_sha(revision: str) -> bool:
    return len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)


def snapshot(
    repo: str,
    revision: str,
    allow_patterns: tuple[str, ...] | None = None,
    snapshot_download: Callable[..., str] | None = None,
) -> Path:
    """Download (or reuse the cache of) ``repo`` at ``revision`` and return its local folder.

    ``snapshot_download`` defaults to ``huggingface_hub.snapshot_download``, imported here so the
    module stays importable without the package. Tests pass a fake.
    """
    if not is_pinned_sha(revision):
        raise ValueError(f"revision must be a full 40-character commit sha, got {revision!r}")
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as hub_snapshot

        snapshot_download = hub_snapshot
    kwargs: dict[str, Any] = {"repo_id": repo, "revision": revision}
    if allow_patterns is not None:
        kwargs["allow_patterns"] = list(allow_patterns)
    return Path(snapshot_download(**kwargs))


def resolve_model_dir(
    local_finder: Callable[[], Path | None],
    snapshot_download: Callable[..., str] | None = None,
    repo: str = UNSLOTH_REPO,
    revision: str = UNSLOTH_REVISION,
) -> tuple[Path, dict[str, Any]]:
    """HF snapshot first; if the download fails, an attached Kaggle model folder.

    ``local_finder`` returns the folder of an attached Kaggle model, or None. Returns
    ``(path, source)`` where ``source`` records repo, revision and where the weights came from,
    so the job can write it to its metrics.
    """
    try:
        path = snapshot(
            repo, revision, allow_patterns=UNSLOTH_ALLOW, snapshot_download=snapshot_download
        )
    except Exception as exc:  # network, gating or cache failure: try the attached model
        reason = f"{type(exc).__name__}: {str(exc)[:300]}"
        local = local_finder()
        if local is None:
            raise RuntimeError(
                f"snapshot_download of {repo}@{revision} failed ({reason}) and no Llama model "
                "folder is attached under /kaggle/input"
            ) from exc
        return Path(local), {
            "source": "kaggle_input",
            "repo": repo,
            "revision": revision,
            "path": str(local),
            "snapshot_error": reason,
        }
    return path, {
        "source": "hf_snapshot",
        "repo": repo,
        "revision": revision,
        "path": str(path),
    }


def _unpermute(w: Any, n_heads: int, dim1: int, dim2: int) -> Any:
    """Inverse of the HF rotary permutation: HF rows -> Meta interleaved rows.

    transformers' convert_llama_weights_to_hf applies ``view(n, d1/n/2, 2, d2).transpose(1, 2)``
    to Meta q/k weights. This is the inverse, view(n, 2, d1/n/2, d2).transpose(1, 2).
    """
    return w.reshape(n_heads, 2, dim1 // n_heads // 2, dim2).swapaxes(1, 2).reshape(dim1, dim2)


def hf_to_meta_state_dict(
    hf_sd: dict[str, Any], params: dict[str, Any], tie_word_embeddings: bool = False
) -> dict[str, Any]:
    """Map a transformers Llama state dict onto Meta's key names and rotary layout.

    ``params`` is the Meta params dict (dim, n_heads, n_kv_heads, n_layers). Unknown keys raise,
    so a new HF layout is not converted silently. ``lm_head.weight`` is mapped when present.

    Llama 3.2 1B sets ``tie_word_embeddings=true`` and its HF checkpoint has no lm_head. Pass
    ``tie_word_embeddings=True`` (the config value) and ``output.weight`` becomes a copy of
    ``tok_embeddings.weight``, the Meta layout that ExecuTorch's llama model expects. A copy,
    not an alias, so the two tensors can be saved and quantised independently.
    """
    dim = params["dim"]
    n_heads = params["n_heads"]
    n_kv = params["n_kv_heads"]
    head_dim = dim // n_heads
    out: dict[str, Any] = {}
    for key, value in hf_sd.items():
        if key in _HF_TOP_MAP:
            out[_HF_TOP_MAP[key]] = value
            continue
        if not key.startswith("model.layers."):
            raise ValueError(f"unexpected key in the HF Llama checkpoint: {key}")
        rest = key[len("model.layers.") :]
        idx, _, suffix = rest.partition(".")
        if suffix not in _HF_LAYER_MAP or not idx.isdigit():
            raise ValueError(f"unexpected key in the HF Llama checkpoint: {key}")
        meta_key = f"layers.{idx}.{_HF_LAYER_MAP[suffix]}"
        if suffix == "self_attn.q_proj.weight":
            value = _unpermute(value, n_heads, dim, dim)
        elif suffix == "self_attn.k_proj.weight":
            value = _unpermute(value, n_kv, n_kv * head_dim, dim)
        out[meta_key] = value
    if "output.weight" not in out and tie_word_embeddings:
        out["output.weight"] = copy.deepcopy(out["tok_embeddings.weight"])
    return out


def check_meta_shapes(meta_sd: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Shape check of the converted state dict against the Meta params. Returns the hidden size.

    Raises ValueError on the first mismatch, so a wrong conversion fails before export.
    """
    dim = params["dim"]
    head_dim = dim // params["n_heads"]
    hidden = meta_hidden_dim(params)
    expected = {
        "tok_embeddings.weight": (params["vocab_size"], dim),
        "output.weight": (params["vocab_size"], dim),
        "norm.weight": (dim,),
        "layers.0.attention.wq.weight": (dim, dim),
        "layers.0.attention.wk.weight": (params["n_kv_heads"] * head_dim, dim),
        "layers.0.attention.wv.weight": (params["n_kv_heads"] * head_dim, dim),
        "layers.0.attention.wo.weight": (dim, dim),
        "layers.0.feed_forward.w1.weight": (hidden, dim),
        "layers.0.feed_forward.w2.weight": (dim, hidden),
        "layers.0.feed_forward.w3.weight": (hidden, dim),
        "layers.0.attention_norm.weight": (dim,),
        "layers.0.ffn_norm.weight": (dim,),
    }
    for key, shape in expected.items():
        if key not in meta_sd:
            raise ValueError(f"converted checkpoint lacks {key}")
        got = tuple(meta_sd[key].shape)
        if got != shape:
            raise ValueError(f"{key} has shape {got}, expected {shape}")
    layers = {k.split(".")[1] for k in meta_sd if k.startswith("layers.")}
    if len(layers) != params["n_layers"]:
        raise ValueError(
            f"converted checkpoint has {len(layers)} layers, expected {params['n_layers']}"
        )
    return {"hidden_dim": hidden, "layers": len(layers)}


def meta_hidden_dim(params: dict[str, Any]) -> int:
    """Feed-forward width from Meta's rule: 2/3 of 4*dim, scaled, rounded up to multiple_of."""
    hidden = 4 * params["dim"]
    hidden = int(2 * hidden / 3)
    mult = params.get("ffn_dim_multiplier")
    if mult is not None:
        hidden = int(mult * hidden)
    multiple = params["multiple_of"]
    return multiple * ((hidden + multiple - 1) // multiple)


def params_from_hf_config(config: dict[str, Any]) -> dict[str, Any]:
    """Meta params from a transformers config.json. Raises when the config is not the 1B shape
    this module was written for (so a different checkpoint is not exported by mistake)."""
    meta = {
        "dim": config["hidden_size"],
        "n_layers": config["num_hidden_layers"],
        "n_heads": config["num_attention_heads"],
        "n_kv_heads": config["num_key_value_heads"],
        "vocab_size": config["vocab_size"],
        "ffn_dim_multiplier": META_PARAMS_1B["ffn_dim_multiplier"],
        "multiple_of": META_PARAMS_1B["multiple_of"],
        "norm_eps": config["rms_norm_eps"],
        "rope_theta": config["rope_theta"],
        "use_scaled_rope": (config.get("rope_scaling") or {}).get("rope_type") == "llama3",
    }
    if config.get("model_type") != "llama":
        raise ValueError(f"expected a llama config, got model_type={config.get('model_type')!r}")
    if meta_hidden_dim(meta) != config["intermediate_size"]:
        raise ValueError(
            f"Meta hidden size {meta_hidden_dim(meta)} != intermediate_size "
            f"{config['intermediate_size']}; the ffn multiplier does not fit this checkpoint"
        )
    if not meta["use_scaled_rope"]:
        raise ValueError("expected llama3 rope scaling in the config")
    return meta
