"""Model-free tests for rakshak.llama_source: pinned sources, fallback, HF-to-Meta conversion.

No network: the downloader is a fake, or a fake ``huggingface_hub`` module put in sys.modules.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from rakshak import llama_source as ls

CONFIG_1B = {
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 8192,
    "rms_norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "vocab_size": 128256,
    "head_dim": 64,
    "rope_scaling": {"rope_type": "llama3", "factor": 32.0},
    "model_type": "llama",
}

TINY = {
    "dim": 8,
    "n_layers": 1,
    "n_heads": 2,
    "n_kv_heads": 1,
    "vocab_size": 16,
    "ffn_dim_multiplier": 1.5,
    "multiple_of": 4,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "use_scaled_rope": True,
}


def _hf_permute(w, n_heads, dim1, dim2):
    """transformers' Meta -> HF rotary permutation (convert_llama_weights_to_hf)."""
    return w.reshape(n_heads, dim1 // n_heads // 2, 2, dim2).swapaxes(1, 2).reshape(dim1, dim2)


def _fake_downloader(calls, result, error=None):
    def fake(**kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error
        return result

    return fake


def _no_local():
    raise AssertionError("the attached Kaggle model must not be looked up")


def test_pins_are_full_commit_shas():
    assert ls.is_pinned_sha(ls.UNSLOTH_REVISION)
    for _repo, revision, _name in ls.PREBUILT_PTE.values():
        assert ls.is_pinned_sha(revision)
    assert not ls.is_pinned_sha("main")
    assert not ls.is_pinned_sha("5a8abab")


def test_prebuilt_table_names_pte_files_and_repos():
    repos = {repo for repo, _, _ in ls.PREBUILT_PTE.values()}
    assert repos == {
        "executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET",
        "executorch-community/Llama-3.2-1B-Instruct-QLORA_INT4_EO8-ET",
    }
    for _repo, _rev, name in ls.PREBUILT_PTE.values():
        assert name.endswith(".pte") and name.startswith("Llama-3.2-1B-Instruct")


def test_snapshot_rejects_a_branch_name():
    with pytest.raises(ValueError, match="40-character"):
        ls.snapshot("unsloth/Llama-3.2-1B-Instruct", "main", snapshot_download=lambda **k: "x")


def test_resolve_prefers_the_pinned_snapshot(tmp_path):
    calls = []
    fake = _fake_downloader(calls, str(tmp_path))
    path, src = ls.resolve_model_dir(_no_local, snapshot_download=fake)
    assert path == tmp_path
    assert calls == [
        {
            "repo_id": ls.UNSLOTH_REPO,
            "revision": ls.UNSLOTH_REVISION,
            "allow_patterns": list(ls.UNSLOTH_ALLOW),
        }
    ]
    assert src["source"] == "hf_snapshot"
    assert src["repo"] == "unsloth/Llama-3.2-1B-Instruct"
    assert src["revision"] == ls.UNSLOTH_REVISION
    assert "snapshot_error" not in src


def test_resolve_falls_back_to_attached_kaggle_model(tmp_path):
    calls = []
    fake = _fake_downloader(calls, "", error=OSError("network unreachable"))
    local = tmp_path / "kaggle_llama"
    path, src = ls.resolve_model_dir(lambda: local, snapshot_download=fake)
    assert path == local
    assert src["source"] == "kaggle_input"
    assert src["revision"] == ls.UNSLOTH_REVISION
    assert "network unreachable" in src["snapshot_error"]


def test_resolve_raises_when_snapshot_and_attachment_both_fail():
    fake = _fake_downloader([], "", error=OSError("403 gated"))
    with pytest.raises(RuntimeError) as info:
        ls.resolve_model_dir(lambda: None, snapshot_download=fake)
    message = str(info.value)
    assert ls.UNSLOTH_REPO in message and ls.UNSLOTH_REVISION in message
    assert "403 gated" in message


def test_default_downloader_is_huggingface_hub_snapshot_download(tmp_path, monkeypatch):
    calls = []
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = _fake_downloader(calls, str(tmp_path))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    path, src = ls.resolve_model_dir(_no_local)
    assert path == Path(tmp_path)
    assert calls[0]["repo_id"] == ls.UNSLOTH_REPO
    assert calls[0]["revision"] == ls.UNSLOTH_REVISION
    assert src["source"] == "hf_snapshot"


def test_unpermute_inverts_the_transformers_rotary_permutation():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 8))
    meta = ls._unpermute(w, 2, 8, 8)
    assert np.array_equal(_hf_permute(meta, 2, 8, 8), w)
    assert np.array_equal(ls._unpermute(_hf_permute(w, 2, 8, 8), 2, 8, 8), w)


def _tiny_hf_state_dict(seed=1):
    rng = np.random.default_rng(seed)
    dim, n_kv, head_dim = TINY["dim"], TINY["n_kv_heads"], TINY["dim"] // TINY["n_heads"]
    hidden = ls.meta_hidden_dim(TINY)
    return {
        "model.embed_tokens.weight": rng.standard_normal((TINY["vocab_size"], dim)),
        "model.norm.weight": rng.standard_normal((dim,)),
        "model.layers.0.self_attn.q_proj.weight": rng.standard_normal((dim, dim)),
        "model.layers.0.self_attn.k_proj.weight": rng.standard_normal((n_kv * head_dim, dim)),
        "model.layers.0.self_attn.v_proj.weight": rng.standard_normal((n_kv * head_dim, dim)),
        "model.layers.0.self_attn.o_proj.weight": rng.standard_normal((dim, dim)),
        "model.layers.0.mlp.gate_proj.weight": rng.standard_normal((hidden, dim)),
        "model.layers.0.mlp.up_proj.weight": rng.standard_normal((hidden, dim)),
        "model.layers.0.mlp.down_proj.weight": rng.standard_normal((dim, hidden)),
        "model.layers.0.input_layernorm.weight": rng.standard_normal((dim,)),
        "model.layers.0.post_attention_layernorm.weight": rng.standard_normal((dim,)),
    }


def test_hf_to_meta_maps_keys_and_unpermutes_q_and_k():
    hf = _tiny_hf_state_dict()
    meta = ls.hf_to_meta_state_dict(hf, TINY)
    assert set(meta) == {
        "tok_embeddings.weight",
        "norm.weight",
        "layers.0.attention.wq.weight",
        "layers.0.attention.wk.weight",
        "layers.0.attention.wv.weight",
        "layers.0.attention.wo.weight",
        "layers.0.feed_forward.w1.weight",
        "layers.0.feed_forward.w2.weight",
        "layers.0.feed_forward.w3.weight",
        "layers.0.attention_norm.weight",
        "layers.0.ffn_norm.weight",
    }
    q_hf = hf["model.layers.0.self_attn.q_proj.weight"]
    assert np.array_equal(_hf_permute(meta["layers.0.attention.wq.weight"], 2, 8, 8), q_hf)
    k_hf = hf["model.layers.0.self_attn.k_proj.weight"]
    assert np.array_equal(
        _hf_permute(meta["layers.0.attention.wk.weight"], 1, 4, 8), k_hf
    )
    assert np.array_equal(
        meta["layers.0.attention.wv.weight"], hf["model.layers.0.self_attn.v_proj.weight"]
    )
    assert np.array_equal(
        meta["layers.0.feed_forward.w2.weight"], hf["model.layers.0.mlp.down_proj.weight"]
    )


def test_hf_to_meta_maps_lm_head_only_when_present():
    hf = _tiny_hf_state_dict()
    assert "output.weight" not in ls.hf_to_meta_state_dict(hf, TINY)
    hf["lm_head.weight"] = np.zeros((TINY["vocab_size"], TINY["dim"]))
    assert "output.weight" in ls.hf_to_meta_state_dict(hf, TINY)


def test_hf_to_meta_ties_output_to_embedding_when_lm_head_absent():
    hf = _tiny_hf_state_dict()
    assert "lm_head.weight" not in hf
    meta = ls.hf_to_meta_state_dict(hf, TINY, tie_word_embeddings=True)
    assert np.array_equal(meta["output.weight"], hf["model.embed_tokens.weight"])
    assert meta["output.weight"] is not meta["tok_embeddings.weight"]
    meta["output.weight"][0, 0] = 99.0
    assert meta["tok_embeddings.weight"][0, 0] != 99.0, "a copy, not an alias"
    assert ls.check_meta_shapes(meta, TINY)["layers"] == 1


def test_hf_to_meta_prefers_lm_head_over_tying_when_both_present():
    hf = _tiny_hf_state_dict()
    lm_head = np.full((TINY["vocab_size"], TINY["dim"]), 7.0)
    hf["lm_head.weight"] = lm_head
    meta = ls.hf_to_meta_state_dict(hf, TINY, tie_word_embeddings=True)
    assert np.array_equal(meta["output.weight"], lm_head)


def test_hf_to_meta_does_not_tie_unless_asked():
    meta = ls.hf_to_meta_state_dict(_tiny_hf_state_dict(), TINY, tie_word_embeddings=False)
    assert "output.weight" not in meta


def test_check_meta_shapes_requires_output_weight():
    meta = ls.hf_to_meta_state_dict(_tiny_hf_state_dict(), TINY)
    with pytest.raises(ValueError, match="output.weight"):
        ls.check_meta_shapes(meta, TINY)


def test_hf_to_meta_rejects_unknown_keys():
    hf = _tiny_hf_state_dict()
    hf["model.layers.0.self_attn.rotary_emb.inv_freq"] = np.zeros(2)
    with pytest.raises(ValueError, match="unexpected key"):
        ls.hf_to_meta_state_dict(hf, TINY)


def test_check_meta_shapes_accepts_a_correct_conversion_and_rejects_a_wrong_one():
    meta = ls.hf_to_meta_state_dict(_tiny_hf_state_dict(), TINY, tie_word_embeddings=True)
    info = ls.check_meta_shapes(meta, TINY)
    assert info == {"hidden_dim": ls.meta_hidden_dim(TINY), "layers": 1}
    meta["layers.0.feed_forward.w1.weight"] = np.zeros((3, TINY["dim"]))
    with pytest.raises(ValueError, match="w1"):
        ls.check_meta_shapes(meta, TINY)


def test_meta_hidden_dim_matches_the_1b_checkpoint():
    assert ls.meta_hidden_dim(ls.META_PARAMS_1B) == 8192


def test_params_from_pinned_1b_config_equal_meta_params():
    assert ls.params_from_hf_config(CONFIG_1B) == ls.META_PARAMS_1B


def test_params_reject_other_model_types_and_hidden_sizes():
    with pytest.raises(ValueError, match="llama config"):
        ls.params_from_hf_config({**CONFIG_1B, "model_type": "qwen2"})
    with pytest.raises(ValueError, match="intermediate_size"):
        ls.params_from_hf_config({**CONFIG_1B, "intermediate_size": 4096})
    with pytest.raises(ValueError, match="llama3 rope"):
        ls.params_from_hf_config({**CONFIG_1B, "rope_scaling": None})
