"""Llama 3.2 1B Instruct on Kaggle CPU: two independent units, each with its own load check.

Launch (internet on, no model attachment; the sources are pinned Hugging Face snapshots):
  python3 kaggle/build_kernel.py --job llama_export --internet
  kaggle/launch.sh llama_export --internet

Units, run in this order. A failure in one never skips the other:
- prebuilt: the two executorch-community .pte files (SpinQuant INT4 EO8, QLoRA INT4 EO8) at pinned
  revisions. Each is downloaded, its size and sha256 recorded, and run in a subprocess for the
  16-token check. They are not copied to /kaggle/working (they are on the Hub, pinned).
- export: our own 8da4w XNNPACK export of unsloth/Llama-3.2-1B-Instruct. ExecuTorch 1.5.1 has no
  llama3_2 HF converter, so the weights are converted to Meta's format and shape-checked here,
  then exported with executorch's export_llm module and the .pte is run for the same check.
  The .pte, tokenizer.json and config.json are copied to /kaggle/working only when export, load
  and generation all succeed.

Install (shared, run once on first need by whichever unit needs it): torch==2.14.0 (CPU index),
executorch==1.5.1 and pytorch-tokenizers==1.5.0, all from PyPI. No source checkout is needed. The
executorch 1.5.1 wheel ships extension/llm/export/export_llm.py and the llama example code, and it
requires pytorch-tokenizers>=1.5.0. The earlier failure came from installing the tokenizers
directory of a source clone, which is a git submodule that was not initialised. Each pip step's
outcome is recorded in export_metrics.json under "install".

Sanity check: each .pte is run in a fresh process with ExecuTorch 1.5.1's official Python runner,
executorch.extension.llm.runner.TextLLMRunner, on CPU. It gets a short Llama 3 chat-formatted
prompt, greedy decoding (temperature 0) and 16 new tokens. The tokenizer is the repo's
tokenizer.json (the .pte's own repo if it has one, else the unsloth base model). The check records
load time, TTFT, decode tokens/s and the output text, plus the forward method's metadata (input
count, sizes, dtypes) so a later error can be traced.

Writes to /kaggle/working: export_metrics.json (always), the export YAML and export.log, and, only
when the export unit succeeds, the exported .pte, tokenizer.json and config.json.
"""

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from _common import job_status, write_metrics

from rakshak.llama_source import (
    META_PARAMS_1B,
    PREBUILT_ALLOW,
    PREBUILT_PTE,
    UNSLOTH_ALLOW,
    UNSLOTH_REPO,
    UNSLOTH_REVISION,
    check_meta_shapes,
    hf_to_meta_state_dict,
    params_from_hf_config,
    snapshot,
)

EXECUTORCH_PIN = "executorch==1.5.1"
TOKENIZERS_PIN = "pytorch-tokenizers==1.5.0"
TORCH_PIN = "torch==2.14.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
EXPORT_MODULE = "executorch.extension.llm.export.export_llm"
INSTALL_STEPS = (
    ("torch", [TORCH_PIN, "--extra-index-url", TORCH_INDEX]),
    ("executorch", [EXECUTORCH_PIN]),
    ("pytorch-tokenizers", [TOKENIZERS_PIN]),
)
PTE_NAME = "Llama-3.2-1B-Instruct-8da4w-xnnpack.pte"
YAML_NAME = "llama1b_8da4w_xnnpack.yaml"
LOG_NAME = "export.log"
METRICS_NAME = "export_metrics.json"
MIN_RAM_GB = 20
MIN_FREE_DISK_GB = 12
GEN_TOKENS = 16
CHAT_PROMPT = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful assistant.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "In one sentence, what does a smoke detector do?<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)
SANITY_TIMEOUT_S = 900
SANITY_PREFIX = "SANITY_JSON "

# Runs in a fresh process. argv: pte, tokenizer.json, n_new, prompt.
SANITY_SCRIPT = r'''
import json, sys, time, traceback
PREFIX = "SANITY_JSON "
pte, tokenizer, n_new, prompt = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
out = {"path": pte, "tokenizer": tokenizer, "runner": "TextLLMRunner", "load": False,
       "prompt": prompt}
try:
    from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner
    t0 = time.perf_counter()
    runner = TextLLMRunner(pte, tokenizer)
    out["load"] = True
    out["load_s"] = round(time.perf_counter() - t0, 2)
    # Forward-method metadata, read through the plain pybindings so a later error can be traced.
    # Best effort: a failure here is recorded and does not stop the generation below.
    try:
        from executorch.extension.pybindings.portable_lib import _load_for_executorch
        mod = _load_for_executorch(pte)
        out["method_names"] = sorted(str(m) for m in mod.method_names())
        meta = mod.method_meta("forward")
        inputs = []
        for i in range(meta.num_inputs()):
            info = meta.input_tensor_meta(i)
            inputs.append({"sizes": list(info.sizes()), "dtype": int(info.dtype())})
        out["forward_meta"] = {"num_inputs": meta.num_inputs(),
                               "num_outputs": meta.num_outputs(), "inputs": inputs}
        del mod
    except Exception as exc:
        out["forward_meta_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    stats = {}
    def on_stats(s):
        # Stats times are milliseconds; copied out here because the object may be reused.
        stats.update(num_prompt_tokens=s.num_prompt_tokens,
                     num_generated_tokens=s.num_generated_tokens,
                     inference_start_ms=s.inference_start_ms,
                     prompt_eval_end_ms=s.prompt_eval_end_ms,
                     first_token_ms=s.first_token_ms,
                     inference_end_ms=s.inference_end_ms)
    pieces = []
    config = GenerationConfig(echo=False, max_new_tokens=n_new, temperature=0.0,
                              num_bos=0, num_eos=0)
    try:
        t1 = time.perf_counter()
        runner.generate(prompt, config, pieces.append, on_stats)
        out["generate_wall_s"] = round(time.perf_counter() - t1, 3)
        out["text"] = "".join(pieces)
        out["tokens_emitted"] = len(pieces)
        out["stats"] = stats
        if stats.get("first_token_ms") is not None:
            ttft_ms = stats["first_token_ms"] - stats["inference_start_ms"]
            out["ttft_s"] = round(ttft_ms / 1000.0, 4)
            decode_s = (stats["inference_end_ms"] - stats["first_token_ms"]) / 1000.0
            gen = stats["num_generated_tokens"]
            if gen > 1 and decode_s > 0:
                out["decode_tokens_per_s"] = round((gen - 1) / decode_s, 2)
            total_s = (stats["inference_end_ms"] - stats["inference_start_ms"]) / 1000.0
            if total_s > 0:
                out["overall_tokens_per_s"] = round(gen / total_s, 2)
    except Exception as exc:
        out["generate_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        out["generate_trace"] = traceback.format_exc()[-600:]
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    out["trace"] = traceback.format_exc()[-800:]
print(PREFIX + json.dumps(out))
'''


def _yaml(checkpoint: Path, params: Path) -> str:
    """Config from docs/EXECUTORCH_PLAN.md section 2.3, with the max_seq_length keys used by
    the v1.5.1 examples/models/llama config files."""
    return f"""base:
  model_class: llama3_2
  checkpoint: {checkpoint}
  params: {params}
  metadata: '{{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}}'
model:
  use_kv_cache: True
  use_sdpa_with_kv_cache: True
export:
  output_name: {PTE_NAME}
  max_seq_length: 2048
  max_context_length: 2048
quantization:
  qmode: 8da4w
  group_size: 32
  embedding_quantize: 4,32
backend:
  xnnpack:
    enabled: True
    extended_ops: True
"""


def _mem_total_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        return None
    return None


def _resources(ctx) -> dict:
    return {
        "ram_gb": _mem_total_gb(),
        "free_disk_gb": round(shutil.disk_usage(ctx.work).free / 1024**3, 1),
        "cpus": os.cpu_count(),
    }


def _top_files(folder: Path) -> list[dict]:
    """Name and size of each top-level file of a snapshot (symlinks followed)."""
    return [
        {"name": p.name, "bytes": p.resolve().stat().st_size}
        for p in sorted(folder.iterdir())
        if p.is_file()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pip(ctx, args, label):
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *args]
    ctx.log(f"{label}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed: {proc.stderr.strip()[-600:]}")


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _versions() -> dict:
    return {
        "executorch": _version("executorch"),
        "torch": _version("torch"),
        "torchao": _version("torchao"),
        "pytorch_tokenizers": _version("pytorch-tokenizers"),
        "safetensors": _version("safetensors"),
        "huggingface_hub": _version("huggingface-hub"),
    }


def _install(ctx, steps: list) -> None:
    """Run the pip steps in order. Each outcome is appended to steps; the first failure stops
    the list and is raised."""
    for label, args in INSTALL_STEPS:
        start = time.perf_counter()
        try:
            _pip(ctx, args, label)
        except Exception as exc:
            steps.append({"step": label, "args": args, "ok": False, "error": str(exc)[:600]})
            ctx.log(f"install step {label} FAILED: {exc}")
            raise
        steps.append({"step": label, "args": args, "ok": True,
                      "seconds": round(time.perf_counter() - start, 1)})
        ctx.log(f"install step {label} ok")


class Install:
    """Runs the pip steps once, on first need. A failure is kept and re-raised by every later
    ensure() call, so each unit records the reason in its own result."""

    def __init__(self, ctx, steps: list):
        self.ctx = ctx
        self.steps = steps
        self.done = False
        self.error = None
        self.versions = None

    def ensure(self) -> dict:
        if not self.done:
            self.done = True
            try:
                _install(self.ctx, self.steps)
                self.versions = _versions()
                self.ctx.log(f"versions: {self.versions}")
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"[:600]
        if self.error is not None:
            raise RuntimeError(f"install failed: {self.error}")
        return self.versions


def _convert_to_meta(ctx, unsloth: Path, scratch: Path):
    """HF safetensors -> Meta-format torch checkpoint and params.json, with shape checks."""
    import torch
    from safetensors.torch import load_file

    config = json.loads((unsloth / "config.json").read_text())
    params = params_from_hf_config(config)
    if {k: params[k] for k in META_PARAMS_1B} != META_PARAMS_1B:
        raise ValueError("config.json does not match the Llama 3.2 1B params")
    hf_sd = load_file(str(unsloth / "model.safetensors"))
    # Llama 3.2 1B ties the output head to the embedding: the HF checkpoint has no lm_head, so
    # output.weight is a copy of the embedding. The flag is passed separately and is not written
    # into params.json, which ExecuTorch's llama model reads as its ModelArgs.
    tie = bool(config.get("tie_word_embeddings", False))
    lm_head_in_checkpoint = "lm_head.weight" in hf_sd
    meta_sd = hf_to_meta_state_dict(hf_sd, params, tie_word_embeddings=tie)
    del hf_sd
    info = check_meta_shapes(meta_sd, params)
    ckpt = scratch / "llama32_1b_meta.pth"
    torch.save(meta_sd, ckpt)
    dtype = str(next(iter(meta_sd.values())).dtype)
    del meta_sd
    params_path = scratch / "params.json"
    params_path.write_text(json.dumps(params, indent=2) + "\n")
    output_from = "lm_head.weight" if lm_head_in_checkpoint else "tied copy of tok_embeddings"
    ctx.log(f"converted: {info['layers']} layers, hidden {info['hidden_dim']}, dtype {dtype}, "
            f"tie_word_embeddings {tie}, output.weight from {output_from}")
    return ckpt, params_path, {**info, "dtype": dtype, "checkpoint_bytes": ckpt.stat().st_size,
                               "tie_word_embeddings": tie, "output_weight_from": output_from}


def _run_export(ctx, export_dir: Path, config: Path, log_path: Path):
    """Run export_llm in its own process, from the installed executorch wheel. Returns
    (exit_code, wall_s, peak_rss_kb)."""
    cmd = [sys.executable, "-m", EXPORT_MODULE, "--config", str(config)]
    ctx.log(f"export: {' '.join(cmd)} (cwd {export_dir})")
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(cmd, cwd=str(export_dir), stdout=log_file, stderr=subprocess.STDOUT)
        # wait4 returns the rusage of this child alone, so the peak RSS is the export's own.
        _, status, usage = os.wait4(proc.pid, 0)
    wall = time.perf_counter() - start
    return os.waitstatus_to_exitcode(status), round(wall, 2), usage.ru_maxrss


def _delegation_lines(log_path: Path, limit: int = 20):
    if not log_path.exists():
        return []
    hits = [
        ln.strip()
        for ln in log_path.read_text(errors="replace").splitlines()
        if "delegat" in ln.lower()
    ]
    return hits[:limit]


def _sanity(scratch: Path, pte: Path, tokenizer: Path) -> dict:
    """Load one .pte with the ExecuTorch Python runner in a fresh process and generate
    GEN_TOKENS tokens from CHAT_PROMPT."""
    folder = scratch / "sanity"
    folder.mkdir(exist_ok=True)
    script = folder / "pte_sanity.py"
    script.write_text(SANITY_SCRIPT)
    try:
        proc = subprocess.run(
            [sys.executable, str(script), str(pte), str(tokenizer), str(GEN_TOKENS), CHAT_PROMPT],
            capture_output=True,
            text=True,
            timeout=SANITY_TIMEOUT_S,
            cwd=str(folder),
        )
    except subprocess.TimeoutExpired:
        return {"load": False, "error": f"timeout after {SANITY_TIMEOUT_S} s"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(SANITY_PREFIX):
            return json.loads(line[len(SANITY_PREFIX):])
    return {"load": False, "error": f"exit {proc.returncode}: {proc.stderr.strip()[-400:]}"}


def _prebuilt_entry(ctx, scratch: Path, install: Install, key: str, repo: str, revision: str,
                    fname: str) -> dict:
    """Download one prebuilt .pte at its pinned revision, record size and sha256, load-check it.
    Nothing is copied to the output. A failure is recorded on the entry and does not raise."""
    entry = {"key": key, "repo": repo, "revision": revision, "file": fname, "ok": False}
    try:
        local = snapshot(repo, revision, allow_patterns=PREBUILT_ALLOW)
        src = local / fname
        if not src.is_file():
            raise FileNotFoundError(f"{fname} is not in the snapshot")
    except Exception as exc:
        entry["reason"] = f"download: {type(exc).__name__}: {str(exc)[:300]}"
        ctx.log(f"prebuilt {key} download failed: {entry['reason']}")
        return entry
    entry["bytes"] = src.stat().st_size
    entry["mib"] = round(entry["bytes"] / 1024**2, 1)
    entry["sha256"] = _sha256(src)
    ctx.log(f"prebuilt {key}: {entry['mib']} MiB sha256 {entry['sha256'][:12]}")
    try:
        tokenizer, entry["tokenizer_source"] = _tokenizer_for(src)
    except Exception as exc:
        entry["sanity"] = {"load": False, "error": f"tokenizer: {exc}"[:400]}
        entry["reason"] = str(entry["sanity"]["error"])[:400]
        ctx.log(f"prebuilt {key}: tokenizer unavailable: {entry['reason']}")
        return entry
    try:
        install.ensure()
    except Exception as exc:
        entry["sanity"] = {"load": False, "error": f"not run: {exc}"[:400]}
    else:
        entry["sanity"] = _sanity(scratch, src, tokenizer)
    sanity = entry["sanity"]
    entry["ok"] = _sanity_ok(sanity)
    if not entry["ok"]:
        entry["reason"] = str(sanity.get("error") or sanity.get("generate_error")
                              or "no tokens generated")[:400]
    ctx.log(f"prebuilt {key}: load {sanity.get('load')} tokens {sanity.get('tokens_emitted')} "
            f"ttft {sanity.get('ttft_s')} s decode {sanity.get('decode_tokens_per_s')} tok/s")
    return entry


def _sanity_ok(sanity: dict) -> bool:
    """A .pte passes when it loads and the runner generates at least one token."""
    return bool(sanity.get("load")) and "generate_error" not in sanity and bool(
        sanity.get("tokens_emitted"))


def _tokenizer_for(model_file: Path) -> tuple[Path, str]:
    """The tokenizer.json of the .pte's own snapshot when it has one, else the unsloth base
    model's (the same Llama 3 vocabulary of 128256 tokens). Returns (path, where it came from)."""
    own = model_file.parent / "tokenizer.json"
    if own.is_file():
        return own, "pte_repo"
    base = snapshot(UNSLOTH_REPO, UNSLOTH_REVISION, allow_patterns=("tokenizer.json",))
    return base / "tokenizer.json", "unsloth_base"


def _unit_prebuilt(ctx, scratch: Path, install: Install) -> dict:
    entries = [
        _prebuilt_entry(ctx, scratch, install, key, *spec) for key, spec in PREBUILT_PTE.items()
    ]
    verified = sum(1 for e in entries if e["ok"])
    res = {"ok": verified == len(entries), "verified": verified, "of": len(entries),
           "entries": entries}
    if not res["ok"]:
        failed = [f"{e['key']}: {e['reason']}" for e in entries if not e["ok"]]
        res["reason"] = "; ".join(failed)[:400]
    return res


def _unit_export(ctx, scratch: Path, install: Install, resources: dict) -> dict:
    res: dict = {"ok": False}
    ram_gb, free_gb = resources["ram_gb"], resources["free_disk_gb"]
    if ram_gb is None or ram_gb < MIN_RAM_GB:
        res["reason"] = f"needs >= {MIN_RAM_GB} GB RAM, kernel has {ram_gb} GB"
        return res
    if free_gb < MIN_FREE_DISK_GB:
        res["reason"] = f"needs >= {MIN_FREE_DISK_GB} GB free disk, has {free_gb} GB"
        return res

    try:
        unsloth = snapshot(UNSLOTH_REPO, UNSLOTH_REVISION, allow_patterns=UNSLOTH_ALLOW)
    except Exception as exc:
        res["reason"] = (f"download {UNSLOTH_REPO}@{UNSLOTH_REVISION[:12]}: "
                         f"{type(exc).__name__}: {str(exc)[:300]}")
        return res
    res["source"] = {"repo": UNSLOTH_REPO, "revision": UNSLOTH_REVISION,
                     "path": str(unsloth), "files": _top_files(unsloth)}
    ctx.log(f"unsloth snapshot {UNSLOTH_REVISION[:12]} at {unsloth}")

    try:
        install.ensure()
    except Exception as exc:
        res["reason"] = str(exc)[:400]
        return res

    try:
        ckpt, params_path, convert_info = _convert_to_meta(ctx, unsloth, scratch)
    except Exception as exc:
        res["reason"] = f"HF to Meta conversion: {type(exc).__name__}: {exc}"[:400]
        return res
    res["conversion"] = {"ok": True, **convert_info}

    export_dir = scratch / "export"
    export_dir.mkdir(exist_ok=True)
    config = ctx.work / YAML_NAME
    config.write_text(_yaml(ckpt, params_path))
    log_path = ctx.work / LOG_NAME
    rc, wall, peak_kb = _run_export(ctx, export_dir, config, log_path)
    res.update(
        exit_code=rc,
        wall_s=wall,
        peak_rss_mb=round(peak_kb / 1024, 1) if peak_kb else None,
        delegation_lines=_delegation_lines(log_path),
    )
    if rc != 0:
        res["reason"] = f"export_llm exited {rc}; see {LOG_NAME}"
        return res

    found = sorted(p for p in export_dir.rglob(PTE_NAME) if p.is_file())
    if not found:
        res["reason"] = f"export finished but {PTE_NAME} was not found under {export_dir}"
        return res
    pte = found[0]
    sanity = _sanity(scratch, pte, unsloth / "tokenizer.json")
    res["pte"] = {
        "name": PTE_NAME,
        "source": f"export of {UNSLOTH_REPO}@{UNSLOTH_REVISION[:12]}",
        "bytes": pte.stat().st_size,
        "mib": round(pte.stat().st_size / 1024**2, 1),
        "sha256": _sha256(pte),
        "sanity": sanity,
    }
    if not _sanity_ok(sanity):
        detail = sanity.get("error") or sanity.get("generate_error") or "no tokens generated"
        res["reason"] = f"exported .pte did not load or generate: {str(detail)[:300]}"
        return res

    shutil.copy2(pte, ctx.work / PTE_NAME)
    for name in ("tokenizer.json", "config.json"):
        shutil.copy2(unsloth / name, ctx.work / name)
    res["saved_to_output"] = [PTE_NAME, "tokenizer.json", "config.json"]
    res["ok"] = True
    return res


def _run_unit(ctx, metrics: dict, name: str, fn) -> bool:
    """Run one unit and record its result. An exception is recorded on the unit and never
    raised, so the other unit still runs."""
    start = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:
        result = {"ok": False, "reason": f"{type(exc).__name__}: {str(exc)[:400]}",
                  "trace": traceback.format_exc()[-800:]}
    result["seconds"] = round(time.perf_counter() - start, 2)
    metrics["units"][name] = result
    if result["ok"]:
        ctx.log(f"unit {name}: ok")
    else:
        ctx.log(f"unit {name}: FAILED: {result.get('reason')}")
    write_metrics(METRICS_NAME, {**metrics, "status": "running"})
    ctx.save()
    return result["ok"]


def run(ctx):
    metrics = {
        "job": "llama_export",
        "status": "running",
        "pins": {
            "executorch": EXECUTORCH_PIN,
            "pytorch_tokenizers": TOKENIZERS_PIN,
            "torch": TORCH_PIN,
            "torch_index": TORCH_INDEX,
        },
        "quantization": {
            "qmode": "8da4w",
            "group_size": 32,
            "embedding_quantize": "4,32",
            "backend": "xnnpack",
        },
        "install": [],
        "units": {},
    }
    resources = _resources(ctx)
    metrics["resources"] = resources
    ctx.log(f"RAM {resources['ram_gb']} GB, free disk {resources['free_disk_gb']} GB")

    install = Install(ctx, metrics["install"])
    scratch = ctx.scratch
    _run_unit(ctx, metrics, "prebuilt", lambda: _unit_prebuilt(ctx, scratch, install))
    _run_unit(ctx, metrics, "export", lambda: _unit_export(ctx, scratch, install, resources))
    metrics["versions"] = install.versions

    units = metrics["units"]
    ok = sum(1 for u in units.values() if u["ok"])
    status = job_status(ok, len(units) - ok)
    prebuilt, export = units["prebuilt"], units["export"]
    metrics["status"] = status
    write_metrics(METRICS_NAME, metrics)
    if "verified" in prebuilt:
        pre_part = f"prebuilt {prebuilt['verified']}/{prebuilt['of']} verified"
    else:
        pre_part = f"prebuilt failed: {prebuilt.get('reason')}"
    if export["ok"]:
        exp_part = f"pte {export['pte']['mib']} MiB"
    else:
        exp_part = f"export failed: {export.get('reason')}"
    ctx.results["summary"] = f"{pre_part} | {exp_part}"[:200]
    ctx.log(f"status {status}: {ctx.results['summary']}")
    return status
