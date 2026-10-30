"""ExecuTorch export of Llama 3.2 1B Instruct to .pte (XNNPACK, 8da4w, group size 32) on Kaggle CPU.

Launch (high-RAM CPU kernel; needs the Kaggle licence consent in docs/EXECUTORCH_PLAN.md):
  kaggle/launch.sh llama_export --internet \
      --models metaresearch/llama-3.2/pytorch/1b-instruct

Pins (docs/EXECUTORCH_PLAN.md): executorch==1.5.1, git tag v1.5.1 for the export_llm module and
the tokenizers package, torch==2.14.0 (the torch pin in the v1.5.1 install_executorch.py).
Input: the Meta-native checkpoint folder (consolidated.00.pth, params.json, tokenizer.model).

Writes to /kaggle/working: the .pte, tokenizer.model, params.json, the export YAML, export.log
and export_metrics.json (exit code, wall time, peak RSS of the export process only, .pte size,
package versions, delegation lines). If the kernel lacks RAM or disk, or the checkpoint is
missing, the job writes export_metrics.json with the reason and returns False. It does not
crash.
"""

import importlib.metadata
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from _common import write_metrics

EXECUTORCH_PIN = "executorch==1.5.1"
TORCH_PIN = "torch==2.14.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
TAG = "v1.5.1"
REPO_URL = "https://github.com/pytorch/executorch.git"
PTE_NAME = "llama3_2_1b_8da4w_xnnpack.pte"
YAML_NAME = "llama1b_8da4w_xnnpack.yaml"
LOG_NAME = "export.log"
MIN_RAM_GB = 20
MIN_FREE_DISK_GB = 8
CKPT_FILES = ("consolidated.00.pth", "params.json", "tokenizer.model")
INPUT_ROOT = Path("/kaggle/input")
SEARCH_DEPTH = 7


def _yaml(checkpoint_dir: Path) -> str:
    """Config from docs/EXECUTORCH_PLAN.md section 2.3, with the max_seq_length keys used by
    the v1.5.1 examples/models/llama/config files."""
    ck = checkpoint_dir / "consolidated.00.pth"
    params = checkpoint_dir / "params.json"
    return f"""base:
  model_class: llama3_2
  checkpoint: {ck}
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


def find_checkpoint(root=INPUT_ROOT, depth=SEARCH_DEPTH):
    """First folder under root holding the three Meta-native files, or None."""
    root = Path(root)
    if not root.is_dir():
        return None
    base = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        if len(Path(dirpath).parts) - base >= depth:
            dirnames[:] = []
        if set(CKPT_FILES) <= set(filenames):
            return Path(dirpath)
    return None


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


def _install(ctx, scratch: Path) -> Path:
    _pip(ctx, [TORCH_PIN, "--extra-index-url", TORCH_INDEX], "torch")
    _pip(ctx, [EXECUTORCH_PIN], "executorch")
    clone = scratch / "executorch"
    if clone.exists():
        shutil.rmtree(clone)
    proc = subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", TAG, REPO_URL, str(clone)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git clone {TAG} failed: {proc.stderr.strip()[-400:]}")
    _pip(ctx, [str(clone / "extension" / "llm" / "tokenizers")], "pytorch-tokenizers (source)")
    return clone


def _run_export(ctx, clone: Path, config: Path, log_path: Path):
    """Run export_llm in its own process. Returns (exit_code, wall_s, peak_rss_kb)."""
    cmd = [sys.executable, "-m", "extension.llm.export.export_llm", "--config", str(config)]
    ctx.log(f"export: {' '.join(cmd)} (cwd {clone})")
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(cmd, cwd=str(clone), stdout=log_file, stderr=subprocess.STDOUT)
        # wait4 returns the rusage of this child alone, so the peak RSS is the export's own.
        _, status, usage = os.wait4(proc.pid, 0)
    wall = time.perf_counter() - start
    proc.returncode = os.waitstatus_to_exitcode(status)
    return proc.returncode, round(wall, 2), usage.ru_maxrss


def _delegation_lines(log_path: Path, limit: int = 20):
    if not log_path.exists():
        return []
    hits = [
        ln.strip()
        for ln in log_path.read_text(errors="replace").splitlines()
        if "delegat" in ln.lower()
    ]
    return hits[:limit]


def _fail(ctx, metrics, reason):
    metrics["status"] = "failed"
    metrics["reason"] = reason
    write_metrics("export_metrics.json", metrics)
    ctx.results["summary"] = f"export failed: {reason}"[:200]
    ctx.log(f"FAILED: {reason}")
    return False


def run(ctx):
    metrics = {
        "job": "llama_export",
        "status": "running",
        "pins": {
            "executorch": EXECUTORCH_PIN,
            "torch": TORCH_PIN,
            "tag": TAG,
            "torch_index": TORCH_INDEX,
        },
        "quantization": {
            "qmode": "8da4w",
            "group_size": 32,
            "embedding_quantize": "4,32",
            "backend": "xnnpack",
        },
    }
    ram_gb = _mem_total_gb()
    free_gb = round(shutil.disk_usage(ctx.work).free / 1024**3, 1)
    metrics["resources"] = {"ram_gb": ram_gb, "free_disk_gb": free_gb, "cpus": os.cpu_count()}
    ctx.log(f"RAM {ram_gb} GB, free disk {free_gb} GB")
    if ram_gb is None or ram_gb < MIN_RAM_GB:
        return _fail(ctx, metrics, f"needs >= {MIN_RAM_GB} GB RAM, kernel has {ram_gb} GB")
    if free_gb < MIN_FREE_DISK_GB:
        return _fail(ctx, metrics, f"needs >= {MIN_FREE_DISK_GB} GB free disk, has {free_gb} GB")

    checkpoint = find_checkpoint()
    if checkpoint is None:
        metrics["checkpoint"] = None
        return _fail(
            ctx,
            metrics,
            "no folder under /kaggle/input holds "
            f"{', '.join(CKPT_FILES)}; attach --models "
            "metaresearch/llama-3.2/pytorch/1b-instruct after accepting the licence",
        )
    metrics["checkpoint"] = str(checkpoint)
    ctx.log(f"checkpoint: {checkpoint}")

    scratch = ctx.scratch
    try:
        clone = _install(ctx, scratch)
    except RuntimeError as exc:
        return _fail(ctx, metrics, f"install: {exc}")
    metrics["versions"] = {
        "executorch": _version("executorch"),
        "torch": _version("torch"),
        "torchao": _version("torchao"),
        "pytorch_tokenizers": _version("pytorch-tokenizers"),
    }
    ctx.log(f"versions: {metrics['versions']}")

    config = ctx.work / YAML_NAME
    config.write_text(_yaml(checkpoint))
    log_path = ctx.work / LOG_NAME
    rc, wall, peak_kb = _run_export(ctx, clone, config, log_path)
    metrics.update(
        exit_code=rc,
        wall_s=wall,
        peak_rss_mb=round(peak_kb / 1024, 1) if peak_kb else None,
        delegation_lines=_delegation_lines(log_path),
    )
    if rc != 0:
        return _fail(ctx, metrics, f"export_llm exited {rc}; see {LOG_NAME}")

    found = [
        p for p in (clone / PTE_NAME, ctx.work / PTE_NAME, *clone.rglob(PTE_NAME)) if p.is_file()
    ]
    if not found:
        return _fail(ctx, metrics, f"export finished but {PTE_NAME} was not found")
    pte = found[0]
    if pte.parent != ctx.work:
        shutil.copy2(pte, ctx.work / PTE_NAME)
    for name in ("tokenizer.model", "params.json"):
        shutil.copy2(checkpoint / name, ctx.work / name)
    size = (ctx.work / PTE_NAME).stat().st_size
    metrics.update(status="ok", pte_file=PTE_NAME, pte_bytes=size, pte_mib=round(size / 1024**2, 1))
    write_metrics("export_metrics.json", metrics)
    ctx.results["summary"] = (
        f"pte {metrics['pte_mib']} MiB peak_rss {metrics['peak_rss_mb']} MB wall {wall}s"
    )
    ctx.log(f"wrote {PTE_NAME} ({metrics['pte_mib']} MiB)")
    return True
