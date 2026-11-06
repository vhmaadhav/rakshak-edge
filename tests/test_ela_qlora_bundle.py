"""QLoRA must ship the unchanged evaluation and interval helpers into its private kernel."""

import importlib.util
from pathlib import Path


def test_qlora_bundle_includes_evaluation_dependencies():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("qlora_builder", root / "kaggle/build_kernel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = module.bundle_names("ela_qlora")
    assert "kaggle/jobs/llama_ela_eval.py" in names
    assert "kaggle/jobs/ela_recompute.py" in names
    assert "kaggle/jobs/ela_data.py" not in names


def test_conditional_export_bundles_recipe_but_no_prebuilt_substitution():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("export_builder", root / "kaggle/build_kernel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = module.bundle_names("ela_export")
    assert "kaggle/jobs/llama_export.py" in names
    assert "src/rakshak/llm/ela_export.py" in names
