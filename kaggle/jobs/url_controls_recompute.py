"""Recover frozen URL statistics from exact completed test caches; no model execution."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from _common import write_metrics
from url_controls_cal import ids_hash, sha
from url_controls_eval import cached_report

PARENT_TREE = "4fdfe6994123ec28645a3383841f9e251bd27e5e2cacf91235aa3db2bb65a3ec"
CONTROL_SHA = "88b20cd71febccdc1ed81d8c2a9ec0f1b984b9eed9f44c389f357b5bfcc4ac7d"
SCORE_HASHES = {
    "evaluation_manifest.json": "54e4fa8f9f1d6128ac31f6515b698d768fa8facf98931b00cb3acfd64bf91509",
    "url_e0_phreshphish_s1_test.parquet": (
        "e9d56967d5fd72d230bd9eed8dd52bc0a4ba5f3bec5964a8c9c21011db8edc7f"
    ),
    "url_e0_phreshphish_s2_test.parquet": (
        "af1e6166b8a2b0fb60595c4f88d0ff20e5b97cdedd4840975d448ff29f9f908f"
    ),
    "url_e0_phreshphish_s3_test.parquet": (
        "26d4b55d460de7819542182538ae452f575dcc30ebef8c227950de979bef986f"
    ),
    "url_e0_url_domain_s1_test.parquet": (
        "747cfd00cdba4d6165d91f6f8edd1732cf7ae0fde61fd1ee9fb7aede9bc046c8"
    ),
    "url_e0_url_domain_s2_test.parquet": (
        "b048ab0eeeb2391812a9189839342e49a23418625a5805fa790ca943e770aa27"
    ),
    "url_e0_url_domain_s3_test.parquet": (
        "6c89d833ae6b089c86723e750af788768b0b340e9686ddfa2cfd38e388e45edc"
    ),
}


def source(root):
    matches = []
    for file in Path(root).rglob("evaluation_manifest.json"):
        if sha(file) != SCORE_HASHES["evaluation_manifest.json"]:
            continue
        frozen = json.loads(file.read_text())
        result = json.loads(file.with_name("results.json").read_text())
        if (
            result["job"] != "url_controls_eval"
            or result["tree_sha256"] != PARENT_TREE
            or result["status"] != "error"
            or not result["error"].startswith("FileNotFoundError:")
            or "no split files match ['url_temporal']" not in result["error"]
            or frozen["tree_sha256"] != PARENT_TREE
            or frozen["parent_sha256"] != CONTROL_SHA
        ):
            raise ValueError("exact temporal-check failure source required")
        for name, digest in SCORE_HASHES.items():
            path = file.parent / name
            if path.is_symlink() or not path.is_file() or sha(path) != digest:
                raise ValueError("immutable saved scores/threshold manifest required")
        matches.append((file.parent, frozen))
    if len(matches) != 1:
        raise ValueError("one exact source snapshot required")
    return matches[0]


def cached_scores(file, ids, split):
    data = pd.read_parquet(file)
    if (
        data.id.astype(str).tolist() != list(map(str, ids))
        or set(data["split"]) != {split}
        or not np.isfinite(data.logit).all()
    ):
        raise ValueError("complete ordered original finite cached scores required")
    return 1.0/(1.0+np.exp(-np.clip(data.logit.to_numpy(), -60.0, 60.0)))


def recover_report(ds, scores, calibration, manifest):
    """Adapt the saved top-level manifest to the original per-dataset controls."""
    return cached_report(ds, scores, calibration, manifest["controls"])


def run(ctx):
    import url_student as base
    from url_teacher_kd import controls

    from rakshak.eval.metrics import threshold_for_fpr, threshold_max_f1

    if ctx.argv or not str(ctx.work).startswith("/kaggle/working"):
        raise ValueError("fixed private Kaggle cache-only recovery required")
    parent, frozen = source(ctx.input_root)
    control_folder, original = controls(ctx.input_root, CONTROL_SHA, "phreshphish")
    controls(ctx.input_root, CONTROL_SHA, "url_domain")
    root = base.D.resolve_data_root(marker=base.PHRESH_FILE)
    if any(sha(root / name) != value for name, value in frozen["input_files_sha256"].items()):
        raise ValueError("original frozen input data/split files changed")
    # Do not synthesize or infer a missing temporal split from a historical narrative.
    if base.D.list_split_files(root, ["url_temporal"]):
        raise ValueError("original absent temporal-file state changed")
    cfg = base.parse_args(["--models", "charcnn", "--seeds", "1", "2", "3", "--device", "cuda"])
    datasets = [base.build_phreshphish(root, cfg, lambda urls: np.empty((len(urls), 0)))]
    datasets += base.build_run_datasets(root, cfg, lambda urls: np.empty((len(urls), 0)), ctx)
    if {ds.name for ds in datasets} != {"phreshphish", "url_domain"}:
        raise ValueError("all frozen measurable datasets required")
    shutil.copyfile(parent / "evaluation_manifest.json", ctx.work / "evaluation_manifest.json")
    shutil.copyfile(parent / "results.json", ctx.work / "original_eval_results.json")
    recovery = {
        "schema": "rakshak-url-e0-cache-recovery-v1",
        "parent_tree": PARENT_TREE,
        "parent_kernel": "maadhavvh/rakshak-url-controls-eval-e0eval-20261102",
        "parent_kernel_version": 1,
        "score_hashes": SCORE_HASHES,
        "control_manifest_sha256": CONTROL_SHA,
        "tree_sha256": ctx.results["tree_sha256"],
        "git_sha": ctx.results["git_sha"],
        "no_model_execution": True,
        "no_new_threshold_fit": True,
        "temporal_status": "UNMEASURABLE: url_temporal file absent from frozen inputs",
    }
    with (ctx.work / "recovery_manifest.json").open("x") as stream:
        json.dump(recovery, stream, indent=2)
    result = {}
    for ds in datasets:
        calibration, scores_all = {}, {}
        for seed in (1, 2, 3):
            entry = original["controls"][ds.name][str(seed)]
            record = frozen["controls"][ds.name][str(seed)]
            if record["parent"] != entry or record["cal_max_abs_logit_error"] > 1e-5:
                raise ValueError("original parity/threshold binding changed")
            for part in ("fit", "cal", "test"):
                if (
                    ids_hash(ds.row_ids[getattr(ds, part)])
                    != entry["metadata"][f"{part}_ids_sha256"]
                ):
                    raise ValueError("original ordered fit/cal/test IDs changed")
            cal = cached_scores(
                control_folder / entry["cal_logits"]["file"], ds.row_ids[ds.cal], "cal"
            )
            expected = {
                "max_f1": threshold_max_f1(ds.labels[ds.cal], cal)[0],
                "cal_fpr_1pct": threshold_for_fpr(ds.labels[ds.cal], cal, 0.01)["threshold"],
            }
            if expected != record["thresholds"]:
                raise ValueError("frozen cal-only threshold reconstruction changed")
            calibration[seed] = cal
            scores_all[f"s{seed}"] = cached_scores(
                parent / f"url_e0_{ds.name}_s{seed}_test.parquet", ds.row_ids[ds.test], "test"
            )
        result[ds.name] = recover_report(ds, scores_all, calibration, frozen)
    write_metrics(
        "metrics.json",
        {
            "controls": result,
            "manifest_sha256": sha(ctx.work / "evaluation_manifest.json"),
            "recovery_manifest_sha256": sha(ctx.work / "recovery_manifest.json"),
            "teacher_gate": "UNMEASURABLE",
            "kd_gate": "UNMEASURABLE",
            "reason": "fixed Phresh teacher failed nonfinite loss; no selected teacher/targets/KD",
            "url_temporal": {"status": recovery["temporal_status"], "n": None, "positives": None},
            "hf_publish": "HF publish pending",
            "shipment_approved": False,
        },
    )
    ctx.results["summary"] = (
        "six exact cached URL controls reported; teacher/KD/temporal unmeasurable"
    )
    return True
