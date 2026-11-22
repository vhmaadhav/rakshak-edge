#!/usr/bin/env python3
"""Measure real Android share-to-verdict requests and summarize benchmark APK timings."""

import argparse
import json
import math
import shlex
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path


PACKAGE = "org.rakshak.edge.benchmark"
ACTIVITY = f"{PACKAGE}/org.rakshak.edge.MainActivity"
SAMPLE_ID = "org.rakshak.edge.perf.SAMPLE_ID"
MESSAGES = (
    "Your appointment is confirmed for 10 AM tomorrow. Please arrive 10 minutes early.",
    "Urgent: Your bank account will be blocked. Update your KYC immediately at https://example.invalid/kyc",
    "Aapka parcel kal aayega. Delivery ka time shaam ko confirm karenge.",
    "You have won a prize. Pay the claim fee today and install the APK for your reward.",
)


def adb(*args, timeout=30):
    return subprocess.run(
        ["adb", *args], check=True, capture_output=True, text=True, timeout=timeout
    ).stdout.strip()


def shell(*args):
    return adb("shell", shlex.join(args))


def read_samples():
    samples = []
    for line in adb("logcat", "-d", "-v", "raw", "-s", "RakshakPerf:I", "*:S").splitlines():
        if not line.startswith("{"):
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(sample, dict) and "elapsed_ms" in sample:
            samples.append(sample)
    return samples


def share(message, *, cold, timeout):
    if cold:
        shell("am", "force-stop", PACKAGE)
    sample_id = time.time_ns()
    output = shell(
        "am", "start", "-W", "-n", ACTIVITY, "-a", "android.intent.action.SEND", "-t", "text/plain",
        "--es", "android.intent.extra.TEXT", message, "--el", SAMPLE_ID, str(sample_id),
    )
    if "Error:" in output or "Status: timeout" in output:
        raise RuntimeError(output)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for sample in reversed(read_samples()):
            if sample.get("id") != sample_id:
                continue
            expected = "cold_share" if cold else "warm_share"
            if sample["path"] != expected:
                raise RuntimeError(f"Expected {expected}, got {sample['path']}. Keep the app open for warm runs.")
            if sample["outcome"] != "verdict":
                raise RuntimeError("The app could not render a verdict. Check the benchmark APK for the detector asset.")
            return sample
        time.sleep(0.15)
    raise RuntimeError("No verdict timing received. Keep the phone unlocked and install the benchmark APK.")


def percentiles(values):
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return None
    return {
        "n": len(ordered),
        "p50": round(ordered[math.ceil(len(ordered) * 0.50) - 1], 3),
        "p95": round(ordered[math.ceil(len(ordered) * 0.95) - 1], 3),
    }


def summarize(samples):
    grouped = defaultdict(list)
    for sample in samples:
        path = sample["path"]
        stages = sample.get("stages_ms", {})
        load = {"check": "detector_load", "explain": "llm_load"}.get(path)
        if load is not None:
            path += "_cold" if load in stages else "_warm"
        grouped[path].append(sample)

    report = {}
    for path, rows in sorted(grouped.items()):
        submitted = [r for r in rows if r.get("frame_submitted")]
        verdicts = [r for r in submitted if r["outcome"] == "verdict"]
        generated = [r for r in rows if r["outcome"] in ("llm", "partial") and (r.get("llm_points") or 0) > 0]
        generated_frames = [r for r in generated if r.get("frame_submitted")]
        stages_ms = defaultdict(list)
        for row in rows:
            for name, millis in row.get("stages_ms", {}).items():
                stages_ms[name].append(millis)
        report[path] = {
            "outcomes": dict(Counter(r["outcome"] for r in rows)),
            "samples_without_frame_commit": len(rows) - len(submitted),
            "request_to_verdict_frame_ms": percentiles(r["elapsed_ms"] for r in verdicts),
            "request_to_llm_frame_ms": percentiles(r["elapsed_ms"] for r in generated_frames),
            "llm_first_token_ms": percentiles(r.get("first_token_ms") for r in generated),
            "llm_first_verified_point_ms": percentiles(r.get("first_verified_point_ms") for r in generated),
            "process_peak_rss_kib": max((r.get("process_peak_rss_kib") or 0 for r in rows), default=0) or None,
            "stages_ms": {name: percentiles(values) for name, values in sorted(stages_ms.items())},
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="send synthetic SMS through the app's Share entry point")
    run.add_argument("--mode", choices=("cold", "warm", "both"), default="both")
    run.add_argument("--runs", type=int, default=30, help="samples per mode")
    run.add_argument("--warmup", type=int, default=5, help="discarded warm Share samples")
    run.add_argument("--timeout", type=float, default=15, help="max seconds to wait for each timing record")
    report = commands.add_parser("report", help="summarize current logcat, including manual Check and Explain taps")
    for command in (run, report):
        command.add_argument("--out", type=Path, help="save the summary and individual samples as JSON")
    args = parser.parse_args()

    try:
        if adb("get-state") != "device":
            raise RuntimeError("ADB cannot see a ready device.")
        if not shell("pm", "path", PACKAGE).startswith("package:"):
            raise RuntimeError(f"Install the {PACKAGE} APK first.")
        device = {key: shell("getprop", prop) for key, prop in (
            ("model", "ro.product.model"), ("android", "ro.build.version.release"),
        )}
        if args.command == "run":
            if args.runs < 1 or args.warmup < 0 or args.timeout <= 0:
                parser.error("--runs and --timeout must be positive and --warmup must be nonnegative")
            samples = []
            if args.mode in ("cold", "both"):
                for index in range(args.runs):
                    samples.append(share(MESSAGES[index % len(MESSAGES)], cold=True, timeout=args.timeout))
            if args.mode in ("warm", "both"):
                share(MESSAGES[0], cold=True, timeout=args.timeout)
                for index in range(-args.warmup, args.runs):
                    sample = share(MESSAGES[index % len(MESSAGES)], cold=False, timeout=args.timeout)
                    if index >= 0:
                        samples.append(sample)
        else:
            samples = read_samples()
            if not samples:
                raise RuntimeError("No timing records. Open the benchmark APK and check a message first.")

        summary = {"device": device, "groups": summarize(samples)}
        print(json.dumps(summary, indent=2))
        if args.out:
            args.out.write_text(json.dumps({**summary, "samples": samples}, indent=2) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        detail = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) else str(error)
        parser.exit(1, f"Benchmark failed: {detail or error}\n")


if __name__ == "__main__":
    main()
