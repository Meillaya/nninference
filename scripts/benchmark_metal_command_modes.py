#!/usr/bin/env python3
"""Compare Metal logits persistent command modes without mixing rankings."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

COMMAND_MODES = ("per_iter", "batched")


def requested_kernels(raw: str) -> list[str]:
    kernels = [item.strip() for item in raw.split(",") if item.strip()]
    if not kernels:
        raise SystemExit("--kernels must contain at least one kernel")
    return kernels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="artifacts/metal/gate3/full_hi/fixture.bin")
    parser.add_argument("--metallib", default="zig-out/metal/kernels.metallib")
    parser.add_argument("--cli", default="zig-out/bin/metal_logits_v1")
    parser.add_argument("--out", default="artifacts/benchmarks/metal_command_modes.json")
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--kernels", default="scalar,threadgroup")
    parser.add_argument("--buffer-mode", choices=["copy", "nocopy"], default="nocopy")
    parser.add_argument("--kernel-repeats", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=10)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
    args.kernel_list = requested_kernels(args.kernels)
    return args


def ensure_prerequisites(args: argparse.Namespace) -> None:
    if args.no_build:
        return
    subprocess.run(["zig", "build", "-Denable-metal=true"], check=True)
    subprocess.run(["zig", "build", "-Denable-metal=true", "metal-lib"], check=True)


def positive_finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def mode_artifact_path(args: argparse.Namespace, out: Path, mode: str) -> Path:
    if args.artifact_dir:
        artifact_dir = Path(args.artifact_dir)
    else:
        artifact_dir = out.with_suffix("")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir / f"{mode}_reuse.json"


def run_reuse_report(args: argparse.Namespace, out: Path, mode: str) -> dict[str, Any]:
    artifact = mode_artifact_path(args, out, mode)
    cmd = [
        sys.executable,
        "scripts/benchmark_metal_logits_reuse.py",
        "--no-build",
        "--fixture",
        args.fixture,
        "--metallib",
        args.metallib,
        "--cli",
        args.cli,
        "--out",
        str(artifact),
        "--kernels",
        args.kernels,
        "--buffer-mode",
        args.buffer_mode,
        "--kernel-repeats",
        str(args.kernel_repeats),
        "--benchmark-iters",
        str(args.benchmark_iters),
        "--samples",
        str(args.samples),
        "--benchmark-command-mode",
        mode,
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise SystemExit(
            f"reusable fixture benchmark for command mode {mode} failed with rc={proc.returncode}\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )
    return json.loads(artifact.read_text()) | {"artifact": str(artifact), "command": cmd}


def collect_failures(mode: str, report: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    if report.get("verdict") != "pass":
        reasons.append(f"{mode}: verdict={report.get('verdict')} reasons={report.get('failure_reasons')}")
    settings = report.get("settings", {})
    if settings.get("benchmark_command_mode") != mode:
        reasons.append(f"{mode}: settings benchmark_command_mode mismatch")
    if int(settings.get("samples", -1)) != args.samples:
        reasons.append(f"{mode}: sample count setting mismatch")
    if int(settings.get("benchmark_iters", -1)) != args.benchmark_iters:
        reasons.append(f"{mode}: benchmark iters setting mismatch")
    if int(settings.get("kernel_repeats", -1)) != args.kernel_repeats:
        reasons.append(f"{mode}: kernel repeats setting mismatch")
    for sample in report.get("samples", []):
        header = sample.get("header", {})
        if header.get("benchmark_command_mode") != mode:
            reasons.append(f"{mode}: sample {sample.get('sample_index')} header command mode mismatch")
        for row in sample.get("rows", []):
            kernel = row.get("kernel")
            if row.get("persistent_command_mode") != mode:
                reasons.append(f"{mode}: {kernel} persistent command mode mismatch")
            if row.get("top1_match") is not True or row.get("top20_set_match") is not True:
                reasons.append(f"{mode}: {kernel} top-k mismatch")
            if int(row.get("mismatches", -1)) != 0:
                reasons.append(f"{mode}: {kernel} mismatches={row.get('mismatches')}")
            if args.buffer_mode == "nocopy" and row.get("used_no_copy_buffers") is not True:
                reasons.append(f"{mode}: {kernel} no-copy evidence missing")
            if float(row.get("fixture_load_ms", -1.0)) != 0.0:
                reasons.append(f"{mode}: {kernel} per-row fixture_load_ms should be 0")
            for key in ("persistent_setup_ms", "persistent_elapsed_ms", "persistent_ms_per_iter", "persistent_ms_per_kernel_repeat"):
                if not positive_finite(row.get(key)):
                    reasons.append(f"{mode}: {kernel} invalid {key}")
    return reasons


def summaries_by_variant(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["variant_id"]): row
        for row in report.get("ranked_by_persistent_ms_per_kernel_repeat_samples", [])
        if row.get("kernel") != "auto"
    }


def median_metric(row: dict[str, Any], metric: str) -> float:
    return float(row[metric]["median_ms"])


def compare_modes(per_iter: dict[str, Any], batched: dict[str, Any]) -> list[dict[str, Any]]:
    per_variants = summaries_by_variant(per_iter)
    batched_variants = summaries_by_variant(batched)
    comparisons: list[dict[str, Any]] = []
    for variant_id in sorted(set(per_variants) & set(batched_variants)):
        per_row = per_variants[variant_id]
        batched_row = batched_variants[variant_id]
        per_kernel_ms = median_metric(per_row, "persistent_ms_per_kernel_repeat")
        batched_kernel_ms = median_metric(batched_row, "persistent_ms_per_kernel_repeat")
        per_iter_ms = median_metric(per_row, "persistent_ms_per_iter")
        batched_iter_ms = median_metric(batched_row, "persistent_ms_per_iter")
        comparisons.append(
            {
                "variant_id": variant_id,
                "kernel": per_row["kernel"],
                "buffer_mode": per_row["buffer_mode"],
                "per_iter_persistent_ms_per_kernel_repeat_median": per_kernel_ms,
                "batched_persistent_ms_per_kernel_repeat_median": batched_kernel_ms,
                "delta_ms_per_kernel_repeat_batched_minus_per_iter": batched_kernel_ms - per_kernel_ms,
                "relative_delta_per_kernel_repeat": (batched_kernel_ms - per_kernel_ms) / per_kernel_ms,
                "per_iter_persistent_ms_per_iter_median": per_iter_ms,
                "batched_persistent_ms_per_iter_median": batched_iter_ms,
                "delta_ms_per_iter_batched_minus_per_iter": batched_iter_ms - per_iter_ms,
                "relative_delta_per_iter": (batched_iter_ms - per_iter_ms) / per_iter_ms,
            }
        )
    return comparisons


def main() -> None:
    args = parse_args()
    ensure_prerequisites(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    reports = {mode: run_reuse_report(args, out, mode) for mode in COMMAND_MODES}
    failure_reasons = [reason for mode in COMMAND_MODES for reason in collect_failures(mode, reports[mode], args)]
    comparisons = compare_modes(reports["per_iter"], reports["batched"])
    if not comparisons:
        failure_reasons.append("no shared concrete variants were available for command-mode comparison")

    report = {
        "mode": "metal-command-mode-comparison",
        "verdict": "pass" if not failure_reasons else "fail",
        "failure_reasons": failure_reasons,
        "settings": {
            "fixture": args.fixture,
            "metallib": args.metallib,
            "cli": args.cli,
            "kernels": args.kernel_list,
            "buffer_mode": args.buffer_mode,
            "kernel_repeats": args.kernel_repeats,
            "benchmark_iters": args.benchmark_iters,
            "samples": args.samples,
            "command_modes": list(COMMAND_MODES),
        },
        "semantics_note": (
            "per_iter and batched use different command-buffer/wait semantics; each mode keeps its own ranking, "
            "and cross-mode deltas are diagnostic rather than a default-promotion criterion."
        ),
        "per_mode_artifacts": {mode: reports[mode]["artifact"] for mode in COMMAND_MODES},
        "per_mode_rankings": {
            mode: reports[mode].get("ranked_by_persistent_ms_per_kernel_repeat_samples", []) for mode in COMMAND_MODES
        },
        "comparisons": comparisons,
        "reports_by_command_mode": reports,
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failure_reasons:
        raise SystemExit(f"command-mode comparison failed validation: {failure_reasons}")


if __name__ == "__main__":
    main()
