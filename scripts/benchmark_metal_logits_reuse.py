#!/usr/bin/env python3
"""Benchmark multiple Metal logits kernels after one shared fixture load."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


CONCRETE_KERNELS = {"scalar", "threadgroup"}
VALID_KERNELS = CONCRETE_KERNELS | {"auto"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="artifacts/metal/gate3/full_hi/fixture.bin")
    parser.add_argument("--metallib", default="zig-out/metal/kernels.metallib")
    parser.add_argument("--cli", default="zig-out/bin/metal_logits_v1")
    parser.add_argument("--out", default="artifacts/benchmarks/metal_logits_reuse.json")
    parser.add_argument("--kernels", default="scalar,threadgroup")
    parser.add_argument("--buffer-mode", choices=["copy", "nocopy"], default="nocopy")
    parser.add_argument("--compare-mode", "--comparison-mode", choices=["full", "topk"], default="full")
    parser.add_argument("--kernel-repeats", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=10)
    parser.add_argument("--benchmark-command-mode", choices=["per_iter", "batched"], default="per_iter")
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="number of reusable-fixture matrix subprocess samples to collect",
    )
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
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


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "median_ms": statistics.median(values),
    }


def parse_ndjson(stdout: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    header: dict[str, Any] | None = None
    footer: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("matrix") == "begin":
            header = record
        elif record.get("matrix") == "end":
            footer = record
        else:
            rows.append(record)
    if header is None:
        raise SystemExit("matrix benchmark did not emit a begin header")
    if footer is None:
        raise SystemExit("matrix benchmark did not emit an end footer")
    if int(footer.get("kernel_count", -1)) != len(rows):
        raise SystemExit("matrix benchmark footer kernel_count did not match row count")
    return header, rows, footer


def requested_kernels(args: argparse.Namespace) -> list[str]:
    kernels = [item.strip() for item in args.kernels.split(",") if item.strip()]
    if not kernels:
        raise SystemExit("--kernels must contain at least one kernel")
    invalid = [kernel for kernel in kernels if kernel not in VALID_KERNELS]
    if invalid:
        expected = "scalar, threadgroup, or auto"
        raise SystemExit(f"--kernels contains unsupported kernel(s): {invalid}; expected {expected}")
    return kernels


def validate(args: argparse.Namespace, header: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    requested = requested_kernels(args)
    actual = [row.get("kernel") for row in rows]
    if actual != requested:
        reasons.append(f"row kernel order {actual} did not match requested {requested}")
    if header.get("buffer_mode") != args.buffer_mode:
        reasons.append("header buffer mode mismatch")
    if int(header.get("kernel_repeats", -1)) != args.kernel_repeats:
        reasons.append("header kernel repeats mismatch")
    if int(header.get("benchmark_iters", -1)) != args.benchmark_iters:
        reasons.append("header benchmark iters mismatch")
    if header.get("benchmark_command_mode") != args.benchmark_command_mode:
        reasons.append("header benchmark command mode mismatch")
    if header.get("compare_mode", "full") != args.compare_mode:
        reasons.append("header compare mode mismatch")
    if float(header.get("shared_fixture_load_ms", -1.0)) <= 0.0:
        reasons.append("shared fixture load time was not positive")
    for row in rows:
        kernel = row.get("kernel")
        actual_kernel = row.get("actual_kernel")
        if kernel == "auto":
            if actual_kernel not in CONCRETE_KERNELS:
                reasons.append(f"{kernel}: auto row did not report a concrete actual_kernel")
        elif actual_kernel != kernel:
            reasons.append(f"{kernel}: actual_kernel mismatch")
        if row.get("buffer_mode") != args.buffer_mode or row.get("actual_buffer_mode") != args.buffer_mode:
            reasons.append(f"{kernel}: buffer mode mismatch")
        if args.buffer_mode == "nocopy" and row.get("used_no_copy_buffers") is not True:
            reasons.append(f"{kernel}: no-copy evidence missing")
        if args.buffer_mode == "copy" and row.get("used_no_copy_buffers") is not False:
            reasons.append(f"{kernel}: copy mode unexpectedly used no-copy buffers")
        if row.get("compare_mode", "full") != args.compare_mode:
            reasons.append(f"{kernel}: compare mode mismatch")
        if bool(row.get("full_compare_ran", True)) != (args.compare_mode == "full"):
            reasons.append(f"{kernel}: full_compare_ran mismatch")
        if row.get("top1_match") is not True or row.get("top20_set_match") is not True:
            reasons.append(f"{kernel}: top-k mismatch")
        if row.get("expected_top1") != row.get("actual_top1"):
            reasons.append(f"{kernel}: expected_top1/actual_top1 mismatch")
        mismatches = row.get("mismatches")
        if args.compare_mode == "full":
            if int(mismatches if mismatches is not None else -1) != 0:
                reasons.append(f"{kernel}: mismatches={mismatches}")
        elif mismatches is not None:
            reasons.append(f"{kernel}: topk compare mode should not emit full mismatches")
        if int(row.get("kernel_repeats", -1)) != args.kernel_repeats:
            reasons.append(f"{kernel}: kernel repeats mismatch")
        if int(row.get("benchmark_iters", -1)) != args.benchmark_iters:
            reasons.append(f"{kernel}: benchmark iters mismatch")
        if row.get("persistent_command_mode") != args.benchmark_command_mode:
            reasons.append(f"{kernel}: benchmark command mode mismatch")
        if float(row.get("fixture_load_ms", -1.0)) != 0.0:
            reasons.append(f"{kernel}: per-row fixture_load_ms should be 0 for reusable-fixture mode")
        for timing_key in (
            "persistent_setup_ms",
            "persistent_elapsed_ms",
            "persistent_ms_per_iter",
            "persistent_ms_per_kernel_repeat",
        ):
            if timing_key in row and not positive_finite(row.get(timing_key)):
                reasons.append(f"{kernel}: invalid {timing_key}")
    return reasons


def rankable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("kernel") != "auto"]


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        args.cli,
        args.metallib,
        "--fixture",
        args.fixture,
        "--expect-topk",
        "--kernel-repeats",
        str(args.kernel_repeats),
        "--benchmark-iters",
        str(args.benchmark_iters),
        "--benchmark-command-mode",
        args.benchmark_command_mode,
        "--compare-mode",
        args.compare_mode,
        "--buffer-mode",
        args.buffer_mode,
        "--matrix-kernels",
        args.kernels,
    ]


def run_sample(args: argparse.Namespace, cmd: list[str], sample_index: int) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wall_ms = (time.perf_counter() - start) * 1000.0
    if proc.returncode != 0:
        raise SystemExit(
            f"matrix benchmark sample {sample_index} failed with rc={proc.returncode}\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )
    header, rows, footer = parse_ndjson(proc.stdout)
    return {
        "sample_index": sample_index,
        "cli_wall_ms": wall_ms,
        "shared_fixture_load_ms": float(header["shared_fixture_load_ms"]),
        "per_row_fixture_load_ms": 0.0,
        "avoided_duplicate_fixture_load_estimate_ms": float(header["shared_fixture_load_ms"])
        * max(0, len(rows) - 1),
        "failure_reasons": validate(args, header, rows),
        "header": header,
        "footer": footer,
        "rows": rows,
        "persistent_setup_ms": summarize([float(row["persistent_setup_ms"]) for row in rows]),
        "persistent_elapsed_ms": summarize([float(row["persistent_elapsed_ms"]) for row in rows]),
        "persistent_ms_per_iter": summarize([float(row["persistent_ms_per_iter"]) for row in rows]),
        "persistent_ms_per_kernel_repeat": summarize([float(row["persistent_ms_per_kernel_repeat"]) for row in rows]),
    }


def row_key(row: dict[str, Any]) -> str:
    return f"{row['kernel']}/{row['buffer_mode']}"


def aggregate_row_summaries(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values_by_key: dict[str, dict[str, list[float]]] = {}
    metadata_by_key: dict[str, dict[str, Any]] = {}
    for sample in samples:
        for row in sample["rows"]:
            if row.get("kernel") == "auto":
                continue
            key = row_key(row)
            bucket = values_by_key.setdefault(
                key,
                {
                    "persistent_setup_ms": [],
                    "persistent_elapsed_ms": [],
                    "persistent_ms_per_iter": [],
                    "persistent_ms_per_kernel_repeat": [],
                },
            )
            for metric in bucket:
                bucket[metric].append(float(row[metric]))
            metadata_by_key.setdefault(
                key,
                {
                    "kernel": row["kernel"],
                    "buffer_mode": row["buffer_mode"],
                    "actual_kernel": row["actual_kernel"],
                },
            )
    summaries: list[dict[str, Any]] = []
    for key, metrics in values_by_key.items():
        summary = metadata_by_key[key] | {
            "variant_id": key,
            "persistent_setup_ms": summarize(metrics["persistent_setup_ms"]),
            "persistent_elapsed_ms": summarize(metrics["persistent_elapsed_ms"]),
            "persistent_ms_per_iter": summarize(metrics["persistent_ms_per_iter"]),
            "persistent_ms_per_kernel_repeat": summarize(metrics["persistent_ms_per_kernel_repeat"]),
        }
        summaries.append(summary)
    return sorted(summaries, key=lambda item: item["persistent_ms_per_kernel_repeat"]["median_ms"])


def actual_kernels_seen(samples: list[dict[str, Any]]) -> dict[str, list[str]]:
    seen: dict[str, set[str]] = {}
    for sample in samples:
        for row in sample["rows"]:
            seen.setdefault(str(row["kernel"]), set()).add(str(row["actual_kernel"]))
    return {kernel: sorted(actuals) for kernel, actuals in sorted(seen.items())}


def has_auto_diagnostic(samples: list[dict[str, Any]]) -> bool:
    return any(row.get("kernel") == "auto" for sample in samples for row in sample["rows"])


def collect_failure_reasons(samples: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for sample in samples:
        for reason in sample["failure_reasons"]:
            reasons.append(f"sample {sample['sample_index']}: {reason}")
    first_header = samples[0]["header"]
    first_rows = samples[0]["rows"]
    for sample in samples[1:]:
        header = sample["header"]
        for key in ("rows", "cols", "buffer_mode", "kernel_repeats", "benchmark_iters", "benchmark_command_mode", "compare_mode"):
            if header.get(key) != first_header.get(key):
                reasons.append(f"sample {sample['sample_index']}: header {key} changed")
        rows = sample["rows"]
        if [row.get("kernel") for row in rows] != [row.get("kernel") for row in first_rows]:
            reasons.append(f"sample {sample['sample_index']}: row kernels changed")
        for row, first_row in zip(rows, first_rows, strict=False):
            for key in ("kernel", "buffer_mode", "actual_buffer_mode", "expected_top1", "actual_top1", "compare_mode", "full_compare_ran"):
                if row.get(key) != first_row.get(key):
                    reasons.append(f"sample {sample['sample_index']}: row {row.get('kernel')} {key} changed")
    return reasons


def main() -> None:
    args = parse_args()
    ensure_prerequisites(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_command(args)
    samples = [run_sample(args, cmd, sample_index) for sample_index in range(args.samples)]
    first = samples[0]
    failure_reasons = collect_failure_reasons(samples)
    ranked_rows = sorted(rankable_rows(first["rows"]), key=lambda row: float(row["persistent_ms_per_kernel_repeat"]))
    ranked_samples = aggregate_row_summaries(samples)
    includes_auto = has_auto_diagnostic(samples)
    shared_fixture_load_ms = first["shared_fixture_load_ms"]
    report = {
        "mode": "reusable-fixture-matrix",
        "verdict": "pass" if not failure_reasons else "fail",
        "failure_reasons": failure_reasons,
        "sample_count": args.samples,
        "settings": {
            "samples": args.samples,
            "fixture": args.fixture,
            "metallib": args.metallib,
            "cli": args.cli,
            "kernels": requested_kernels(args),
            "buffer_mode": args.buffer_mode,
            "kernel_repeats": args.kernel_repeats,
            "benchmark_iters": args.benchmark_iters,
            "benchmark_command_mode": args.benchmark_command_mode,
            "compare_mode": args.compare_mode,
        },
        "includes_auto_diagnostic": includes_auto,
        "actual_kernels_seen_by_kernel": actual_kernels_seen(samples),
        "ranking_confidence": "unranked"
        if includes_auto
        else ("medium" if args.samples >= 7 and args.benchmark_iters >= 10 else "low"),
        "ranking_note": "auto rows are diagnostic-only and excluded from persistent timing rankings",
        "command": cmd,
        "cli_wall_ms": first["cli_wall_ms"],
        "cli_wall_ms_summary": summarize([float(sample["cli_wall_ms"]) for sample in samples]),
        "shared_fixture_load_ms": shared_fixture_load_ms,
        "shared_fixture_load_ms_summary": summarize([float(sample["shared_fixture_load_ms"]) for sample in samples]),
        "per_row_fixture_load_ms": 0.0,
        "avoided_duplicate_fixture_load_estimate_ms": shared_fixture_load_ms * max(0, len(first["rows"]) - 1),
        "avoided_duplicate_fixture_load_estimate_ms_summary": summarize(
            [float(sample["avoided_duplicate_fixture_load_estimate_ms"]) for sample in samples]
        ),
        "header": first["header"],
        "footer": first["footer"],
        "rows": first["rows"],
        "samples": samples,
        "persistent_setup_ms": summarize(
            [float(row["persistent_setup_ms"]) for sample in samples for row in sample["rows"]]
        ),
        "persistent_elapsed_ms": summarize(
            [float(row["persistent_elapsed_ms"]) for sample in samples for row in sample["rows"]]
        ),
        "persistent_ms_per_iter": summarize(
            [float(row["persistent_ms_per_iter"]) for sample in samples for row in sample["rows"]]
        ),
        "persistent_ms_per_kernel_repeat": summarize(
            [float(row["persistent_ms_per_kernel_repeat"]) for sample in samples for row in sample["rows"]]
        ),
        "ranked_by_persistent_ms_per_kernel_repeat": [
            {
                "kernel": row["kernel"],
                "buffer_mode": row["buffer_mode"],
                "persistent_ms_per_kernel_repeat": row["persistent_ms_per_kernel_repeat"],
                "actual_kernel": row["actual_kernel"],
            }
            for row in ranked_rows
        ],
        "ranked_by_persistent_ms_per_kernel_repeat_samples": ranked_samples,
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failure_reasons:
        raise SystemExit(f"reusable fixture benchmark failed validation: {failure_reasons}")


if __name__ == "__main__":
    main()
