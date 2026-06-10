#!/usr/bin/env python3
"""Benchmark multiple Metal logits kernels after one shared fixture load."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--kernel-repeats", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=10)
    parser.add_argument("--no-build", action="store_true")
    return parser.parse_args()


def ensure_prerequisites(args: argparse.Namespace) -> None:
    if args.no_build:
        return
    subprocess.run(["zig", "build", "-Denable-metal=true"], check=True)
    subprocess.run(["zig", "build", "-Denable-metal=true", "metal-lib"], check=True)


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
        raise SystemExit(f"--kernels contains unsupported kernel(s): {invalid}; expected scalar, threadgroup, or auto")
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
        if row.get("top1_match") is not True or row.get("top20_set_match") is not True:
            reasons.append(f"{kernel}: top-k mismatch")
        if int(row.get("mismatches", -1)) != 0:
            reasons.append(f"{kernel}: mismatches={row.get('mismatches')}")
        if int(row.get("kernel_repeats", -1)) != args.kernel_repeats:
            reasons.append(f"{kernel}: kernel repeats mismatch")
        if int(row.get("benchmark_iters", -1)) != args.benchmark_iters:
            reasons.append(f"{kernel}: benchmark iters mismatch")
        if float(row.get("fixture_load_ms", -1.0)) != 0.0:
            reasons.append(f"{kernel}: per-row fixture_load_ms should be 0 for reusable-fixture mode")
        if float(row.get("persistent_ms_per_kernel_repeat", 0.0)) <= 0.0:
            reasons.append(f"{kernel}: invalid persistent timing")
    return reasons


def rankable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("kernel") != "auto"]


def main() -> None:
    args = parse_args()
    ensure_prerequisites(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.cli,
        args.metallib,
        "--fixture",
        args.fixture,
        "--expect-topk",
        "--kernel-repeats",
        str(args.kernel_repeats),
        "--benchmark-iters",
        str(args.benchmark_iters),
        "--buffer-mode",
        args.buffer_mode,
        "--matrix-kernels",
        args.kernels,
    ]
    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wall_ms = (time.perf_counter() - start) * 1000.0
    if proc.returncode != 0:
        raise SystemExit(f"matrix benchmark failed with rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}")

    header, rows, footer = parse_ndjson(proc.stdout)
    failure_reasons = validate(args, header, rows)
    ranked = sorted(rankable_rows(rows), key=lambda row: float(row["persistent_ms_per_kernel_repeat"]))
    shared_fixture_load_ms = float(header["shared_fixture_load_ms"])
    report = {
        "mode": "reusable-fixture-matrix",
        "verdict": "pass" if not failure_reasons else "fail",
        "failure_reasons": failure_reasons,
        "includes_auto_diagnostic": any(row.get("kernel") == "auto" for row in rows),
        "ranking_note": "auto rows are diagnostic-only and excluded from persistent timing rankings",
        "command": cmd,
        "cli_wall_ms": wall_ms,
        "shared_fixture_load_ms": shared_fixture_load_ms,
        "per_row_fixture_load_ms": 0.0,
        "avoided_duplicate_fixture_load_estimate_ms": shared_fixture_load_ms * max(0, len(rows) - 1),
        "header": header,
        "footer": footer,
        "rows": rows,
        "persistent_ms_per_kernel_repeat": summarize([float(row["persistent_ms_per_kernel_repeat"]) for row in rows]),
        "ranked_by_persistent_ms_per_kernel_repeat": [
            {
                "kernel": row["kernel"],
                "buffer_mode": row["buffer_mode"],
                "persistent_ms_per_kernel_repeat": row["persistent_ms_per_kernel_repeat"],
                "actual_kernel": row["actual_kernel"],
            }
            for row in ranked
        ],
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failure_reasons:
        raise SystemExit(f"reusable fixture benchmark failed validation: {failure_reasons}")


if __name__ == "__main__":
    main()
