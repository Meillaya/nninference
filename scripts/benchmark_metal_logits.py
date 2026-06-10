#!/usr/bin/env python3
"""Benchmark the fixture-driven Metal LM-head projection prototype."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

MAGIC = b"NNLGFIX1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="artifacts/metal/gate3/full_hi/fixture.bin")
    parser.add_argument("--metallib", default="zig-out/metal/kernels.metallib")
    parser.add_argument("--cli", default="zig-out/bin/metal_logits_v1")
    parser.add_argument("--out", default="artifacts/benchmarks/metal_logits_benchmark.json")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-repeats", type=int, default=2)
    parser.add_argument("--kernel", choices=["scalar", "threadgroup", "threadgroup128", "auto"], default="scalar")
    parser.add_argument("--buffer-mode", choices=["copy", "nocopy"], default="copy")
    parser.add_argument("--compare-mode", "--comparison-mode", choices=["full", "topk"], default="full")
    parser.add_argument("--kernel-repeats", type=int, default=5)
    parser.add_argument(
        "--benchmark-command-mode",
        choices=["per_iter", "batched", "session"],
        default="per_iter",
        help="Persistent benchmark command-buffer mode to request when --persistent-iters is set; session is retained row-major and copy-backed only",
    )
    parser.add_argument(
        "--persistent-iters",
        type=int,
        default=0,
        help="Also run one in-process persistent-buffer benchmark with this many command-buffer iterations",
    )
    parser.add_argument(
        "--persistent-samples",
        type=int,
        default=1,
        help="Number of persistent benchmark samples to collect when --persistent-iters is set",
    )
    parser.add_argument(
        "--persistent-only",
        action="store_true",
        help="Skip one-shot CLI repeat measurements and rely on persistent benchmark records for correctness/timing",
    )
    parser.add_argument("--no-build", action="store_true", help="Skip prerequisite zig build steps")
    return parser.parse_args()


def ensure_prerequisites(args: argparse.Namespace) -> None:
    if args.no_build:
        return
    if shutil.which("zig") is None:
        raise SystemExit("zig not found on PATH; cannot build Metal benchmark prerequisites")
    subprocess.run(["zig", "build", "-Denable-metal=true", "metal-lib"], check=True)
    subprocess.run(["zig", "build", "-Denable-metal=true"], check=True)


def load_fixture_views(path: Path):
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) != 24 or header[:8] != MAGIC:
        raise ValueError(f"bad fixture header: {path}")
    rows, cols, atol, rtol = struct.unpack("<IIff", header[8:24])
    hidden_offset = 24
    weights_offset = hidden_offset + cols * 4
    expected_offset = weights_offset + rows * cols * 4
    hidden = np.memmap(path, dtype="<f4", mode="r", offset=hidden_offset, shape=(cols,))
    weights = np.memmap(path, dtype="<f4", mode="r", offset=weights_offset, shape=(rows, cols))
    expected = np.memmap(path, dtype="<f4", mode="r", offset=expected_offset, shape=(rows,))
    return rows, cols, atol, rtol, hidden, weights, expected


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "median_ms": statistics.median(values),
    }


def summarize_or_none(values: list[float]) -> dict[str, float] | None:
    return summarize(values) if values else None


def numeric_field(records: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def run_metal(args: argparse.Namespace) -> tuple[list[float], list[float], list[dict]]:
    cmd = [
        args.cli,
        args.metallib,
        "--fixture",
        args.fixture,
        "--expect-topk",
        "--kernel",
        args.kernel,
        "--buffer-mode",
        args.buffer_mode,
        "--kernel-repeats",
        str(args.kernel_repeats),
        "--compare-mode",
        args.compare_mode,
    ]
    for _ in range(args.warmup):
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    wall_ms: list[float] = []
    kernel_ms: list[float] = []
    records: list[dict] = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
        elapsed = (time.perf_counter() - start) * 1000.0
        record = json.loads(proc.stdout)
        wall_ms.append(elapsed)
        kernel_ms.append(float(record["elapsed_ms"]))
        records.append(record)
    return wall_ms, kernel_ms, records


def run_persistent_metal_once(args: argparse.Namespace) -> tuple[float, dict]:
    cmd = [
        args.cli,
        args.metallib,
        "--fixture",
        args.fixture,
        "--expect-topk",
        "--kernel",
        args.kernel,
        "--buffer-mode",
        args.buffer_mode,
        "--kernel-repeats",
        str(args.kernel_repeats),
        "--compare-mode",
        args.compare_mode,
        "--benchmark-command-mode",
        args.benchmark_command_mode,
        "--benchmark-iters",
        str(args.persistent_iters),
    ]
    start = time.perf_counter()
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    wall_ms = (time.perf_counter() - start) * 1000.0
    return wall_ms, json.loads(proc.stdout)


def run_persistent_metal(args: argparse.Namespace) -> list[tuple[float, dict]]:
    if args.persistent_iters <= 0:
        return []
    if args.persistent_samples <= 0:
        raise SystemExit("--persistent-samples must be positive when --persistent-iters is set")
    return [run_persistent_metal_once(args) for _ in range(args.persistent_samples)]


def run_cpu_fixture(path: Path, repeats: int):
    rows, cols, atol, rtol, hidden, weights, expected = load_fixture_views(path)
    times: list[float] = []
    last = None
    for _ in range(repeats):
        start = time.perf_counter()
        actual = weights @ hidden
        elapsed = (time.perf_counter() - start) * 1000.0
        times.append(elapsed)
        last = actual
    assert last is not None
    diff = np.abs(last - expected)
    rel = diff / np.maximum(np.abs(expected), 1.0e-12)
    mismatches = np.logical_and(diff > atol, rel > rtol).sum().item()
    return {
        "rows": rows,
        "cols": cols,
        "tolerance": {"max_abs": float(atol), "max_rel": float(rtol)},
        "cpu_numpy_ms": summarize(times),
        "cpu_numpy_max_abs_diff": float(diff.max()),
        "cpu_numpy_max_rel_diff": float(rel.max()),
        "cpu_numpy_mismatches": int(mismatches),
        "cpu_numpy_top1": int(np.argmax(last)),
    }


def persistent_summary(persistent_runs: list[tuple[float, dict]], args: argparse.Namespace) -> dict | None:
    if not persistent_runs:
        return None
    records = [record for _, record in persistent_runs]
    return {
        "count": len(persistent_runs),
        "wall_ms": summarize([wall_ms for wall_ms, _ in persistent_runs]),
        "setup_ms": summarize([float(record["persistent_setup_ms"]) for record in records]),
        "elapsed_ms": summarize([float(record["persistent_elapsed_ms"]) for record in records]),
        "persistent_ms_per_iter": summarize([float(record["persistent_ms_per_iter"]) for record in records]),
        "persistent_ms_per_kernel_repeat": summarize([float(record["persistent_ms_per_kernel_repeat"]) for record in records]),
        "host_compare_ms": summarize([float(record["host_compare_ms"]) for record in records]),
        "full_per_logit_diff_ms": summarize_or_none(numeric_field(records, "full_per_logit_diff_ms")),
        "expected_topk_selection_ms": summarize_or_none(numeric_field(records, "expected_topk_selection_ms")),
        "actual_topk_selection_ms": summarize_or_none(numeric_field(records, "actual_topk_selection_ms")),
        "topk_selection_total_ms": summarize_or_none(numeric_field(records, "topk_selection_total_ms")),
        "wall_ms_per_iter": summarize([wall_ms / args.persistent_iters for wall_ms, _ in persistent_runs]),
        "wall_ms_per_kernel_repeat": summarize([wall_ms / (args.persistent_iters * args.kernel_repeats) for wall_ms, _ in persistent_runs]),
        "actual_kernels_seen": sorted({record.get("actual_kernel") for record in records if record.get("actual_kernel") is not None}),
        "command_modes_seen": sorted(
            {
                record.get("persistent_command_mode")
                for record in records
                if record.get("persistent_command_mode") is not None
            }
        ),
        "used_no_copy_all": all(bool(record.get("used_no_copy_buffers", False)) for record in records),
    }


def validate_actual_kernel(args: argparse.Namespace, records: list[dict], persistent_runs: list[tuple[float, dict]]) -> tuple[str, list[str]]:
    actual_kernels = [record.get("actual_kernel") for record in records]
    actual_kernels.extend(record.get("actual_kernel") for _, record in persistent_runs)

    if args.persistent_only:
        missing = [index for index, value in enumerate(actual_kernels) if value not in {"scalar", "threadgroup", "threadgroup128"}]
        if missing:
            raise SystemExit("--persistent-only requires every persistent Metal run to report a concrete actual_kernel")

    if args.kernel == "auto":
        missing = [index for index, value in enumerate(actual_kernels) if value not in {"scalar", "threadgroup", "threadgroup128"}]
        if missing:
            raise SystemExit("requested --kernel auto but at least one Metal run did not report a concrete actual_kernel")
        if len(set(actual_kernels)) != 1:
            raise SystemExit(f"requested --kernel auto but resolved kernels were inconsistent: {actual_kernels}")
    else:
        mismatched = [value for value in actual_kernels if value is not None and value != args.kernel]
        if mismatched:
            raise SystemExit(f"requested --kernel {args.kernel} but at least one Metal run reported actual_kernel={mismatched[0]}")

    seen = sorted({value for value in actual_kernels if value is not None})
    return (seen[0] if len(seen) == 1 else args.kernel), seen


def main() -> None:
    args = parse_args()
    if args.persistent_only and args.persistent_iters <= 0:
        raise SystemExit("--persistent-only requires --persistent-iters")
    ensure_prerequisites(args)

    fixture = Path(args.fixture)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, cols, *_ = load_fixture_views(fixture)
    if args.persistent_only:
        metal_wall, metal_kernel, records = [], [], []
    else:
        metal_wall, metal_kernel, records = run_metal(args)
    persistent_runs = run_persistent_metal(args)
    if persistent_runs:
        wrong_command_mode = [
            record.get("persistent_command_mode")
            for _, record in persistent_runs
            if record.get("persistent_command_mode") != args.benchmark_command_mode
        ]
        if wrong_command_mode:
            raise SystemExit(
                "requested --benchmark-command-mode "
                f"{args.benchmark_command_mode} but persistent run reported {wrong_command_mode[0]}"
            )
    persistent = persistent_runs[-1] if persistent_runs else None
    metal_per_repeat = [float(record.get("elapsed_ms_per_repeat", record["elapsed_ms"])) for record in records]
    fixture_load_ms = [float(record["fixture_load_ms"]) for record in records if "fixture_load_ms" in record]
    bridge_wall_ms = [float(record["metal_bridge_wall_ms"]) for record in records if "metal_bridge_wall_ms" in record]
    host_compare_ms = [float(record["host_compare_ms"]) for record in records if "host_compare_ms" in record]
    full_per_logit_diff_ms = numeric_field(records, "full_per_logit_diff_ms")
    expected_topk_selection_ms = numeric_field(records, "expected_topk_selection_ms")
    actual_topk_selection_ms = numeric_field(records, "actual_topk_selection_ms")
    topk_selection_total_ms = numeric_field(records, "topk_selection_total_ms")
    measured_total_ms = [float(record["total_cli_measured_ms"]) for record in records if "total_cli_measured_ms" in record]
    cpu = run_cpu_fixture(fixture, args.cpu_repeats)
    if args.buffer_mode == "nocopy":
        failed = [record for record in records if not record.get("used_no_copy_buffers", False)]
        persistent_failed = any(not record.get("used_no_copy_buffers", False) for _, record in persistent_runs)
        if failed or persistent_failed:
            raise SystemExit("requested --buffer-mode nocopy but at least one Metal run did not use no-copy buffers")
    host_overhead = [wall - kernel for wall, kernel in zip(metal_wall, metal_kernel)]
    no_copy_count = sum(1 for record in records if record.get("used_no_copy_buffers", False))
    persistent_used_no_copy = None if persistent is None else bool(persistent[1].get("used_no_copy_buffers", False))
    actual_kernel, actual_kernels_seen = validate_actual_kernel(args, records, persistent_runs)
    persistent_samples_summary = persistent_summary(persistent_runs, args)
    metal_last_record = records[-1] if records else (persistent[1] if persistent is not None else None)
    actual_used_no_copy_all = None if not records else no_copy_count == len(records)

    report = {
        "fixture": str(fixture),
        "fixture_size_bytes": fixture.stat().st_size,
        "rows": rows,
        "cols": cols,
        "metal_cli": args.cli,
        "metallib": args.metallib,
        "kernel": args.kernel,
        "requested_kernel": args.kernel,
        "actual_kernel": actual_kernel,
        "actual_kernels_seen": actual_kernels_seen,
        "persistent_only": args.persistent_only,
        "requested_buffer_mode": args.buffer_mode,
        "buffer_mode": args.buffer_mode,
        "compare_mode": args.compare_mode,
        "benchmark_command_mode": args.benchmark_command_mode,
        "actual_used_no_copy_count": no_copy_count,
        "actual_used_no_copy_all": actual_used_no_copy_all,
        "persistent_actual_used_no_copy": persistent_used_no_copy,
        "metal_wall_ms": summarize_or_none(metal_wall),
        "kernel_repeats_per_cli_run": args.kernel_repeats,
        "metal_command_buffer_total_ms": summarize_or_none(metal_kernel),
        "metal_command_buffer_per_repeat_ms": summarize_or_none(metal_per_repeat),
        "metal_host_load_and_transfer_overhead_ms": summarize_or_none(host_overhead),
        "metal_cli_fixture_load_ms": summarize_or_none(fixture_load_ms),
        "metal_cli_bridge_wall_ms": summarize_or_none(bridge_wall_ms),
        "metal_cli_host_compare_ms": summarize_or_none(host_compare_ms),
        "metal_cli_full_per_logit_diff_ms": summarize_or_none(full_per_logit_diff_ms),
        "metal_cli_expected_topk_selection_ms": summarize_or_none(expected_topk_selection_ms),
        "metal_cli_actual_topk_selection_ms": summarize_or_none(actual_topk_selection_ms),
        "metal_cli_topk_selection_total_ms": summarize_or_none(topk_selection_total_ms),
        "metal_cli_measured_total_ms": summarize_or_none(measured_total_ms),
        "metal_last_record": metal_last_record,
        "persistent_metal": None
        if persistent is None
        else {
            "wall_ms": persistent[0],
            "record": persistent[1],
            "wall_ms_per_iter": persistent[0] / args.persistent_iters,
            "wall_ms_per_kernel_repeat": persistent[0] / (args.persistent_iters * args.kernel_repeats),
        },
        "persistent_metal_samples": [
            {
                "wall_ms": wall_ms,
                "record": record,
                "wall_ms_per_iter": wall_ms / args.persistent_iters,
                "wall_ms_per_kernel_repeat": wall_ms / (args.persistent_iters * args.kernel_repeats),
            }
            for wall_ms, record in persistent_runs
        ]
        if len(persistent_runs) > 1
        else None,
        "persistent_metal_summary": persistent_samples_summary,
        "cpu_reference": cpu,
        "bridge_baseline_artifact": "artifacts/benchmarks/bridge_baseline.json",
        "local_app_baseline": "skipped: no fair local app/CLI baseline configured for identical Qwen3.5-0.8B settings",
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
