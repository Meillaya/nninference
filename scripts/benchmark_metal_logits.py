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
    parser.add_argument("--kernel", choices=["scalar", "threadgroup"], default="scalar")
    parser.add_argument("--kernel-repeats", type=int, default=5)
    parser.add_argument(
        "--persistent-iters",
        type=int,
        default=0,
        help="Also run one in-process persistent-buffer benchmark with this many command-buffer iterations",
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


def run_metal(args: argparse.Namespace) -> tuple[list[float], list[float], list[dict]]:
    cmd = [
        args.cli,
        args.metallib,
        "--fixture",
        args.fixture,
        "--expect-topk",
        "--kernel",
        args.kernel,
        "--kernel-repeats",
        str(args.kernel_repeats),
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


def run_persistent_metal(args: argparse.Namespace) -> tuple[float, dict] | None:
    if args.persistent_iters <= 0:
        return None
    cmd = [
        args.cli,
        args.metallib,
        "--fixture",
        args.fixture,
        "--expect-topk",
        "--kernel",
        args.kernel,
        "--kernel-repeats",
        str(args.kernel_repeats),
        "--benchmark-iters",
        str(args.persistent_iters),
    ]
    start = time.perf_counter()
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    wall_ms = (time.perf_counter() - start) * 1000.0
    return wall_ms, json.loads(proc.stdout)


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


def main() -> None:
    args = parse_args()
    ensure_prerequisites(args)

    fixture = Path(args.fixture)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, cols, *_ = load_fixture_views(fixture)
    metal_wall, metal_kernel, records = run_metal(args)
    persistent = run_persistent_metal(args)
    metal_per_repeat = [float(record.get("elapsed_ms_per_repeat", record["elapsed_ms"])) for record in records]
    fixture_load_ms = [float(record["fixture_load_ms"]) for record in records if "fixture_load_ms" in record]
    bridge_wall_ms = [float(record["metal_bridge_wall_ms"]) for record in records if "metal_bridge_wall_ms" in record]
    host_compare_ms = [float(record["host_compare_ms"]) for record in records if "host_compare_ms" in record]
    measured_total_ms = [float(record["total_cli_measured_ms"]) for record in records if "total_cli_measured_ms" in record]
    cpu = run_cpu_fixture(fixture, args.cpu_repeats)
    host_overhead = [wall - kernel for wall, kernel in zip(metal_wall, metal_kernel)]

    report = {
        "fixture": str(fixture),
        "fixture_size_bytes": fixture.stat().st_size,
        "rows": rows,
        "cols": cols,
        "metal_cli": args.cli,
        "metallib": args.metallib,
        "kernel": args.kernel,
        "metal_wall_ms": summarize(metal_wall),
        "kernel_repeats_per_cli_run": args.kernel_repeats,
        "metal_command_buffer_total_ms": summarize(metal_kernel),
        "metal_command_buffer_per_repeat_ms": summarize(metal_per_repeat),
        "metal_host_load_and_transfer_overhead_ms": summarize(host_overhead),
        "metal_cli_fixture_load_ms": summarize(fixture_load_ms) if fixture_load_ms else None,
        "metal_cli_bridge_wall_ms": summarize(bridge_wall_ms) if bridge_wall_ms else None,
        "metal_cli_host_compare_ms": summarize(host_compare_ms) if host_compare_ms else None,
        "metal_cli_measured_total_ms": summarize(measured_total_ms) if measured_total_ms else None,
        "metal_last_record": records[-1],
        "persistent_metal": None
        if persistent is None
        else {
            "wall_ms": persistent[0],
            "record": persistent[1],
            "wall_ms_per_iter": persistent[0] / args.persistent_iters,
            "wall_ms_per_kernel_repeat": persistent[0] / (args.persistent_iters * args.kernel_repeats),
        },
        "cpu_reference": cpu,
        "bridge_baseline_artifact": "artifacts/benchmarks/bridge_baseline.json",
        "local_app_baseline": "skipped: no fair local app/CLI baseline configured for identical Qwen3.5-0.8B settings",
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
