#!/usr/bin/env python3
"""Run a small benchmark matrix for the Metal LM-head logits prototype.

This wrapper intentionally reuses benchmark_metal_logits.py so each row keeps the
same correctness checks, CPU reference comparison, and benchmark JSON schema.
It exists to make kernel/buffer-mode comparisons reproducible without changing
prototype defaults based on one-off shell commands.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

KERNELS = ("scalar", "threadgroup")
BUFFER_MODES = ("copy", "nocopy")


def parse_csv(value: str, choices: tuple[str, ...], name: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    invalid = [item for item in items if item not in choices]
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid {name}: {', '.join(invalid)}")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="artifacts/metal/gate3/full_hi/fixture.bin")
    parser.add_argument("--metallib", default="zig-out/metal/kernels.metallib")
    parser.add_argument("--cli", default="zig-out/bin/metal_logits_v1")
    parser.add_argument("--benchmark-script", default="scripts/benchmark_metal_logits.py")
    parser.add_argument("--out", default="artifacts/benchmarks/metal_logits_matrix.json")
    parser.add_argument("--artifact-dir", default="artifacts/benchmarks/matrix_runs")
    parser.add_argument("--kernels", default=",".join(KERNELS), help="Comma-separated subset: scalar,threadgroup")
    parser.add_argument("--buffer-modes", default=",".join(BUFFER_MODES), help="Comma-separated subset: copy,nocopy")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cpu-repeats", type=int, default=1)
    parser.add_argument("--kernel-repeats", type=int, default=5)
    parser.add_argument("--persistent-iters", type=int, default=3)
    parser.add_argument("--persistent-samples", type=int, default=1)
    parser.add_argument("--no-build", action="store_true", help="Skip the one upfront Metal build")
    return parser.parse_args()


def ensure_prerequisites(args: argparse.Namespace) -> None:
    if args.no_build:
        return
    if shutil.which("zig") is None:
        raise SystemExit("zig not found on PATH; cannot build Metal benchmark prerequisites")
    subprocess.run(["zig", "build", "-Denable-metal=true", "metal-lib"], check=True)
    subprocess.run(["zig", "build", "-Denable-metal=true"], check=True)


def stat_ms(report: dict[str, Any], key: str, stat: str) -> float | None:
    value = report.get(key)
    if not isinstance(value, dict):
        return None
    result = value.get(stat)
    return float(result) if result is not None else None


def mean_ms(report: dict[str, Any], key: str) -> float | None:
    return stat_ms(report, key, "mean_ms")


def median_ms(report: dict[str, Any], key: str) -> float | None:
    return stat_ms(report, key, "median_ms")


def persistent_per_repeat(report: dict[str, Any]) -> float | None:
    summary = report.get("persistent_metal_summary")
    if isinstance(summary, dict):
        per_repeat = summary.get("persistent_ms_per_kernel_repeat")
        if isinstance(per_repeat, dict) and per_repeat.get("median_ms") is not None:
            return float(per_repeat["median_ms"])
    persistent = report.get("persistent_metal")
    if not isinstance(persistent, dict):
        return None
    record = persistent.get("record")
    if not isinstance(record, dict):
        return None
    value = record.get("persistent_ms_per_kernel_repeat")
    return float(value) if value is not None else None


def positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0 and value == value and value not in (float("inf"), float("-inf"))


def row_passed(report: dict[str, Any], kernel: str, buffer_mode: str, repeats: int, kernel_repeats: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    record = report.get("metal_last_record", {})
    cpu = report.get("cpu_reference", {})
    wall = report.get("metal_wall_ms", {})
    per_repeat = report.get("metal_command_buffer_per_repeat_ms", {})

    if report.get("kernel") != kernel:
        reasons.append(f"top-level kernel={report.get('kernel')} expected={kernel}")
    if report.get("requested_buffer_mode") != buffer_mode or report.get("buffer_mode") != buffer_mode:
        reasons.append("requested/top-level buffer mode mismatch")
    if record.get("kernel") != kernel or record.get("buffer_mode") != buffer_mode:
        reasons.append("last-record kernel or buffer mode mismatch")
    if record.get("rows") != report.get("rows") or record.get("cols") != report.get("cols"):
        reasons.append("last-record dimensions mismatch report dimensions")
    if not record.get("top1_match", False):
        reasons.append("top1 mismatch")
    if record.get("expected_top1") != record.get("actual_top1"):
        reasons.append("expected/actual top1 differ")
    if not record.get("top20_set_match", False):
        reasons.append("top20 mismatch")
    if int(record.get("mismatches", -1)) != 0:
        reasons.append(f"mismatches={record.get('mismatches')}")
    if cpu.get("cpu_numpy_mismatches") != 0:
        reasons.append(f"cpu reference mismatches={cpu.get('cpu_numpy_mismatches')}")
    if cpu.get("cpu_numpy_top1") != record.get("expected_top1"):
        reasons.append("cpu reference top1 differs from expected top1")
    if int(report.get("kernel_repeats_per_cli_run", -1)) != kernel_repeats:
        reasons.append("top-level kernel repeats mismatch")
    if int(record.get("kernel_repeats", -1)) != kernel_repeats:
        reasons.append("last-record kernel repeats mismatch")
    if int(wall.get("count", -1)) < repeats:
        reasons.append("wall sample count below requested repeats")
    for label, value in (
        ("wall median", wall.get("median_ms")),
        ("command per-repeat median", per_repeat.get("median_ms")),
        ("command per-repeat min", per_repeat.get("min_ms")),
        ("command per-repeat max", per_repeat.get("max_ms")),
        ("elapsed", record.get("elapsed_ms")),
        ("elapsed per-repeat", record.get("elapsed_ms_per_repeat")),
    ):
        if not positive_number(value):
            reasons.append(f"invalid timing: {label}")

    if buffer_mode == "copy":
        if report.get("actual_used_no_copy_count") != 0:
            reasons.append("copy row reported no-copy one-shot use")
        if report.get("persistent_actual_used_no_copy") is True:
            reasons.append("copy persistent run reported no-copy use")
        if record.get("actual_buffer_mode") != "copy" or record.get("used_no_copy_buffers") is not False:
            reasons.append("copy last record actual mode mismatch")
    else:
        if not report.get("actual_used_no_copy_all", False):
            reasons.append("not all one-shot runs used no-copy buffers")
        if report.get("actual_used_no_copy_count") != wall.get("count"):
            reasons.append("no-copy actual count differs from wall count")
        if report.get("persistent_actual_used_no_copy") is False:
            reasons.append("persistent run did not use no-copy buffers")
        if record.get("actual_buffer_mode") != "nocopy" or record.get("used_no_copy_buffers") is not True:
            reasons.append("nocopy last record actual mode mismatch")
    summary = report.get("persistent_metal_summary")
    if isinstance(summary, dict):
        if int(summary.get("count", -1)) < 1:
            reasons.append("persistent summary has no samples")
        per_repeat_summary = summary.get("persistent_ms_per_kernel_repeat", {})
        if not positive_number(per_repeat_summary.get("median_ms")):
            reasons.append("invalid persistent per-repeat summary median")
    return not reasons, reasons


def run_row(args: argparse.Namespace, kernel: str, buffer_mode: str, row_out: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        args.benchmark_script,
        "--fixture",
        args.fixture,
        "--metallib",
        args.metallib,
        "--cli",
        args.cli,
        "--out",
        str(row_out),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--cpu-repeats",
        str(args.cpu_repeats),
        "--kernel",
        kernel,
        "--buffer-mode",
        buffer_mode,
        "--kernel-repeats",
        str(args.kernel_repeats),
        "--persistent-iters",
        str(args.persistent_iters),
        "--persistent-samples",
        str(args.persistent_samples),
        "--no-build",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return {
            "kernel": kernel,
            "buffer_mode": buffer_mode,
            "ok": False,
            "command": cmd,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    report = json.loads(row_out.read_text())
    ok, reasons = row_passed(report, kernel, buffer_mode, args.repeats, args.kernel_repeats)
    return {
        "variant_id": f"{kernel}/{buffer_mode}",
        "kernel": kernel,
        "buffer_mode": buffer_mode,
        "status": "pass" if ok else "fail",
        "ok": ok,
        "failure_reasons": reasons,
        "artifact": str(row_out),
        "wall_mean_ms": mean_ms(report, "metal_wall_ms"),
        "wall_median_ms": median_ms(report, "metal_wall_ms"),
        "measured_total_mean_ms": mean_ms(report, "metal_cli_measured_total_ms"),
        "measured_total_median_ms": median_ms(report, "metal_cli_measured_total_ms"),
        "bridge_wall_mean_ms": mean_ms(report, "metal_cli_bridge_wall_ms"),
        "bridge_wall_median_ms": median_ms(report, "metal_cli_bridge_wall_ms"),
        "command_buffer_per_repeat_mean_ms": mean_ms(report, "metal_command_buffer_per_repeat_ms"),
        "command_buffer_per_repeat_median_ms": median_ms(report, "metal_command_buffer_per_repeat_ms"),
        "persistent_ms_per_kernel_repeat": persistent_per_repeat(report),
        "persistent_metal_summary": report.get("persistent_metal_summary"),
        "actual_used_no_copy_all": report.get("actual_used_no_copy_all"),
        "persistent_actual_used_no_copy": report.get("persistent_actual_used_no_copy"),
        "last_record": report.get("metal_last_record"),
    }


def rank(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    ranked = [row for row in rows if row.get("ok") and row.get(key) is not None]
    return sorted(ranked, key=lambda row: float(row[key]))


def delta_entry(rows: list[dict[str, Any]], label: str, faster_id: tuple[str, str], slower_id: tuple[str, str], metric: str) -> dict[str, Any]:
    lookup = {(row.get("kernel"), row.get("buffer_mode")): row for row in rows if row.get("ok")}
    faster = lookup.get(faster_id)
    slower = lookup.get(slower_id)
    if faster is None or slower is None:
        return {"label": label, "metric": metric, "available": False}
    fast_value = faster.get(metric)
    slow_value = slower.get(metric)
    if fast_value is None or slow_value is None:
        return {"label": label, "metric": metric, "available": False}
    delta = float(fast_value) - float(slow_value)
    relative = delta / float(slow_value) if float(slow_value) != 0 else None
    return {
        "label": label,
        "metric": metric,
        "available": True,
        "candidate": f"{faster_id[0]}/{faster_id[1]}",
        "baseline": f"{slower_id[0]}/{slower_id[1]}",
        "candidate_ms": float(fast_value),
        "baseline_ms": float(slow_value),
        "delta_ms": delta,
        "relative_delta": relative,
    }


def interaction_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    return {
        "metric": metric,
        "buffer_effect_nocopy_vs_copy_by_kernel": [
            delta_entry(rows, "scalar nocopy vs copy", ("scalar", "nocopy"), ("scalar", "copy"), metric),
            delta_entry(rows, "threadgroup nocopy vs copy", ("threadgroup", "nocopy"), ("threadgroup", "copy"), metric),
        ],
        "kernel_effect_threadgroup_vs_scalar_by_buffer": [
            delta_entry(rows, "threadgroup vs scalar with copy", ("threadgroup", "copy"), ("scalar", "copy"), metric),
            delta_entry(rows, "threadgroup vs scalar with nocopy", ("threadgroup", "nocopy"), ("scalar", "nocopy"), metric),
        ],
    }


def main() -> None:
    args = parse_args()
    kernels = parse_csv(args.kernels, KERNELS, "kernels")
    buffer_modes = parse_csv(args.buffer_modes, BUFFER_MODES, "buffer-modes")
    ensure_prerequisites(args)

    out = Path(args.out)
    artifact_dir = Path(args.artifact_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for kernel in kernels:
        for buffer_mode in buffer_modes:
            row_out = artifact_dir / f"{kernel}_{buffer_mode}.json"
            rows.append(run_row(args, kernel, buffer_mode, row_out))

    failed = [row for row in rows if not row.get("ok")]
    rankable = [row for row in rows if row.get("ok") and row.get("persistent_ms_per_kernel_repeat") is not None]
    verdict = "pass" if not failed else "fail"
    if not rankable:
        ranking_confidence = "unranked"
    elif args.repeats >= 7 and args.persistent_iters >= 10:
        ranking_confidence = "medium"
    else:
        ranking_confidence = "low"

    summary = {
        "matrix_version": 1,
        "verdict": verdict,
        "ranking_confidence": ranking_confidence,
        "fixture": args.fixture,
        "metallib": args.metallib,
        "cli": args.cli,
        "rows": rows,
        "ranked_by_measured_total_median_ms": [
            {"kernel": row["kernel"], "buffer_mode": row["buffer_mode"], "measured_total_median_ms": row["measured_total_median_ms"], "artifact": row["artifact"]}
            for row in rank(rows, "measured_total_median_ms")
        ],
        "ranked_by_measured_total_mean_ms": [
            {"kernel": row["kernel"], "buffer_mode": row["buffer_mode"], "measured_total_mean_ms": row["measured_total_mean_ms"], "artifact": row["artifact"]}
            for row in rank(rows, "measured_total_mean_ms")
        ],
        "ranked_by_persistent_ms_per_kernel_repeat": [
            {"kernel": row["kernel"], "buffer_mode": row["buffer_mode"], "persistent_ms_per_kernel_repeat": row["persistent_ms_per_kernel_repeat"], "artifact": row["artifact"]}
            for row in rank(rows, "persistent_ms_per_kernel_repeat")
        ],
        "interaction": [
            interaction_summary(rows, "measured_total_median_ms"),
            interaction_summary(rows, "persistent_ms_per_kernel_repeat"),
        ],
        "promotion_note": "medium-confidence directional ranking when repeats >= 7 and persistent_iters >= 10; require follow-on review before default changes",
        "settings": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "cpu_repeats": args.cpu_repeats,
            "kernel_repeats": args.kernel_repeats,
            "persistent_iters": args.persistent_iters,
            "persistent_samples": args.persistent_samples,
        },
    }
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    if failed:
        raise SystemExit(f"benchmark matrix failed for {len(failed)} row(s); see {out}")


if __name__ == "__main__":
    main()
