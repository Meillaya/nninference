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

DEFAULT_KERNELS = ("scalar", "threadgroup")
KERNEL_CHOICES = ("scalar", "threadgroup", "threadgroup128", "auto")
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
    parser.add_argument("--kernels", default=",".join(DEFAULT_KERNELS), help="Comma-separated subset: scalar,threadgroup,threadgroup128,auto; auto is diagnostic-only")
    parser.add_argument("--buffer-modes", default=",".join(BUFFER_MODES), help="Comma-separated subset: copy,nocopy")
    parser.add_argument("--compare-mode", "--comparison-mode", choices=["full", "topk"], default="full")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cpu-repeats", type=int, default=1)
    parser.add_argument("--kernel-repeats", type=int, default=5)
    parser.add_argument("--persistent-iters", type=int, default=3)
    parser.add_argument("--persistent-samples", type=int, default=1)
    parser.add_argument("--persistent-only", action="store_true", help="Skip one-shot row measurements and rank persistent samples only")
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


def persistent_summary_ms(report: dict[str, Any], key: str, stat: str = "median_ms") -> float | None:
    summary = report.get("persistent_metal_summary")
    if isinstance(summary, dict):
        metric = summary.get(key)
        if isinstance(metric, dict) and metric.get(stat) is not None:
            return float(metric[stat])
    persistent = report.get("persistent_metal")
    if not isinstance(persistent, dict):
        return None
    record = persistent.get("record")
    if not isinstance(record, dict):
        return None
    fallback_keys = {
        "setup_ms": "persistent_setup_ms",
        "persistent_ms_per_kernel_repeat": "persistent_ms_per_kernel_repeat",
        "wall_ms_per_kernel_repeat": None,
    }
    record_key = fallback_keys.get(key)
    if record_key is None:
        value = persistent.get(key)
    else:
        value = record.get(record_key)
    return float(value) if value is not None else None


def positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0 and value == value and value not in (float("inf"), float("-inf"))


def row_passed(
    report: dict[str, Any],
    kernel: str,
    buffer_mode: str,
    repeats: int,
    kernel_repeats: int,
    compare_mode: str,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    record = report.get("metal_last_record", {})
    cpu = report.get("cpu_reference", {})
    wall = report.get("metal_wall_ms") or {}
    per_repeat = report.get("metal_command_buffer_per_repeat_ms") or {}
    persistent_only = bool(report.get("persistent_only", False))

    if report.get("kernel") != kernel:
        reasons.append(f"top-level kernel={report.get('kernel')} expected={kernel}")
    actual_kernel = report.get("actual_kernel")
    actual_kernels_seen = report.get("actual_kernels_seen", [])
    if kernel == "auto":
        if actual_kernel not in {"scalar", "threadgroup", "threadgroup128"}:
            reasons.append(f"auto row reported non-concrete actual_kernel={actual_kernel}")
        if not actual_kernels_seen or any(item not in {"scalar", "threadgroup", "threadgroup128"} for item in actual_kernels_seen):
            reasons.append(f"auto row reported invalid actual_kernels_seen={actual_kernels_seen}")
    else:
        if persistent_only and actual_kernel is None:
            reasons.append("persistent-only row did not report a concrete actual_kernel")
        if persistent_only and not actual_kernels_seen:
            reasons.append("persistent-only row did not report actual_kernels_seen")
        if actual_kernel is not None and actual_kernel != kernel:
            reasons.append(f"actual_kernel={actual_kernel} expected={kernel}")
        if actual_kernels_seen and any(item != kernel for item in actual_kernels_seen):
            reasons.append(f"actual_kernels_seen={actual_kernels_seen} expected only {kernel}")
    if report.get("requested_buffer_mode") != buffer_mode or report.get("buffer_mode") != buffer_mode:
        reasons.append("requested/top-level buffer mode mismatch")
    if record.get("kernel") != kernel or record.get("buffer_mode") != buffer_mode:
        reasons.append("last-record kernel or buffer mode mismatch")
    if report.get("compare_mode", "full") != record.get("compare_mode", "full") or report.get("compare_mode", "full") != compare_mode:
        reasons.append("comparison mode mismatch")
    if bool(record.get("full_compare_ran", True)) != (compare_mode == "full"):
        reasons.append("full_compare_ran mismatch")
    record_actual_kernel = record.get("actual_kernel")
    if kernel == "auto":
        if record_actual_kernel not in {"scalar", "threadgroup", "threadgroup128"}:
            reasons.append(f"auto last record reported non-concrete actual_kernel={record_actual_kernel}")
    else:
        if persistent_only and record_actual_kernel is None:
            reasons.append("persistent-only last record did not report a concrete actual_kernel")
        elif record_actual_kernel is not None and record_actual_kernel != kernel:
            reasons.append(f"last-record actual_kernel={record_actual_kernel} expected={kernel}")
    if record.get("rows") != report.get("rows") or record.get("cols") != report.get("cols"):
        reasons.append("last-record dimensions mismatch report dimensions")
    if not record.get("top1_match", False):
        reasons.append("top1 mismatch")
    if record.get("expected_top1") != record.get("actual_top1"):
        reasons.append("expected/actual top1 differ")
    if not record.get("top20_set_match", False):
        reasons.append("top20 mismatch")
    mismatches = record.get("mismatches")
    if compare_mode == "full":
        if int(mismatches if mismatches is not None else -1) != 0:
            reasons.append(f"mismatches={mismatches}")
        if not positive_number(record.get("full_per_logit_diff_ms")):
            reasons.append("invalid full_per_logit_diff_ms")
    elif mismatches is not None:
        reasons.append(f"topk comparison mode unexpectedly emitted mismatches={mismatches}")
    elif record.get("full_per_logit_diff_ms") is not None:
        reasons.append("topk comparison mode unexpectedly emitted full_per_logit_diff_ms")
    for timing_key in ("host_compare_ms", "expected_topk_selection_ms", "actual_topk_selection_ms", "topk_selection_total_ms"):
        if not positive_number(record.get(timing_key)):
            reasons.append(f"invalid timing: {timing_key}")
    expected_topk = float(record.get("expected_topk_selection_ms", 0.0))
    actual_topk = float(record.get("actual_topk_selection_ms", 0.0))
    topk_total = float(record.get("topk_selection_total_ms", -1.0))
    if abs((expected_topk + actual_topk) - topk_total) > 0.001:
        reasons.append("top-k timing components do not sum to topk_selection_total_ms")
    if cpu.get("cpu_numpy_mismatches") != 0:
        reasons.append(f"cpu reference mismatches={cpu.get('cpu_numpy_mismatches')}")
    if cpu.get("cpu_numpy_top1") != record.get("expected_top1"):
        reasons.append("cpu reference top1 differs from expected top1")
    if int(report.get("kernel_repeats_per_cli_run", -1)) != kernel_repeats:
        reasons.append("top-level kernel repeats mismatch")
    if int(record.get("kernel_repeats", -1)) != kernel_repeats:
        reasons.append("last-record kernel repeats mismatch")
    if not persistent_only and int(wall.get("count", -1)) < repeats:
        reasons.append("wall sample count below requested repeats")
    timing_checks = [
        ("elapsed", record.get("elapsed_ms")),
        ("elapsed per-repeat", record.get("elapsed_ms_per_repeat")),
    ]
    if not persistent_only:
        timing_checks.extend(
            [
                ("wall median", wall.get("median_ms")),
                ("command per-repeat median", per_repeat.get("median_ms")),
                ("command per-repeat min", per_repeat.get("min_ms")),
                ("command per-repeat max", per_repeat.get("max_ms")),
            ]
        )
    for label, value in timing_checks:
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
        if not persistent_only and not report.get("actual_used_no_copy_all", False):
            reasons.append("not all one-shot runs used no-copy buffers")
        if not persistent_only and report.get("actual_used_no_copy_count") != wall.get("count"):
            reasons.append("no-copy actual count differs from wall count")
        if report.get("persistent_actual_used_no_copy") is False:
            reasons.append("persistent run did not use no-copy buffers")
        if record.get("actual_buffer_mode") != "nocopy" or record.get("used_no_copy_buffers") is not True:
            reasons.append("nocopy last record actual mode mismatch")
    summary = report.get("persistent_metal_summary")
    if isinstance(summary, dict):
        if int(summary.get("count", -1)) < 1:
            reasons.append("persistent summary has no samples")
        summary_actual = summary.get("actual_kernels_seen", [])
        if kernel == "auto":
            if not summary_actual or any(item not in {"scalar", "threadgroup", "threadgroup128"} for item in summary_actual):
                reasons.append(f"auto persistent summary has invalid actual_kernels_seen={summary_actual}")
        elif summary_actual and any(item != kernel for item in summary_actual):
            reasons.append(f"persistent summary actual_kernels_seen={summary_actual} expected only {kernel}")
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
        "--compare-mode",
        args.compare_mode,
        "--persistent-iters",
        str(args.persistent_iters),
        "--persistent-samples",
        str(args.persistent_samples),
        "--no-build",
    ]
    if args.persistent_only:
        cmd.append("--persistent-only")
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
    ok, reasons = row_passed(report, kernel, buffer_mode, args.repeats, args.kernel_repeats, args.compare_mode)
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
        "fixture_load_median_ms": median_ms(report, "metal_cli_fixture_load_ms"),
        "bridge_wall_mean_ms": mean_ms(report, "metal_cli_bridge_wall_ms"),
        "bridge_wall_median_ms": median_ms(report, "metal_cli_bridge_wall_ms"),
        "host_compare_median_ms": median_ms(report, "metal_cli_host_compare_ms"),
        "full_per_logit_diff_median_ms": median_ms(report, "metal_cli_full_per_logit_diff_ms"),
        "expected_topk_selection_median_ms": median_ms(report, "metal_cli_expected_topk_selection_ms"),
        "actual_topk_selection_median_ms": median_ms(report, "metal_cli_actual_topk_selection_ms"),
        "topk_selection_total_median_ms": median_ms(report, "metal_cli_topk_selection_total_ms"),
        "persistent_host_compare_median_ms": persistent_summary_ms(report, "host_compare_ms"),
        "persistent_full_per_logit_diff_median_ms": persistent_summary_ms(report, "full_per_logit_diff_ms"),
        "persistent_expected_topk_selection_median_ms": persistent_summary_ms(report, "expected_topk_selection_ms"),
        "persistent_actual_topk_selection_median_ms": persistent_summary_ms(report, "actual_topk_selection_ms"),
        "persistent_topk_selection_total_median_ms": persistent_summary_ms(report, "topk_selection_total_ms"),
        "command_buffer_total_median_ms": median_ms(report, "metal_command_buffer_total_ms"),
        "command_buffer_per_repeat_mean_ms": mean_ms(report, "metal_command_buffer_per_repeat_ms"),
        "command_buffer_per_repeat_median_ms": median_ms(report, "metal_command_buffer_per_repeat_ms"),
        "persistent_setup_median_ms": persistent_summary_ms(report, "setup_ms"),
        "persistent_ms_per_kernel_repeat": persistent_per_repeat(report),
        "persistent_wall_ms_per_kernel_repeat": persistent_summary_ms(report, "wall_ms_per_kernel_repeat"),
        "persistent_metal_summary": report.get("persistent_metal_summary"),
        "actual_used_no_copy_all": report.get("actual_used_no_copy_all"),
        "persistent_actual_used_no_copy": report.get("persistent_actual_used_no_copy"),
        "last_record": report.get("metal_last_record"),
    }


def rank(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    ranked = [row for row in rows if row.get("ok") and row.get(key) is not None]
    return sorted(ranked, key=lambda row: float(row[key]))


def confidence_label(args: argparse.Namespace, kernels: list[str], rankable: list[dict[str, Any]]) -> str:
    if not rankable or "auto" in kernels:
        return "unranked"
    if args.persistent_only:
        if args.persistent_samples >= 7 and args.persistent_iters >= 10:
            return "medium"
        return "low"
    if args.repeats >= 7 and args.persistent_iters >= 10:
        return "medium"
    return "low"


def promotion_note(args: argparse.Namespace) -> str:
    if args.persistent_only:
        return (
            "medium-confidence persistent-only directional ranking when persistent_samples >= 7 "
            "and persistent_iters >= 10; measured-total rankings are intentionally omitted; "
            "auto rows are diagnostic-only and require explicit scalar/threadgroup/threadgroup128 confirmation before default changes"
        )
    return (
        "medium-confidence measured-total directional ranking when repeats >= 7 and persistent_iters >= 10; "
        "auto rows are diagnostic-only and require explicit scalar/threadgroup/threadgroup128 confirmation before default changes"
    )


def share(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def bottleneck_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    dominant_counts: dict[str, int] = {}
    for row in rows:
        if not row.get("ok"):
            continue
        measured = row.get("measured_total_median_ms")
        buckets = {
            "fixture_load": row.get("fixture_load_median_ms"),
            "bridge_wall": row.get("bridge_wall_median_ms"),
            "host_compare": row.get("host_compare_median_ms"),
            "persistent_host_compare": row.get("persistent_host_compare_median_ms"),
            "persistent_full_per_logit_diff": row.get("persistent_full_per_logit_diff_median_ms"),
            "persistent_topk_selection_total": row.get("persistent_topk_selection_total_median_ms"),
        }
        observed = {key: value for key, value in buckets.items() if value is not None}
        dominant = max(observed, key=lambda key: float(observed[key])) if observed else None
        if dominant is not None:
            dominant_counts[dominant] = dominant_counts.get(dominant, 0) + 1
        entries.append(
            {
                "variant_id": row.get("variant_id"),
                "measured_total_median_ms": measured,
                "dominant_observed_bucket": dominant,
                "nonexclusive_shares_of_measured_total": {key: share(value, measured) for key, value in buckets.items()},
                "command_buffer_total_median_ms": row.get("command_buffer_total_median_ms"),
                "command_buffer_per_repeat_median_ms": row.get("command_buffer_per_repeat_median_ms"),
                "persistent_ms_per_kernel_repeat": row.get("persistent_ms_per_kernel_repeat"),
                "persistent_setup_median_ms": row.get("persistent_setup_median_ms"),
                "persistent_wall_ms_per_kernel_repeat": row.get("persistent_wall_ms_per_kernel_repeat"),
                "full_per_logit_diff_median_ms": row.get("full_per_logit_diff_median_ms"),
                "topk_selection_total_median_ms": row.get("topk_selection_total_median_ms"),
                "persistent_host_compare_median_ms": row.get("persistent_host_compare_median_ms"),
                "persistent_full_per_logit_diff_median_ms": row.get("persistent_full_per_logit_diff_median_ms"),
                "persistent_topk_selection_total_median_ms": row.get("persistent_topk_selection_total_median_ms"),
            }
        )
    likely_focus = max(dominant_counts, key=dominant_counts.get) if dominant_counts else None
    return {
        "note": "Measured-total shares are non-exclusive because bridge_wall includes GPU command execution and setup; use this summary for bottleneck triage, not accounting totals.",
        "likely_next_focus": likely_focus,
        "dominant_bucket_counts": dominant_counts,
        "rows": entries,
    }


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
    kernels = parse_csv(args.kernels, KERNEL_CHOICES, "kernels")
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
    ranking_confidence = confidence_label(args, kernels, rankable)

    summary = {
        "matrix_version": 1,
        "verdict": verdict,
        "ranking_confidence": ranking_confidence,
        "includes_auto_diagnostic": "auto" in kernels,
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
        "ranked_by_bridge_wall_median_ms": [
            {"kernel": row["kernel"], "buffer_mode": row["buffer_mode"], "bridge_wall_median_ms": row["bridge_wall_median_ms"], "artifact": row["artifact"]}
            for row in rank(rows, "bridge_wall_median_ms")
        ],
        "ranked_by_fixture_load_median_ms": [
            {"kernel": row["kernel"], "buffer_mode": row["buffer_mode"], "fixture_load_median_ms": row["fixture_load_median_ms"], "artifact": row["artifact"]}
            for row in rank(rows, "fixture_load_median_ms")
        ],
        "ranked_by_command_buffer_per_repeat_median_ms": [
            {"kernel": row["kernel"], "buffer_mode": row["buffer_mode"], "command_buffer_per_repeat_median_ms": row["command_buffer_per_repeat_median_ms"], "artifact": row["artifact"]}
            for row in rank(rows, "command_buffer_per_repeat_median_ms")
        ],
        "ranked_by_command_buffer_total_median_ms": [
            {"kernel": row["kernel"], "buffer_mode": row["buffer_mode"], "command_buffer_total_median_ms": row["command_buffer_total_median_ms"], "artifact": row["artifact"]}
            for row in rank(rows, "command_buffer_total_median_ms")
        ],
        "ranked_by_persistent_setup_median_ms": [
            {"kernel": row["kernel"], "buffer_mode": row["buffer_mode"], "persistent_setup_median_ms": row["persistent_setup_median_ms"], "artifact": row["artifact"]}
            for row in rank(rows, "persistent_setup_median_ms")
        ],
        "ranked_by_persistent_wall_ms_per_kernel_repeat": [
            {"kernel": row["kernel"], "buffer_mode": row["buffer_mode"], "persistent_wall_ms_per_kernel_repeat": row["persistent_wall_ms_per_kernel_repeat"], "artifact": row["artifact"]}
            for row in rank(rows, "persistent_wall_ms_per_kernel_repeat")
        ],
        "interaction": [
            interaction_summary(rows, "measured_total_median_ms"),
            interaction_summary(rows, "persistent_ms_per_kernel_repeat"),
        ],
        "bottleneck_summary": bottleneck_summary(rows),
        "promotion_note": promotion_note(args),
        "settings": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "cpu_repeats": args.cpu_repeats,
            "kernel_repeats": args.kernel_repeats,
            "compare_mode": args.compare_mode,
            "persistent_iters": args.persistent_iters,
            "persistent_samples": args.persistent_samples,
            "persistent_only": args.persistent_only,
        },
    }
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    if failed:
        raise SystemExit(f"benchmark matrix failed for {len(failed)} row(s); see {out}")


if __name__ == "__main__":
    main()
