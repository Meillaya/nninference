#!/usr/bin/env python3
"""Summarize Metal command-mode benchmark artifacts without rerunning them.

The command-mode benchmark reports are intentionally verbose because they retain
per-sample correctness evidence. This script distills a chosen set of reports
into a compact JSON/Markdown regression summary that can be committed alongside
worklog decisions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

COMMAND_MODES = ("per_iter", "batched")
DEFAULT_INPUTS = (
    "artifacts/benchmarks/g080_post_topk_command/command_modes_full_samples7.json",
    "artifacts/benchmarks/g081_tg128/command_modes_full_samples7.json",
    "artifacts/benchmarks/g082_tg128x4/command_modes_full_samples7.json",
    "artifacts/benchmarks/g083_tg128r2/command_modes_full_samples7.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Command-mode benchmark JSON artifacts to summarize.",
    )
    parser.add_argument(
        "--out-json",
        default="artifacts/benchmarks/g084_regression_summary/summary.json",
        help="Path for the machine-readable summary.",
    )
    parser.add_argument(
        "--out-md",
        default="artifacts/benchmarks/g084_regression_summary/summary.md",
        help="Path for the Markdown summary.",
    )
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number):
        return number
    return None


def exact_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def median_ms(row: dict[str, Any], metric: str = "persistent_ms_per_kernel_repeat") -> float | None:
    value = row.get(metric, {})
    if not isinstance(value, dict):
        return None
    return finite_float(value.get("median_ms"))


def artifact_label(path: Path) -> str:
    parent = path.parent.name
    if parent.endswith("_runs") and path.parent.parent.name:
        return path.parent.parent.name
    return parent or path.stem


def load_artifact(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"missing artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit(f"artifact {path} must contain a JSON object")
    return raw


def collect_validation_failures(path: Path, report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    label = artifact_label(path)
    if report.get("mode") != "metal-command-mode-comparison":
        failures.append(f"{label}: unexpected mode {report.get('mode')!r}")
    if report.get("verdict") != "pass":
        failures.append(f"{label}: verdict={report.get('verdict')!r}")
    if report.get("failure_reasons"):
        failures.append(f"{label}: failure_reasons={report.get('failure_reasons')!r}")

    settings = report.get("settings", {})
    if settings.get("compare_mode") != "full":
        failures.append(f"{label}: compare_mode is not full")

    rankings = report.get("per_mode_rankings", {})
    for mode in COMMAND_MODES:
        rows = rankings.get(mode)
        if not isinstance(rows, list) or not rows:
            failures.append(f"{label}: missing {mode} rankings")
            continue
        for row in rows:
            variant = row.get("variant_id", row.get("kernel", "<unknown>"))
            if median_ms(row) is None or median_ms(row) <= 0.0:
                failures.append(f"{label}: {mode} {variant} invalid median persistent_ms_per_kernel_repeat")
            if row.get("top1_match") is False or row.get("top20_set_match") is False:
                failures.append(f"{label}: {mode} {variant} summary reports top-k mismatch")

    nested = report.get("reports_by_command_mode")
    if not isinstance(nested, dict):
        failures.append(f"{label}: missing reports_by_command_mode evidence")
        nested = {}
    for mode in COMMAND_MODES:
        nested_report = nested.get(mode)
        if not isinstance(nested_report, dict):
            failures.append(f"{label}: missing nested {mode} report")
            continue
        samples = nested_report.get("samples")
        if not isinstance(samples, list) or not samples:
            failures.append(f"{label}: nested {mode} report has no samples")
            continue
        for sample in samples:
            rows = sample.get("rows") if isinstance(sample, dict) else None
            sample_index = sample.get("sample_index") if isinstance(sample, dict) else "<invalid>"
            if not isinstance(rows, list) or not rows:
                failures.append(f"{label}: {mode} sample {sample_index} has no rows")
                continue
            for row in rows:
                kernel = row.get("kernel", "<unknown>")
                if row.get("top1_match") is not True or row.get("top20_set_match") is not True:
                    failures.append(f"{label}: {mode} sample {sample_index} {kernel} top-k mismatch")
                mismatches = row.get("mismatches")
                parsed_mismatches = exact_int(mismatches)
                if parsed_mismatches is None:
                    failures.append(f"{label}: {mode} sample {sample_index} {kernel} malformed mismatches={mismatches!r}")
                elif parsed_mismatches != 0:
                    failures.append(f"{label}: {mode} sample {sample_index} {kernel} mismatches={mismatches}")
                if row.get("full_compare_ran") is not True:
                    failures.append(f"{label}: {mode} sample {sample_index} {kernel} did not run full compare")

    return failures


def ranking_rows(report: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    rows = report.get("per_mode_rankings", {}).get(mode, [])
    if not isinstance(rows, list):
        return []
    concrete = [row for row in rows if row.get("kernel") != "auto" and median_ms(row) is not None]
    return sorted(concrete, key=lambda row: median_ms(row) or math.inf)


def summarize_artifact(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    per_mode: dict[str, Any] = {}
    for mode in COMMAND_MODES:
        rows = ranking_rows(report, mode)
        winner = rows[0] if rows else None
        per_mode[mode] = {
            "winner": None
            if winner is None
            else {
                "variant_id": winner.get("variant_id"),
                "kernel": winner.get("kernel"),
                "buffer_mode": winner.get("buffer_mode"),
                "median_ms_per_kernel_repeat": median_ms(winner),
            },
            "rankings": [
                {
                    "variant_id": row.get("variant_id"),
                    "kernel": row.get("kernel"),
                    "buffer_mode": row.get("buffer_mode"),
                    "median_ms_per_kernel_repeat": median_ms(row),
                }
                for row in rows
            ],
        }

    return {
        "label": artifact_label(path),
        "path": str(path),
        "settings": report.get("settings", {}),
        "per_mode": per_mode,
    }


def best_by_mode(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for mode in COMMAND_MODES:
        candidates: list[dict[str, Any]] = []
        for artifact in artifacts:
            winner = artifact["per_mode"][mode]["winner"]
            if winner is None:
                continue
            candidates.append(
                {
                    "artifact": artifact["label"],
                    **winner,
                }
            )
        candidates.sort(key=lambda row: row["median_ms_per_kernel_repeat"])
        best[mode] = candidates[0] if candidates else None
    return best


def relative_delta(new: float, baseline: float) -> float:
    return (new - baseline) / baseline


def candidate_delta_against_baseline(
    rows: list[dict[str, Any]],
    *,
    candidate_kernel: str,
) -> tuple[dict[str, Any], dict[str, Any], float] | None:
    candidate = next((row for row in rows if row["kernel"] == candidate_kernel), None)
    baselines = [row for row in rows if row["kernel"] != candidate_kernel]
    if candidate is None or not baselines:
        return None
    baseline = baselines[0]
    delta = relative_delta(candidate["median_ms_per_kernel_repeat"], baseline["median_ms_per_kernel_repeat"])
    return candidate, baseline, delta


def derive_conclusions(artifacts: list[dict[str, Any]]) -> list[str]:
    conclusions: list[str] = []
    best = best_by_mode(artifacts)
    for mode in COMMAND_MODES:
        winner = best.get(mode)
        if winner:
            conclusions.append(
                f"{mode}: best recorded winner is {winner['variant_id']} from {winner['artifact']} "
                f"at {winner['median_ms_per_kernel_repeat']:.3f} ms/kernel-repeat."
            )

    by_label = {artifact["label"]: artifact for artifact in artifacts}
    g083 = by_label.get("g083_tg128r2")
    if g083:
        for mode in COMMAND_MODES:
            rows = g083["per_mode"][mode]["rankings"]
            comparison = candidate_delta_against_baseline(rows, candidate_kernel="threadgroup128r2")
            if comparison:
                _candidate, baseline, delta = comparison
                conclusions.append(
                    f"g083 {mode}: threadgroup128r2 was {delta * 100.0:+.2f}% versus the best retained baseline "
                    f"{baseline['variant_id']}, supporting the rollback."
                )
    g082 = by_label.get("g082_tg128x4")
    if g082:
        for mode in COMMAND_MODES:
            rows = g082["per_mode"][mode]["rankings"]
            comparison = candidate_delta_against_baseline(rows, candidate_kernel="threadgroup128x4")
            if comparison:
                _candidate, baseline, delta = comparison
                conclusions.append(
                    f"g082 {mode}: threadgroup128x4 was {delta * 100.0:+.2f}% versus the best retained baseline "
                    f"{baseline['variant_id']}, below the keep threshold."
                )

    conclusions.append(
        "Next direction: keep scalar/threadgroup/threadgroup128 as comparison baselines and shift from "
        "near-duplicate LM-head reductions toward benchmark integration, dispatch/readback overhead, or a "
        "qualitatively different prepacked/tiled layout hypothesis."
    )
    return conclusions


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Metal benchmark regression summary",
        "",
        f"Verdict: **{summary['verdict']}**",
        "",
        "## Winners by artifact",
        "",
        "| artifact | mode | winner | median ms/kernel-repeat | kernels |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for artifact in summary["artifacts"]:
        kernels = ",".join(artifact["settings"].get("kernels", []))
        for mode in COMMAND_MODES:
            winner = artifact["per_mode"][mode]["winner"] or {}
            value = winner.get("median_ms_per_kernel_repeat")
            value_text = "" if value is None else f"{value:.6f}"
            lines.append(
                f"| {artifact['label']} | {mode} | {winner.get('variant_id', '')} | {value_text} | {kernels} |"
            )

    lines.extend(["", "## Conclusions", ""])
    lines.extend(f"- {item}" for item in summary["conclusions"])
    if summary["failure_reasons"]:
        lines.extend(["", "## Validation failures", ""])
        lines.extend(f"- {item}" for item in summary["failure_reasons"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in (args.inputs or DEFAULT_INPUTS)]
    reports = [(path, load_artifact(path)) for path in input_paths]
    failure_reasons = [
        failure
        for path, report in reports
        for failure in collect_validation_failures(path, report)
    ]
    artifacts = [summarize_artifact(path, report) for path, report in reports]
    summary = {
        "mode": "metal-benchmark-regression-summary",
        "verdict": "pass" if not failure_reasons else "fail",
        "failure_reasons": failure_reasons,
        "inputs": [str(path) for path in input_paths],
        "artifacts": artifacts,
        "best_by_mode": best_by_mode(artifacts),
        "conclusions": derive_conclusions(artifacts),
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    out_md.write_text(markdown_report(summary))
    print(json.dumps(summary, indent=2))
    if failure_reasons:
        raise SystemExit(f"benchmark regression summary failed validation: {failure_reasons}")


if __name__ == "__main__":
    main()
