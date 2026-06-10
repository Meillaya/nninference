#!/usr/bin/env python3
# ─── How to run ───
# uv run python scripts/autoptimize_report.py
"""Report writers for the autonomous optimization harness."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, TypedDict


class PlannedCommand(TypedDict):
    name: str
    command: str
    mode: str


class BenchmarkRow(TypedDict):
    name: str
    scope: str
    mode: str
    prompt: str | None
    samples: int
    median_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    tokens_per_second: float | None
    ttft_seconds: float | None
    correctness_ref: str
    confidence: str
    claim_allowed: bool
    note: str


ReportJson = dict[str, Any]


def planned_correctness_commands(commands: list[str] | None = None) -> list[PlannedCommand]:
    if commands:
        return [{"name": f"custom_{index}", "command": command, "mode": "custom"} for index, command in enumerate(commands)]
    return [
        {"name": "hf_alignment", "command": "uv run python scripts/run_alignment_tests.py", "mode": "required"},
        {"name": "zig_build", "command": "zig build", "mode": "required"},
    ]


def benchmark_scope_rows() -> list[BenchmarkRow]:
    return [
        _row("HF bridge first-token path", "hf_bridge", True, "Current CPU surface delegates model forward to the Hugging Face bridge."),
        _row("Metal LM-head sidecar", "metal_sidecar", False, "Sidecar timings are not full native transformer throughput."),
        _row("Microbenchmark fixtures", "microbenchmark", False, "Synthetic fixture measurements require explicit labels."),
    ]


def write_reports(run_dir: Path, report: ReportJson) -> None:
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (reports / "report.md").write_text(_markdown(report))
    (reports / "charts.svg").write_text(_svg(report))
    (reports / "report.html").write_text(_html(report))


def _row(name: str, scope: str, claim_allowed: bool, note: str) -> BenchmarkRow:
    return {
        "name": name,
        "scope": scope,
        "mode": "dry_run_plan",
        "prompt": None,
        "samples": 0,
        "median_ms": None,
        "p50_ms": None,
        "p95_ms": None,
        "tokens_per_second": None,
        "ttft_seconds": None,
        "correctness_ref": "planned_correctness_gates",
        "confidence": "low",
        "claim_allowed": claim_allowed,
        "note": note,
    }


def _markdown(report: ReportJson) -> str:
    commands = "\n".join(f"- `{item['command']}`" for item in report["correctness"]["planned_commands"])
    results = "\n".join(f"- `{item['command']}` exit={item['exit_code']}" for item in report["correctness"].get("results", [])) or "- Not executed in dry-run mode."
    gaps = "\n".join(f"- {gap}" for gap in report["known_gaps"])
    lmstudio = report["comparators"]["lmstudio"]
    return f"""# Autonomous optimization run {report['run_id']}

## Executive summary

Dry-run harness artifacts were generated for budget `{report['budget']}` ({report['budget_seconds']} seconds). This report is a production harness scaffold, not a performance win claim.

## What was optimized

The current milestone optimized harness orchestration and evidence capture, not model math.

## Claim boundaries

- Current local CPU metrics are **HF bridge** scoped.
- Current Metal metrics are **Metal LM-head sidecar** scoped.
- This is **not full native transformer** inference until a future gate proves that surface.
- LM Studio comparisons require local measurement metadata before any ranking claim.

## Hardware/software/model environment

Machine: `{report['machine'].get('platform')}`. Model scope: `{report['model'].get('scope')}`.

## Baseline metrics

No baseline benchmark is promoted by this dry-run. Benchmark rows are schema placeholders with explicit scopes.

## Optimization timeline

Events are recorded in `events.jsonl`; lifecycle state includes queued, baseline, experiment, verify, checkpoint, rollback, and complete semantics.

## Promoted changes

{json.dumps(report['promotions'], indent=2)}

## Rejected/rolled-back changes and reasons

{json.dumps(report['rollbacks'], indent=2)}

## Correctness evidence

Planned gate commands:

{commands}

Executed gate results:

{results}

## LM Studio comparison

Status: `{lmstudio['status']}`  
Reason: {lmstudio['reason']}

## Charts

See `charts.svg` for scoped dry-run chart placeholders.

## Benchmark scopes

- HF bridge
- Metal LM-head sidecar
- microbenchmark

## Known gaps

{gaps}

## Reproduction commands

- `uv run python scripts/autoptimize.py --budget {report['budget']} --run-id {report['run_id']} --dry-run --output-dir artifacts/optimization`
"""


def _svg(report: ReportJson) -> str:
    labels = [row["scope"] for row in report["benchmarks"]]
    bars = "".join(
        f'<text x="40" y="{70 + index * 36}" font-size="16">{html.escape(label)}</text>'
        f'<rect x="250" y="{54 + index * 36}" width="{180 + index * 50}" height="22" fill="#2563eb" />'
        for index, label in enumerate(labels)
    )
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="720" height="220"><text x="40" y="32" font-size="22">Optimization scopes</text>{bars}</svg>\n'


def _html(report: ReportJson) -> str:
    markdown = html.escape(_markdown(report))
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(str(report['run_id']))}</title></head><body><pre>{markdown}</pre></body></html>\n"
