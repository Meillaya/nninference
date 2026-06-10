#!/usr/bin/env python3
# ─── How to run ───
# uv run python scripts/autoptimize.py --budget 3h --run-id smoke --dry-run
"""Autonomous optimization harness entrypoint for nninference."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from autoptimize_lmstudio import detect_lmstudio
from autoptimize_report import ReportJson, benchmark_scope_rows, planned_correctness_commands, write_reports

BUDGET_SECONDS: Final[dict[str, int]] = {
    "3h": 10_800,
    "6h": 21_600,
    "9h": 32_400,
    "12h": 43_200,
}
DEFAULT_OUTPUT_DIR: Final = Path("artifacts/optimization")
PROMOTION_THRESHOLDS: Final = {"median_improvement_pct": 3.0, "max_secondary_regression_pct": 5.0, "min_samples": 7}


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    budget: str
    budget_seconds: int
    run_id: str
    dry_run: bool
    correctness_only: bool
    output_dir: Path
    resume: bool
    simulate_failed_experiment: bool
    commands: list[str]

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_id


def parse_args() -> HarnessConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", default="3h", choices=tuple(BUDGET_SECONDS), help="Wall-clock budget tier, maximum 12h.")
    parser.add_argument("--run-id", default=f"run-{int(time.time())}", help="Stable run artifact id.")
    parser.add_argument("--dry-run", action="store_true", help="Plan and write artifacts without running expensive optimization.")
    parser.add_argument("--correctness-only", action="store_true", help="Record or execute only correctness gates.")
    parser.add_argument("--resume", action="store_true", help="Append to an existing run event log and write a checkpoint.")
    parser.add_argument("--simulate-failed-experiment", action="store_true", help="Dry-run a failed experiment and rollback record for harness QA.")
    parser.add_argument("--command", action="append", default=[], help="Override correctness command; repeatable. Intended for tests/smoke runs.")
    parser.add_argument("--output-dir", "--out-dir", default=str(DEFAULT_OUTPUT_DIR), help="Root directory for run artifacts.")
    args = parser.parse_args()
    budget = str(args.budget)
    return HarnessConfig(
        budget=budget,
        budget_seconds=BUDGET_SECONDS[budget],
        run_id=str(args.run_id),
        dry_run=bool(args.dry_run),
        correctness_only=bool(args.correctness_only),
        output_dir=Path(str(args.output_dir)),
        resume=bool(args.resume),
        simulate_failed_experiment=bool(args.simulate_failed_experiment),
        commands=list(args.command),
    )


def build_report(config: HarnessConfig, correctness_results: list[dict[str, str | int | float]], rollbacks: list[dict[str, str]]) -> ReportJson:
    lmstudio = detect_lmstudio().to_json()
    return {
        "run_id": config.run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "budget": config.budget,
        "budget_seconds": config.budget_seconds,
        "dry_run": config.dry_run,
        "machine": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor()},
        "git": {"status_short": _cmd_text(["git", "status", "--short"]), "commit": _cmd_text(["git", "rev-parse", "HEAD"])},
        "model": {"id": "Qwen/Qwen3.5-0.8B", "local_dir": "Qwen3.5-0.8B", "scope": "HF bridge plus Metal LM-head sidecar"},
        "correctness": {"planned_commands": planned_correctness_commands(config.commands), "results": correctness_results},
        "benchmarks": benchmark_scope_rows(),
        "comparators": {"lmstudio": lmstudio},
        "claim_boundaries": {
            "current_scope": "HF bridge plus Metal LM-head sidecar; not full native transformer inference",
            "requires_local_lmstudio_measurement": True,
            "sidecar_is_end_to_end": False,
        },
        "promotions": [],
        "rollbacks": rollbacks,
        "promotion_thresholds": PROMOTION_THRESHOLDS,
        "artifacts": {
            "config": "config.json",
            "manifest": "manifest.json",
            "events": "events.jsonl",
            "report_json": "reports/report.json",
            "report_md": "reports/report.md",
            "charts_svg": "reports/charts.svg",
            "report_html": "reports/report.html",
        },
        "known_gaps": [
            "No long-duration optimization loop has been run yet.",
            "LM Studio live generation comparison is optional and skipped when unavailable.",
            "Current metrics must remain scoped as HF bridge, Metal sidecar, or microbenchmark evidence.",
        ],
    }


def write_run(config: HarnessConfig) -> ReportJson:
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", {
        "budget": config.budget,
        "budget_seconds": config.budget_seconds,
        "run_id": config.run_id,
        "dry_run": config.dry_run,
        "correctness_only": config.correctness_only,
        "resume": config.resume,
        "state_machine": ["queued", "baseline", "experiment", "verify", "checkpoint", "rollback", "complete", "failed"],
    })
    events_path = run_dir / "events.jsonl"
    if not config.resume:
        events_path.write_text("")
    _append_event(events_path, {"event": "queued", "run_id": config.run_id, "budget": config.budget})
    correctness_results = _run_correctness(config) if config.correctness_only and not config.dry_run else []
    rollbacks = []
    if config.simulate_failed_experiment:
        rollbacks.append({"experiment_id": "simulated", "reason": "simulated_failed_experiment"})
        _append_event(events_path, {"event": "rollback", "reason": "simulated_failed_experiment"})
    _append_event(events_path, {"event": "complete", "run_id": config.run_id})
    _append_event(events_path, {"event": "checkpoint", "run_id": config.run_id})
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    _write_json(checkpoint_dir / "last_green.json", {"run_id": config.run_id, "status": "checkpoint"})
    manifest = {
        "run_id": config.run_id,
        "created_unix": time.time(),
        "status": "dry_run_complete" if config.dry_run else "complete",
        "plan": ".omo/plans/20260610-inference-optimization-harness.md",
        "resumed": config.resume,
    }
    _write_json(run_dir / "manifest.json", manifest)
    report = build_report(config, correctness_results, rollbacks)
    write_reports(run_dir, report)
    return report


def _run_correctness(config: HarnessConfig) -> list[dict[str, str | int | float]]:
    commands = planned_correctness_commands(config.commands)
    results: list[dict[str, str | int | float]] = []
    for planned in commands:
        start = time.perf_counter()
        proc = subprocess.run(planned["command"], shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        elapsed = time.perf_counter() - start
        results.append({
            "name": planned["name"],
            "command": planned["command"],
            "exit_code": proc.returncode,
            "elapsed_seconds": elapsed,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        })
        if proc.returncode != 0:
            break
    return results


def _append_event(path: Path, event: dict[str, str]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps({"timestamp_unix": time.time(), **event}) + "\n")


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _cmd_text(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() or proc.stderr.strip() or None


def main() -> int:
    config = parse_args()
    report = write_run(config)
    failed = [item for item in report["correctness"]["results"] if item["exit_code"] != 0]
    print(json.dumps({"run_id": report["run_id"], "run_dir": str(config.run_dir), "status": "fail" if failed else "pass"}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
