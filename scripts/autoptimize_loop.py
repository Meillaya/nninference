#!/usr/bin/env python3
# ─── How to run ───
# uv run python scripts/autoptimize_loop.py --budget 3h --run-id opt-3h

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

BUDGET_SECONDS: Final[dict[str, int]] = {
    "3h": 10_800,
    "6h": 21_600,
    "9h": 32_400,
    "12h": 43_200,
}
DEFAULT_OUTPUT_DIR: Final = Path("artifacts/optimization")
DEFAULT_INTERVAL_SECONDS: Final = 300.0
DEFAULT_COMMANDS: Final[tuple[str, ...]] = (
    "zig build",
    "uv run python scripts/run_alignment_tests.py",
    "uv run python scripts/benchmark_bridge.py --warmup 0 --repeats 1",
    "uv run python scripts/benchmark_metal_logits_reuse.py --no-build --samples 1 "
    "--kernels scalar,threadgroup,threadgroup128 --buffer-mode copy --compare-mode topk "
    "--kernel-repeats 3 --benchmark-iters 4 --out {iteration_dir}/metal_reuse.json",
)


@dataclass(frozen=True, slots=True)
class LoopConfig:
    budget: str
    budget_seconds: int
    run_id: str
    output_dir: Path
    interval_seconds: float
    max_iterations: int | None
    commands: tuple[str, ...]

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_id


def parse_args() -> LoopConfig:
    parser = argparse.ArgumentParser(description="Long-running autonomous optimization loop runner.")
    parser.add_argument("--budget", default="3h", choices=tuple(BUDGET_SECONDS))
    parser.add_argument("--run-id", default=f"opt-{int(time.time())}")
    parser.add_argument("--output-dir", "--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--command", action="append", default=[])
    args = parser.parse_args()
    budget = str(args.budget)
    commands = tuple(args.command) if args.command else DEFAULT_COMMANDS
    return LoopConfig(
        budget=budget,
        budget_seconds=BUDGET_SECONDS[budget],
        run_id=str(args.run_id),
        output_dir=Path(str(args.output_dir)),
        interval_seconds=max(float(args.interval_seconds), 0.0),
        max_iterations=args.max_iterations,
        commands=commands,
    )


def run_loop(config: LoopConfig) -> int:
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    events_path.write_text("")
    _write_json(run_dir / "config.json", _config_json(config))
    started = time.time()
    deadline = started + config.budget_seconds
    _append_event(events_path, {"event": "queued", "run_id": config.run_id, "budget": config.budget})
    _append_event(events_path, {"event": "baseline", "status": "capturing", "command_count": len(config.commands)})
    _write_manifest(config, "running", started, 0)
    iteration = 0
    while time.time() < deadline and _can_iterate(config, iteration):
        iteration += 1
        status = _run_iteration(config, events_path, iteration)
        if status != "pass":
            _append_event(events_path, {"event": "rollback", "iteration": iteration, "reason": "command_failed"})
            _append_event(events_path, {"event": "failed", "run_id": config.run_id, "iteration": iteration})
            _write_manifest(config, "failed", started, iteration)
            _write_report(config, "failed", started, iteration)
            return 1
        _append_event(events_path, {"event": "checkpoint", "run_id": config.run_id, "iteration": iteration})
        _write_checkpoint(config, iteration)
        _write_manifest(config, "running", started, iteration)
        if _can_iterate(config, iteration) and time.time() < deadline:
            _sleep_until_next_iteration(config, events_path, deadline)
    _append_event(events_path, {"event": "complete", "run_id": config.run_id, "iterations": iteration})
    _write_manifest(config, "complete", started, iteration)
    _write_report(config, "complete", started, iteration)
    print(json.dumps({"run_id": config.run_id, "run_dir": str(run_dir), "status": "complete", "iterations": iteration}, indent=2))
    return 0


def _run_iteration(config: LoopConfig, events_path: Path, iteration: int) -> str:
    iteration_dir = config.run_dir / "iterations" / f"{iteration:04d}"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    _append_event(events_path, {"event": "experiment", "iteration": iteration, "status": "started"})
    results = []
    for index, template in enumerate(config.commands):
        command = template.format(run_id=config.run_id, run_dir=config.run_dir, iteration=iteration, iteration_dir=iteration_dir)
        start = time.perf_counter()
        proc = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        elapsed = time.perf_counter() - start
        result = {
            "index": index,
            "command": command,
            "exit_code": proc.returncode,
            "elapsed_seconds": elapsed,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
        results.append(result)
        _append_event(events_path, {"event": "command", "iteration": iteration, "index": index, "exit_code": proc.returncode})
        if proc.returncode != 0:
            break
    status = "pass" if all(item["exit_code"] == 0 for item in results) else "fail"
    _write_json(iteration_dir / "commands.json", {"iteration": iteration, "status": status, "commands": results})
    _append_event(events_path, {"event": "verify", "iteration": iteration, "status": status, "command_count": len(results)})
    return status


def _sleep_until_next_iteration(config: LoopConfig, events_path: Path, deadline: float) -> None:
    if config.interval_seconds <= 0:
        return
    sleep_seconds = min(config.interval_seconds, max(deadline - time.time(), 0.0))
    _append_event(events_path, {"event": "heartbeat", "sleep_seconds": round(sleep_seconds, 3)})
    time.sleep(sleep_seconds)


def _can_iterate(config: LoopConfig, completed_iterations: int) -> bool:
    return config.max_iterations is None or completed_iterations < config.max_iterations


def _config_json(config: LoopConfig) -> dict[str, object]:
    return {
        "budget": config.budget,
        "budget_seconds": config.budget_seconds,
        "run_id": config.run_id,
        "interval_seconds": config.interval_seconds,
        "max_iterations": config.max_iterations,
        "commands": list(config.commands),
        "state_machine": ["queued", "baseline", "experiment", "command", "verify", "checkpoint", "rollback", "complete", "failed"],
    }


def _write_manifest(config: LoopConfig, status: str, started: float, iterations_completed: int) -> None:
    _write_json(config.run_dir / "manifest.json", {
        "run_id": config.run_id,
        "status": status,
        "started_unix": started,
        "updated_unix": time.time(),
        "iterations_completed": iterations_completed,
        "budget_seconds": config.budget_seconds,
    })


def _write_checkpoint(config: LoopConfig, iteration: int) -> None:
    checkpoint_dir = config.run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    _write_json(checkpoint_dir / "last_green.json", {"run_id": config.run_id, "iteration": iteration, "status": "checkpoint"})


def _write_report(config: LoopConfig, status: str, started: float, iterations_completed: int) -> None:
    report_dir = config.run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "run_id": config.run_id,
        "status": status,
        "iterations_completed": iterations_completed,
        "elapsed_seconds": time.time() - started,
        "budget_seconds": config.budget_seconds,
        "machine": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor()},
    }
    _write_json(report_dir / "summary.json", report)
    (report_dir / "report.md").write_text(_report_markdown(report) + "\n")


def _report_markdown(report: Mapping[str, object]) -> str:
    elapsed = report["elapsed_seconds"]
    if not isinstance(elapsed, int | float):
        raise TypeError("elapsed_seconds must be numeric")
    return "\n".join([
        f"# Optimization loop {report['run_id']}",
        "",
        f"- Status: `{report['status']}`",
        f"- Iterations completed: `{report['iterations_completed']}`",
        f"- Elapsed seconds: `{float(elapsed):.3f}`",
        f"- Budget seconds: `{report['budget_seconds']}`",
    ])


def _append_event(path: Path, event: dict[str, str | int | float]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps({"timestamp_unix": time.time(), **event}) + "\n")


def _write_json(path: Path, data: Mapping[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> int:
    return run_loop(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
