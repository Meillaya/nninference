from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTOPTIMIZE = ROOT / "scripts" / "autoptimize.py"
REQUIRED_CHARTS = (
    "Throughput over time",
    "Latency distribution",
    "TTFT",
    "Speedup waterfall",
    "Correctness pass/fail timeline",
    "Promotion/rollback timeline",
)


class AutoptimizeLifecycleReportTestCase(unittest.TestCase):
    def run_autoptimize(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(AUTOPTIMIZE), *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def assert_success(self, proc: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


class TestLifecycleEvents(AutoptimizeLifecycleReportTestCase):
    def test_run_events_record_required_lifecycle_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "optimization"

            proc = self.run_autoptimize("--budget", "3h", "--run-id", "lifecycle", "--dry-run", "--output-dir", str(out_dir))
            self.assert_success(proc)

            events = [json.loads(line)["event"] for line in (out_dir / "lifecycle" / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events, ["queued", "baseline", "experiment", "verify", "checkpoint", "complete"])

    def test_failed_experiment_records_rollback_before_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "optimization"

            proc = self.run_autoptimize(
                "--budget",
                "3h",
                "--run-id",
                "rollback-lifecycle",
                "--dry-run",
                "--simulate-failed-experiment",
                "--output-dir",
                str(out_dir),
            )
            self.assert_success(proc)

            events = [json.loads(line)["event"] for line in (out_dir / "rollback-lifecycle" / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events, ["queued", "baseline", "experiment", "verify", "rollback", "checkpoint", "complete"])


class TestRequiredChartSections(AutoptimizeLifecycleReportTestCase):
    def test_report_artifacts_name_every_required_chart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "optimization"

            proc = self.run_autoptimize("--dry-run", "--run-id", "charts", "--output-dir", str(out_dir))
            self.assert_success(proc)

            reports = out_dir / "charts" / "reports"
            report_md = (reports / "report.md").read_text()
            charts_svg = (reports / "charts.svg").read_text()
            report_html = (reports / "report.html").read_text()
            for chart in REQUIRED_CHARTS:
                self.assertIn(chart, report_md)
                self.assertIn(chart, charts_svg)
                self.assertIn(chart, report_html)


if __name__ == "__main__":
    unittest.main()
