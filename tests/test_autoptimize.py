from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTOPTIMIZE = ROOT / "scripts" / "autoptimize.py"


class AutoptimizeCliTestCase(unittest.TestCase):
    def run_autoptimize(self, *args: str, path: str | None = None) -> subprocess.CompletedProcess[str]:
        env = None
        if path is not None:
            env = {"PATH": path}
        return subprocess.run(
            [sys.executable, str(AUTOPTIMIZE), *args],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def assert_success(self, proc: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


class TestDryRunCreatesArtifacts(AutoptimizeCliTestCase):
    def test_dry_run_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "optimization"
            proc = self.run_autoptimize(
                "--budget",
                "3h",
                "--run-id",
                "ulw-smoke",
                "--dry-run",
                "--output-dir",
                str(out_dir),
            )
            self.assert_success(proc)

            run_dir = out_dir / "ulw-smoke"
            expected = [
                "config.json",
                "manifest.json",
                "events.jsonl",
                "reports/report.json",
                "reports/report.md",
                "reports/charts.svg",
                "reports/report.html",
            ]
            for relative in expected:
                self.assertTrue((run_dir / relative).exists(), msg=f"missing {relative}")

            config = json.loads((run_dir / "config.json").read_text())
            self.assertEqual(config["budget"], "3h")
            self.assertEqual(config["budget_seconds"], 10_800)
            self.assertTrue(config["dry_run"])


class TestBudgetParsing(AutoptimizeCliTestCase):
    def test_budget_presets_parse_and_enforce_12h_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "optimization"
            for budget, seconds in (("3h", 10_800), ("6h", 21_600), ("9h", 32_400), ("12h", 43_200)):
                proc = self.run_autoptimize(
                    "--budget",
                    budget,
                    "--run-id",
                    f"run-{budget}",
                    "--dry-run",
                    "--output-dir",
                    str(out_dir),
                )
                self.assert_success(proc)
                config = json.loads((out_dir / f"run-{budget}" / "config.json").read_text())
                self.assertEqual(config["budget_seconds"], seconds)

            rejected = self.run_autoptimize("--budget", "13h", "--dry-run", "--output-dir", str(out_dir))
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("budget", rejected.stderr.lower())
            self.assertIn("12h", rejected.stderr)


class TestLMStudioMissing(AutoptimizeCliTestCase):
    def test_missing_lm_studio_returns_skipped_with_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "optimization"
            proc = self.run_autoptimize(
                "--budget",
                "3h",
                "--run-id",
                "lmstudio-skip",
                "--dry-run",
                "--output-dir",
                str(out_dir),
                path="/usr/bin:/bin",
            )
            self.assert_success(proc)
            report = json.loads((out_dir / "lmstudio-skip" / "reports" / "report.json").read_text())
            self.assertEqual(report["comparators"]["lmstudio"]["status"], "skipped_with_reason")
            self.assertIn("lms", report["comparators"]["lmstudio"]["reason"].lower())


class TestClaimBoundaries(AutoptimizeCliTestCase):
    def test_correctness_only_records_alignment_and_zig_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "optimization"
            proc = self.run_autoptimize(
                "--budget",
                "3h",
                "--run-id",
                "correctness-plan",
                "--dry-run",
                "--correctness-only",
                "--output-dir",
                str(out_dir),
            )
            self.assert_success(proc)
            report = json.loads((out_dir / "correctness-plan" / "reports" / "report.json").read_text())
            commands = [item["command"] for item in report["correctness"]["planned_commands"]]
            self.assertIn("uv run python scripts/run_alignment_tests.py", commands)
            self.assertIn("zig build", commands)
            scopes = {row["scope"] for row in report["benchmarks"]}
            self.assertIn("hf_bridge", scopes)
            self.assertIn("metal_sidecar", scopes)
            self.assertTrue(report["claim_boundaries"]["requires_local_lmstudio_measurement"])
            self.assertNotIn("faster than LM Studio", (out_dir / "correctness-plan" / "reports" / "report.md").read_text())


class TestReportArtifacts(AutoptimizeCliTestCase):
    def test_report_contains_required_sections_and_claim_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "optimization"
            proc = self.run_autoptimize("--dry-run", "--output-dir", str(out_dir), "--run-id", "report")
            self.assert_success(proc)
            report_md = (out_dir / "report" / "reports" / "report.md").read_text()
            for section in ("Executive summary", "Claim boundaries", "Correctness evidence", "Known gaps"):
                self.assertIn(section, report_md)
            self.assertIn("HF bridge", report_md)
            self.assertIn("Metal LM-head sidecar", report_md)
            self.assertIn("not full native transformer", report_md)


class TestWorklogContract(unittest.TestCase):
    def test_worklog_mentions_autoptimize_artifacts_and_known_gaps(self) -> None:
        worklog = (ROOT / "worklog.md").read_text()
        self.assertIn("autoptimize", worklog.lower())
        self.assertIn("artifacts/optimization", worklog)
        self.assertIn("known gaps", worklog.lower())


if __name__ == "__main__":
    unittest.main()

class TestLifecycleContracts(AutoptimizeCliTestCase):
    def test_resume_reuses_existing_event_log_and_records_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "optimization"
            first = self.run_autoptimize("--budget", "3h", "--run-id", "resume", "--dry-run", "--output-dir", str(out_dir))
            self.assert_success(first)
            second = self.run_autoptimize("--budget", "3h", "--run-id", "resume", "--dry-run", "--resume", "--output-dir", str(out_dir))
            self.assert_success(second)
            events = (out_dir / "resume" / "events.jsonl").read_text().splitlines()
            self.assertGreaterEqual(len(events), 2)
            self.assertEqual(json.loads(events[-1])["event"], "checkpoint")

    def test_failed_experiment_can_record_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "optimization"
            proc = self.run_autoptimize(
                "--budget",
                "3h",
                "--run-id",
                "rollback",
                "--dry-run",
                "--simulate-failed-experiment",
                "--output-dir",
                str(out_dir),
            )
            self.assert_success(proc)
            report = json.loads((out_dir / "rollback" / "reports" / "report.json").read_text())
            self.assertEqual(report["rollbacks"][0]["reason"], "simulated_failed_experiment")
            self.assertGreaterEqual(report["promotion_thresholds"]["median_improvement_pct"], 3.0)


class TestCorrectnessExecution(AutoptimizeCliTestCase):
    def test_correctness_only_run_executes_commands_when_not_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "optimization"
            proc = self.run_autoptimize(
                "--budget",
                "3h",
                "--run-id",
                "correctness-run",
                "--correctness-only",
                "--command",
                f"{sys.executable} -c 'print(42)'",
                "--output-dir",
                str(out_dir),
            )
            self.assert_success(proc)
            report = json.loads((out_dir / "correctness-run" / "reports" / "report.json").read_text())
            results = report["correctness"]["results"]
            self.assertEqual(results[0]["exit_code"], 0)
            self.assertIn("42", results[0]["stdout"])


class TestLMStudioAvailableMetadata(AutoptimizeCliTestCase):
    def test_fake_lm_studio_cli_records_version_and_process_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake = tmp_path / "lms"
            fake.write_text("#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'lms 1.2.3'; exit 0; fi\nif [ \"$1\" = \"ps\" ]; then echo 'qwen-loaded'; exit 0; fi\nexit 0\n")
            fake.chmod(0o755)
            out_dir = tmp_path / "optimization"
            proc = self.run_autoptimize("--dry-run", "--run-id", "lmstudio-available", "--output-dir", str(out_dir), path=f"{tmp_path}:/usr/bin:/bin")
            self.assert_success(proc)
            lmstudio = json.loads((out_dir / "lmstudio-available" / "reports" / "report.json").read_text())["comparators"]["lmstudio"]
            self.assertEqual(lmstudio["status"], "available_unmeasured")
            self.assertEqual(lmstudio["version"], "lms 1.2.3")
            self.assertIn("qwen-loaded", lmstudio["loaded_models"])


class TestRequiredReportSchema(AutoptimizeCliTestCase):
    def test_report_json_contains_required_plan_schema_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "optimization"
            proc = self.run_autoptimize("--dry-run", "--run-id", "schema", "--output-dir", str(out_dir))
            self.assert_success(proc)
            report = json.loads((out_dir / "schema" / "reports" / "report.json").read_text())
            for key in ("created_at", "machine", "git", "model", "promotions", "rollbacks", "promotion_thresholds"):
                self.assertIn(key, report)
            row = report["benchmarks"][0]
            for key in ("samples", "median_ms", "p50_ms", "p95_ms", "tokens_per_second", "ttft_seconds", "correctness_ref", "confidence"):
                self.assertIn(key, row)
