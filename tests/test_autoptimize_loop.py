from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOP = ROOT / "scripts" / "autoptimize_loop.py"


class AutoptimizeLoopTestCase(unittest.TestCase):
    def run_loop(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(LOOP), *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def assert_success(self, proc: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(proc.returncode, 0, msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


class TestAutoptimizeLoop(AutoptimizeLoopTestCase):
    def test_loop_runs_multiple_iterations_and_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "optimization"

            proc = self.run_loop(
                "--budget",
                "3h",
                "--run-id",
                "loop-pass",
                "--output-dir",
                str(out_dir),
                "--max-iterations",
                "2",
                "--interval-seconds",
                "0",
                "--command",
                f"{sys.executable} -c 'print(123)'",
            )
            self.assert_success(proc)

            run_dir = out_dir / "loop-pass"
            events = [json.loads(line)["event"] for line in (run_dir / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events.count("experiment"), 2)
            self.assertEqual(events.count("verify"), 2)
            self.assertEqual(events.count("checkpoint"), 2)
            self.assertEqual(events[-1], "complete")
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["iterations_completed"], 2)

    def test_loop_rolls_back_and_fails_on_command_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "optimization"

            proc = self.run_loop(
                "--budget",
                "3h",
                "--run-id",
                "loop-fail",
                "--output-dir",
                str(out_dir),
                "--max-iterations",
                "2",
                "--interval-seconds",
                "0",
                "--command",
                f"{sys.executable} -c 'raise SystemExit(7)'",
            )
            self.assertNotEqual(proc.returncode, 0)

            run_dir = out_dir / "loop-fail"
            events = [json.loads(line)["event"] for line in (run_dir / "events.jsonl").read_text().splitlines()]
            self.assertIn("rollback", events)
            self.assertEqual(events[-1], "failed")
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")


if __name__ == "__main__":
    unittest.main()
