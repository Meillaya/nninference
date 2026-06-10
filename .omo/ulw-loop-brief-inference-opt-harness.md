Execute .omo/plans/20260610-inference-optimization-harness.md to completion.

User-visible deliverable: a production-grade autonomous inference optimization harness for nninference, implemented with uv-managed Python, verified by tests and real CLI/tmux QA, with reports and run artifacts.

Success criteria:
1. Harness CLI and state: `uv run python scripts/autoptimize.py --budget 3h --run-id ulw-smoke --dry-run --output-dir artifacts/optimization` creates a run directory with config.json, manifest.json, events.jsonl, reports/report.json, reports/report.md, charts.svg, and report.html without touching model/inference code. Automated test first: tests/test_autoptimize_budget.py::test_budget_presets_parse_and_enforce_12h_ceiling and tests/test_autoptimize_dry_run.py::test_dry_run_creates_run_artifacts. Manual QA channel: tmux transcript at .omo/ulw-loop/evidence/harness-dry-run-tmux.txt.
2. LM Studio comparator unavailable path: when PATH is forced to omit `lms`, the harness records `lmstudio.status=skipped_with_reason` and does not fail. Automated test first: tests/test_lmstudio_adapter.py::test_lmstudio_unavailable_emits_skipped_with_reason. Manual QA channel: tmux transcript at .omo/ulw-loop/evidence/lmstudio-skip-tmux.txt.
3. Report claim boundaries: generated reports label current local metrics as HF bridge / Metal sidecar / microbenchmark, never unqualified end-to-end or faster-than-LM-Studio. Automated test first: tests/test_report_artifacts.py::test_report_contains_required_sections_and_claim_boundaries. Manual QA channel: tmux transcript at .omo/ulw-loop/evidence/report-boundary-tmux.txt.
4. Regression gates: existing alignment/build surfaces remain runnable and the harness records them as gated commands in dry-run/correctness-only mode without changing sampling defaults. Automated test first: tests/test_autoptimize_correctness.py::test_correctness_plan_includes_alignment_and_zig_build_commands. Manual QA channel: tmux transcript at .omo/ulw-loop/evidence/correctness-plan-tmux.txt.
5. Documentation and audit: worklog.md documents commands, decisions, verification, known gaps, and artifacts; each verified milestone is committed using repository commit style and Lore trailers. Automated test first: tests/test_worklog_contract.py::test_worklog_mentions_autoptimize_artifacts_and_known_gaps. Manual QA channel: tmux transcript at .omo/ulw-loop/evidence/worklog-contract-tmux.txt.

Constraints:
- No non-uv Python dependency management.
- Test-first for production changes; capture RED and GREEN outputs.
- Keep changes small/reversible.
- Use optional-but-first-class LM Studio adapter; missing LM Studio is skip-with-reason.
- Preserve claim boundaries from the approved plan.
- Run final review gate before completion.
