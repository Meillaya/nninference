# Plan: production-grade autonomous inference optimization harness

Status: approved for planning output; implementation not started.
Created: 2026-06-10
Scope owner: nninference

## 1. Objective

Build a production-grade optimization harness around `nninference` that can autonomously run bounded optimization passes, preserve correctness, compare against local industry inference engines such as LM Studio, and produce frontier-lab-quality reports after each run.

The harness must support fixed wall-clock budgets such as `3h`, `6h`, `9h`, and `12h`, with resumable run state, checkpointed evidence, rollback to the last green state, and report artifacts suitable for performance review.

The Kimi K2.6 blog is an inspiration source for long-horizon autonomous optimization and report style, not a benchmark source to copy blindly. Its Qwen3.5-0.8B Mac/Zig case claims a move from about 15 tok/s to about 193 tok/s and about 20% faster than LM Studio, but this repo must only claim locally reproduced measurements.

## 2. Non-negotiable claim boundaries

Current repo state is not yet a full native transformer inference engine:

- CPU path: `src/main.zig` invokes `scripts/hf_prefill_bridge.py` through `uv run python`.
- Correctness gate: `scripts/run_alignment_tests.py` checks Hugging Face alignment for prompts:
  - `Hi,`
  - `The capital of China is`
  - `What is 1+1?`
- Metal path: opt-in LM-head sidecar through `src/metal_logits_test.zig`, `src/metal_bridge.h`, `src/metal_bridge.m`, and `metal/vector_add.metal`.
- Existing reports already warn that local metrics are sidecar/HF-bridge scoped, not full end-to-end token throughput.

Therefore every generated graph/report must separate:

1. End-to-end generation throughput, if measured.
2. Hugging Face bridge latency.
3. Zig process overhead.
4. Metal LM-head sidecar timings.
5. LM Studio local server/API metrics.
6. Any synthetic/microbenchmark timings.

No report may say “faster than LM Studio” unless the same local machine, model, quantization, prompt set, context, sampling config, and measurement window were used, and the result passed confidence gates.

## 3. External references to pin in implementation docs

- Kimi K2.6 blog: https://www.kimi.com/blog/kimi-k2-6
- LM Studio OpenAI-compatible endpoints: https://lmstudio.ai/docs/developer/openai-compat
- LM Studio local API server: https://lmstudio.ai/docs/developer/core/server
- LM Studio CLI: https://lmstudio.ai/docs/cli
- LM Studio `lms load`: https://lmstudio.ai/docs/cli/local-models/load
- LM Studio REST API endpoints/stats: https://lmstudio.ai/docs/developer/rest/endpoints

## 4. Recommended architecture

Add a harness layer in `scripts/` before changing inference internals.

Primary new entrypoint:

- `scripts/autoptimize.py`

Supporting modules should stay small and reviewable. Prefer a simple module split over a monolith once the file approaches project limits:

- `scripts/optimize_config.py` — typed config, budget presets, run IDs.
- `scripts/optimize_state.py` — event log, checkpoints, resume/rollback.
- `scripts/optimize_commands.py` — command runner with timeout, env, captured stdout/stderr.
- `scripts/optimize_metrics.py` — metric parsing, schema validation, confidence gates.
- `scripts/optimize_lmstudio.py` — optional LM Studio adapter.
- `scripts/optimize_report.py` — JSON/Markdown/SVG/HTML report generation.

Do not introduce non-`uv` Python dependency management. If dependencies are needed, add them through `uv` only and justify them in `worklog.md` during implementation. Prefer standard library first for the initial harness.

## 5. Run model

### 5.1 Budget tiers

Supported tiers:

- `3h` — shakedown/default implementation target.
- `6h` — medium optimization run.
- `9h` — long run with multiple optimization families.
- `12h` — maximum standard run, matching the long-horizon inspiration envelope.

Hard rules:

- 12h ceiling unless a future user explicitly authorizes longer.
- Every subprocess has its own timeout.
- Every optimization slice must end with verification or rollback.
- Every run is resumable from state, not just logs.

### 5.2 State machine

Each run uses this lifecycle:

```text
queued -> baseline -> experiment -> verify -> checkpoint -> continue|rollback|complete|failed
```

Persist all state under:

```text
artifacts/optimization/<run-id>/
```

Suggested tree:

```text
artifacts/optimization/<run-id>/
  config.json
  events.jsonl
  manifest.json
  baseline/
  experiments/
    <experiment-id>/
      patch.diff
      commands.jsonl
      metrics.json
      verdict.json
      stdout.log
      stderr.log
  checkpoints/
  lmstudio/
  reports/
    report.json
    report.md
    charts.svg
    report.html   # optional
```

## 6. Correctness gates

No performance result is promotable unless correctness passes.

Minimum gates:

1. Hugging Face alignment:
   - `uv run python scripts/run_alignment_tests.py`
   - Prompts must include `Hi,`, `The capital of China is`, `What is 1+1?`.
   - Prefill logits alignment must remain within the existing accepted tolerance.
   - Greedy next-token behavior must match Hugging Face behavior.
2. Zig build/test:
   - `zig build`
   - any existing Zig tests/build steps.
3. Sampling contract:
   - default `temperature=0.6`
   - default `top_p=0.95`
   - default `top_k=20`
   - greedy mode available for deterministic tests.
4. Report claim boundary:
   - sidecar metrics cannot be promoted as end-to-end token/sec.
   - LM Studio comparisons require local measurement metadata.

Future implementation should add regression tests before harness cleanup/refactors.

## 7. LM Studio comparator design

LM Studio support should be optional-but-first-class.

### 7.1 Detection

The harness checks:

- `lms --version`
- `lms ps`
- whether local server is reachable.
- whether the requested model is loaded.

If unavailable, the run does not fail by default. It emits:

```json
{
  "lmstudio": {
    "status": "skipped_with_reason",
    "reason": "lms CLI not found or local server unavailable"
  }
}
```

### 7.2 Fairness metadata

Each LM Studio comparison must record:

- LM Studio version.
- runtime/backend if available.
- model identifier.
- quantization / file name if available.
- context length.
- prompt set.
- max tokens.
- temperature/top-p/top-k or greedy mode.
- warmup count.
- measured runs count.
- whether model load time is included or excluded.
- tokens/sec, TTFT, total latency when exposed by API.

### 7.3 Comparison modes

Keep modes separate in charts:

- cold start including load.
- warm server generation.
- one-shot CLI invocation if used.
- OpenAI-compatible local server request.
- nninference HF-bridge mode.
- nninference native/sidecar modes.

Never mix those modes into one ranking without clear labeling.

## 8. Optimization loop policy

The autonomous loop should optimize only through safe, evidence-backed changes.

Per experiment:

1. Select one hypothesis.
2. Snapshot current state.
3. Apply the smallest viable change.
4. Build/test.
5. Run correctness gates.
6. Run benchmark suite with enough samples.
7. Compare against baseline and last green checkpoint.
8. Promote only if improvement and correctness thresholds pass.
9. Otherwise rollback and record failure reason.

Promotion criteria for performance-sensitive changes:

- correctness gates pass.
- no new product-code warnings/errors.
- measured median improvement exceeds noise threshold.
- minimum sample count met.
- benchmark mode is comparable to baseline.
- report schema validation passes.

Initial threshold recommendation:

- Promote if median improves by at least 3% on the target metric and no secondary metric regresses by more than 5%, unless the experiment is explicitly marked exploratory.

## 9. Metrics schema

Use stable JSON so reports and future agents can consume results.

Required high-level fields:

```json
{
  "run_id": "...",
  "created_at": "...",
  "budget": "3h|6h|9h|12h",
  "machine": {},
  "git": {},
  "model": {},
  "correctness": {},
  "benchmarks": [],
  "comparators": {},
  "promotions": [],
  "rollbacks": [],
  "known_gaps": []
}
```

For every benchmark row:

```json
{
  "name": "...",
  "scope": "end_to_end|hf_bridge|zig_process|metal_sidecar|lmstudio|microbenchmark",
  "mode": "cold|warm|reuse|server|cli|fixture",
  "prompt": "...",
  "samples": 0,
  "median_ms": 0.0,
  "p50_ms": 0.0,
  "p95_ms": 0.0,
  "tokens_per_second": null,
  "ttft_seconds": null,
  "correctness_ref": "...",
  "confidence": "low|medium|high",
  "claim_allowed": false
}
```

## 10. Report requirements

Reports must be generated after each run and after final completion.

Required artifacts:

- `report.json` — machine-readable complete result.
- `report.md` — human-readable narrative.
- `charts.svg` — portable chart bundle.
- `report.html` — optional but preferred for frontier-lab-quality presentation.

Required report sections:

1. Executive summary.
2. What was optimized.
3. Claim boundaries and comparability warnings.
4. Hardware/software/model environment.
5. Baseline metrics.
6. Optimization timeline.
7. Promoted changes.
8. Rejected/rolled-back changes and reasons.
9. Correctness evidence.
10. LM Studio comparison, if available.
11. Charts:
    - throughput over time.
    - latency distribution.
    - TTFT if available.
    - speedup waterfall.
    - correctness pass/fail timeline.
    - experiment promotion/rollback timeline.
12. Known gaps.
13. Reproduction commands.

Report style should emulate a frontier-lab benchmark writeup: visual, structured, precise, but conservative about claims.

## 11. Test plan for implementation

Add tests before broad implementation.

Recommended test IDs:

- `tests/test_alignment_contract.py::test_canonical_prompts_prefill_logits_and_greedy_match_hf`
- `tests/test_cli_sampling_contract.py::test_infer_cpu_v1_reports_default_sampling_params`
- `tests/test_benchmark_validation.py::test_reuse_report_rejects_missing_correctness_fields`
- `tests/test_command_mode_fairness.py::test_command_mode_report_keeps_rankings_separate`
- `tests/test_summary_validation.py::test_summary_rejects_stale_or_malformed_nested_evidence`
- `tests/test_industry_comparison.py::test_report_labels_local_metrics_as_sidecar_not_token_throughput`
- `tests/test_autoptimize_budget.py::test_budget_presets_parse_and_enforce_12h_ceiling`
- `tests/test_autoptimize_resume.py::test_run_state_can_resume_from_events_jsonl`
- `tests/test_autoptimize_rollback.py::test_failed_experiment_restores_last_green_checkpoint`
- `tests/test_lmstudio_adapter.py::test_lmstudio_unavailable_emits_skipped_with_reason`
- `tests/test_report_artifacts.py::test_report_contains_required_sections_and_charts`

If pytest is not currently configured, add the smallest uv-managed test setup needed and document it.

## 12. Implementation milestones

Each tested milestone should be committed using the repository Lore commit protocol and documented in `worklog.md`.

### Milestone 0 — Baseline preservation

- Verify current alignment/build/benchmark commands.
- Record baseline evidence in `worklog.md`.
- Commit only if repository policy requires tracking planning/baseline artifacts.

Exit criteria:

- baseline commands documented.
- no product-code changes.
- current correctness artifacts understood.

### Milestone 1 — Harness skeleton

- Add `scripts/autoptimize.py` with `--budget`, `--run-id`, `--dry-run`, `--resume`.
- Add state/event writing under `artifacts/optimization/<run-id>/`.
- Add unit tests for budget parsing and run-state creation.

Exit criteria:

- dry run creates valid config/manifest/events.
- `3h/6h/9h/12h` parse correctly.
- 12h ceiling enforced.

### Milestone 2 — Command runner and correctness gates

- Wrap existing alignment/build commands.
- Capture stdout/stderr/exit code/duration.
- Fail fast on correctness failures.
- Add stale-artifact protections.

Exit criteria:

- alignment and Zig gates run through harness.
- failures are recorded in structured JSON.
- product code unchanged unless required by tests.

### Milestone 3 — Benchmark ingestion

- Run existing benchmark/reporting scripts through harness.
- Normalize metrics into schema.
- Add confidence/noise fields.
- Prevent sidecar metrics from being reported as end-to-end throughput.

Exit criteria:

- report JSON validates.
- existing Kimi-style local warnings preserved.

### Milestone 4 — LM Studio adapter

- Detect `lms` and local server.
- Support skipped-with-reason behavior.
- If available, run a small prompt set through local API.
- Capture LM Studio metadata and stats.

Exit criteria:

- tests pass both unavailable and available/manual paths.
- report clearly separates LM Studio modes.

### Milestone 5 — Report generation

- Generate Markdown and SVG charts.
- Optionally generate HTML report.
- Include all required sections and reproduction commands.

Exit criteria:

- report artifacts render/read cleanly.
- charts use separated modes and claim boundaries.

### Milestone 6 — Autonomous optimization loop

- Implement experiment/checkpoint/rollback lifecycle.
- Add promotion thresholds.
- Support resume after interruption.
- Keep patches small and reversible.

Exit criteria:

- synthetic failing experiment rolls back.
- passing experiment promotes only after correctness and benchmark gates.
- run can resume from events.

### Milestone 7 — First 3h shakedown

- Run the `3h` tier.
- Produce full artifact bundle.
- Review results and known gaps.

Exit criteria:

- full report produced.
- no false performance claims.
- all verification evidence recorded in `worklog.md`.

## 13. Manual QA plan

Use these channels only when relevant:

- Terminal/tmux: run harness and watch progress/logs.
- HTTP: query LM Studio local server if comparator is enabled.
- Browser: inspect generated HTML report if implemented.

Manual QA scenarios:

1. Run dry run with `--budget 3h --dry-run`.
2. Run correctness-only mode.
3. Simulate missing LM Studio.
4. Simulate failed benchmark schema.
5. Simulate interrupted run and resume.
6. Simulate experiment failure and rollback.
7. Inspect final Markdown/SVG/HTML report.

## 14. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Stale artifacts treated as fresh | Run-specific directories, timestamps, manifest validation. |
| Misleading LM Studio comparison | Mandatory metadata, separated modes, local-only claims. |
| Sidecar metrics presented as full inference | Schema `scope` field and report claim gates. |
| Long run wastes time after first failure | Per-slice timeouts and fail-fast correctness gates. |
| Autonomy makes broad risky changes | One-hypothesis experiments, small diffs, rollback. |
| Benchmark noise | Sample thresholds, medians/p95, promotion threshold. |
| Dependency sprawl | Standard library first; uv-only dependencies if justified. |
| Reports look polished but lack evidence | Report schema requires command logs, correctness refs, and reproduction commands. |

## 15. Open decisions before implementation

Recommended defaults are already selected, but user may override before implementation:

1. Default first run: `3h` shakedown.
2. LM Studio comparator: optional-but-first-class, skipped when unavailable.
3. Scope order: harness/reporting first, deeper native transformer work later.
4. Benchmark positioning: compare only against locally measured LM Studio, not the Kimi blog number unless exact config is reproduced.

## 16. Definition of done for implementation

Implementation is complete only when:

- `scripts/autoptimize.py` or equivalent exists and is documented.
- Budget tiers `3h/6h/9h/12h` work.
- Correctness gates run from the harness.
- LM Studio adapter handles available and unavailable cases.
- Reports produce JSON + Markdown + SVG/HTML artifacts.
- Claim-boundary checks prevent misleading comparisons.
- A 3h shakedown or explicit shorter smoke-equivalent has been run.
- `worklog.md` records commands, decisions, verification, and known gaps.
- Relevant Python/Zig tests pass.
- Milestone commits use Lore protocol.

## 17. Immediate next action

Start implementation at Milestone 0 and Milestone 1:

1. Re-run baseline alignment/build checks.
2. Add minimal tests for budget parsing and state creation.
3. Implement harness skeleton with dry-run.
4. Verify and commit the tested milestone.

