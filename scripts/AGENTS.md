# scripts/ KNOWLEDGE

## OVERVIEW

`scripts/` is a uv-run Python CLI collection for model download, HF reference inference, alignment gates, fixture generation, and Metal benchmark orchestration.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Model download | `download_model.py` | Downloads `Qwen/Qwen3.5-0.8B` into the root snapshot dir. |
| HF prefill bridge | `hf_prefill_bridge.py` | Contract consumed by `src/main.zig`; emits metadata and candidates. |
| Required alignment gate | `run_alignment_tests.py` | Fixed prompts, logits diff, greedy next-token checks. |
| Bridge baseline timing | `benchmark_bridge.py` | Benchmarks current HF bridge path. |
| Metal fixture generation | `generate_checkpoint_logits_fixture.py` | Produces `NNLGFIX1` row-major fixture data. |
| Metal benchmark wrappers | `benchmark_metal_logits*.py`, `benchmark_metal_command_modes.py` | Spawn Zig/Metal test binary and validate JSON/NDJSON. |
| Benchmark synthesis | `summarize_metal_benchmarks.py`, `consolidate_benchmark_findings.py` | Read artifact JSON and write summary evidence. |

## CONVENTIONS

- Run Python through `uv run python ...` for dependency-aware execution.
- Keep scripts executable as CLIs with explicit arguments; there is no package import surface to preserve.
- Prefer Python standard library for report/synthesis helpers unless a dependency is already in `pyproject.toml`.
- Alignment prompts are canonical: `Hi,`, `The capital of China is`, `What is 1+1?`.
- Benchmark scripts write JSON evidence under `artifacts/benchmarks/`; alignment writes under `artifacts/alignment/`.
- Wrapper scripts must fail fast with explicit `verdict` / `failure_reasons` when mode, buffer, kernel, count, or positive-timing invariants drift.
- Preserve timing schema checks, especially top-k timing component consistency and `metal_logits_v1` field semantics.
- Prefer named milestone outputs with `--out` and `--artifact-dir`; do not overwrite canonical baselines for exploratory runs.

## ANTI-PATTERNS

- Do not add pip/requirements/poetry flows; update `pyproject.toml` and `uv.lock` with uv if dependencies change.
- Do not weaken alignment comparisons to make a run pass.
- Do not hide setup, fixture load, process startup, or top-k host selection time inside Metal kernel timing.
- Do not describe local LM-head sidecar benchmark numbers as end-to-end Qwen token/s throughput.
- Do not make nested benchmark wrappers silently change the defaults of the underlying benchmark script.

## REQUIRED CHECKS BY CHANGE TYPE

```bash
python3 -m py_compile scripts/*.py
uv run python scripts/run_alignment_tests.py
uv run python scripts/summarize_metal_benchmarks.py
```

Use targeted benchmark commands from `worklog.md` or the edited script's `--help` for performance changes.
