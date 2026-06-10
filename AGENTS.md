# PROJECT KNOWLEDGE BASE

**Generated:** 2026-06-10T21:36:17Z
**Commit:** f3cc27e
**Branch:** ultragoal/kimi-metal-iteration

## OVERVIEW

`nninference` is a Zig CLI plus Python/Hugging Face reference harness for `Qwen/Qwen3.5-0.8B` first-token inference alignment. CPU inference currently goes through a Zig `infer_cpu_v1` surface with a Python HF prefill bridge; Metal work is an opt-in LM-head/logits projection sidecar, not full native transformer inference.

## STRUCTURE

```text
.
├── src/                 # Zig CLI, Metal test binary, Objective-C Metal bridge
├── scripts/             # uv-run Python reference, alignment, fixture, benchmark CLIs
├── metal/               # Metal kernels compiled only when -Denable-metal=true
├── docs/                # design/audit notes for Metal session and packed-layout work
├── artifacts/           # generated alignment, fixture, benchmark evidence (ignored)
├── Qwen3.5-0.8B/        # local HF model snapshot
├── build.zig            # Zig build graph and custom Metal steps
├── pyproject.toml       # uv-managed Python dependency surface
├── uv.lock              # pinned Python dependency lock
└── worklog.md           # running process, command, decision, verification log
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| `infer_cpu_v1` CLI, flags, sampling | `src/main.zig` | Defaults: `temperature=0.6`, `top_p=0.95`, `top_k=20`; `--greedy` aligns deterministic checks. |
| HF reference prefill and candidates | `scripts/hf_prefill_bridge.py` | Called by Zig via `uv run python ...`; emits metadata plus candidates. |
| Required alignment gate | `scripts/run_alignment_tests.py` | Builds Zig, checks fixed prompts, writes `artifacts/alignment/report.json`. |
| Model download | `scripts/download_model.py` | Downloads `Qwen/Qwen3.5-0.8B` into `Qwen3.5-0.8B/`. |
| Zig build targets | `build.zig` | `zig build`; opt-in `-Denable-metal=true` for Metal targets. |
| Metal bridge/runtime | `src/metal_bridge.m`, `src/metal_bridge.h` | Objective-C ownership and buffer lifetime rules matter. |
| Metal logits validation | `src/metal_logits_test.zig` | Produces JSON/NDJSON used by benchmark scripts. |
| Metal kernels | `metal/vector_add.metal` | Rebuild metallib after edits via `zig build -Denable-metal=true metal-lib`. |
| Benchmark evidence | `artifacts/benchmarks/` | Generated evidence; do not hand-edit to fake results. |
| Decision history | `worklog.md` | Record commands, outcomes, known gaps, rollback decisions. |

## CODE MAP

| Symbol / Surface | Type | Location | Role |
|------------------|------|----------|------|
| `main` | Zig CLI | `src/main.zig` | Parses CLI flags, calls HF bridge, selects next token. |
| `runBridge` | Zig subprocess | `src/main.zig` | Executes `uv run python scripts/hf_prefill_bridge.py`. |
| `selectToken` | Zig sampler | `src/main.zig` | Greedy or temperature/top-p/top-k candidate selection. |
| `main` | Python harness | `scripts/run_alignment_tests.py` | Required logits + greedy alignment gate. |
| `runCase` | Zig Metal test | `src/metal_logits_test.zig` | Runs Metal LM-head projection and JSON comparison report. |
| `run_metal` / bridge APIs | Objective-C | `src/metal_bridge.m` | Metal command/buffer/session execution. |

## ACTIVE TASK REQUIREMENTS

- Download Hugging Face model `Qwen/Qwen3.5-0.8B` into this repository/current directory.
- Manage Python dependencies with `uv`.
- Implement `infer_cpu_v1` in Zig for CPU inference.
- `infer_cpu_v1` must support sampling parameters:
  - `temperature` default `0.6`
  - `top_p` default `0.95`
  - `top_k` default `20`
  - greedy mode for deterministic alignment tests.
- Test prompts:
  - `Hi,`
  - `The capital of China is`
  - `What is 1+1?`
- Perform logits alignment tests against the default Hugging Face Python implementation for the prefill phase.
- Verify greedy sampling alignment against Hugging Face Python generation/next-token behavior.
- Initialize git and commit after every tested milestone.
- Document process, commands, decisions, and verification in `worklog.md`.
- Use subagents aggressively for parallelizable research, architecture, implementation-review, and verification tasks, with a practical cap of 16 simultaneous agents.

## ENGINEERING CONSTRAINTS

- Keep changes small, reviewable, and reversible.
- Prefer simple, direct implementation over broad abstraction.
- Do not introduce non-`uv` Python dependency management.
- Record known gaps honestly in `worklog.md` and commit messages.
- Before claiming completion, run the relevant Python reference tests, Zig build/tests, and alignment checks, then record evidence.

## CONVENTIONS

- Python dependency management is `uv` only. Use `uv run python ...`; do not add `requirements.txt`, pipenv, poetry, or ad hoc global installs.
- This is scripts-first Python, not an installable package (`[tool.uv] package = false`).
- Verification is command/script driven; `tests/` is currently empty and no pytest config exists.
- Keep generated model files, `.venv`, `.zig-cache`, `zig-out`, `.omx`, and benchmark/alignment artifacts out of commits unless explicitly required.
- Benchmark run folders use `artifacts/benchmarks/gNN_*` naming; keep JSON schemas stable because summarizers validate them.
- `Qwen3.5-0.8B/` is a local model snapshot, not source code. Avoid manual edits except model refresh/download operations.
- Commit messages follow the Lore protocol from the workspace instructions; include honest `Tested:` / `Not-tested:` trailers.

## ANTI-PATTERNS (THIS PROJECT)

- Do not introduce non-`uv` Python dependency management.
- Do not claim full native Qwen inference when the path is still an HF bridge plus optional Metal LM-head sidecar.
- Do not promote Metal kernels, no-copy buffers, retained sessions, or packed layouts to defaults without explicit alignment and ownership evidence.
- Do not reinterpret `NNLGFIX1` row-major fixtures as packed data.
- Do not hide setup, fixture loading, or host top-k selection time inside kernel timing.
- Do not retain no-copy Metal session pointers past caller-owned memory lifetimes.
- Do not hand-edit benchmark or alignment artifacts to make gates pass.

## COMMANDS

```bash
uv run python scripts/download_model.py
zig build
zig build run -- --prompt "Hi," --greedy --json
uv run python scripts/run_alignment_tests.py
zig build -Dtarget=x86_64-linux --summary all
zig build -Denable-metal=true metal-lib
zig build -Denable-metal=true metal-smoke
zig build -Denable-metal=true metal-logits-test -- --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel-repeats 2
python3 -m py_compile scripts/*.py
```

## VERIFICATION BASELINE

- CPU/alignment changes: run `zig build` and `uv run python scripts/run_alignment_tests.py`.
- Python script changes: run `python3 -m py_compile scripts/*.py`; add the targeted `uv run python scripts/<script>.py ...` gate when behavior changes.
- Metal bridge/kernel changes on macOS: run `zig build -Denable-metal=true metal-lib`, `metal-smoke`, and a full fixture `metal-logits-test`.
- Cross-target build sanity when build graph or portable Zig changes: `zig build -Dtarget=x86_64-linux --summary all`.
- Record command lines, outcomes, decisions, and known gaps in `worklog.md` before claiming completion.

## NOTES

- `.github/workflows` and `Makefile` are absent; do not invent CI-only commands as required gates.
- `worklog.md` is long and operational; append dated entries rather than rewriting history.
- Existing artifacts may be large and ignored. Treat them as evidence snapshots, not source of truth for code behavior.
