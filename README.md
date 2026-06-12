# nninference

`nninference` is a small Zig CLI plus Python/Hugging Face reference harness for checking first-token inference alignment against `Qwen/Qwen3.5-0.8B`.

The current CPU path exposes an `infer_cpu_v1` Zig command that calls a Python Hugging Face prefill bridge, then applies the Zig-side token selection logic. Metal code in this repository is an opt-in logits/LM-head validation sidecar, not a full native transformer runtime.

## Repository layout

- `src/` - Zig CLI entry point, Metal validation binary, and Objective-C Metal bridge.
- `scripts/` - `uv`-managed Python helpers for model download, HF reference prefill, alignment checks, fixtures, and benchmark summaries.
- `metal/` - Metal kernels built only when `-Denable-metal=true` is supplied.
- `docs/` - design and audit notes.
- `tests/` - Python tests for optimization/reporting helpers.

Generated artifacts, local model snapshots, virtual environments, Zig build outputs, and benchmark results are intentionally ignored.

## Prerequisites

- Zig installed and available as `zig`
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for Python dependency management
- Optional on macOS: Xcode command line tools for Metal targets

## Setup

```bash
uv sync
uv run python scripts/download_model.py
```

The model download creates a local `Qwen3.5-0.8B/` snapshot, which is not committed.

## Basic usage

```bash
zig build
zig build run -- --prompt "Hi," --greedy --json
```

Useful alignment and sanity checks:

```bash
uv run python scripts/run_alignment_tests.py
python3 -m py_compile scripts/*.py
```

Optional Metal checks on macOS:

```bash
zig build -Denable-metal=true metal-lib
zig build -Denable-metal=true metal-smoke
```

## Publishing notes

This repository is source-first. Do not publish local generated directories such as `.venv/`, `.zig-cache/`, `zig-out/`, `artifacts/`, or `Qwen3.5-0.8B/`. Benchmark and alignment outputs should be regenerated from scripts when needed rather than committed as source.
