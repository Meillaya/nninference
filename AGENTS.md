# Project Instructions: nninference

## Active Task Requirements
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

## Engineering Constraints
- Keep changes small, reviewable, and reversible.
- Prefer simple, direct implementation over broad abstraction.
- Do not introduce non-`uv` Python dependency management.
- Record known gaps honestly in `worklog.md` and commit messages.
- Before claiming completion, run the relevant Python reference tests, Zig build/tests, and alignment checks, then record evidence.
