# Metal packed-weight/session boundary audit

Date: 2026-06-10  
Scope: G085 read-only feasibility audit for a future opt-in packed/tiled LM-head weight layout and retained Metal session boundary.

## Decision summary

The next safe optimization class is **not** another row/threadgroup-only LM-head kernel. G081-G083 showed that near-duplicate reductions are correct but below the keep threshold. A future code-changing iteration should instead attack a qualitatively different boundary:

1. retain a Metal session across repeated logits calls, and
2. optionally feed that session a prepacked/tiled weight buffer that has its own explicit fixture/layout metadata.

This audit authorizes only future **split, opt-in prototypes**. It does not change current runtime defaults, kernel auto-selection, fixture ABI, or HF bridge behavior. It specifically rejects implementing packed/tiled weights and retained sessions as one combined first patch.

## Current boundaries

### Public Metal bridge API

`src/metal_bridge.h` exposes synchronous C functions only:

- `nn_metal_run_logits_matmul(...)` consumes `hidden`, `weights_row_major`, and `logits_out` per call.
- `nn_metal_benchmark_logits_matmul_persistent(...)` and `_persistent_batched(...)` reuse buffers only inside one synchronous benchmark call.
- The header explicitly states that “persistent” means repeated command-buffer submission inside one synchronous call, **not** a retained async buffer API.
- The no-copy contract says caller-owned memory must remain valid and unmodified until the call returns.

Implication: a retained session needs a new opt-in API; overloading the existing functions would blur lifetime ownership and make rollback harder.

### Objective-C bridge implementation

`src/metal_bridge.m` currently performs these steps per call or per benchmark invocation:

- create `MTLDevice`;
- load the metallib;
- create a compute pipeline;
- allocate/wrap hidden, row-major weights, logits, and dims buffers;
- copy row-major weights in copy mode or wrap caller memory in no-copy mode;
- create a command queue/buffer/encoder;
- dispatch and wait synchronously.

Reusable helpers already exist for a future session boundary:

- `nn_create_logits_pipeline` selects/probes `auto` and creates a pipeline.
- `nn_make_logits_dispatch_config` computes dispatch shape.
- `nn_encode_logits_dispatches` encodes repeated dispatches.
- `nn_fill_benchmark_result` and `nn_fill_actual_kernel` preserve reporting fields.

Implication: the smallest retained-session prototype should factor these helpers into a session object instead of duplicating setup logic.

### Metal kernel/layout boundary

`metal/vector_add.metal` currently provides:

- `logits_matmul` scalar row-major projection;
- `logits_matmul_tg` row-major 256-threadgroup reduction;
- `logits_matmul_tg128` row-major 128-threadgroup reduction.

All LM-head kernels consume the same four-buffer ABI:

1. hidden vector,
2. row-major weights,
3. logits output,
4. `uint2(rows, cols)` dims.

Implication: a packed/tiled layout is a distinct ABI and must not masquerade as `weights_row_major`. It should use a new explicit kernel label, fixture/header metadata, and validator path.

### Zig fixture and CLI boundary

`src/metal_logits_test.zig` defines fixture magic `NNLGFIX1` and a `Fixture` with `hidden`, `weights`, and `expected` f32 arrays. The default CLI behavior remains:

- `kernel=scalar`,
- `buffer_mode=copy`,
- `compare_mode=full`,
- `benchmark_command_mode=per_iter`.

The matrix mode loads a fixture once and emits a shared-load header plus per-kernel JSON rows. It is the best current seam for comparing variants while preserving full correctness evidence.

Implication: a packed prototype should first add a **separate generated fixture or pack step** while keeping the existing `NNLGFIX1` fixture as the correctness source of truth. The packed buffer must be derivable from row-major weights and validated against unchanged `expected` logits.

### Python benchmark/reporting boundary

Current scripts already enforce safety properties:

- `scripts/benchmark_metal_logits_reuse.py` validates matrix rows, actual kernel labels, actual buffer mode, no-copy evidence, full compare, top-1/top-20, and positive timing.
- `scripts/benchmark_metal_command_modes.py` keeps `per_iter` and `batched` rankings separate and says cross-mode deltas are diagnostic, not default-promotion criteria.
- `scripts/summarize_metal_benchmarks.py` now summarizes G080-G083 and validates nested per-sample correctness evidence before drawing conclusions.

Implication: a future prototype should extend the existing reusable-fixture and command-mode reports additively. It should not weaken full comparison or mix command-mode rankings.

## Minimal safe prototype shape

A safe future prototype should be split into independent gates so lifetime/session risk and layout/kernel risk are not debugged simultaneously.

### Gate A — Retained session without packed layout

Goal: retain device/library/pipeline/queue/buffers across repeated calls while still consuming row-major weights.

Expected write scope:

- `src/metal_bridge.h` / `src/metal_bridge.m`: new opaque session API, not a replacement for current synchronous functions.
- `src/metal_logits_test.zig`: opt-in CLI benchmark mode that creates a session and runs repeated calls.
- Python benchmark wrappers: additive opt-in flags and schema fields only.

Required safety properties:

- Existing synchronous APIs remain unchanged.
- Default CLI remains scalar/copy/full/per_iter.
- The session API owns any Metal buffers it allocates.
- No-copy session mode must either deep-copy/own the buffer or document and enforce a synchronous caller lifetime; it must not retain no-copy pointers after return unless the API explicitly exposes ownership/lifetime rules.
- Session teardown must be explicit and safe on all error paths.

Keep threshold:

- Correctness must pass all default and explicit gates.
- At least 3% median improvement in repeated-call wall/setup metrics versus the current reusable-fixture path before promoting beyond experimental status.
- No regression in per-kernel correctness or HF alignment.

### Gate B — Packed/tiled layout after session API

Goal: derive a packed weight buffer from row-major fixture weights and run a separate packed kernel. This gate is not authorized until Gate A is correct and measured, because layout changes imply a new kernel/ABI rather than simple preprocessing.

Expected write scope:

- New packing code in Zig or Python benchmark tooling.
- New ignored fixture/version, e.g. `NNLGPCK1`, with explicit metadata for original rows/cols, tile rows, tile cols, dtype, and packing order.
- New explicit Metal kernel label, e.g. `logits_matmul_packed_tiled`, never added to `auto` initially.
- New layout metadata in the test/benchmark report (`layout=row_major|packed_tiled`, tile sizes, packed byte count).

Required safety properties:

- Row-major fixture remains the source of truth.
- Packed buffer is generated from the fixture and checked for dimensions/byte count.
- `NNLGFIX1` must never be reinterpreted as packed data; packed readers/writers need a separate magic/version.
- Packed kernel results are compared against the unchanged expected logits with full comparison as default.
- `topk` remains diagnostic only.
- `auto` and defaults stay unchanged until a separate promotion gate.

Keep threshold:

- Full-vocab top-1/top-20 and zero-mismatch tolerance pass.
- HF alignment remains unchanged.
- Median `persistent_ms_per_kernel_repeat` improves by at least 3% in both `per_iter` and `batched` command modes against the best retained row-major baseline with samples >= 7.
- If setup/packing time is included, report it separately and do not hide it inside kernel timing.
- Claims must be limited to LM-head matmul fixture correctness unless a separate HF-forward/generation gate proves a broader claim; the Metal full fixture is not a full transformer validation.

## Risks and mitigations

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Retained no-copy pointers outlive caller memory | Current no-copy contract is synchronous only | Keep retained session copy-backed first; only add retained no-copy with explicit owner object/lifetime tests |
| Packed layout changes correctness silently | A packed ABI can confuse row-major validators | Use explicit layout labels, separate magic/version, separate kernels, and unchanged row-major expected logits |
| Setup wins hide kernel regressions | Session reuse can make wall time look better while kernel gets slower | Report setup, packing, dispatch, host-compare, and per-kernel-repeat separately |
| Memory/disk pressure increases | The full row-major fixture is already roughly 971 MiB and a packed duplicate can be materially worse | Track packed byte count, conversion time, mmap/no-copy behavior, and keep generated packed artifacts ignored |
| HF-vs-matmul claims get conflated | LM-head fixture correctness is narrower than full HF forward/generation alignment | Phrase packed-kernel evidence as LM-head matmul correctness unless separate HF gates are added |
| Command-mode rankings are mixed | Existing reports warn per-iter and batched semantics differ | Preserve separate child reports and treat cross-mode deltas as diagnostic |
| Default path drifts | User requires HF bridge and existing defaults preserved | Keep all new features behind explicit flags and run default scalar/copy/full gates |
| Linux/default build accidentally gains Metal dependency | CPU path must remain portable | Keep `-Denable-metal=true` gating and run Linux cross-build after every prototype |

## Required validation matrix for a future code-changing prototype

Baseline/default gates:

```bash
zig build
zig build -Dtarget=x86_64-linux --summary all
zig build -Denable-metal=true metal-lib
zig build -Denable-metal=true metal-smoke
zig build -Denable-metal=true metal-logits-test -- \
  --fixture artifacts/metal/gate3/full_hi/fixture.bin \
  --expect-topk \
  --kernel-repeats 2 \
  --compare-mode full
uv run python scripts/run_alignment_tests.py
```

Explicit prototype gate, replacing `<candidate>` and flags with the new opt-in mode:

```bash
zig build -Denable-metal=true metal-logits-test -- \
  --fixture artifacts/metal/gate3/full_hi/fixture.bin \
  --expect-topk \
  --kernel <candidate> \
  --buffer-mode nocopy \
  --kernel-repeats 2 \
  --compare-mode full
```

Benchmark gate:

```bash
uv run python scripts/benchmark_metal_command_modes.py \
  --no-build \
  --samples 7 \
  --kernel-repeats 5 \
  --benchmark-iters 10 \
  --kernels scalar,threadgroup,threadgroup128,<candidate> \
  --buffer-mode nocopy \
  --compare-mode full \
  --out artifacts/benchmarks/<goal>/command_modes_full_samples7.json \
  --artifact-dir artifacts/benchmarks/<goal>/command_modes_full_samples7_runs
```

Rollback criteria:

- any HF alignment or greedy mismatch;
- any default scalar/copy/full drift;
- any full-vocab mismatch or top-1/top-20 mismatch;
- retained no-copy lifetime cannot be made explicit and testable;
- candidate fails the >=3% median improvement threshold in either command mode;
- schema changes break existing benchmark scripts or reports.

## G085 conclusion

Proceed next with **Gate A: retained session without packed layout** before attempting packed/tiled weights. This isolates lifetime and setup overhead first. Only after the session boundary is correct and measured should a packed/tiled layout be introduced as Gate B.

Critic caveat: Gate A is an implementation direction only because it is split, opt-in, and row-major. Gate B remains a separate later prototype requiring an explicit fixture/version and layout contract. Combining retained sessions, no-copy lifetime extension, and packed/tiled layout in one patch is rejected.
