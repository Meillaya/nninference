# src/ KNOWLEDGE

## OVERVIEW

`src/` owns the Zig executable surfaces and the Objective-C/Metal bridge used by opt-in logits tests.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| CPU CLI and sampling | `main.zig` | `infer_cpu_v1`; parses flags and delegates prefill logits to Python. |
| Token candidate selection | `main.zig` | `selectToken` implements greedy and temperature/top-p/top-k sampling. |
| Metal logits test CLI | `metal_logits_test.zig` | Built as `metal_logits_v1` only under `-Denable-metal=true`. |
| Metal smoke binary | `metal_smoke.zig` | Capability check for vector-add path. |
| Objective-C Metal bridge | `metal_bridge.m`, `metal_bridge.h` | Owns Metal device, buffers, command modes, retained-session behavior. |

## CONVENTIONS

- Keep the default CPU path independent of Metal; Metal targets stay behind `-Denable-metal=true`.
- `main.zig` calls `uv run python scripts/hf_prefill_bridge.py`; update `scripts/AGENTS.md` rules when changing that bridge contract.
- Preserve CLI defaults unless the task explicitly changes them: `temperature=0.6`, `top_p=0.95`, `top_k=20`, non-greedy by default.
- Treat `metal_bridge.h` + `metal_bridge.m` as a versioned C ABI; mirror struct/function changes across declarations, implementation, and Zig callers.
- JSON output from CLIs is consumed by Python validation scripts; keep fields such as `fixture_load_ms`, `metal_bridge_wall_ms`, `host_compare_ms`, `persistent_*`, and `actual_kernel` stable or update all consumers in the same change.
- Keep `metal_logits_v1` defaults stable: scalar kernel, copy buffers, full comparison, one-shot/per-iteration behavior unless flags request otherwise.
- Rebuild `zig-out/metal/kernels.metallib` after editing `metal/vector_add.metal` or bridge expectations that require new Metal functions.

## ANTI-PATTERNS

- Do not silently replace the HF bridge with a partial native path and still call it aligned CPU inference.
- Do not change Metal defaults from scalar/copy/full/per-iteration behavior without alignment and benchmark evidence.
- Do not silently change `auto` kernel fallback behavior; wrappers expect concrete `actual_kernel` / `actual_kernels_seen` reporting.
- Do not let retained/no-copy Metal sessions outlive caller-owned memory.
- Do not reinterpret `NNLGFIX1` row-major fixtures as packed/tiled weight layouts.

## REQUIRED CHECKS BY CHANGE TYPE

```bash
zig build
uv run python scripts/run_alignment_tests.py
zig build -Dtarget=x86_64-linux --summary all
zig build -Denable-metal=true metal-lib
zig build -Denable-metal=true metal-smoke
zig build -Denable-metal=true metal-logits-test -- --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel-repeats 2
```
