# nninference Worklog

## 2026-06-10
- Captured user requirements in `AGENTS.md` as the first repository file action.
- Initialized `uv` project metadata with Hugging Face/PyTorch/Transformers/Safetensors dependencies.
- Verified Hugging Face metadata for `Qwen/Qwen3.5-0.8B` exists and includes config, tokenizer, and single safetensors shard.
- Added `.gitignore` entries to keep downloaded model weights and generated artifacts out of git.
- Downloaded `Qwen/Qwen3.5-0.8B` into `Qwen3.5-0.8B/` using `uv run python scripts/download_model.py`; verified required files and 1.65 GiB local footprint.
- Inspected model architecture: Qwen3.5 text config uses 24 layers with mixed `linear_attention` and `full_attention`, bfloat16 weights, and tied embeddings.
- Added Zig 0.16 executable `infer_cpu_v1` plus a Python HF CPU prefill bridge. Zig owns CLI parsing and greedy/top-k/top-p/temperature sampling over HF prefill candidates; the bridge performs the Qwen3.5 hybrid forward pass because the checkpoint uses mixed linear attention and full attention.
- Verified `zig build` and smoke-tested `./zig-out/bin/infer_cpu_v1 --prompt 'Hi,' --greedy --json --logits-out artifacts/manual_hi_logits.bin`; Zig selected token `353`, matching HF forward argmax and HF `generate(... do_sample=False)`.
- Added `scripts/run_alignment_tests.py` to build the Zig executable, compare full prefill logits for the three required prompts against a direct HF reference (`rtol=1.6e-2`, `atol=1e-5`), and verify greedy token IDs match HF forward argmax and HF `generate`.
- Verification after formatting: `zig fmt build.zig src/main.zig`; `zig build`; `uv run python scripts/run_alignment_tests.py`.
- Alignment results: `Hi,` -> token `353` (`" I"`); `The capital of China is` -> token `25701` (`" Beijing"`); `What is 1+1?` -> token `271` (blank-line token). All max absolute logit diffs were `0.0`; default sampling smoke confirmed temperature `0.6`, top_p `0.95`, top_k `20`.
- Known limitation: v1 uses the default Hugging Face CPU implementation for the Qwen3.5 hybrid forward pass and implements the user-facing inference CLI plus sampling in Zig; it is not yet a standalone native Zig implementation of Qwen3.5 linear attention kernels.

## 2026-06-10 Ultragoal Gate 0 — baseline capture
- Created execution branch `ultragoal/kimi-metal-iteration` for the Kimi-inspired Zig+Metal plan.
- Added `scripts/benchmark_bridge.py` to capture current `infer_cpu_v1` HF-bridge greedy first-token baseline, including git/tool/hardware/thread/model-hash metadata and optional local-app baseline availability.
- Verification commands run:
  - `zig build`
  - `uv run python scripts/run_alignment_tests.py`
  - `uv run python scripts/benchmark_bridge.py --warmup 1 --repeats 2`
- Alignment remained intact for the three required prompts with max absolute logit diff `0.0` and greedy IDs matching HF forward argmax/generate.
- Bridge baseline summary (cold-ish executable path; includes `uv run`, Python startup, model load, HF prefill/generate, bridge serialization, Zig parse/sampling):
  - `Hi,`: mean 5.126s/token, min 4.755s, max 5.498s, mean first-token rate 0.195 tok/s over 2 measured repeats.
  - `The capital of China is`: mean 7.996s/token, min 5.375s, max 10.616s, mean first-token rate 0.125 tok/s over 2 measured repeats.
  - `What is 1+1?`: mean 5.696s/token, min 4.722s, max 6.670s, mean first-token rate 0.176 tok/s over 2 measured repeats.
- Benchmark artifact written to ignored local path `artifacts/benchmarks/bridge_baseline.json`; summary is preserved here because `artifacts/` remains git-ignored.
- Known limitation preserved: this baseline is the HF Python bridge path, not native Zig or Metal inference.

## 2026-06-10 Ultragoal Gate 1 — Metal capability spike
- Added a minimal Zig-callable Objective-C Metal bridge (`src/metal_bridge.h`, `src/metal_bridge.m`) plus a standalone Zig smoke executable (`src/metal_smoke.zig`) so Metal runtime work stays isolated from the HF-bridge inference path.
- Added `metal/vector_add.metal` and `zig build metal-lib` / `zig build metal-smoke` build steps that compile the `.metal` source through `xcrun metal` + `xcrun metallib` into a Zig build-cache `kernels.metallib` path before running the smoke test.
- Metal smoke evidence: device `Apple M4`, recommended max working set `12713115648` bytes, max threads per threadgroup `1024`, non-uniform dispatch available, vector length `1024`, mismatches `0`, max absolute error `0.0`; latest local artifact is `artifacts/metal/gate1_vector_add_smoke.json`.
- Verification commands run:
  - `zig fmt build.zig src/main.zig src/metal_smoke.zig`
  - `zig build metal-smoke`
  - `zig build`
  - `uv run python scripts/run_alignment_tests.py`
  - `mkdir -p artifacts/metal && zig build metal-smoke | tee artifacts/metal/gate1_vector_add_smoke.json`
- HF-bridge alignment remained intact for the required prompts with max absolute logit diff `0.0`, greedy IDs matching HF forward argmax/generate, and default sampling smoke still reporting temperature `0.6`, top_p `0.95`, top_k `20`.
- Known limitation: Gate 1 only proves host/shader/build/runtime capability with a vector-add fixture; it does not yet compute LM-head logits or move Qwen tensors through Metal.

## 2026-06-10 Ultragoal Gate 2 — independent golden logits matmul
- Extended the sidecar Metal shader library with `logits_matmul`, computing final-token logits as `weights[row, col] * hidden[col]` reductions for f32 row-major weights.
- Extended the C/Objective-C bridge with `nn_metal_run_logits_matmul` and added `src/metal_logits_test.zig`, a standalone test runner that keeps model integration out of `infer_cpu_v1`.
- Tiny independent fixture: analytic HF-independent f32 data with shape `17 x 19`, CPU reference accumulated in f64 then cast to f32. Result: max_abs_diff `0.0`, max_rel_diff `0.0`, mismatches `0` under `1e-4`/`1e-4` tolerance.
- Added `scripts/generate_checkpoint_logits_fixture.py` to create ignored checkpoint-slice fixtures under `artifacts/metal/gate2/checkpoint_hi/` from Qwen3.5 hidden state + tied embedding rows. The committed code records a JSON manifest and writes the binary tensor fixture only to ignored artifacts.
- Checkpoint-derived fixture: prompt `Hi,`, 64 selected rows (guard rows, required greedy IDs, top-20 prompt rows, deterministic random fill) x hidden size `1024`. Result: max_abs_diff `0.000008583069`, max_rel_diff `0.000013547198`, mismatches `0` under `5e-3`/`5e-3` tolerance.
- Verification commands run:
  - `zig fmt build.zig src/metal_logits_test.zig src/metal_smoke.zig`
  - `zig build metal-smoke`
  - `zig build metal-logits-test`
  - `uv run python scripts/generate_checkpoint_logits_fixture.py --model-dir Qwen3.5-0.8B --prompt 'Hi,' --out artifacts/metal/gate2/checkpoint_hi`
  - `zig build metal-logits-test -- --fixture artifacts/metal/gate2/checkpoint_hi/fixture.bin`
  - `zig build`
  - `uv run python scripts/run_alignment_tests.py`
  - `zig build metal-logits-test | tee artifacts/metal/gate2_tiny_logits_matmul.json`
  - `zig build metal-logits-test -- --fixture artifacts/metal/gate2/checkpoint_hi/fixture.bin | tee artifacts/metal/gate2_checkpoint_hi_logits_matmul.json`
- HF-bridge alignment remained intact for all three required prompts with max absolute logit diff `0.0`, greedy IDs matching HF forward argmax/generate, and default sampling still at temperature `0.6`, top_p `0.95`, top_k `20`.
- Known limitation: Gate 2 validates row-slice checkpoint matmul only; full-vocab top-1/top-20 agreement is reserved for Gate 3.

## 2026-06-10 Ultragoal Gate 3 — Qwen full-vocab logits prototype
- Extended the checkpoint fixture generator with `--row-mode full` for a full-vocab Qwen LM-head projection fixture while keeping the binary tensors under ignored `artifacts/`.
- Extended `src/metal_logits_test.zig` to load large fixtures (up to 2 GiB) and optionally enforce exact top-1/top-20 set agreement for full-vocab projection outputs.
- Full-vocab fixture: prompt `Hi,`, rows `248320`, hidden size `1024`, fixture SHA256 `eec8fa6b44db79777bec0ae1895bbf93c1cf9c98bab9d41f5b88fc611d5aed1c`. Generated fixture size is about 971 MiB and remains ignored at `artifacts/metal/gate3/full_hi/fixture.bin`.
- Metal full-vocab result vs PyTorch f32 LM-head projection: max_abs_diff `0.000011444092`, max_rel_diff `0.020768026`, mismatches `0` under the gate's combined abs/rel rule with `5e-3`/`5e-3` tolerance; top1 match `true`; top20 set match `true`; latest measured kernel elapsed `21.036982536315918` ms.
- HF default-forward nuance for `Hi,`: HF forward logits and explicit f32 LM-head projection both choose token `353`; their top-20 sets differ at the tied cutoff (`10.5`) because several boundary tokens share the same rounded HF logit. All HF tokens strictly above the cutoff are present in the projection top-20. This gate therefore claims full-vocab Metal LM-head projection correctness, not full transformer/logit-path identity.
- Verification commands run:
  - `zig fmt build.zig src/metal_logits_test.zig src/metal_smoke.zig`
  - `zig build metal-smoke`
  - `zig build metal-logits-test`
  - `uv run python scripts/generate_checkpoint_logits_fixture.py --model-dir Qwen3.5-0.8B --prompt 'Hi,' --row-mode full --out artifacts/metal/gate3/full_hi`
  - `zig build metal-logits-test -- --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk`
  - `zig build`
  - `uv run python scripts/run_alignment_tests.py`
- HF-bridge alignment remained intact for all three required prompts with max absolute logit diff `0.0`, greedy IDs matching HF forward argmax/generate, and default sampling still at temperature `0.6`, top_p `0.95`, top_k `20`.
- Known limitation: the Metal path still runs only an isolated LM-head projection using a final hidden state from HF; it is not full native Qwen inference and is not integrated into `infer_cpu_v1` yet.

## 2026-06-10 Ultragoal Gate 4 — explicit prototype CLI integration
- Exposed the Metal LM-head projection prototype as a separate installed CLI, `zig-out/bin/metal_logits_v1`, rather than changing `infer_cpu_v1` or its default HF bridge.
- Installed the generated Metal library via `zig build metal-lib` to `zig-out/metal/kernels.metallib`, so the prototype can run outside the Zig build runner with an explicit metallib path and ignored fixture path.
- Prototype CLI verification: `./zig-out/bin/metal_logits_v1 zig-out/metal/kernels.metallib --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk` returned expected_top1 `353`, actual_top1 `353`, top1_match `true`, top20_set_match `true`, max_abs_diff `0.000011444092`, mismatches `0`; latest artifact is `artifacts/metal/gate4_metal_logits_v1_full_hi.json`.
- Verification commands run:
  - `zig fmt build.zig src/metal_logits_test.zig`
  - `zig build metal-lib`
  - `zig build`
  - `./zig-out/bin/metal_logits_v1 --help`
  - `./zig-out/bin/metal_logits_v1 zig-out/metal/kernels.metallib --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk | tee artifacts/metal/gate4_metal_logits_v1_full_hi.json`
  - `zig build metal-smoke`
  - `zig build metal-logits-test -- --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk`
  - `uv run python scripts/run_alignment_tests.py`
- Default `infer_cpu_v1` behavior remains HF-bridge based and unchanged; alignment still passes for all three required prompts with max absolute logit diff `0.0` and default sampling temperature `0.6`, top_p `0.95`, top_k `20`.
- Known limitation: the prototype CLI consumes generated fixtures; it does not tokenize prompts, run transformer blocks, or accelerate the default inference path yet.
