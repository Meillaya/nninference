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
