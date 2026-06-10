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

## 2026-06-10 Ultragoal Gate 5 — throughput baseline
- Added `scripts/benchmark_metal_logits.py` to benchmark the fixture-driven Metal LM-head prototype, CPU NumPy/Accelerate reference projection, Metal command-buffer time, CLI wall time, and estimated host/load/transfer overhead.
- Benchmark artifact: `artifacts/benchmarks/metal_logits_benchmark.json` (ignored). Fixture: `artifacts/metal/gate3/full_hi/fixture.bin`, size `1018116120` bytes, rows `248320`, cols `1024`.
- Metal prototype baseline over 3 repeats: CLI wall mean `2137.514 ms`, min `2127.880 ms`, max `2150.459 ms`; Metal command-buffer mean `24.318 ms`, min `21.355 ms`, max `29.660 ms`; estimated host fixture-load/copy/verification overhead mean `2113.196 ms`.
- CPU reference baseline over 2 repeats using NumPy/Accelerate on the same fixture: mean `34.605 ms`, min `14.148 ms`, max `55.061 ms`, max_abs_diff `0.0`, top1 `353`.
- Current bridge baseline remains the earlier `artifacts/benchmarks/bridge_baseline.json`; local-app baseline remains skipped because no fair local app/CLI baseline is configured for identical Qwen3.5-0.8B settings.
- Verification commands run:
  - `uv run python scripts/benchmark_metal_logits.py --warmup 1 --repeats 3 --cpu-repeats 2`
  - `zig build`
  - `uv run python scripts/run_alignment_tests.py`
- Interpretation: the current prototype proves correctness but is not yet a throughput win end-to-end; the dominant optimization target is avoiding repeated ~971 MiB fixture load/copy and separating persistent-buffer benchmark timing from one-shot CLI overhead.

## 2026-06-10 Ultragoal Gate 5 — repeat-dispatch measurement optimization
- Added `--kernel-repeats N` to `metal_logits_v1` and the Metal bridge, dispatching the same logits kernel repeatedly after a single fixture load and buffer setup. This preserves correctness while giving a cleaner amortized command-buffer timing signal.
- Updated `scripts/benchmark_metal_logits.py` to pass `--kernel-repeats` and report both total command-buffer time and per-repeat command-buffer time.
- Correctness check with `--kernel-repeats 5`: expected_top1 `353`, actual_top1 `353`, top1_match `true`, top20_set_match `true`, max_abs_diff `0.000011444092`, mismatches `0`; artifact `artifacts/metal/gate5_kernel_repeats_full_hi.json`.
- Benchmark after repeat-dispatch change (`--warmup 1 --repeats 3 --cpu-repeats 2 --kernel-repeats 5`): amortized Metal command-buffer per-repeat mean `17.891 ms`, min `15.760 ms`, max `20.938 ms`; one-shot CLI wall mean `2458.763 ms`; host/load/copy overhead remains dominant at mean `2369.306 ms`.
- Compared with the prior Gate 5 baseline mean command-buffer time `24.318 ms`, repeat-dispatch measurement lowers the amortized command-buffer elapsed time per dispatch by about 26% while leaving the end-to-end fixture CLI dominated by host overhead.
- Verification commands run:
  - `zig build metal-logits-test -- --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel-repeats 5`
  - `zig build`
  - `./zig-out/bin/metal_logits_v1 zig-out/metal/kernels.metallib --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel-repeats 5 | tee artifacts/metal/gate5_kernel_repeats_full_hi.json`
  - `uv run python scripts/benchmark_metal_logits.py --warmup 1 --repeats 3 --cpu-repeats 2 --kernel-repeats 5`
  - `uv run python scripts/run_alignment_tests.py`
- Next optimization target remains persistent fixture/buffer ownership or live integration; the current CLI still reloads a ~971 MiB fixture for every process invocation.

## 2026-06-10 Gate 6 review fixes
- Independent code review requested changes for sidecar isolation, benchmark reproducibility, fixture-generator safety, and performance-claim wording.
- Fixed default build isolation by gating all Metal targets and installs behind explicit `-Denable-metal=true`; default `zig build` remains CPU/HF-bridge only. Verified `zig build -Dtarget=x86_64-linux --summary all` succeeds and installs only `infer_cpu_v1` for a Linux target.
- Updated benchmark reproducibility so `scripts/benchmark_metal_logits.py` builds prerequisites with `zig build -Denable-metal=true metal-lib` and `zig build -Denable-metal=true` unless `--no-build` is explicitly used.
- Removed unnecessary `trust_remote_code=True` from checkpoint fixture generation and switched the LM-head fixture source to `model.get_output_embeddings().weight`.
- Corrected the Metal probe's non-uniform dispatch capability field to avoid overstating support; current smoke output reports `supports_non_uniform_threadgroups: false` rather than inferring support from `supportsFamily:`.
- Reworded Gate 5 worklog claims from kernel speedup to amortized command-buffer elapsed per dispatch; no GPU timestamp/counter evidence is claimed.
- Verification commands run:
  - `zig build`
  - `zig build -Dtarget=x86_64-linux --summary all`
  - `zig build -Denable-metal=true metal-lib`
  - `zig build -Denable-metal=true`
  - `zig build -Denable-metal=true metal-smoke`
  - `zig build -Denable-metal=true metal-logits-test -- --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel-repeats 2`
  - `uv run python scripts/generate_checkpoint_logits_fixture.py --model-dir Qwen3.5-0.8B --prompt 'Hi,' --row-mode subset --out artifacts/metal/gate2/checkpoint_hi`
  - `uv run python scripts/generate_checkpoint_logits_fixture.py --model-dir Qwen3.5-0.8B --prompt 'Hi,' --row-mode full --out artifacts/metal/gate3/full_hi`
  - `zig build -Denable-metal=true metal-logits-test -- --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel-repeats 5`
  - `uv run python scripts/benchmark_metal_logits.py --warmup 1 --repeats 2 --cpu-repeats 1 --kernel-repeats 5`
  - `uv run python scripts/run_alignment_tests.py`
- Latest post-fix benchmark: amortized Metal command-buffer per-repeat mean `14.640 ms`; CLI wall mean `2220.244 ms`; host/load/copy overhead mean `2147.046 ms`; CPU NumPy reference `62.637 ms` for one repeat. HF alignment still reports max_abs_diff `0.0` for all three required prompts.

## 2026-06-10 Ultragoal Gate 6 — final verification report
- Final independent review found six issues and requested changes. All were addressed in commit `840e052`: explicit Metal opt-in, reproducible benchmark prerequisites, no unnecessary `trust_remote_code`, output-embedding LM-head fixtures, corrected capability reporting, and tighter performance wording.
- Final verification matrix after review fixes:
  - `zig build` — pass; default build remains CPU/HF-bridge only.
  - `zig build -Dtarget=x86_64-linux --summary all` — pass; installs only `infer_cpu_v1` for Linux target.
  - `zig build -Denable-metal=true metal-smoke` — pass; Apple M4 smoke, vector length `1024`, mismatches `0`, max_abs_error `0`.
  - `zig build -Denable-metal=true metal-logits-test -- --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel-repeats 2` — pass; max_abs_diff `0.000011444092`, mismatches `0`, expected_top1/actual_top1 `353`, top20_set_match `true`.
  - `uv run python scripts/run_alignment_tests.py` — pass; all three required prompts have max_abs_diff `0.0`, greedy IDs match HF forward/generate, default sampling remains temperature `0.6`, top_p `0.95`, top_k `20`.
  - `uv run python scripts/benchmark_metal_logits.py --warmup 1 --repeats 2 --cpu-repeats 1 --kernel-repeats 5` — pass after explicit Metal prerequisite build; latest amortized Metal command-buffer per-repeat mean `14.640 ms`, CLI wall mean `2220.244 ms`, host/load/copy overhead mean `2147.046 ms`.
- Completed scope:
  - Preserved the existing HF-bridge `infer_cpu_v1` correctness path.
  - Added explicit, macOS-only Zig+Objective-C+Metal sidecar support under `-Denable-metal=true`.
  - Validated tiny f32 matmul, Qwen row-slice matmul, and full-vocab Qwen f32 LM-head projection for prompt `Hi,`.
  - Exposed a non-default fixture-driven prototype CLI `metal_logits_v1`.
  - Measured current throughput and one amortized command-buffer timing improvement.
- Remaining risks/gaps:
  - `metal_logits_v1` is fixture-driven and does not tokenize prompts, run transformer blocks, or replace `infer_cpu_v1`.
  - Full-vocab Metal validation was performed for prompt `Hi,`; row-slice validation covers only that prompt as well.
  - Performance claims are limited to command-buffer and one-shot fixture CLI measurements; no GPU timestamp/counter profiling or persistent-buffer runtime exists yet.
  - The default HF bridge still depends on Python/PyTorch/Transformers and is not a standalone native Qwen runtime.

## 2026-06-10 Resumed ultragoal G051 — mandatory 12h loop baseline
- Resumed the Kimi/Metal ultragoal after the earlier premature completion. The corrected interpretation is now explicit: the `12h_plus` budget authorizes continued safe autonomous optimization even after the initial staged Gates 0-6 pass; completion of the old goal list is not by itself a stop condition.
- Added follow-on ultragoal stories G051-G055 by steering the existing durable plan: baseline reconstruction, persistent in-process Metal benchmark mode, optimized LM-head kernel investigation, fixture/memory-layout evaluation, and a final quality gate. The active Codex goal now points back to `.omx/ultragoal/goals.json` / `.omx/ultragoal/ledger.jsonl`.
- Reconstructed current baseline from commit `0d44cd9` on branch `ultragoal/kimi-metal-iteration`. Evidence log: `artifacts/benchmarks/resume_g051_baseline.log` (ignored artifact).
- Verification commands run:
  - `zig build`
  - `zig build -Denable-metal=true metal-logits-test -- --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel-repeats 2`
  - `uv run python scripts/run_alignment_tests.py`
  - `uv run python scripts/benchmark_metal_logits.py --warmup 1 --repeats 2 --cpu-repeats 1 --kernel-repeats 5`
- Correctness baseline remains green: full-vocab Metal fixture for prompt `Hi,` matched expected top-1 `353`, top-20 set matched, `mismatches=0`, `max_abs_diff=0.000011444092`; HF alignment for `Hi,`, `The capital of China is`, and `What is 1+1?` still reports max absolute logit diff `0.0` with greedy token IDs matching HF.
- Resumed benchmark baseline: one-shot Metal CLI wall mean `2412.188 ms`; command-buffer total mean for 5 repeats `93.969 ms`; amortized command-buffer per-repeat mean `18.794 ms`; estimated host/load/transfer overhead mean `2318.219 ms`; CPU NumPy/Accelerate reference for one repeat `66.897 ms`.
- Next optimization target: add a reversible in-process benchmark mode so `metal_logits_v1` can reuse a loaded fixture and Metal buffers across multiple measured iterations. This should improve measurement quality and isolate persistent-buffer/kernel behavior before any integration into the default HF bridge.

## 2026-06-10 Resumed ultragoal G052 — persistent in-process Metal benchmark mode
- Added an opt-in persistent benchmark path for the sidecar LM-head prototype: `metal_logits_v1 ... --benchmark-iters N`. This keeps the existing one-shot correctness path unchanged, but adds a C bridge path that creates the Metal device/library/pipeline/command queue and shared buffers once, copies the fixture into Metal buffers once, then runs `N` measured command-buffer iterations against those persistent buffers.
- Updated `scripts/benchmark_metal_logits.py` with `--persistent-iters N`; the default benchmark remains backward-compatible and reports `persistent_metal: null` unless the option is requested.
- Correctness remained green for the full-vocab Qwen fixture: expected/actual top-1 `353`, top-20 set match `true`, `mismatches=0`, `max_abs_diff=0.000011444092`.
- Verification commands run:
  - `zig fmt src/metal_logits_test.zig`
  - `zig build`
  - `zig build -Denable-metal=true`
  - `zig build -Dtarget=x86_64-linux --summary all`
  - `zig build -Denable-metal=true metal-logits-test -- --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel-repeats 2`
  - `./zig-out/bin/metal_logits_v1 zig-out/metal/kernels.metallib --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel-repeats 2 --benchmark-iters 3`
  - `uv run python scripts/run_alignment_tests.py`
  - `uv run python scripts/benchmark_metal_logits.py --warmup 1 --repeats 2 --cpu-repeats 1 --kernel-repeats 5 --persistent-iters 3`
- Latest persistent-mode sample: setup `97.151 ms`; measured persistent loop `200.001 ms` for `3` iterations × `5` kernel repeats; `persistent_ms_per_kernel_repeat=13.333 ms`. The same benchmark run's one-shot command-buffer per-repeat mean was `20.802 ms`, while the one-shot CLI wall mean remained `2352.716 ms` because the process still loads the ~971 MiB fixture once.
- Interpretation: this is a measurement/runtime-ownership milestone rather than a full end-to-end throughput win. It proves buffer/pipeline reuse inside one process and gives a cleaner persistent command-buffer signal; it does not yet remove the initial fixture file read or integrate Metal into `infer_cpu_v1`.
- Next optimization target: use this persistent measurement path to evaluate kernel-level changes or reduce fixture load/transfer overhead further, while keeping the one-shot CLI and HF bridge as rollback/reference paths.

## 2026-06-10 Resumed ultragoal G053 — opt-in threadgroup LM-head kernel prototype
- Added an alternate Metal LM-head kernel, `logits_matmul_tg`, exposed through `metal_logits_v1 --kernel threadgroup`. The existing scalar row-per-thread kernel remains the default and rollback path (`--kernel scalar`).
- The threadgroup kernel assigns one Metal threadgroup per output row and reduces partial dot products across 256 threads. This is intentionally opt-in because it increases dispatch/threadgroup pressure and may be platform-sensitive.
- Updated the Objective-C bridge and benchmark harness so both one-shot and persistent benchmark paths accept a kernel selection. Rebuilt the shader library explicitly with `zig build -Denable-metal=true metal-lib`; note that `zig build -Denable-metal=true` installs the CLI but does not by itself refresh `zig-out/metal/kernels.metallib`.
- Correctness results:
  - Tiny scalar and threadgroup fixtures passed with `mismatches=0`.
  - Full-vocab threadgroup fixture passed: expected/actual top-1 `353`, top-20 set match `true`, `mismatches=0`, `max_abs_diff=0.0000019073486`.
  - HF bridge alignment still passed for the three required prompts with max absolute logit diff `0.0`.
- Comparable benchmark samples with `--kernel-repeats 3 --persistent-iters 3`:
  - Scalar persistent path: `persistent_ms_per_kernel_repeat=13.458 ms`; one-shot command-buffer per-repeat mean `19.152 ms`.
  - Threadgroup persistent path: `persistent_ms_per_kernel_repeat=13.125 ms`; one-shot command-buffer per-repeat mean `19.473 ms`.
- Interpretation: the threadgroup kernel is correct and gives a small (~2.5%) persistent-loop improvement in this sample, but it is below the predeclared threshold for a stable performance win and the one-shot metric is slightly worse/noisy. Keep it as an experimental opt-in kernel for future profiling rather than making it the default.
- Verification commands run:
  - `zig fmt src/metal_logits_test.zig`
  - `zig build -Denable-metal=true`
  - `zig build -Denable-metal=true metal-lib`
  - `./zig-out/bin/metal_logits_v1 zig-out/metal/kernels.metallib --kernel scalar`
  - `./zig-out/bin/metal_logits_v1 zig-out/metal/kernels.metallib --kernel threadgroup`
  - `./zig-out/bin/metal_logits_v1 zig-out/metal/kernels.metallib --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel threadgroup --kernel-repeats 1`
  - `uv run python scripts/benchmark_metal_logits.py --warmup 1 --repeats 2 --cpu-repeats 1 --kernel scalar --kernel-repeats 3 --persistent-iters 3 --out artifacts/benchmarks/g053_scalar_compare.json`
  - `uv run python scripts/benchmark_metal_logits.py --warmup 1 --repeats 2 --cpu-repeats 1 --kernel threadgroup --kernel-repeats 3 --persistent-iters 3 --out artifacts/benchmarks/g053_threadgroup_compare.json`
  - `zig build`
  - `zig build -Dtarget=x86_64-linux --summary all`
  - `./zig-out/bin/metal_logits_v1 zig-out/metal/kernels.metallib --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel scalar --kernel-repeats 2`
  - `./zig-out/bin/metal_logits_v1 zig-out/metal/kernels.metallib --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel threadgroup --kernel-repeats 2`
  - `uv run python scripts/run_alignment_tests.py`
- Next optimization target: G054 should focus on fixture memory layout/transfer overhead, because process-level fixture load still dominates end-to-end wall time.

## 2026-06-10 Resumed ultragoal G054 — fixture load and transfer bottleneck instrumentation
- Added benchmark clarity instrumentation to `metal_logits_v1` JSON output: `fixture_load_ms`, `metal_bridge_wall_ms`, `host_compare_ms`, and `total_cli_measured_ms`. These fields are also summarized by `scripts/benchmark_metal_logits.py` when present.
- The implementation uses Zig 0.16's `std.Io.Clock.awake.now(io)` timing API. Initial attempts with older `std.time.Timer` / `std.time.nanoTimestamp` APIs failed under the installed Zig toolchain and were corrected before commit.
- Latest instrumented full-vocab scalar check remained correct: expected/actual top-1 `353`, top-20 set match `true`, `mismatches=0`, `max_abs_diff=0.000011444092`.
- Latest instrumented benchmark (`--kernel scalar --kernel-repeats 5 --persistent-iters 3`) produced this bottleneck split:
  - One-shot CLI wall mean: `2384.427 ms`.
  - CLI-measured total mean: `2127.364 ms`.
  - Fixture load mean: `1941.055 ms`.
  - Metal bridge wall mean: `162.553 ms`.
  - Host compare mean: `22.918 ms`.
  - Command-buffer per-repeat mean: `16.102 ms`.
  - Persistent sample: fixture load `1742.705 ms`, bridge wall `272.099 ms`, persistent loop `196.435 ms` for `3×5` kernel repeats, `persistent_ms_per_kernel_repeat=13.096 ms`.
- Interpretation: the dominant end-to-end bottleneck is confirmed as fixture file load plus host-side copying/materialization, not the LM-head command-buffer itself. The next throughput-relevant path should avoid the current fixture format's full read-and-copy cycle, memory-map/reuse host arrays, or move toward a real persistent backend that does not reload a ~971 MiB fixture per process.
- Verification commands run:
  - `zig fmt src/metal_logits_test.zig`
  - `zig build`
  - `zig build -Denable-metal=true metal-lib`
  - `zig build -Denable-metal=true`
  - `./zig-out/bin/metal_logits_v1 zig-out/metal/kernels.metallib --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel scalar --kernel-repeats 2`
  - `uv run python scripts/benchmark_metal_logits.py --warmup 1 --repeats 2 --cpu-repeats 1 --kernel scalar --kernel-repeats 5 --persistent-iters 3 --out artifacts/benchmarks/g054_instrumented_breakdown.json`
  - `zig build -Dtarget=x86_64-linux --summary all`
  - `uv run python scripts/run_alignment_tests.py`
- Remaining limitation: this milestone improves measurement clarity rather than removing the bottleneck. It does not change the fixture binary format or the default HF bridge.

## 2026-06-10 Resumed ultragoal G056 — direct fixture loader
- Replaced the `metal_logits_v1` fixture loader's full-file `readFileAlloc` + per-float conversion loop with direct positional reads into the destination `[]f32` buffers for hidden, weights, and expected logits. The fixture binary format remains unchanged.
- Correctness remained green for scalar and threadgroup kernels on the full-vocab Qwen fixture: top-1 `353`, top-20 set match `true`, `mismatches=0`; scalar max abs diff remained `0.000011444092` and threadgroup max abs diff remained `0.0000019073486`.
- Benchmark improvement versus G054 instrumentation:
  - G054 scalar one-shot CLI wall mean: `2384.427 ms`; G056: `813.716 ms` (~65.9% lower).
  - G054 CLI-measured total mean: `2127.364 ms`; G056: `551.057 ms` (~74.1% lower).
  - G054 fixture load mean: `1941.055 ms`; G056: `357.533 ms` (~81.6% lower).
  - G056 command-buffer per-repeat mean was `17.181 ms`, so the end-to-end gain came from reducing host fixture materialization rather than changing kernel math.
  - G056 persistent sample: fixture load `372.467 ms`, persistent loop `216.218 ms` for `3×5` repeats, `persistent_ms_per_kernel_repeat=14.415 ms`.
- Verification commands run:
  - `zig fmt src/metal_logits_test.zig`
  - `zig build -Denable-metal=true`
  - `./zig-out/bin/metal_logits_v1 zig-out/metal/kernels.metallib --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel scalar --kernel-repeats 2`
  - `./zig-out/bin/metal_logits_v1 zig-out/metal/kernels.metallib --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel threadgroup --kernel-repeats 2`
  - `uv run python scripts/benchmark_metal_logits.py --warmup 1 --repeats 2 --cpu-repeats 1 --kernel scalar --kernel-repeats 5 --persistent-iters 3 --out artifacts/benchmarks/g056_direct_loader.json`
  - `zig build`
  - `zig build -Dtarget=x86_64-linux --summary all`
  - `uv run python scripts/run_alignment_tests.py`
- Remaining bottleneck: the CLI still reads hundreds of MiB of fixture data and copies weights into a shared Metal buffer once per process. A true live backend or memory-mapped/persistent process would be needed to remove that remaining cost.

## 2026-06-10 Resumed ultragoal G057 — opt-in no-copy Metal buffers
- Added an opt-in `--buffer-mode nocopy` path for `metal_logits_v1` and `scripts/benchmark_metal_logits.py`. The default remains `--buffer-mode copy`; the no-copy path attempts `newBufferWithBytesNoCopy` for hidden, row-major weights, and logits output, then falls back to the existing copy-backed shared buffers if Metal refuses any allocation.
- Extended the Metal bridge result JSON with `buffer_mode` and `used_no_copy_buffers` so benchmark reports distinguish a requested no-copy experiment from an actual no-copy run.
- Correctness remained green on the full-vocab Qwen LM-head fixture:
  - Copy-backed scalar: top-1 `353`, top-20 set match `true`, `mismatches=0`.
  - No-copy scalar: top-1 `353`, top-20 set match `true`, `mismatches=0`, `used_no_copy_buffers=true`.
  - Final preservation check for no-copy scalar with `--kernel-repeats 2`: top-1 `353`, top-20 set match `true`, `mismatches=0`, `max_abs_diff=0.000011444092`.
  - HF bridge alignment still passed for `"Hi,"`, `"The capital of China is"`, and `"What is 1+1?"` with max absolute logit diff `0.0` and sampling smoke defaults `temperature=0.6`, `top_p=0.95`, `top_k=20`.
- Benchmark comparison with `--kernel scalar --kernel-repeats 5 --persistent-iters 3`:
  - Copy-backed run (`artifacts/benchmarks/g057_copy.json`): wall mean `1117.524 ms`; command-buffer per-repeat mean `28.367 ms`; fixture load mean `389.891 ms`; bridge wall mean `356.288 ms`; measured total mean `769.560 ms`; persistent setup `124.786 ms`; persistent per-kernel-repeat `14.815 ms`.
  - No-copy run (`artifacts/benchmarks/g057_nocopy.json`): wall mean `740.003 ms`; command-buffer per-repeat mean `14.191 ms`; fixture load mean `364.962 ms`; bridge wall mean `88.101 ms`; measured total mean `475.901 ms`; persistent setup `19.934 ms`; persistent per-kernel-repeat `12.806 ms`.
- Interpretation: this is a real host-transfer/setup win for the prototype benchmark path. The no-copy mode removes the large shared-buffer input copy and final output copy when Metal accepts host-backed shared buffers, improving measured total by ~38.2% versus the copy-backed G057 sample and reducing bridge wall time by ~75.3%. It remains opt-in until a follow-on milestone validates whether it should become the default benchmark mode.
- Verification commands run:
  - `zig fmt src/metal_logits_test.zig`
  - `zig build -Denable-metal=true`
  - `./zig-out/bin/metal_logits_v1 zig-out/metal/kernels.metallib --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel scalar --buffer-mode copy --kernel-repeats 2`
  - `./zig-out/bin/metal_logits_v1 zig-out/metal/kernels.metallib --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel scalar --buffer-mode nocopy --kernel-repeats 2`
  - `uv run python scripts/benchmark_metal_logits.py --warmup 1 --repeats 2 --cpu-repeats 1 --kernel scalar --buffer-mode copy --kernel-repeats 5 --persistent-iters 3 --out artifacts/benchmarks/g057_copy.json`
  - `uv run python scripts/benchmark_metal_logits.py --warmup 1 --repeats 2 --cpu-repeats 1 --kernel scalar --buffer-mode nocopy --kernel-repeats 5 --persistent-iters 3 --out artifacts/benchmarks/g057_nocopy.json`
  - `zig build`
  - `zig build -Dtarget=x86_64-linux --summary all`
  - `./zig-out/bin/metal_logits_v1 zig-out/metal/kernels.metallib --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel scalar --buffer-mode nocopy --kernel-repeats 2`
  - `uv run python scripts/run_alignment_tests.py`
- Next optimization target: run a no-copy-first defaulting experiment or a higher-repeat scalar/threadgroup matrix before promoting any default behavior; keep copy-backed mode as the rollback path.

## 2026-06-10 Resumed ultragoal G058 — no-copy default validation rejected; strict opt-in retained
- Tested a no-copy-first defaulting hypothesis after G057 showed a clear opt-in no-copy win. A temporary local default change made no-flag `metal_logits_v1` and no-flag `benchmark_metal_logits.py` use `nocopy`; default correctness and benchmark metrics matched explicit no-copy.
- Independent review found a real promotion blocker: the G057 no-copy request could silently fall back to copy-backed buffers, so a default promotion could pass correctness and benchmark gates while not actually exercising no-copy. It also noted lifetime/alignment documentation gaps for `newBufferWithBytesNoCopy`.
- Decision: do **not** promote no-copy to the default yet. Reverted the temporary default change so `copy` remains the no-flag CLI/benchmark behavior and explicit rollback baseline. Kept no-copy as an opt-in throughput experiment.
- Hardened the opt-in no-copy contract instead:
  - `nn_metal_run_logits_matmul` and persistent benchmark mode now fail if `use_no_copy_buffers` is requested but any no-copy Metal buffer cannot be created, rather than silently falling back to copy-backed buffers.
  - Moved no-copy output zeroing after successful buffer creation, avoiding output clobber on setup failure.
  - Documented the synchronous host-memory lifetime contract in `src/metal_bridge.h`.
  - Added `actual_buffer_mode` to `metal_logits_v1` JSON.
  - Added benchmark report fields `requested_buffer_mode`, `actual_used_no_copy_count`, `actual_used_no_copy_all`, and `persistent_actual_used_no_copy`; the benchmark script now exits nonzero if `--buffer-mode nocopy` was requested but any measured run did not actually use no-copy buffers.
- Correctness and benchmark evidence:
  - Default no-flag full-vocab scalar remains copy-backed and correct: `buffer_mode=copy`, `actual_buffer_mode=copy`, `used_no_copy_buffers=false`, top-1 `353`, top-20 set match `true`, `mismatches=0`.
  - Strict no-copy full-vocab scalar remains correct: `buffer_mode=nocopy`, `actual_buffer_mode=nocopy`, `used_no_copy_buffers=true`, top-1 `353`, top-20 set match `true`, `mismatches=0`.
  - Persistent strict no-copy check remained correct with `persistent_ms_per_kernel_repeat=13.100 ms` in the direct CLI sample.
  - Strict benchmark comparison:
    - Copy (`artifacts/benchmarks/g058_copy_strict.json`): measured total mean `582.781 ms`, bridge wall mean `199.747 ms`, wall mean `852.367 ms`, persistent per-kernel-repeat `13.016 ms`.
    - No-copy (`artifacts/benchmarks/g058_nocopy_strict.json`): measured total mean `473.286 ms`, bridge wall mean `87.373 ms`, wall mean `737.882 ms`, persistent per-kernel-repeat `12.839 ms`, `actual_used_no_copy_all=true`, `persistent_actual_used_no_copy=true`.
  - HF bridge alignment still passed for `"Hi,"`, `"The capital of China is"`, and `"What is 1+1?"` with max absolute logit diff `0.0`; sampling smoke defaults remain `temperature=0.6`, `top_p=0.95`, `top_k=20`.
- Verification commands run:
  - `python3 -m py_compile scripts/benchmark_metal_logits.py`
  - `zig fmt src/metal_logits_test.zig`
  - `zig build -Denable-metal=true`
  - `./zig-out/bin/metal_logits_v1 zig-out/metal/kernels.metallib --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel scalar --kernel-repeats 2`
  - `./zig-out/bin/metal_logits_v1 zig-out/metal/kernels.metallib --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel scalar --buffer-mode nocopy --kernel-repeats 2`
  - `./zig-out/bin/metal_logits_v1 zig-out/metal/kernels.metallib --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel scalar --buffer-mode nocopy --kernel-repeats 5 --benchmark-iters 3`
  - `uv run python scripts/benchmark_metal_logits.py --warmup 1 --repeats 2 --cpu-repeats 1 --kernel scalar --buffer-mode copy --kernel-repeats 5 --persistent-iters 3 --out artifacts/benchmarks/g058_copy_strict.json`
  - `uv run python scripts/benchmark_metal_logits.py --warmup 1 --repeats 2 --cpu-repeats 1 --kernel scalar --buffer-mode nocopy --kernel-repeats 5 --persistent-iters 3 --out artifacts/benchmarks/g058_nocopy_strict.json`
  - `zig build`
  - `zig build -Dtarget=x86_64-linux --summary all`
  - `uv run python scripts/run_alignment_tests.py`
- Next optimization target: continue from strict opt-in no-copy and compare kernel variants under this cleaner measurement contract; do not default no-copy until alignment/ownership requirements are made portable or explicitly page-aligned.

## 2026-06-10 Resumed ultragoal G059 — strict Metal logits benchmark matrix harness
- Added `scripts/benchmark_metal_logits_matrix.py`, a small wrapper that runs the existing `benchmark_metal_logits.py` across the 2×2 matrix of `scalar`/`threadgroup` kernels and `copy`/`nocopy` buffer modes.
- The harness keeps the previous benchmark script as the single row executor, so each cell still runs the existing Metal full-vocab top-k correctness check, CPU fixture reference, strict no-copy actual-mode enforcement, and persistent benchmark path.
- Added aggregate safeguards and decision-quality output:
  - `matrix_version`, `verdict`, and `ranking_confidence`.
  - Per-cell `variant_id`, `status`, failure reasons, mean/median timing summaries, actual no-copy evidence, and artifact path.
  - Median measured-total ranking and persistent per-kernel-repeat ranking.
  - Interaction summaries for no-copy-vs-copy within each kernel and threadgroup-vs-scalar within each buffer mode.
  - A promotion note that low-repeat runs are directional only unless `repeats >= 7` and `persistent_iters >= 10`.
- Independent review recommendations were incorporated: fail the matrix if any row fails correctness or buffer-mode evidence; rank only after all rows pass; keep wall/bridge timing separate from persistent command-buffer timing to avoid confusing host-transfer improvements with kernel improvements.
- Validation run:
  - `python3 -m py_compile scripts/benchmark_metal_logits.py scripts/benchmark_metal_logits_matrix.py`
  - `uv run python scripts/benchmark_metal_logits_matrix.py --no-build --warmup 1 --repeats 2 --cpu-repeats 1 --kernel-repeats 5 --persistent-iters 3 --out artifacts/benchmarks/g059_matrix.json --artifact-dir artifacts/benchmarks/g059_matrix_runs`
- Matrix verdict: `pass`; ranking confidence: `low` because this run used only two repeats and one persistent aggregate per cell.
- Latest matrix highlights from `artifacts/benchmarks/g059_matrix.json`:
  - All four variants passed full-vocab correctness: top-1 `353`, top-20 set match `true`, `mismatches=0`.
  - `nocopy` rows reported `actual_used_no_copy_all=true` and `persistent_actual_used_no_copy=true`; `copy` rows reported no no-copy use.
  - Median measured total ranking in this noisy sample: `scalar/nocopy` (`751.788 ms`), `threadgroup/nocopy` (`775.943 ms`), `threadgroup/copy` (`1596.812 ms`), `scalar/copy` (`1660.020 ms`).
  - Persistent per-kernel-repeat ranking: `threadgroup/nocopy` (`15.010 ms`), `scalar/nocopy` (`16.298 ms`), `scalar/copy` (`31.651 ms`), `threadgroup/copy` (`33.023 ms`).
  - Interaction summary shows a strong no-copy host/setup effect in both kernels, while the threadgroup-vs-scalar effect is mixed/noisy and should not drive a default change from this low-repeat sample.
- Interpretation: the benchmark matrix harness is now a reusable gate for future default or kernel changes. The result supports continuing to investigate no-copy and threadgroup interactions, but does not yet justify promoting either default because the sample is low-confidence and device-specific.
- Next optimization target: use the matrix harness with higher repeats or add repeated persistent samples to reduce noise before making any kernel/default change.

## 2026-06-10 Resumed ultragoal G060 — higher-confidence Metal logits matrix
- Ran the strict matrix harness at the planned higher-confidence settings before changing any defaults:
  - `repeats=7`
  - `persistent_iters=10`
  - `kernel_repeats=5`
  - variants: `scalar/copy`, `scalar/nocopy`, `threadgroup/copy`, `threadgroup/nocopy`
- During the first G060 run, the harness correctly passed all rows but still labeled ranking confidence as `low` despite meeting the high-repeat threshold. Fixed the harness confidence label so runs with `repeats >= 7` and `persistent_iters >= 10` report `ranking_confidence="medium"`; lower-repeat runs remain `low`.
- Final validation command:
  - `python3 -m py_compile scripts/benchmark_metal_logits_matrix.py`
  - `uv run python scripts/benchmark_metal_logits_matrix.py --no-build --warmup 1 --repeats 7 --cpu-repeats 1 --kernel-repeats 5 --persistent-iters 10 --out artifacts/benchmarks/g060_matrix_r7_p10.json --artifact-dir artifacts/benchmarks/g060_matrix_r7_p10_runs`
- Final matrix verdict: `pass`; ranking confidence: `medium`.
- Correctness evidence:
  - All four variants passed full-vocab correctness: top-1 `353`, top-20 set match `true`, `mismatches=0`.
  - `nocopy` rows reported `actual_used_no_copy_all=true` and `persistent_actual_used_no_copy=true`.
  - `copy` rows reported `actual_used_no_copy_all=false` and `persistent_actual_used_no_copy=false`.
- Median measured-total ranking from `artifacts/benchmarks/g060_matrix_r7_p10.json`:
  1. `threadgroup/nocopy`: `482.850 ms`
  2. `scalar/nocopy`: `485.332 ms`
  3. `threadgroup/copy`: `545.500 ms`
  4. `scalar/copy`: `550.653 ms`
- Persistent per-kernel-repeat ranking:
  1. `threadgroup/nocopy`: `10.876 ms`
  2. `threadgroup/copy`: `11.072 ms`
  3. `scalar/nocopy`: `12.256 ms`
  4. `scalar/copy`: `12.514 ms`
- Interaction interpretation:
  - No-copy improves median measured-total time in both kernels by ~11-12%, confirming a host/setup effect that composes with either kernel.
  - Threadgroup improves persistent per-kernel-repeat time by ~11% under both copy and no-copy, but only improves median measured-total by ~0.5-0.9%; host/fixture overhead still dominates end-to-end CLI timing.
  - No-copy has only ~1.8-2.1% persistent-loop effect, so its main win is not kernel math; it is bridge/setup transfer reduction.
- Decision: do not change defaults in this milestone. The matrix now supports a follow-on goal to promote `threadgroup` as the kernel default **only if** an explicit full verification/review gate accepts the device-specific risk, and to keep `nocopy` opt-in until buffer ownership/alignment portability is resolved.
- Next optimization target: evaluate a conservative default change or runtime auto-selection policy using the matrix evidence, while preserving `--kernel scalar` and `--buffer-mode copy` rollback paths.

## 2026-06-10 Resumed ultragoal G061 — opt-in Metal logits auto-kernel prototype
- Added an explicit `--kernel auto` mode for `metal_logits_v1` and `scripts/benchmark_metal_logits.py` without changing no-flag behavior: default CLI and benchmark execution remain `scalar` + `copy`.
- Auto-kernel selection is conservative and reversible:
  - Explicit `--kernel scalar` still runs `actual_kernel=scalar`.
  - Explicit `--kernel threadgroup` still runs `actual_kernel=threadgroup`.
  - Explicit `--kernel auto` probes the threadgroup LM-head pipeline and selects `threadgroup` only when `maxTotalThreadsPerThreadgroup >= 256`; otherwise it falls back to `scalar`.
  - Added `nn_metal_probe_logits_kernel` so the capability decision is based on the actual logits pipeline rather than a generic device probe.
- Reporting and benchmark schema updates:
  - `metal_logits_v1` JSON now includes both requested `kernel` and resolved `actual_kernel`.
  - `benchmark_metal_logits.py` now accepts `--kernel auto`, writes `requested_kernel`, `actual_kernel`, and `actual_kernels_seen`, and fails auto-mode runs if any one-shot or persistent record omits a concrete `actual_kernel` or resolves inconsistently.
  - Existing explicit scalar/threadgroup matrix rows remain comparable; `auto` is a diagnostic opt-in, not a default-promotion or matrix-ranking axis.
- Review feedback incorporated:
  - Architect guidance: keep auto opt-in, preserve scalar/threadgroup hard overrides, resolve through capability evidence, report requested vs resolved kernel.
  - Code-review guidance: require concrete/invariant `actual_kernel` in auto benchmark artifacts and document auto behavior in CLI help.
- Correctness and smoke evidence:
  - Default no-flag full-vocab run preserved `kernel=scalar`, `actual_kernel=scalar`, `buffer_mode=copy`, `actual_buffer_mode=copy`, top-1 `353`, top-20 set match `true`, `mismatches=0`.
  - Explicit scalar full-vocab run preserved `actual_kernel=scalar`, top-1 `353`, top-20 set match `true`, `mismatches=0`.
  - Explicit threadgroup full-vocab run preserved `actual_kernel=threadgroup`, top-1 `353`, top-20 set match `true`, `mismatches=0`.
  - Auto full-vocab run on Apple M4 resolved to `actual_kernel=threadgroup`, top-1 `353`, top-20 set match `true`, `mismatches=0`.
  - Auto persistent run on Apple M4 resolved to `actual_kernel=threadgroup`, top-1 `353`, top-20 set match `true`, `mismatches=0`, persistent per-kernel-repeat `12.709 ms` in the direct CLI sample.
- Focused benchmark evidence:
  - `artifacts/benchmarks/g061_auto_kernel/auto_nocopy_benchmark.json`: requested `auto`, resolved `actual_kernel=threadgroup`, `actual_kernels_seen=["threadgroup"]`, `actual_used_no_copy_all=true`, `persistent_actual_used_no_copy=true`, median measured total `526.004 ms`, persistent per-kernel-repeat `15.693 ms`.
  - `artifacts/benchmarks/g061_auto_kernel/threadgroup_nocopy_benchmark.json`: requested `threadgroup`, resolved `actual_kernel=threadgroup`, `actual_kernels_seen=["threadgroup"]`, `actual_used_no_copy_all=true`, `persistent_actual_used_no_copy=true`, median measured total `540.231 ms`, persistent per-kernel-repeat `15.044 ms`.
  - `artifacts/benchmarks/g061_auto_kernel/default_required_benchmark.json`: requested/default `scalar`, resolved `actual_kernel=scalar`, median measured total `638.200 ms`, command-buffer per-repeat median `19.222 ms`.
- Full verification commands run:
  - `zig fmt src/metal_logits_test.zig`
  - `python3 -m py_compile scripts/benchmark_metal_logits.py`
  - `zig build -Denable-metal=true`
  - Direct CLI full-vocab checks for no-flag default, explicit scalar, explicit threadgroup, auto, and auto persistent.
  - `uv run python scripts/benchmark_metal_logits.py --warmup 1 --repeats 2 --cpu-repeats 1 --kernel auto --buffer-mode nocopy --kernel-repeats 5 --persistent-iters 3 --out artifacts/benchmarks/g061_auto_kernel/auto_nocopy_benchmark.json`
  - `uv run python scripts/benchmark_metal_logits.py --warmup 1 --repeats 2 --cpu-repeats 1 --kernel threadgroup --buffer-mode nocopy --kernel-repeats 5 --persistent-iters 3 --out artifacts/benchmarks/g061_auto_kernel/threadgroup_nocopy_benchmark.json`
  - `zig build`
  - `zig build -Dtarget=x86_64-linux --summary all`
  - `zig build -Denable-metal=true metal-smoke`
  - `zig build -Denable-metal=true metal-logits-test -- --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel-repeats 2`
  - `uv run python scripts/run_alignment_tests.py`
  - `uv run python scripts/benchmark_metal_logits.py --warmup 1 --repeats 2 --cpu-repeats 1 --kernel-repeats 5 --out artifacts/benchmarks/g061_auto_kernel/default_required_benchmark.json`
- HF bridge alignment remained intact for the three required prompts with max absolute logit diff `0.0`; sampling smoke remained `temperature=0.6`, `top_p=0.95`, `top_k=20`, candidate count `20`, selected token `353`.
- Decision: keep the auto mode because it is opt-in, auditable, correct, and rollback-safe. Do not promote threadgroup or no-copy defaults from this milestone; use auto only as a diagnostic path for future staged throughput experiments.
- Next optimization target: reduce repeated auto/benchmark setup overhead by adding a lower-overhead kernel-resolution/probe path or cache so auto diagnostics do not add avoidable library/pipeline setup work to each CLI process.

## 2026-06-10 Resumed ultragoal G062 — bridge-side auto-kernel resolution
- Follow-on hypothesis from G061: `--kernel auto` was safe but did a Zig-side threadgroup pipeline probe before calling the Metal bridge, then the bridge loaded the library and built a pipeline again for the actual run. That made auto diagnostics auditable but structurally duplicated setup work.
- Implemented the smallest reversible cleanup:
  - Removed the separate `nn_metal_probe_logits_kernel` API added in G061.
  - Added bridge-side auto resolution inside normal logits pipeline creation.
  - `auto` now opens the Metal library once per run, attempts the threadgroup pipeline in that same library, chooses it only when `maxTotalThreadsPerThreadgroup >= 256`, and otherwise falls back to scalar.
  - Added `actual_kernel_name` to `NnMetalSmokeResult` so `metal_logits_v1` still reports concrete `actual_kernel` from the bridge result.
  - Kept default/no-flag behavior unchanged: `scalar` + `copy`; explicit `scalar` and `threadgroup` remain hard overrides.
- Direct correctness evidence after the change:
  - Default no-flag full-vocab run: `kernel=scalar`, `actual_kernel=scalar`, top-1 `353`, top-20 set match `true`, `mismatches=0`.
  - Explicit scalar full-vocab run: `actual_kernel=scalar`, top-1 `353`, top-20 set match `true`, `mismatches=0`.
  - Explicit threadgroup full-vocab run: `actual_kernel=threadgroup`, top-1 `353`, top-20 set match `true`, `mismatches=0`.
  - Auto full-vocab run on Apple M4: `kernel=auto`, bridge-reported `actual_kernel=threadgroup`, top-1 `353`, top-20 set match `true`, `mismatches=0`.
  - Auto persistent full-vocab run: `actual_kernel=threadgroup`, top-1 `353`, top-20 set match `true`, `mismatches=0`.
- Focused sequential benchmark evidence with `--no-build` to avoid build/parallel contention:
  - `artifacts/benchmarks/g062_auto_probe/auto_nocopy_seq.json`: requested `auto`, resolved `actual_kernel=threadgroup`, `actual_kernels_seen=["threadgroup"]`, `actual_used_no_copy_all=true`, median bridge wall `97.197 ms`, median measured total `506.939 ms`, persistent setup `16.699 ms`, persistent per-kernel-repeat `13.634 ms`.
  - `artifacts/benchmarks/g062_auto_probe/threadgroup_nocopy_seq.json`: requested `threadgroup`, resolved `actual_kernel=threadgroup`, `actual_kernels_seen=["threadgroup"]`, `actual_used_no_copy_all=true`, median bridge wall `110.580 ms`, median measured total `554.606 ms`, persistent setup `26.131 ms`, persistent per-kernel-repeat `13.978 ms`.
  - Interpretation: absolute timings remain noisy on this machine, but auto no longer has a structural extra pre-run probe and is comparable to explicit threadgroup in the same sequential run. This is a setup-path cleanup, not a default promotion or stable throughput-win claim.
- Required/default benchmark gate after the change:
  - `artifacts/benchmarks/g062_auto_probe/default_required_benchmark.json`: requested/default `scalar`, resolved `actual_kernel=scalar`, top-1 `353`, top-20 set match `true`, `mismatches=0`. Timing was unusually noisy (`metal_cli_bridge_wall_ms` median `671.159 ms`), so it is recorded as correctness evidence, not performance evidence.
- Full verification commands run:
  - `zig fmt src/metal_logits_test.zig`
  - `python3 -m py_compile scripts/benchmark_metal_logits.py`
  - `zig build -Denable-metal=true`
  - Direct CLI full-vocab checks for no-flag default, explicit scalar, explicit threadgroup, auto, and auto persistent.
  - `uv run python scripts/benchmark_metal_logits.py --no-build --warmup 1 --repeats 3 --cpu-repeats 1 --kernel auto --buffer-mode nocopy --kernel-repeats 5 --persistent-iters 3 --out artifacts/benchmarks/g062_auto_probe/auto_nocopy_seq.json`
  - `uv run python scripts/benchmark_metal_logits.py --no-build --warmup 1 --repeats 3 --cpu-repeats 1 --kernel threadgroup --buffer-mode nocopy --kernel-repeats 5 --persistent-iters 3 --out artifacts/benchmarks/g062_auto_probe/threadgroup_nocopy_seq.json`
  - `zig build`
  - `zig build -Dtarget=x86_64-linux --summary all`
  - `zig build -Denable-metal=true metal-smoke`
  - `zig build -Denable-metal=true metal-logits-test -- --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel-repeats 2`
  - `uv run python scripts/benchmark_metal_logits.py --no-build --warmup 1 --repeats 2 --cpu-repeats 1 --kernel-repeats 5 --out artifacts/benchmarks/g062_auto_probe/default_required_benchmark.json`
  - `uv run python scripts/run_alignment_tests.py`
- HF bridge alignment remained intact for `"Hi,"`, `"The capital of China is"`, and `"What is 1+1?"` with max absolute logit diff `0.0`; sampling smoke remained `temperature=0.6`, `top_p=0.95`, `top_k=20`, candidate count `20`, selected token `353`.
- Decision: keep the bridge-side auto resolution cleanup because it removes redundant setup structure while preserving defaults, rollback paths, and alignment. Do not claim a stable throughput win due noisy absolute measurements.
- Next optimization target: reduce benchmark timing noise and improve decision quality by recording process/thermal/noise context or adding repeated persistent samples in the matrix harness before any further default/promotion work.

## 2026-06-10 Resumed ultragoal G063 — repeated persistent benchmark samples
- Follow-on hypothesis from G062: single persistent benchmark samples were too noisy to support confident kernel/default decisions. Added an explicit multi-sample path to improve decision quality without changing default benchmark behavior.
- Implemented tooling changes:
  - `scripts/benchmark_metal_logits.py` now accepts `--persistent-samples N` when `--persistent-iters` is enabled.
  - Default remains `--persistent-samples 1`, preserving the existing `persistent_metal` object for compatibility.
  - Multi-sample runs add `persistent_metal_samples` and `persistent_metal_summary` with count, wall/setup/elapsed/per-iter/per-kernel-repeat summaries, actual kernels seen, and no-copy evidence.
  - `scripts/benchmark_metal_logits_matrix.py` now accepts explicit `--persistent-samples`, passes it through to each row, validates persistent summaries, and ranks persistent timing by the summarized median when present.
- Direct benchmark validation:
  - `artifacts/benchmarks/g063_persistent_samples/auto_samples2.json`: requested `auto`, resolved `actual_kernel=threadgroup`, `actual_kernels_seen=["threadgroup"]`, `actual_used_no_copy_all=true`, two persistent samples captured, persistent per-kernel-repeat summary median `16.109 ms`, min `15.400 ms`, max `16.817 ms`.
  - Default compatibility check `artifacts/benchmarks/g063_persistent_samples/default_samples1.json`: omitted `--persistent-samples`, preserved non-null `persistent_metal`, kept `persistent_metal_samples=null`, emitted `persistent_metal_summary.count=1`, resolved `actual_kernel=scalar`, and preserved correctness.
- Matrix validation:
  - `artifacts/benchmarks/g063_persistent_samples/matrix_samples2.json`: verdict `pass`, ranking confidence `low` because repeats/samples were intentionally small for validation.
  - Rows: `scalar/nocopy` and `threadgroup/nocopy` both passed top-1/top-20 correctness and actual no-copy evidence.
  - Matrix persistent ranking used summary medians: `threadgroup/nocopy` `13.076 ms` vs `scalar/nocopy` `17.524 ms` per kernel repeat in this validation run.
  - This is a tooling validation sample, not a default-promotion claim.
- Full verification commands run:
  - `python3 -m py_compile scripts/benchmark_metal_logits.py scripts/benchmark_metal_logits_matrix.py`
  - `uv run python scripts/benchmark_metal_logits.py --no-build --warmup 1 --repeats 2 --cpu-repeats 1 --kernel auto --buffer-mode nocopy --kernel-repeats 5 --persistent-iters 2 --persistent-samples 2 --out artifacts/benchmarks/g063_persistent_samples/auto_samples2.json`
  - `uv run python scripts/benchmark_metal_logits_matrix.py --no-build --warmup 1 --repeats 2 --cpu-repeats 1 --kernel-repeats 5 --persistent-iters 2 --persistent-samples 2 --kernels scalar,threadgroup --buffer-modes nocopy --out artifacts/benchmarks/g063_persistent_samples/matrix_samples2.json --artifact-dir artifacts/benchmarks/g063_persistent_samples/matrix_runs`
  - `uv run python scripts/benchmark_metal_logits.py --no-build --warmup 1 --repeats 1 --cpu-repeats 1 --kernel scalar --buffer-mode copy --kernel-repeats 2 --persistent-iters 1 --out artifacts/benchmarks/g063_persistent_samples/default_samples1.json`
  - `zig build`
  - `zig build -Dtarget=x86_64-linux --summary all`
  - `zig build -Denable-metal=true metal-smoke`
  - `zig build -Denable-metal=true metal-logits-test -- --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel-repeats 2`
  - `uv run python scripts/run_alignment_tests.py`
- HF bridge alignment remained intact for all three required prompts with max absolute logit diff `0.0`; sampling smoke remained `temperature=0.6`, `top_p=0.95`, `top_k=20`, candidate count `20`, selected token `353`.
- Decision: keep repeated persistent samples as opt-in measurement infrastructure. Do not change defaults or promote kernels based on this validation-sized run.
- Next optimization target: use the improved matrix sampling to run a higher-confidence focused comparison, then decide whether a small integration convenience (for example auto diagnostics in matrix via explicit `--kernels auto`) is worthwhile without changing defaults.

## 2026-06-10 Resumed ultragoal G064 — sampled focused Metal logits matrix
- Ran the G063 multi-sample matrix path as an evidence-only milestone before any further implementation or default decision.
- Command:
  - `uv run python scripts/benchmark_metal_logits_matrix.py --no-build --warmup 1 --repeats 5 --cpu-repeats 1 --kernel-repeats 5 --persistent-iters 5 --persistent-samples 5 --kernels scalar,threadgroup --buffer-modes nocopy --out artifacts/benchmarks/g064_sampled_focus/matrix_r5_p5_s5.json --artifact-dir artifacts/benchmarks/g064_sampled_focus/runs`
- Matrix verdict: `pass`; ranking confidence remains `low` because this was a focused sampled run below the existing medium-confidence threshold (`repeats >= 7` and `persistent_iters >= 10`).
- Correctness evidence:
  - `scalar/nocopy` and `threadgroup/nocopy` rows both passed top-1/top-20 checks with `mismatches=0` and `actual_used_no_copy_all=true`.
  - `scalar/nocopy` actual kernels seen: `["scalar"]`.
  - `threadgroup/nocopy` actual kernels seen: `["threadgroup"]`.
- One-shot measured-total evidence:
  - `scalar/nocopy` median measured total: `546.638 ms`.
  - `threadgroup/nocopy` median measured total: `568.926 ms`.
  - Interaction delta: threadgroup was `+22.288 ms` (`+4.08%`) slower on measured total in this run, likely reflecting remaining host/fixture noise rather than pure kernel math.
- Repeated persistent kernel evidence:
  - `scalar/nocopy` persistent per-kernel-repeat median across 5 samples: `16.145 ms` (min `13.904 ms`, max `16.578 ms`).
  - `threadgroup/nocopy` persistent per-kernel-repeat median across 5 samples: `13.027 ms` (min `12.622 ms`, max `14.120 ms`).
  - Interaction delta: threadgroup was `-3.118 ms` (`-19.31%`) faster on persistent per-kernel-repeat.
- Interpretation:
  - The sampled persistent metric strengthens evidence that the threadgroup kernel improves kernel-loop math on Apple M4.
  - The one-shot measured-total metric remains dominated by fixture/host/process noise and does not support default promotion by itself.
  - No defaults changed. `scalar` + `copy` remains the default; `threadgroup`, `nocopy`, and `auto` remain explicit/diagnostic paths.
- Next optimization target: add an explicit matrix option for `auto` diagnostics or isolate fixture/host timing further, but only if it improves decision quality without affecting default inference behavior.

## 2026-06-10 Resumed ultragoal G065 — explicit auto diagnostics in the matrix harness
- Added explicit `auto` support to `scripts/benchmark_metal_logits_matrix.py` without changing the default matrix:
  - Default `--kernels` remains `scalar,threadgroup` for historical comparability.
  - Parser now accepts `auto` only when requested explicitly.
  - Matrix summaries include `includes_auto_diagnostic`.
  - Promotion note now states auto rows are diagnostic-only and require explicit scalar/threadgroup confirmation before default changes.
- Added row validation for requested-vs-actual kernel evidence:
  - Auto rows must report concrete `actual_kernel` and `actual_kernels_seen` values in `{scalar, threadgroup}`.
  - Explicit scalar/threadgroup rows must not report mismatched actual kernels.
  - Persistent summaries must also report concrete actual-kernel evidence.
- Validation matrix command:
  - `uv run python scripts/benchmark_metal_logits_matrix.py --no-build --warmup 1 --repeats 2 --cpu-repeats 1 --kernel-repeats 5 --persistent-iters 2 --persistent-samples 2 --kernels auto,threadgroup --buffer-modes nocopy --out artifacts/benchmarks/g065_auto_matrix/matrix_auto_threadgroup.json --artifact-dir artifacts/benchmarks/g065_auto_matrix/runs`
- Validation result:
  - Matrix verdict `pass`; `includes_auto_diagnostic=true`; ranking confidence `low`.
  - `auto/nocopy` row passed correctness, resolved `actual_kernel=threadgroup`, `actual_kernels_seen=["threadgroup"]`, `actual_used_no_copy_all=true`, persistent per-kernel-repeat summary median `14.714 ms`.
  - `threadgroup/nocopy` row passed correctness, resolved `actual_kernel=threadgroup`, `actual_kernels_seen=["threadgroup"]`, `actual_used_no_copy_all=true`, persistent per-kernel-repeat summary median `12.830 ms`.
  - The run confirms auto diagnostics can be included and audited, but does not authorize default promotion.
- Default compatibility check:
  - Import-time parse check confirmed `parse_args().kernels == "scalar,threadgroup"` when no explicit `--kernels` is provided.
- Full verification commands run:
  - `python3 -m py_compile scripts/benchmark_metal_logits.py scripts/benchmark_metal_logits_matrix.py`
  - Default parser check for `scalar,threadgroup`.
  - The explicit auto/threadgroup nocopy matrix command above.
  - `zig build`
  - `zig build -Dtarget=x86_64-linux --summary all`
  - `zig build -Denable-metal=true metal-smoke`
  - `zig build -Denable-metal=true metal-logits-test -- --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel-repeats 2`
  - `uv run python scripts/run_alignment_tests.py`
- HF bridge alignment remained intact for all three required prompts with max absolute logit diff `0.0`; sampling smoke remained `temperature=0.6`, `top_p=0.95`, `top_k=20`, candidate count `20`, selected token `353`.
- Decision: keep explicit auto matrix diagnostics as measurement infrastructure. Keep defaults unchanged.
- Next optimization target: isolate host fixture/load timing further from kernel-loop timing, likely by adding matrix/report fields that rank fixture load, bridge wall, and persistent setup separately for clearer bottleneck targeting.

## 2026-06-10 Resumed ultragoal G066 — timing-breakdown rankings in the Metal matrix
- Added report-only timing-breakdown rankings to `scripts/benchmark_metal_logits_matrix.py` so future optimization decisions can separate host/fixture/setup noise from kernel-loop behavior.
- New per-row fields include:
  - `fixture_load_median_ms`
  - `host_compare_median_ms`
  - `persistent_setup_median_ms`
  - `persistent_wall_ms_per_kernel_repeat`
- New top-level rankings include:
  - `ranked_by_bridge_wall_median_ms`
  - `ranked_by_fixture_load_median_ms`
  - `ranked_by_command_buffer_per_repeat_median_ms`
  - `ranked_by_persistent_setup_median_ms`
  - `ranked_by_persistent_wall_ms_per_kernel_repeat`
- Validation command:
  - `uv run python scripts/benchmark_metal_logits_matrix.py --no-build --warmup 1 --repeats 2 --cpu-repeats 1 --kernel-repeats 5 --persistent-iters 2 --persistent-samples 2 --kernels scalar,threadgroup --buffer-modes nocopy --out artifacts/benchmarks/g066_breakdown/matrix_breakdown.json --artifact-dir artifacts/benchmarks/g066_breakdown/runs`
- Validation result:
  - Matrix verdict `pass`; both rows preserved top-1/top-20 correctness, `mismatches=0`, and actual no-copy evidence.
  - Breakdown winners in this validation run:
    - `ranked_by_bridge_wall_median_ms`: `scalar/nocopy`.
    - `ranked_by_fixture_load_median_ms`: `scalar/nocopy`.
    - `ranked_by_command_buffer_per_repeat_median_ms`: `scalar/nocopy`.
    - `ranked_by_persistent_setup_median_ms`: `threadgroup/nocopy`.
    - `ranked_by_persistent_wall_ms_per_kernel_repeat`: `threadgroup/nocopy`.
  - Existing persistent per-kernel-repeat ranking still favored `threadgroup/nocopy` (`13.488 ms`) over `scalar/nocopy` (`15.636 ms`).
- Interpretation:
  - The new rankings make the host-vs-kernel split explicit: one-shot/host-heavy metrics may favor scalar in noisy samples, while persistent loop metrics favor threadgroup.
  - This supports continuing to isolate host overhead before changing any defaults.
- Verification commands run:
  - `python3 -m py_compile scripts/benchmark_metal_logits_matrix.py`
  - The validation matrix command above plus JSON assertions that all new rankings exist.
  - `zig build`
  - `zig build -Dtarget=x86_64-linux --summary all`
  - `zig build -Denable-metal=true metal-smoke`
  - `zig build -Denable-metal=true metal-logits-test -- --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel-repeats 2`
  - `uv run python scripts/run_alignment_tests.py`
- HF bridge alignment remained intact for all three required prompts with max absolute logit diff `0.0`; sampling smoke remained `temperature=0.6`, `top_p=0.95`, `top_k=20`, candidate count `20`, selected token `353`.
- Decision: keep the timing-breakdown rankings. No runtime/default behavior changed.
- Next optimization target: reduce host/fixture dominance in benchmarks, likely by adding a fixture-load amortization or matrix mode that runs multiple variant executions after one build and uses persistent summaries as the primary kernel metric.

## 2026-06-10 Resumed ultragoal G067 — matrix bottleneck summary
- Added a report-only `bottleneck_summary` to `scripts/benchmark_metal_logits_matrix.py` using existing timing fields.
- The summary records, per passing row:
  - measured-total median,
  - dominant observed timing bucket among fixture load, bridge wall, and host compare,
  - non-exclusive shares of measured total for those buckets,
  - command-buffer total/per-repeat medians,
  - persistent setup, persistent per-kernel-repeat, and persistent wall per kernel repeat.
- Added `ranked_by_command_buffer_total_median_ms` alongside the G066 breakdown rankings.
- Validation command:
  - `uv run python scripts/benchmark_metal_logits_matrix.py --no-build --warmup 1 --repeats 2 --cpu-repeats 1 --kernel-repeats 5 --persistent-iters 2 --persistent-samples 2 --kernels scalar,threadgroup --buffer-modes nocopy --out artifacts/benchmarks/g067_bottleneck/matrix_bottleneck.json --artifact-dir artifacts/benchmarks/g067_bottleneck/runs`
- Validation result:
  - Matrix verdict `pass`.
  - `bottleneck_summary.likely_next_focus=fixture_load`.
  - `scalar/nocopy`: fixture load share `0.756`, bridge wall share `0.193`, host compare share `0.050`.
  - `threadgroup/nocopy`: fixture load share `0.748`, bridge wall share `0.204`, host compare share `0.046`.
  - Interpretation: measured-total benchmark decisions are still fixture-load dominated in this CLI subprocess harness; persistent metrics remain better for kernel-loop comparisons.
- Verification commands run:
  - `python3 -m py_compile scripts/benchmark_metal_logits_matrix.py`
  - The validation matrix command above plus JSON assertions for `bottleneck_summary`, `likely_next_focus`, and `ranked_by_command_buffer_total_median_ms`.
  - `zig build`
  - `zig build -Dtarget=x86_64-linux --summary all`
  - `zig build -Denable-metal=true metal-smoke`
  - `zig build -Denable-metal=true metal-logits-test -- --fixture artifacts/metal/gate3/full_hi/fixture.bin --expect-topk --kernel-repeats 2`
  - `uv run python scripts/run_alignment_tests.py`
- HF bridge alignment remained intact for all three required prompts with max absolute logit diff `0.0`; sampling smoke remained `temperature=0.6`, `top_p=0.95`, `top_k=20`, candidate count `20`, selected token `353`.
- Decision: keep bottleneck summary. No runtime/default behavior changed.
- Next optimization target: reduce fixture-load dominance in benchmark decision loops, likely with an in-process multi-variant benchmark path or fixture reuse strategy rather than further kernel changes.
