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
