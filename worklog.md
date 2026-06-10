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
