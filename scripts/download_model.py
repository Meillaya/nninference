#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "Qwen/Qwen3.5-0.8B"
LOCAL_DIR = Path("Qwen3.5-0.8B")
REQUIRED = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "model.safetensors-00001-of-00001.safetensors",
}


def main() -> None:
    path = Path(snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "chat_template.jinja",
            "model.safetensors*",
            "*.md",
            "LICENSE",
        ],
    ))
    missing = sorted(name for name in REQUIRED if not (path / name).exists())
    if missing:
        raise SystemExit(f"missing required model files: {missing}")
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    print(f"downloaded {REPO_ID} -> {path} ({total / (1024**3):.2f} GiB)")
    for name in sorted(REQUIRED):
        print(f"ok {name}")


if __name__ == "__main__":
    main()
