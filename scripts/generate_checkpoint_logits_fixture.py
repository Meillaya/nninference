#!/usr/bin/env python3
"""Generate an ignored Qwen checkpoint-slice logits matmul fixture for Metal tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAGIC = b"NNLGFIX1"
REQUIRED_GREEDY_IDS = [353, 25701, 271]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="Qwen3.5-0.8B")
    parser.add_argument("--prompt", default="Hi,")
    parser.add_argument("--out", required=True)
    parser.add_argument("--row-count", type=int, default=64)
    parser.add_argument("--atol", type=float, default=5.0e-3)
    parser.add_argument("--rtol", type=float, default=5.0e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        dtype=torch.float32,
        device_map=None,
    )
    model.eval()

    encoded = tokenizer(args.prompt, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[-1][0, -1, :].to(torch.float32).contiguous()
        logits = outputs.logits[0, -1, :].to(torch.float32).contiguous()
        embedding = model.get_input_embeddings().weight.detach().to(torch.float32).contiguous()

    vocab_size, hidden_size = embedding.shape
    selected: set[int] = set(range(4))
    selected.add(vocab_size - 1)
    selected.update(i for i in REQUIRED_GREEDY_IDS if 0 <= i < vocab_size)
    top20 = torch.topk(logits, k=20).indices.tolist()
    selected.update(int(i) for i in top20)

    rng = random.Random(0xC0DEC0DE)
    while len(selected) < args.row_count:
        selected.add(rng.randrange(vocab_size))
    rows = sorted(selected)

    weights = embedding[rows, :].contiguous()
    expected = torch.matmul(weights, hidden).to(torch.float32).contiguous()

    fixture_path = out_dir / "fixture.bin"
    payload = bytearray()
    payload += MAGIC
    payload += struct.pack("<IIff", len(rows), hidden_size, args.atol, args.rtol)
    payload += hidden.numpy().astype("<f4", copy=False).tobytes()
    payload += weights.numpy().astype("<f4", copy=False).tobytes()
    payload += expected.numpy().astype("<f4", copy=False).tobytes()
    fixture_path.write_bytes(payload)

    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "fixture_format": "NNLGFIX1",
        "prompt": args.prompt,
        "model_dir": args.model_dir,
        "rows": len(rows),
        "cols": hidden_size,
        "vocab_size": vocab_size,
        "row_indices": rows,
        "top20_for_prompt": top20,
        "required_greedy_ids_included": REQUIRED_GREEDY_IDS,
        "tolerance": {"max_abs": args.atol, "max_rel": args.rtol},
        "fixture_bin": fixture_path.name,
        "sha256": digest,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(out_dir / "manifest.json"), "fixture": str(fixture_path), "sha256": digest, "rows": len(rows), "cols": hidden_size}, indent=2))


if __name__ == "__main__":
    main()
