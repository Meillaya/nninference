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
    parser.add_argument("--row-mode", choices=["subset", "full"], default="subset")
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
    top20 = torch.topk(logits, k=20).indices.tolist()
    if args.row_mode == "full":
        rows = list(range(vocab_size))
    else:
        selected: set[int] = set(range(4))
        selected.add(vocab_size - 1)
        selected.update(i for i in REQUIRED_GREEDY_IDS if 0 <= i < vocab_size)
        selected.update(int(i) for i in top20)

        rng = random.Random(0xC0DEC0DE)
        while len(selected) < args.row_count:
            selected.add(rng.randrange(vocab_size))
        rows = sorted(selected)

    weights = embedding[rows, :].contiguous()
    expected = torch.matmul(weights, hidden).to(torch.float32).contiguous()
    full_matmul_for_top = torch.matmul(embedding, hidden).to(torch.float32).contiguous() if args.row_mode == "full" else None

    fixture_path = out_dir / "fixture.bin"
    with fixture_path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIff", len(rows), hidden_size, args.atol, args.rtol))
        hidden.numpy().astype("<f4", copy=False).tofile(f)
        weights.numpy().astype("<f4", copy=False).tofile(f)
        expected.numpy().astype("<f4", copy=False).tofile(f)

    hasher = hashlib.sha256()
    with fixture_path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    matmul_top20 = torch.topk(full_matmul_for_top, k=20).indices.tolist() if full_matmul_for_top is not None else []
    hf_vs_matmul = {}
    if full_matmul_for_top is not None:
        diff = (logits - full_matmul_for_top).abs()
        hf_vs_matmul = {
            "max_abs_diff": float(diff.max().item()),
            "max_rel_diff": float((diff / logits.abs().clamp_min(1.0e-12)).max().item()),
            "hf_top1_matches_matmul_top1": int(top20[0]) == int(matmul_top20[0]),
            "hf_top20_set_matches_matmul_top20_set": set(map(int, top20)) == set(map(int, matmul_top20)),
        }

    manifest = {
        "fixture_format": "NNLGFIX1",
        "expected_kind": "torch.matmul(output_embedding_rows, final_hidden_state)",
        "prompt": args.prompt,
        "model_dir": args.model_dir,
        "row_mode": args.row_mode,
        "rows": len(rows),
        "cols": hidden_size,
        "vocab_size": vocab_size,
        "row_indices": rows if args.row_mode != "full" else "implicit_full_vocab_order",
        "hf_top1_for_prompt": int(top20[0]),
        "hf_top20_for_prompt": [int(i) for i in top20],
        "matmul_top1_for_prompt": int(matmul_top20[0]) if matmul_top20 else None,
        "matmul_top20_for_prompt": [int(i) for i in matmul_top20],
        "hf_logits_vs_matmul": hf_vs_matmul,
        "required_greedy_ids_included": REQUIRED_GREEDY_IDS,
        "tolerance": {"max_abs": args.atol, "max_rel": args.rtol},
        "fixture_bin": fixture_path.name,
        "sha256": digest,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(out_dir / "manifest.json"), "fixture": str(fixture_path), "sha256": digest, "rows": len(rows), "cols": hidden_size}, indent=2))


if __name__ == "__main__":
    main()
