#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="HF CPU prefill bridge for infer_cpu_v1")
    parser.add_argument("--model-dir", default="Qwen3.5-0.8B")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--logits-out")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto").eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, return_dict=True)
        logits = out.logits[0, -1, :].float().cpu()
        forward_argmax = int(torch.argmax(logits).item())
        gen = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
        )
        generate_token = int(gen.sequences[0, -1].item())

    if args.logits_out:
        Path(args.logits_out).parent.mkdir(parents=True, exist_ok=True)
        logits.numpy().astype("<f4", copy=False).tofile(args.logits_out)

    candidate_count = max(1, min(int(args.candidate_count), int(logits.numel())))
    values, indices = torch.topk(logits, k=candidate_count)

    print(json.dumps({
        "vocab_size": int(logits.numel()),
        "input_ids": [int(x) for x in inputs["input_ids"][0].tolist()],
        "hf_forward_argmax_token_id": forward_argmax,
        "hf_generate_token_id": generate_token,
        "hf_generate_text": tokenizer.decode([generate_token]),
    }, separators=(",", ":")))
    print("CANDIDATES")
    for idx, value in zip(indices.tolist(), values.tolist()):
        print(f"{int(idx)}\t{float(value):.9g}")


if __name__ == "__main__":
    main()
