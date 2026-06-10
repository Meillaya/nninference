#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = ["Hi,", "The capital of China is", "What is 1+1?"]
MODEL_DIR = Path("Qwen3.5-0.8B")
ARTIFACT_DIR = Path("artifacts/alignment")
RTOL = 1.6e-2
ATOL = 1e-5


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def parse_zig_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and "selected_token_id" in line:
            return json.loads(line)
    raise AssertionError(f"no infer_cpu_v1 JSON object in stdout:\n{stdout}")


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    run(["zig", "build"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto").eval()

    report: list[dict] = []
    with torch.no_grad():
        for idx, prompt in enumerate(PROMPTS):
            inputs = tokenizer(prompt, return_tensors="pt")
            out = model(**inputs, return_dict=True)
            ref_logits = out.logits[0, -1, :].float().cpu().numpy().astype("<f4", copy=False)
            ref_argmax = int(ref_logits.argmax())
            gen = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                num_beams=1,
                return_dict_in_generate=True,
            )
            ref_generate = int(gen.sequences[0, -1].item())
            ref_path = ARTIFACT_DIR / f"ref_{idx}.bin"
            zig_path = ARTIFACT_DIR / f"zig_{idx}.bin"
            ref_logits.tofile(ref_path)

            proc = run([
                "./zig-out/bin/infer_cpu_v1",
                "--model-dir", str(MODEL_DIR),
                "--prompt", prompt,
                "--greedy",
                "--json",
                "--logits-out", str(zig_path),
            ])
            got = parse_zig_json(proc.stdout)
            zig_logits = np.fromfile(zig_path, dtype="<f4")
            if zig_logits.shape != ref_logits.shape:
                raise AssertionError(f"{prompt!r}: logits shape mismatch {zig_logits.shape} != {ref_logits.shape}")
            close = np.allclose(zig_logits, ref_logits, rtol=RTOL, atol=ATOL)
            max_abs = float(np.max(np.abs(zig_logits - ref_logits)))
            if not close:
                diff_idx = int(np.argmax(np.abs(zig_logits - ref_logits)))
                raise AssertionError(
                    f"{prompt!r}: logits not aligned; max_abs={max_abs} at {diff_idx}, "
                    f"zig={zig_logits[diff_idx]}, ref={ref_logits[diff_idx]}"
                )
            if got["selected_token_id"] != ref_argmax:
                raise AssertionError(f"{prompt!r}: Zig greedy {got['selected_token_id']} != ref argmax {ref_argmax}")
            if got["hf_forward_argmax_token_id"] != ref_argmax:
                raise AssertionError(f"{prompt!r}: bridge argmax {got['hf_forward_argmax_token_id']} != ref argmax {ref_argmax}")
            if got["hf_generate_token_id"] != ref_generate:
                raise AssertionError(f"{prompt!r}: bridge generate {got['hf_generate_token_id']} != ref generate {ref_generate}")
            if ref_argmax != ref_generate:
                raise AssertionError(f"{prompt!r}: HF forward argmax {ref_argmax} != HF generate {ref_generate}")
            report.append({
                "prompt": prompt,
                "token_id": ref_argmax,
                "decoded": tokenizer.decode([ref_argmax]),
                "max_abs_diff": max_abs,
                "rtol": RTOL,
                "atol": ATOL,
            })

    sample = run([
        "./zig-out/bin/infer_cpu_v1",
        "--model-dir", str(MODEL_DIR),
        "--prompt", PROMPTS[0],
        "--json",
        "--seed", "7",
    ])
    sample_json = parse_zig_json(sample.stdout)
    if sample_json["temperature"] != 0.6 or sample_json["top_p"] != 0.95 or sample_json["top_k"] != 20:
        raise AssertionError(f"default sampling parameters changed: {sample_json}")

    result = {"prompts": report, "sampling_smoke": sample_json}
    (ARTIFACT_DIR / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
