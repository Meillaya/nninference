#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import hashlib
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

PROMPTS = ["Hi,", "The capital of China is", "What is 1+1?"]
OUT_DIR = Path("artifacts/benchmarks")


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def cmd_text(cmd: list[str]) -> str | None:
    try:
        proc = run(cmd)
        return proc.stdout.strip() or proc.stderr.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def detect_local_apps() -> dict:
    candidates = {
        "lmstudio_cli": shutil.which("lms") or shutil.which("lmstudio"),
        "llama_cli": shutil.which("llama-cli") or shutil.which("main"),
        "ollama": shutil.which("ollama"),
    }
    return candidates


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata(model_dir: Path) -> dict:
    return {
        "git_commit": cmd_text(["git", "rev-parse", "HEAD"]),
        "git_status_short": cmd_text(["git", "status", "--short"]),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "zig_version": cmd_text(["zig", "version"]),
        "uv_version": cmd_text(["uv", "--version"]),
        "python_version_uv": cmd_text(["uv", "run", "python", "--version"]),
        "transformers_version": cmd_text(["uv", "run", "python", "-c", "import transformers; print(transformers.__version__)"]),
        "torch_version": cmd_text(["uv", "run", "python", "-c", "import torch; print(torch.__version__)"]),
        "torch_threads": cmd_text(["uv", "run", "python", "-c", "import torch; print({'num_threads': torch.get_num_threads(), 'num_interop_threads': torch.get_num_interop_threads()})"]),
        "mac_model": cmd_text(["sysctl", "-n", "hw.model"]),
        "cpu_brand": cmd_text(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_bytes": cmd_text(["sysctl", "-n", "hw.memsize"]),
        "metal_toolchain": {
            "xcrun": shutil.which("xcrun"),
            "metal": cmd_text(["xcrun", "-f", "metal"]) if shutil.which("xcrun") else None,
            "metallib": cmd_text(["xcrun", "-f", "metallib"]) if shutil.which("xcrun") else None,
        },
        "env_threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        },
        "model_files": {
            "config_sha256": file_sha256(model_dir / "config.json"),
            "tokenizer_sha256": file_sha256(model_dir / "tokenizer.json"),
            "safetensors_sha256": file_sha256(model_dir / "model.safetensors-00001-of-00001.safetensors"),
        },
        "optional_local_app_baselines": detect_local_apps(),
    }


def parse_json_line(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and "selected_token_id" in line:
            return json.loads(line)
    raise RuntimeError(f"no JSON result in stdout:\n{stdout}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark current infer_cpu_v1 HF-bridge path")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--model-dir", default="Qwen3.5-0.8B")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run(["zig", "build"])

    rows: list[dict] = []
    for prompt in PROMPTS:
        for phase, count in (("warmup", args.warmup), ("measure", args.repeats)):
            for rep in range(count):
                cmd = [
                    "./zig-out/bin/infer_cpu_v1",
                    "--model-dir", args.model_dir,
                    "--prompt", prompt,
                    "--greedy",
                    "--json",
                ]
                start = time.perf_counter()
                proc = run(cmd)
                elapsed = time.perf_counter() - start
                result = parse_json_line(proc.stdout)
                row = {
                    "phase": phase,
                    "repeat": rep,
                    "prompt": prompt,
                    "elapsed_seconds": elapsed,
                    "first_token_per_second": 1.0 / elapsed if elapsed > 0 else None,
                    "result": result,
                    "stderr_tail": proc.stderr[-1000:],
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))

    measured = [r for r in rows if r["phase"] == "measure"]
    by_prompt: dict[str, dict] = {}
    for prompt in PROMPTS:
        vals = [r["elapsed_seconds"] for r in measured if r["prompt"] == prompt]
        by_prompt[prompt] = {
            "repeats": len(vals),
            "mean_seconds": sum(vals) / len(vals),
            "min_seconds": min(vals),
            "max_seconds": max(vals),
            "mean_first_token_per_second": len(vals) / sum(vals),
        }

    model_dir = Path(args.model_dir)
    report = {
        "benchmark": "current infer_cpu_v1 HF bridge greedy first-token baseline",
        "timestamp_unix": time.time(),
        "prompts": PROMPTS,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "metadata": metadata(model_dir),
        "summary_by_prompt": by_prompt,
        "rows": rows,
    }
    out = OUT_DIR / "bridge_baseline.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
