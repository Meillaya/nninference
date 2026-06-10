#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="Qwen3.5-0.8B")
    parser.add_argument("--token-id", type=int, required=True)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(Path(args.model_dir))
    print(json.dumps({"token_id": args.token_id, "text": tok.decode([args.token_id])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
