#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from safetensors import safe_open

MODEL_DIR = Path('Qwen3.5-0.8B')


def main() -> None:
    cfg = json.loads((MODEL_DIR / 'config.json').read_text())
    text = cfg.get('text_config', {})
    shard = MODEL_DIR / 'model.safetensors-00001-of-00001.safetensors'
    with safe_open(shard, framework='pt') as f:
        keys = list(f.keys())
        full = sum(1 for i, t in enumerate(text.get('layer_types', [])) if t == 'full_attention')
        linear = sum(1 for i, t in enumerate(text.get('layer_types', [])) if t == 'linear_attention')
        print(json.dumps({
            'model_type': cfg.get('model_type'),
            'text_model_type': text.get('model_type'),
            'hidden_size': text.get('hidden_size'),
            'num_hidden_layers': text.get('num_hidden_layers'),
            'vocab_size': text.get('vocab_size'),
            'dtype': text.get('dtype'),
            'layer_types': text.get('layer_types'),
            'linear_attention_layers': linear,
            'full_attention_layers': full,
            'tensor_count': len(keys),
            'has_linear_attention_weights': any('.linear_attn.' in k for k in keys),
            'has_full_attention_weights': any('.self_attn.' in k for k in keys),
        }, indent=2))


if __name__ == '__main__':
    main()
