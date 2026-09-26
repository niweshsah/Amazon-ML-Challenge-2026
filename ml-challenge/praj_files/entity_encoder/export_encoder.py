"""Merge the selected LoRA adapter into a standalone encoder checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import config, save_json
from train import load_model


def export(cfg: dict) -> dict:
    root = Path(cfg["output_dir"])
    adapter = root / "final_adapter"
    if not adapter.exists():
        raise FileNotFoundError(f"Missing selected adapter: {adapter}")
    tokenizer, model = load_model(cfg, adapter)
    model = model.merge_and_unload()
    output = root / "final_encoder"
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    result = {"model": cfg["model"], "model_revision": cfg.get("model_revision"),
              "source_adapter": str(adapter), "final_encoder": str(output),
              "pooling": "L2-normalized CLS token", "embedding_dimension": 1024,
              "max_length": cfg["max_length"]}
    save_json(output / "encoder_manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    print(json.dumps(export(config(args.config)), indent=2))
