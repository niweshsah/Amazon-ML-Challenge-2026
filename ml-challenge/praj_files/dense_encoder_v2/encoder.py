"""Shared BF16 CLS encoder and embedding routines."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoTokenizer

from core import MODEL, MODEL_REVISION


def load_encoder(adapter: Path | None = None, trainable: bool = False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run inside pytorch-qwen container")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16,
                                      attn_implementation="sdpa", revision=MODEL_REVISION,
                                      local_files_only=True)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                 target_modules=["query", "value"], bias="none"))
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return tokenizer, model.cuda()


def tensor_vectors(model, tokenizer, texts: list[str], max_length: int) -> torch.Tensor:
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                       return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model(**tokens).last_hidden_state[:, 0]
    return F.normalize(hidden.float(), dim=1)


@torch.no_grad()
def embed_rows(model, tokenizer, rows: list[dict], path: Path, max_length: int = 96,
               batch_size: int = 128, logger=None) -> dict:
    model.eval()
    dim = model.config.hidden_size
    vectors = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(len(rows), dim))
    started = time.monotonic()
    truncated = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        texts = [row["text"] for row in chunk]
        lengths = tokenizer(texts, truncation=False, add_special_tokens=True)["input_ids"]
        truncated += sum(len(x) > max_length for x in lengths)
        vectors[start:start + len(chunk)] = tensor_vectors(model, tokenizer, texts, max_length).cpu().numpy()
        if logger and (start // batch_size) % 100 == 0:
            logger("embed_progress", rows_done=start + len(chunk), rows_total=len(rows),
                   rows_per_second=(start + len(chunk)) / max(1e-6, time.monotonic() - started),
                   gpu_peak_gb=torch.cuda.max_memory_reserved() / 2**30)
    vectors.flush()
    return {"rows": len(rows), "seconds": time.monotonic() - started,
            "rows_per_second": len(rows) / max(1e-6, time.monotonic() - started),
            "truncated": truncated, "truncation_rate": truncated / max(1, len(rows)),
            "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}
