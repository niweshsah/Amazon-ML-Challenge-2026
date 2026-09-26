"""LoRA contrastive training with resumable state and dev retrieval selection."""
from __future__ import annotations

import math
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from core import metrics, save_json
from encoder import embed_rows, load_encoder, tensor_vectors


def group_batches(rows: list[dict], batch_size: int, seed: int = 42) -> list[list[int]]:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["anchor_id"]].append(i)
    for ids in groups.values():
        rng.shuffle(ids)
    group_ids = list(groups)
    rng.shuffle(group_ids)
    queue = deque(group_ids)
    result = []
    while queue:
        batch = []
        take = min(batch_size, len(queue))
        for _ in range(take):
            sid = queue.popleft()
            batch.append(groups[sid].pop())
            if groups[sid]:
                queue.append(sid)
        if len(batch) == batch_size:
            result.append(batch)
    return result


def restore_training_state(path: Path, optimizer, scheduler, max_length: int,
                           batch_size: int, max_pairs: int) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state["max_length"], state["batch_size"], state["max_pairs"]) != (max_length, batch_size, max_pairs):
        raise ValueError("Resume configuration differs from checkpoint")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    return state


def dev_score(root: Path, model, tokenizer, max_length: int, batch_size: int, logger) -> float:
    import faiss
    queries = pq.read_table(root / "dev_queries.parquet").to_pylist()
    targets = pq.read_table(root / "dev_targets.parquet").to_pylist()
    q_stats = embed_rows(model, tokenizer, queries, root / "dev_query_vectors.npy", max_length, batch_size, logger)
    t_stats = embed_rows(model, tokenizer, targets, root / "dev_target_vectors.npy", max_length, batch_size, logger)
    q = np.load(root / "dev_query_vectors.npy", mmap_mode="r")
    t = np.load(root / "dev_target_vectors.npy", mmap_mode="r")
    index = faiss.IndexFlatIP(t.shape[1])
    index.add(np.asarray(t))
    scores, positions = index.search(np.asarray(q), 20)
    ranked = [[targets[j]["entity_id"] for j in row if j >= 0] for row in positions]
    thresholds = np.arange(0.20, 0.951, 0.025)
    best = max((metrics(queries, ranked, scores, float(x), 20)["macro_f0.5"], float(x))
               for x in thresholds)
    logger("dev_retrieval", macro_f05=best[0], threshold=best[1],
           query_embedding=q_stats, target_embedding=t_stats, pool_size=len(targets))
    model.train()
    return best[0]


def train(root: Path, logger, resume: bool = False, batch_size: int = 128,
          max_length: int = 96, max_minutes: int = 80, max_pairs: int = 200000,
          checkpoint_steps: int = 250) -> dict:
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    rows = pq.read_table(root / "train_pairs.parquet").to_pylist()
    if len(rows) < max_pairs:
        raise ValueError(f"Prepared {len(rows)} pairs; requested {max_pairs}. Rerun prepare with --pairs {max_pairs}.")
    rows = rows[:max_pairs]
    positive_owner = {row["positive_id"]: row["anchor_id"] for row in rows}
    if len(rows) < 200000 and max_pairs >= 200000:
        raise ValueError("Expected at least 200,000 training pairs")
    batches = group_batches(rows, batch_size)
    if not batches:
        raise ValueError("No complete distinct-group batches")
    latest = root / "latest_checkpoint"
    tokenizer, model = load_encoder(latest / "adapter" if resume else None, trainable=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1 - step / len(batches)))
    start_step, best_score, best_step = 0, -1.0, 0
    if resume:
        state = restore_training_state(latest / "state.pt", optimizer, scheduler,
                                       max_length, batch_size, max_pairs)
        start_step, best_score, best_step = state["step"], state["best_score"], state["best_step"]
        logger("resume", step=start_step, best_score=best_score)
    started = time.monotonic()
    profile_start = started
    model.train()
    for step in range(start_step, len(batches)):
        batch = [rows[i] for i in batches[step]]
        anchors = [x["anchor"] for x in batch]
        positives = [x["positive"] for x in batch]
        negatives = [x["negative"] or batch[(i + 1) % len(batch)]["positive"] for i, x in enumerate(batch)]
        negative_owners = [x.get("negative_owner_id") or positive_owner.get(x["negative_id"]) or
                           batch[(i + 1) % len(batch)]["anchor_id"]
                           for i, x in enumerate(batch)]
        if step == start_step:
            lengths = tokenizer(anchors + positives + negatives, truncation=False)["input_ids"]
            logger("train_token_profile", sampled_texts=len(lengths),
                   truncated_at_96=sum(len(x) > 96 for x in lengths) / len(lengths),
                   truncated_at_64=sum(len(x) > 64 for x in lengths) / len(lengths))
        q = tensor_vectors(model, tokenizer, anchors, max_length)
        p = tensor_vectors(model, tokenizer, positives, max_length)
        n = tensor_vectors(model, tokenizer, negatives, max_length)
        logits = q @ torch.cat((p, n), dim=0).T / 0.05
        # The explicit negative can be a positive for another anchor in this batch.
        for i, anchor in enumerate(batch):
            for j, owner in enumerate(negative_owners):
                if owner == anchor["anchor_id"]:
                    logits[i, len(batch) + j] = -1e4
        labels = torch.arange(len(batch), device="cuda")
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        completed = step + 1
        if completed == 50:
            elapsed = time.monotonic() - profile_start
            estimate = elapsed * len(batches) / 50
            if estimate > max_minutes * 60 and max_length == 96:
                max_length = 64
            logger("throughput_profile", steps=50, seconds=elapsed,
                   estimated_train_minutes=estimate / 60, max_length=max_length,
                   action="reduced_to_64" if max_length == 64 else "kept_96")
        if completed == 1 or completed % 20 == 0:
            logger("train_step", step=completed, steps_total=len(batches), loss=float(loss.item()),
                   elapsed_seconds=time.monotonic() - started,
                   gpu_peak_gb=torch.cuda.max_memory_reserved() / 2**30)
        if completed % checkpoint_steps == 0 or completed == len(batches):
            latest.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(latest / "adapter")
            state = {"step": completed, "best_score": best_score, "best_step": best_step,
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                     "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                     "max_length": max_length, "batch_size": batch_size, "max_pairs": max_pairs}
            torch.save(state, latest / "state.pt")
            logger("checkpoint", step=completed, path=str(latest))
        if completed % (checkpoint_steps * 2) == 0 or completed == len(batches):
            score = dev_score(root, model, tokenizer, max_length, batch_size, logger)
            if score > best_score:
                best_score, best_step = score, completed
                model.save_pretrained(root / "best_adapter")
                logger("best_checkpoint", step=completed, macro_f05=score)
            state["best_score"], state["best_step"] = best_score, best_step
            torch.save(state, latest / "state.pt")
        if time.monotonic() - started >= max_minutes * 60:
            logger("training_cap_reached", step=completed)
            break
    result = {"step": completed, "total_steps": len(batches), "best_step": best_step,
              "best_dev_macro_f0.5": best_score, "elapsed_seconds": time.monotonic() - started,
              "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}
    save_json(root / "training_result.json", result)
    return result
