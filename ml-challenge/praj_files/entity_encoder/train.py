from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from common import config, save_json


class PairDataset(Dataset):
    def __init__(self, path: Path, limit: int | None = None):
        self.data = pq.read_table(path, columns=["anchor", "positive", "negative"]).to_pydict()
        self.length = min(len(self.data["anchor"]), limit or len(self.data["anchor"]))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[str, str, str]:
        return tuple(self.data[k][index] for k in ("anchor", "positive", "negative"))


def load_model(cfg: dict, adapter: Path | None = None, trainable: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True,
                                               revision=cfg.get("model_revision"))
    model = AutoModel.from_pretrained(cfg["model"], torch_dtype=torch.bfloat16,
                                      attn_implementation="sdpa",
                                      revision=cfg.get("model_revision"))
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        model = get_peft_model(model, LoraConfig(
            r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
            target_modules=["query", "value"], lora_dropout=0.05,
            bias="none"))
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.to("cuda")
    return tokenizer, model


def encode_tensor(model, tokenizer, texts: list[str], max_length: int):
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(**batch).last_hidden_state[:, 0]
    return F.normalize(result.float(), dim=-1)


@torch.no_grad()
def encode_numpy(model, tokenizer, texts: list[str], max_length: int,
                 batch_size: int = 64) -> np.ndarray:
    model.eval()
    vectors = []
    for start in range(0, len(texts), batch_size):
        vec = encode_tensor(model, tokenizer, texts[start:start + batch_size], max_length)
        vectors.append(vec.cpu().numpy())
    return np.concatenate(vectors) if vectors else np.empty((0, 1024), dtype=np.float32)


def evaluate(cfg: dict, tokenizer, model, label: str) -> dict:
    root = Path(cfg["output_dir"])
    queries = pq.read_table(root / "val_queries.parquet").to_pylist()
    targets = pq.read_table(root / "val_targets.parquet").to_pylist()
    q_vec = encode_numpy(model, tokenizer, [x["text"] for x in queries], cfg["max_length"])
    t_vec = encode_numpy(model, tokenizer, [x["text"] for x in targets], cfg["max_length"])
    target_ids = [x["entity_id"] for x in targets]
    target_index = {id_: i for i, id_ in enumerate(target_ids)}
    scores = q_vec @ t_vec.T
    order = np.argsort(-scores, axis=1)
    truth = [set(filter(None, q["truth"].split(","))) for q in queries]
    ranks, positive_scores, negative_scores = [], [], []
    for i, ids in enumerate(truth):
        gt_indexes = {target_index[x] for x in ids if x in target_index}
        ranked = order[i]
        rank = next((j + 1 for j, index in enumerate(ranked) if index in gt_indexes), None)
        if gt_indexes:
            ranks.append(rank)
            positive_scores.extend(float(scores[i, j]) for j in gt_indexes)
        negative_scores.append(float(scores[i, ranked[0]]) if ranked[0] not in gt_indexes
                               else float(scores[i, ranked[1]]))
    nonempty = len(ranks)
    recall = {f"recall@{k}": sum(r is not None and r <= k for r in ranks) / max(1, nonempty)
              for k in (1, 5, 10, 50)}
    recall["mrr"] = sum(1 / r for r in ranks if r) / max(1, nonempty)
    best = None
    for threshold in np.arange(0.30, 0.951, 0.025):
        fvals = []
        for i, ids in enumerate(truth):
            selected = {target_ids[j] for j in order[i, :50] if scores[i, j] >= threshold}
            if not ids:
                fvals.append(float(not selected))
            elif not selected:
                fvals.append(0.0)
            else:
                hit = len(selected & ids)
                precision, sensitivity = hit / len(selected), hit / len(ids)
                fvals.append(1.25 * precision * sensitivity /
                             (0.25 * precision + sensitivity) if hit else 0.0)
        value = float(np.mean(fvals))
        if best is None or value > best["macro_f0.5"]:
            best = {"threshold": round(float(threshold), 3), "macro_f0.5": value}
    result = {"label": label, "query_count": len(queries), "target_pool_size": len(targets),
              "nonempty_queries": nonempty, **recall, "best_threshold": best,
              "mean_positive_cosine": float(np.mean(positive_scores)) if positive_scores else None,
              "mean_top_negative_cosine": float(np.mean(negative_scores))}
    query_tags = []
    for query in queries:
        tags = {t for t in query.get("tags", "").split(",") if t}
        if "Country: India" in query["text"]:
            tags.add("country_india")
        if "Country: US" in query["text"]:
            tags.add("country_us")
        query_tags.append(tags)
    group_stats = {}
    per_query_f05 = []
    for i, ids in enumerate(truth):
        selected = {target_ids[j] for j in order[i, :50]
                    if scores[i, j] >= best["threshold"]}
        if not ids:
            per_query_f05.append(float(not selected))
        elif not selected:
            per_query_f05.append(0.0)
        else:
            hit = len(selected & ids)
            precision, sensitivity = hit / len(selected), hit / len(ids)
            per_query_f05.append(1.25 * precision * sensitivity /
                                 (0.25 * precision + sensitivity) if hit else 0.0)
    for tag in sorted({t for tags in query_tags for t in tags}):
        subset = [i for i, tags in enumerate(query_tags) if tag in tags]
        valid = [i for i in subset if truth[i]]
        recall5 = sum(any(target_ids[j] in truth[i] for j in order[i, :5]) for i in valid) / max(1, len(valid))
        singleton_accuracy = None
        if tag == "singleton":
            singleton_accuracy = sum(not any(scores[i, j] >= best["threshold"] for j in order[i, :50])
                                     for i in subset) / max(1, len(subset))
        group_stats[tag] = {"queries": len(subset), "recall@5": recall5,
                            "macro_f0.5": float(np.mean([per_query_f05[i] for i in subset])),
                            "singleton_accuracy": singleton_accuracy}
    result["edge_groups"] = group_stats
    save_json(root / f"{label}_evaluation.json", result)
    return result


def train(cfg: dict, smoke_steps: int = 0) -> dict:
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    root = Path(cfg["output_dir"])
    tokenizer, model = load_model(cfg, trainable=True)
    pair_limit = cfg["batch_size"] * max(4, smoke_steps) if smoke_steps else None
    dataset = PairDataset(root / "train_pairs.parquet", pair_limit)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                        drop_last=True, num_workers=0)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=cfg["learning_rate"], weight_decay=0.01)
    accum = cfg["gradient_accumulation"]
    step = 0
    started = time.monotonic()
    log_path = root / ("smoke_log.jsonl" if smoke_steps else "training_log.jsonl")
    save_dir = root / ("smoke_adapter" if smoke_steps else "final_adapter")
    best_score = -1.0
    best_step = None
    model.train()
    with log_path.open("w") as log:
        for epoch in range(cfg["epochs"]):
            for anchors, positives, negatives in loader:
                anchor_vec = encode_tensor(model, tokenizer, list(anchors), cfg["max_length"])
                pos_vec = encode_tensor(model, tokenizer, list(positives), cfg["max_length"])
                neg_vec = encode_tensor(model, tokenizer, list(negatives), cfg["max_length"])
                logits = anchor_vec @ torch.cat((pos_vec, neg_vec), dim=0).T * 20
                labels = torch.arange(len(anchors), device="cuda")
                loss = F.cross_entropy(logits, labels) / accum
                loss.backward()
                step += 1
                if step % accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if step == 1 or step % 20 == 0:
                    event = {"step": step, "epoch": epoch, "loss": loss.detach().item() * accum,
                             "elapsed_sec": round(time.monotonic() - started, 1),
                             "gpu_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
                    log.write(json.dumps(event) + "\n"); log.flush()
                    print(json.dumps(event), flush=True)
                if step % cfg["checkpoint_steps"] == 0 and not smoke_steps:
                    model.save_pretrained(root / "latest_adapter")
                if step % (cfg["checkpoint_steps"] * 2) == 0 and not smoke_steps:
                    metric = evaluate(cfg, tokenizer, model, f"step_{step}")
                    score = metric["best_threshold"]["macro_f0.5"]
                    if score > best_score:
                        best_score, best_step = score, step
                        model.save_pretrained(save_dir)
                    model.train()
                if smoke_steps and step >= smoke_steps:
                    break
                if time.monotonic() - started > cfg["max_train_minutes"] * 60:
                    break
            if smoke_steps and step >= smoke_steps or time.monotonic() - started > cfg["max_train_minutes"] * 60:
                break
    if not smoke_steps:
        metric = evaluate(cfg, tokenizer, model, "last_step")
        score = metric["best_threshold"]["macro_f0.5"]
        if score > best_score:
            best_score, best_step = score, step
            model.save_pretrained(save_dir)
    else:
        model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    result = {"steps": step, "pairs_seen": step * cfg["batch_size"],
              "elapsed_sec": time.monotonic() - started,
              "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
              "adapter": str(save_dir), "best_step": best_step,
              "best_validation_macro_f0.5": best_score if best_step is not None else None}
    save_json(root / ("smoke_result.json" if smoke_steps else "training_result.json"), result)
    if not smoke_steps:
        del model
        torch.cuda.empty_cache()
        best_tokenizer, best_model = load_model(cfg, save_dir)
        result["evaluation"] = evaluate(cfg, best_tokenizer, best_model, "fine_tuned")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["baseline", "smoke", "train", "evaluate"])
    parser.add_argument("--config")
    parser.add_argument("--model", help="Override pretrained model for a baseline comparison")
    parser.add_argument("--smoke-steps", type=int, default=3)
    parser.add_argument("--adapter", help="Adapter path for evaluate; defaults to final_adapter")
    parser.add_argument("--label", help="Output label for evaluate")
    args = parser.parse_args()
    cfg = config(args.config)
    if args.model:
        cfg["model"] = args.model
        cfg.pop("model_revision", None)
    if not torch.cuda.is_available():
        raise RuntimeError("Training and evaluation require CUDA inside ./run_qwen.sh")
    if args.command in {"baseline", "evaluate"}:
        adapter = None if args.command == "baseline" else Path(args.adapter or Path(cfg["output_dir"]) / "final_adapter")
        tokenizer, model = load_model(cfg, adapter)
        label = args.label or (("zero_shot_" + cfg["model"].split("/")[-1].lower())
                               if adapter is None else "fine_tuned")
        result = evaluate(cfg, tokenizer, model, label)
    else:
        result = train(cfg, args.smoke_steps if args.command == "smoke" else 0)
    print(json.dumps(result, indent=2))
