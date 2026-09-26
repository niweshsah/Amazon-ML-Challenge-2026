from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Any

from .checkpoint import valid_embedding_manifest
from .gpu_utils import environment_report
from .io_utils import atomic_json, ensure_layout, load_config, output_root


def build_index(config_path: str | None, split: str, resume: bool = True) -> dict[str, Any]:
    """Build a lightweight catalog for resumable, sharded GPU exact search.

    The vectors remain in FP16 NPY shards. Search loads one target shard at a time,
    so neither 10M targets nor a duplicate float32 FAISS index must fit in RAM.
    """
    config = load_config(config_path)
    ensure_layout(config)
    started = time.perf_counter()
    catalog: dict[str, Any] = {
        "success": False, "split": split, "backend": "sharded_gpu_flat_ip",
        "metric": "cosine", "embedding_dim": config["model"]["output_dim"],
        "sources": {}, "environment": environment_report(),
    }
    for source in ("source2", "source3"):
        directory = output_root(config) / "embeddings" / f"{split}_{source}"
        manifests = []
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Embedding summary not found: {summary_path}")
        active_paths = json.loads(summary_path.read_text()).get("manifests", [])
        for path_text in active_paths:
            path = Path(path_text)
            manifest = json.loads(path.read_text())
            if not valid_embedding_manifest(manifest):
                raise RuntimeError(f"Invalid embedding checkpoint: {path}")
            manifests.append({
                "manifest": str(path), "vectors": manifest["files"][0]["path"],
                "mapping": manifest["files"][1]["path"], "rows": manifest["rows"],
                "unique_texts": manifest["unique_texts"], "shard_id": manifest["shard_id"],
                "vector_checksum": manifest["files"][0]["sha256"],
                "mapping_checksum": manifest["files"][1]["sha256"],
            })
        if not manifests:
            raise FileNotFoundError(f"No completed embedding shards in {directory}")
        catalog["sources"][source] = {"rows": sum(x["rows"] for x in manifests), "shards": manifests}
    catalog["success"] = True
    signature_payload = {
        "backend": catalog["backend"], "metric": catalog["metric"], "embedding_dim": catalog["embedding_dim"],
        "sources": {
            source: [(part["shard_id"], part["vector_checksum"], part["mapping_checksum"]) for part in value["shards"]]
            for source, value in catalog["sources"].items()
        },
    }
    catalog["index_signature"] = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    catalog["build_seconds"] = time.perf_counter() - started
    path = output_root(config) / "indexes" / f"{split}_index.json"
    atomic_json(path, catalog)
    print(f"[build-index] cataloged {sum(len(v['shards']) for v in catalog['sources'].values())} shards at {path}")
    return catalog
