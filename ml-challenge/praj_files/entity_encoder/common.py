from __future__ import annotations

import json
import re
from pathlib import Path

import pyarrow.dataset as ds


HERE = Path(__file__).resolve().parent


def config(path: str | None = None) -> dict:
    cfg = json.loads(Path(path or HERE / "config.json").read_text())
    # The same config works on the host and inside run_qwen.sh.
    if not Path(cfg["data_dir"]).exists():
        cfg["data_dir"] = str(HERE.parent / "datasets" / "preprocessed")
    if not Path(cfg["output_dir"]).parent.exists():
        cfg["output_dir"] = str(HERE / "artifacts")
    return cfg


def rows(path: Path, columns: list[str], batch_size: int = 32768):
    dataset = ds.dataset(path, format="parquet")
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        yield from batch.to_pylist()


def value(x: object) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def record_text(row: dict) -> str:
    # Raw fields preserve Indic combining marks destroyed by the older cleaner.
    fields = [("Name", row.get("business_name")),
              ("Address", row.get("business_address")),
              ("Country", row.get("country"))]
    return " | ".join(f"{key}: {value(v)}" for key, v in fields if value(v))


def source_path(cfg: dict, split: str, source: int) -> Path:
    return Path(cfg["data_dir"]) / f"{split}_source{source}.parquet"


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
