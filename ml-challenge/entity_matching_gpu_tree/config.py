"""Configuration for the sampled entity-matching baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    dataset: Path = ROOT / "entity_matching_100k"
    output: Path = ROOT / "entity_matching_gpu_tree" / "artifacts"
    seed: int = 20260926
    max_queries: int = 100_000
    max_candidates_per_source: int = 32
    feature_batch_size: int = 1_000
    retrieval_progress_every: int = 2_000
    rounds: int = 2_000
    early_stopping_rounds: int = 100
    max_depth: int = 7
    learning_rate: float = 0.045
    subsample: float = 0.82
    colsample_bytree: float = 0.85
    min_child_weight: int = 8
    reg_lambda: float = 5.0
    nthread: int = 8

    def serializable(self) -> dict[str, object]:
        result = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in result.items()}
