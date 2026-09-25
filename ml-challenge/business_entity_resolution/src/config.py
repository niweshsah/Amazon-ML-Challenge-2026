from dataclasses import dataclass, field
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]

@dataclass
class Config:
    # Environment & Paths
    MODE: str = "train"
    DATA_DIR: Path = field(init=False)
    OUTPUT_DIR: Path = PROJECT_DIR / "challenge-dataset" / "output"

    def __post_init__(self):
        if self.MODE not in {"train", "test"}:
            raise ValueError("MODE must be 'train' or 'test'")
        self.DATA_DIR = PROJECT_DIR / "challenge-dataset" / "dataset" / self.MODE
    
    # Preprocessing
    USE_CORE_STEM: bool = True
    
    # Blocking Parameters
    BLOCKING_TOP_K: int = 30
    BLOCKING_THRESHOLD: float = 0.15
    BLOCKING_BATCH_SIZE: int = 256
    CORPUS_CHUNK_SIZE: int = 10_000
    HASHING_FEATURES: int = 2**18
    
    # Model Parameters
    MATCH_THRESHOLD: float = 0.85        # Precision-heavy cutoff for F_0.5