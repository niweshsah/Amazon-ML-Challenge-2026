import os
from dataclasses import dataclass

@dataclass
class Config:
    # Environment & Paths
    DATA_DIR: str = "dataset/test"       # Switch to 'dataset/train' for local dev
    OUTPUT_DIR: str = "output"
    
    # Preprocessing
    USE_CORE_STEM: bool = True
    
    # Blocking Parameters
    BLOCKING_TOP_K: int = 30
    BLOCKING_THRESHOLD: float = 0.15
    
    # Model Parameters
    MATCH_THRESHOLD: float = 0.85        # Precision-heavy cutoff for F_0.5