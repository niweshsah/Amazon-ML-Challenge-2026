from abc import ABC, abstractmethod
import pandas as pd

class BaseBlocker(ABC):
    @abstractmethod
    def fit(self, corpus_df: pd.DataFrame):
        """Build index from Source 2 + Source 3 corpus."""
        pass

    @abstractmethod
    def generate_candidates(self, queries_df: pd.DataFrame, top_k: int, threshold: float) -> pd.DataFrame:
        """Returns candidate_pairs DataFrame with columns: [source1_entity_id, candidate_entity_ids]"""
        pass