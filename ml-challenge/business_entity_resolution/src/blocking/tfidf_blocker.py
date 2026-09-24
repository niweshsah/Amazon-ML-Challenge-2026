import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.blocking.base import BaseBlocker

class TfidfBlocker(BaseBlocker):
    def __init__(self, ngram_range=(3, 5)):
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=ngram_range,
            min_df=1,
            sublinear_tf=True
        )
        self.corpus_ids = None
        self.X_corpus = None

    def fit(self, corpus_df: pd.DataFrame):
        self.corpus_ids = corpus_df['entity_id'].values
        self.X_corpus = self.vectorizer.fit_transform(corpus_df['search_text'])

    def generate_candidates(self, queries_df: pd.DataFrame, top_k: int = 30, threshold: float = 0.15) -> pd.DataFrame:
        X_queries = self.vectorizer.transform(queries_df['search_text'])
        sim_matrix = X_queries.dot(self.X_corpus.T)
        
        results = []
        for row_idx in range(sim_matrix.shape[0]):
            s1_id = queries_df.iloc[row_idx]['entity_id']
            row = sim_matrix.getrow(row_idx)
            
            if len(row.data) == 0:
                results.append({'source1_entity_id': s1_id, 'candidate_entity_ids': ''})
                continue
                
            mask = row.data >= threshold
            valid_indices = row.indices[mask]
            valid_scores = row.data[mask]
            
            if len(valid_scores) > 0:
                top_order = np.argsort(valid_scores)[::-1][:top_k]
                matched_ids = self.corpus_ids[valid_indices[top_order]]
                
                # Order-preserving deduplication
                seen = set()
                dedup = [x for x in matched_ids if not (x in seen or seen.add(x))]
                cand_str = ",".join(dedup)
            else:
                cand_str = ''
                
            results.append({'source1_entity_id': s1_id, 'candidate_entity_ids': cand_str})
            
        return pd.DataFrame(results)