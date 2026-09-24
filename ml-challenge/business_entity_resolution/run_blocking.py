import os
import pandas as pd
from src.config import Config
from src.preprocessing.cleaner import prepare_dataframe
from src.blocking.tfidf_blocker import TfidfBlocker
# from src.blocking.dense_blocker import DenseBlocker  <-- Plug in later!

def main():
    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    print("1. Loading Datasets...")
    s1_df = pd.read_csv(os.path.join(cfg.DATA_DIR, "test_source1.tsv"), sep="\t")
    s2_df = pd.read_csv(os.path.join(cfg.DATA_DIR, "test_source2.tsv"), sep="\t")
    s3_df = pd.read_csv(os.path.join(cfg.DATA_DIR, "test_source3.tsv"), sep="\t")
    
    print("2. Preprocessing Data...")
    s1_df = prepare_dataframe(s1_df)
    corpus_df = prepare_dataframe(pd.concat([s2_df, s3_df], ignore_index=True))
    
    print("3. Fitting Blocker Index...")
    blocker = TfidfBlocker(ngram_range=(3, 5))
    blocker.fit(corpus_df)
    
    print("4. Generating Candidate Pairs...")
    candidate_df = blocker.generate_candidates(
        queries_df=s1_df, 
        top_k=cfg.BLOCKING_TOP_K, 
        threshold=cfg.BLOCKING_THRESHOLD
    )
    
    candidate_file = os.path.join(cfg.OUTPUT_DIR, "candidate_pairs.tsv")
    candidate_df.to_csv(candidate_file, sep="\t", index=False)
    print(f"Saved: {candidate_file}")

if __name__ == "__main__":
    main()