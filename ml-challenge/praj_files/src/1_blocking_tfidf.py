import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import os
import gc

def load_data(file_path):
    print(f"Loading {file_path}...")
    # The challenge states: All files are tab-separated (.tsv)
    # Read them with an explicit tab separator.
    df = pd.read_csv(file_path, sep="\t", dtype=str)
    
    # Fill NaNs with empty string
    df['business_name'] = df['business_name'].fillna('')
    df['business_address'] = df['business_address'].fillna('')
    df['country'] = df['country'].fillna('')
    
    # Create a combined text column for TF-IDF
    df['combined_text'] = df['business_name'].str.lower() + " " + df['business_address'].str.lower()
    return df

def run_blocking(source1_df, target_df, source_name, top_k=20):
    """
    Runs TF-IDF based K-Nearest Neighbors to find candidate matches.
    source1_df: The reference dataset (Source 1)
    target_df: The dataset to match against (Source 2 or 3)
    """
    print(f"\n--- Blocking against {source_name} ---")
    print(f"Source 1 size: {len(source1_df)}, {source_name} size: {len(target_df)}")
    
    start_time = time.time()
    
    # We fit the TF-IDF on both to ensure vocabulary covers both
    print("Fitting TF-IDF Vectorizer (Character 3-grams)...")
    # Character n-grams handle typos better than word n-grams
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), max_features=100000)
    
    # Fit on target data to build the index
    target_tfidf = vectorizer.fit_transform(target_df['combined_text'])
    source1_tfidf = vectorizer.transform(source1_df['combined_text'])
    
    print(f"TF-IDF Shape: Source1 {source1_tfidf.shape}, Target {target_tfidf.shape}")
    
    print(f"Building Nearest Neighbors Index (K={top_k})...")
    # n_jobs=-1 uses all CPU cores
    nn = NearestNeighbors(n_neighbors=top_k, metric='cosine', n_jobs=-1)
    nn.fit(target_tfidf)
    
    print("Querying Nearest Neighbors (this might take a few minutes)...")
    distances, indices = nn.kneighbors(source1_tfidf)
    
    print(f"Blocking completed in {time.time() - start_time:.2f} seconds.")
    
    return distances, indices

def main():
    # Dynamically resolve path so it works regardless of where the script is run from (Docker or Host)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "business_entity_resolution", "challenge-dataset", "dataset", "train"))
    
    # Paths
    s1_path = os.path.join(base_dir, "train_source1.tsv")
    s2_path = os.path.join(base_dir, "train_source2.tsv")
    s3_path = os.path.join(base_dir, "train_source3.tsv")
    
    # Load Source 1
    s1_df = load_data(s1_path)
    
    # ----- Process Source 2 -----
    s2_df = load_data(s2_path)
    s2_distances, s2_indices = run_blocking(s1_df, s2_df, "Source 2", top_k=10)
    
    # Save some RAM
    del s2_df
    gc.collect()
    
    # ----- Process Source 3 -----
    s3_df = load_data(s3_path)
    s3_distances, s3_indices = run_blocking(s1_df, s3_df, "Source 3", top_k=10)
    
    # Save some RAM
    del s3_df
    gc.collect()
    
    print("\nNext steps: We will format these indices into 'candidate_pairs.tsv' and evaluate recall against ground truth!")

if __name__ == "__main__":
    main()
