import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import os
import joblib
import time
import gc
from tqdm import tqdm

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "datasets", "preprocessed"))
    tfidf_out_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "datasets", "tfidf"))
    os.makedirs(tfidf_out_dir, exist_ok=True)
    
    vec_path = os.path.join(tfidf_out_dir, "tfidf_vectorizer.pkl")
    
    if os.path.exists(vec_path):
        print(f"✅ Found existing TF-IDF Vectorizer at {vec_path}. Loading it instantly!")
        vectorizer = joblib.load(vec_path)
    else:
        print("Loading Source 1 to learn vocabulary...")
        df_s1 = pd.read_parquet(os.path.join(preprocessed_dir, "train_source1.parquet"))
        fit_text = df_s1['combined_text_clean']
        
        print("Fitting TF-IDF Vectorizer (Character 3-grams) on Source 1 only...")
        start_time = time.time()
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), max_features=150000, min_df=2)
        vectorizer.fit(fit_text)
        print(f"Vectorizer fitted in {time.time() - start_time:.2f} seconds.")
        
        joblib.dump(vectorizer, vec_path)
        print(f"Saved Vectorizer to {vec_path}")
        
        # FREE MEMORY IMMEDIATELY
        del fit_text
        del df_s1
        gc.collect()
    
    import pyarrow.parquet as pq
    
    datasets_to_process = [
        ("train_s1", "train_source1.parquet"),
        ("train_s2", "train_source2.parquet"),
        ("train_s3", "train_source3.parquet"),
        ("test_s1", "test_source1.parquet"),
        ("test_s2", "test_source2.parquet"),
        ("test_s3", "test_source3.parquet")
    ]
    
    for name, filename in datasets_to_process:
        out_path = os.path.join(tfidf_out_dir, f"{name}_tfidf.npz")
        
        # 1. File-level Checkpointer
        if os.path.exists(out_path):
            print(f"\n✅ {name} already exists. Skipping!")
            continue
            
        import pyarrow.dataset as ds
        print(f"\nTransforming {name} using RAM-saving Chunking...")
        dataset = ds.dataset(os.path.join(preprocessed_dir, filename), format="parquet")
        
        chunk_dir = os.path.join(tfidf_out_dir, f"{name}_chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        
        matrices = []
        # 2. Intra-file Chunking & Checkpointing
        for i, batch in enumerate(tqdm(dataset.to_batches(batch_size=100000), desc=f"Chunking {name}")):
            chunk_path = os.path.join(chunk_dir, f"chunk_{i}.npz")
            
            if os.path.exists(chunk_path):
                # Instantly load from disk if we already processed this chunk
                matrices.append(sp.load_npz(chunk_path))
                continue
                
            df_chunk = batch.to_pandas()
            mat = vectorizer.transform(df_chunk['combined_text_clean'])
            sp.save_npz(chunk_path, mat)
            matrices.append(mat)
            del df_chunk
            
        # Combine the chunks mathematically
        print(f"Stacking chunks for {name}...")
        full_matrix = sp.vstack(matrices)
        
        sp.save_npz(out_path, full_matrix)
        print(f"Saved {name} matrix to {out_path}")
        
        # Cleanup temporary chunks to save disk space
        import shutil
        shutil.rmtree(chunk_dir)
        
        # CLEAR RAM completely before next file
        del full_matrix
        del matrices
        gc.collect()
        
    print("\nAll TF-IDF matrices built and saved successfully!")

if __name__ == "__main__":
    main()
