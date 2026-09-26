import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import torch
import gc

def encode_and_save(model, filename, name, out_dir, preprocessed_dir):
    out_path = os.path.join(out_dir, f"{name}_embeddings.npy")
    
    # Checkpointer
    if os.path.exists(out_path):
        print(f"✅ {name} already exists. Skipping!")
        return
        
    import pyarrow.dataset as ds
    from tqdm import tqdm
    print(f"Encoding {name} using RAM-saving Chunking...")
    
    dataset = ds.dataset(os.path.join(preprocessed_dir, filename), format="parquet")
    
    chunk_dir = os.path.join(out_dir, f"{name}_chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    
    emb_list = []
    
    # Intra-file Chunking & Checkpointing
    for i, batch in enumerate(tqdm(dataset.to_batches(batch_size=100000), desc=f"Chunking {name}")):
        chunk_path = os.path.join(chunk_dir, f"chunk_{i}.npy")
        
        if os.path.exists(chunk_path):
            emb_list.append(np.load(chunk_path))
            continue
            
        df_chunk = batch.to_pandas()
        emb = model.encode(
            df_chunk['combined_text_clean'].tolist(),
            batch_size=512,
            show_progress_bar=False, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            normalize_embeddings=True 
        )
        np.save(chunk_path, emb)
        emb_list.append(emb)
        del df_chunk
        
    print(f"Stacking chunks for {name}...")
    full_emb = np.vstack(emb_list)
    np.save(out_path, full_emb)
    print(f"Saved {name} embeddings to {out_path}\n")
    
    import shutil
    shutil.rmtree(chunk_dir)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "datasets", "preprocessed"))
    embed_out_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "datasets", "dense_embeddings"))
    os.makedirs(embed_out_dir, exist_ok=True)
    
    print("Loading multilingual SentenceTransformer model...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    datasets_to_process = [
        ("train_s1", "train_source1.parquet"),
        ("train_s2", "train_source2.parquet"),
        ("train_s3", "train_source3.parquet"),
        ("test_s1", "test_source1.parquet"),
        ("test_s2", "test_source2.parquet"),
        ("test_s3", "test_source3.parquet")
    ]
    
    import gc
    for name, filename in datasets_to_process:
        # We pass filename directly. encode_and_save handles the chunked loading!
        encode_and_save(model, filename, name, embed_out_dir, preprocessed_dir)
        gc.collect()
        
    print("\nAll Dense Embeddings successfully generated and saved!")

if __name__ == "__main__":
    main()
