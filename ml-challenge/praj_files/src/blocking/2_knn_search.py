import pandas as pd
import scipy.sparse as sp
import numpy as np
import os
import time
from sklearn.neighbors import NearestNeighbors
import gc

def format_duration(seconds):
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def run_knn_for_source(
    s1_tfidf,
    target_tfidf,
    s1_ids,
    target_ids,
    target_name,
    out_tsv_path,
    chunk_dir,
    top_k=20,
    chunk_size=50000,
):
    print(f"\n--- Running KNN Blocking against {target_name} ---")
    start_time = time.time()
    
    # We use cosine distance. n_jobs=-1 uses all CPU cores.
    # Note: If you have cuML installed, this can be swapped to cuml.NearestNeighbors for GPU acceleration!
    try:
        from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
        print("✅ Using GPU-accelerated cuML for KNN!")
        nn = cuNearestNeighbors(n_neighbors=top_k, metric='cosine')
    except ImportError:
        print("⚠️ cuML not found. Falling back to CPU-based sklearn (this may take time).")
        nn = NearestNeighbors(n_neighbors=top_k, metric='cosine', n_jobs=-1)
        
    print(f"Fitting index on {target_name} ({target_tfidf.shape[0]} rows)...", flush=True)
    nn.fit(target_tfidf)
    
    print("Querying Source 1 against index with resumable chunks...", flush=True)
    os.makedirs(chunk_dir, exist_ok=True)
    num_chunks = (s1_tfidf.shape[0] + chunk_size - 1) // chunk_size
    search_start = time.time()
    rows_done = 0

    for chunk_idx in range(num_chunks):
        chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_idx}.npy")
        start_i = chunk_idx * chunk_size
        end_i = min(start_i + chunk_size, s1_tfidf.shape[0])
        current_rows = end_i - start_i

        if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 0:
            rows_done += current_rows
        else:
            chunk = s1_tfidf[start_i:end_i]
            _, chunk_indices = nn.kneighbors(chunk)
            np.save(chunk_file, np.asarray(chunk_indices))
            rows_done += current_rows

        elapsed = time.time() - search_start
        speed = rows_done / elapsed if elapsed > 0 else 0
        remaining_rows = max(0, s1_tfidf.shape[0] - end_i)
        eta_seconds = remaining_rows / speed if speed > 0 else 0
        print(
            f"[knn {target_name}] shard {chunk_idx + 1} "
            f"rows {end_i:,}/{s1_tfidf.shape[0]:,} "
            f"speed {speed:.1f} rows/s "
            f"elapsed {format_duration(elapsed)} "
            f"ETA {format_duration(eta_seconds)} "
            f"batch {chunk_size} output {chunk_file}",
            flush=True,
        )

    print(f"Formatting candidate pairs to {out_tsv_path}...", flush=True)
    tmp_out = out_tsv_path + ".tmp"
    with open(tmp_out, "w", encoding="utf-8") as output_file:
        output_file.write("source1_entity_id\tcandidate_entity_ids\n")
        for chunk_idx in range(num_chunks):
            chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_idx}.npy")
            chunk_indices = np.load(chunk_file)
            start_i = chunk_idx * chunk_size
            end_i = min(start_i + chunk_size, s1_tfidf.shape[0])
            matched_target_ids = target_ids[chunk_indices]
            output_file.writelines(
                f"{s1_id}\t{','.join(candidates)}\n"
                for s1_id, candidates in zip(s1_ids[start_i:end_i], matched_target_ids)
            )

    os.replace(tmp_out, out_tsv_path)
    del nn
    gc.collect()
    print(f"KNN completed in {time.time() - start_time:.2f} seconds.", flush=True)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "datasets", "preprocessed"))
    tfidf_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "datasets", "tfidf"))
    output_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "output"))
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading Source 1 IDs and TF-IDF matrix...")
    df_s1 = pd.read_parquet(os.path.join(preprocessed_dir, "train_source1.parquet"), columns=['entity_id'])
    s1_ids = df_s1['entity_id'].values
    s1_tfidf = sp.load_npz(
        os.path.join(tfidf_dir, "train_s1_tfidf.npz")
    ).astype("float32", copy=False)
    
    s2_out = os.path.join(output_dir, "candidate_pairs_s2.tsv")
    if os.path.exists(s2_out):
        print("\n✅ Source 2 candidates already exist! Loading checkpoint...")
        s2_candidates_df = pd.read_csv(s2_out, sep='\t')
    else:
        print("\nLoading Source 2 IDs and TF-IDF matrix...")
        df_s2 = pd.read_parquet(os.path.join(preprocessed_dir, "train_source2.parquet"), columns=['entity_id'])
        s2_ids = df_s2['entity_id'].values
        s2_tfidf = sp.load_npz(
            os.path.join(tfidf_dir, "train_s2_tfidf.npz")
        ).astype("float32", copy=False)
        
        run_knn_for_source(
            s1_tfidf, s2_tfidf, s1_ids, s2_ids, "Source 2", s2_out,
            os.path.join(output_dir, "knn_s2_chunks"), top_k=20,
        )
        s2_candidates_df = pd.read_csv(s2_out, sep='\t')
        print(f"Saved Source 2 checkpoint to {s2_out}")
        
        del df_s2, s2_tfidf
        gc.collect()
        
    s3_out = os.path.join(output_dir, "candidate_pairs_s3.tsv")
    if os.path.exists(s3_out):
        print("\n✅ Source 3 candidates already exist! Loading checkpoint...")
        s3_candidates_df = pd.read_csv(s3_out, sep='\t')
    else:
        print("\nLoading Source 3 IDs and TF-IDF matrix...")
        df_s3 = pd.read_parquet(os.path.join(preprocessed_dir, "train_source3.parquet"), columns=['entity_id'])
        s3_ids = df_s3['entity_id'].values
        s3_tfidf = sp.load_npz(
            os.path.join(tfidf_dir, "train_s3_tfidf.npz")
        ).astype("float32", copy=False)
        
        run_knn_for_source(
            s1_tfidf, s3_tfidf, s1_ids, s3_ids, "Source 3", s3_out,
            os.path.join(output_dir, "knn_s3_chunks"), top_k=20,
        )
        s3_candidates_df = pd.read_csv(s3_out, sep='\t')
        print(f"Saved Source 3 checkpoint to {s3_out}")
        
        del df_s3, s3_tfidf
        gc.collect()
    
    # ----- Combine and Save -----
    print("\nMerging candidates from Source 2 and Source 3...")
    # Merge on source1_entity_id
    final_candidates = pd.merge(s2_candidates_df, s3_candidates_df, on='source1_entity_id', how='outer', suffixes=('_s2', '_s3'))
    
    # Combine the lists
    def combine_lists(row):
        s2_list = str(row['candidate_entity_ids_s2']) if pd.notna(row['candidate_entity_ids_s2']) and row['candidate_entity_ids_s2'] else ""
        s3_list = str(row['candidate_entity_ids_s3']) if pd.notna(row['candidate_entity_ids_s3']) and row['candidate_entity_ids_s3'] else ""
        combined = [x for x in [s2_list, s3_list] if x]
        return ",".join(combined)
        
    final_candidates['candidate_entity_ids'] = final_candidates.apply(combine_lists, axis=1)
    
    # Drop the intermediate columns
    final_candidates = final_candidates[['source1_entity_id', 'candidate_entity_ids']]
    
    out_path = os.path.join(output_dir, "candidate_pairs.tsv")
    final_candidates.to_csv(out_path, sep='\t', index=False)
    print(f"\nSUCCESS! Candidate pairs saved to {out_path}")

if __name__ == "__main__":
    main()
