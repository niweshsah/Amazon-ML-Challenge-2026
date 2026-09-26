import pandas as pd
import numpy as np
import os
import argparse
import faiss
import time
import gc
import json

def format_duration(seconds):
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def run_faiss_search(
    s1_emb,
    target_emb_path,
    s1_ids,
    target_ids,
    target_name,
    out_tsv_path,
    chunk_dir,
    index_path,
    top_k=20,
    chunk_size=25000,
    similarity_threshold=None,
    force_rebuild_index=False,
):
    print(f"\n--- Running FAISS GPU Search against {target_name} ---", flush=True)
    start_time = time.time()
    
    # Keep a serialized CPU index so future runs can load it instead of adding
    # every target embedding to a fresh index again.
    index_path = os.path.abspath(index_path)
    metadata_path = index_path + ".json"
    target_stat = os.stat(target_emb_path)
    target_emb = np.load(target_emb_path, mmap_mode="r")
    if target_emb.ndim != 2:
        raise ValueError(f"Expected a 2D embedding array in {target_emb_path}")
    dim = int(target_emb.shape[1])
    metadata = {
        "embedding_path": os.path.abspath(target_emb_path),
        "embedding_size": target_stat.st_size,
        "embedding_mtime_ns": target_stat.st_mtime_ns,
        "vector_count": int(target_emb.shape[0]),
        "dimension": dim,
        "metric": "inner_product",
        "index_type": "IndexFlatIP",
    }
    use_saved_index = False
    if not force_rebuild_index and os.path.exists(index_path) and os.path.exists(metadata_path):
        try:
            with open(metadata_path, encoding="utf-8") as metadata_file:
                use_saved_index = json.load(metadata_file) == metadata
        except (OSError, json.JSONDecodeError):
            use_saved_index = False

    if use_saved_index:
        print(f"Loading saved {target_name} FAISS index from {index_path}...", flush=True)
        cpu_index = faiss.read_index(index_path)
        if cpu_index.d != dim or cpu_index.ntotal != target_emb.shape[0]:
            raise ValueError(
                f"Saved index at {index_path} has {cpu_index.ntotal} vectors / dimension "
                f"{cpu_index.d}; expected {target_emb.shape[0]} / {dim}. Use --rebuild-index."
            )
    else:
        reason = "--rebuild-index supplied" if force_rebuild_index else "saved index missing or stale"
        print(f"Building {target_name} FAISS index ({reason})...", flush=True)
        cpu_index = faiss.IndexFlatIP(dim)
        index_batch_size = 100_000
        for start in range(0, target_emb.shape[0], index_batch_size):
            stop = min(start + index_batch_size, target_emb.shape[0])
            cpu_index.add(np.ascontiguousarray(target_emb[start:stop]))
            if stop % 500_000 < index_batch_size or stop == target_emb.shape[0]:
                print(f"[index {target_name}] added {stop:,}/{target_emb.shape[0]:,} vectors", flush=True)
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        temp_index_path = index_path + ".tmp"
        faiss.write_index(cpu_index, temp_index_path)
        os.replace(temp_index_path, index_path)
        temp_metadata_path = metadata_path + ".tmp"
        with open(temp_metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
        os.replace(temp_metadata_path, metadata_path)
        print(f"Saved reusable {target_name} index to {index_path}", flush=True)

    del target_emb
    gc.collect()
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    del cpu_index
    gc.collect()
    
    # 2. Chunk-level search with micro-checkpoints saved to disk
    os.makedirs(chunk_dir, exist_ok=True)
    n_queries = len(s1_ids)
    num_chunks = (n_queries + chunk_size - 1) // chunk_size
    
    print(f"Querying Source 1 ({n_queries:,} rows) against GPU index in {num_chunks} shards...", flush=True)
    search_start = time.time()
    rows_done_this_run = 0
    
    for chunk_idx in range(num_chunks):
        chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_idx}.npz")
        start_i = chunk_idx * chunk_size
        end_i = min(start_i + chunk_size, n_queries)
        current_chunk_rows = end_i - start_i
        
        # Checkpointer: If this chunk was already computed in a previous run, skip it!
        if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 0:
            cached = np.load(chunk_file)
            if cached["indices"].shape[1] >= top_k:
                print(f"[faiss {target_name}] shard {chunk_idx + 1} already cached. Skipping!", flush=True)
                rows_done_this_run += current_chunk_rows
                elapsed = time.time() - search_start
                speed = rows_done_this_run / elapsed if elapsed > 0 else 0
                remaining_rows = max(0, n_queries - end_i)
                eta_seconds = remaining_rows / speed if speed > 0 else 0
                print(
                    f"[faiss {target_name}] shard {chunk_idx + 1} "
                    f"rows {end_i:,}/{n_queries:,} "
                    f"speed {speed:.1f} rows/s "
                    f"elapsed {format_duration(elapsed)} "
                    f"ETA {format_duration(eta_seconds)} "
                    f"batch {chunk_size} output {chunk_file}",
                    flush=True,
                )
                continue
            
        # s1_emb is memory-mapped, so this reads only the active query slice from disk
        query_chunk = np.ascontiguousarray(s1_emb[start_i:end_i])
        
        chunk_scores, chunk_indices = gpu_index.search(query_chunk, top_k)
        
        # Save indices and scores so threshold filtering can resume safely.
        np.savez(chunk_file, indices=chunk_indices, scores=chunk_scores)
        
        # Timing & Metrics
        rows_done_this_run += current_chunk_rows
        elapsed = time.time() - search_start
        speed = rows_done_this_run / elapsed if elapsed > 0 else 0
        remaining_rows = max(0, n_queries - end_i)
        eta_seconds = remaining_rows / speed if speed > 0 else 0
        
        print(
            f"[faiss {target_name}] shard {chunk_idx + 1} "
            f"rows {end_i:,}/{n_queries:,} "
            f"speed {speed:.1f} rows/s "
            f"elapsed {format_duration(elapsed)} "
            f"ETA {format_duration(eta_seconds)} "
            f"batch {chunk_size} "
            f"output {chunk_file}",
            flush=True
        )
        
    # Free GPU index memory now that search is complete
    del gpu_index, cpu_index, res
    gc.collect()
    
    # 3. Stream-format results chunk by chunk to TSV (Rock-solid, minimal RAM)
    print(f"Formatting and saving {target_name} candidates to {out_tsv_path}...", flush=True)
    write_start = time.time()
    tmp_out = out_tsv_path + ".tmp"
    with open(tmp_out, 'w', encoding='utf-8') as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for chunk_idx in range(num_chunks):
            chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_idx}.npz")
            checkpoint = np.load(chunk_file)
            chunk_indices = checkpoint["indices"]
            chunk_scores = checkpoint["scores"]
            
            start_i = chunk_idx * chunk_size
            end_i = min(start_i + chunk_size, n_queries)
            chunk_s1_ids = s1_ids[start_i:end_i]
            
            # Vectorized ID lookup
            matched_target_ids = target_ids[chunk_indices]
            if similarity_threshold is not None:
                matched_target_ids = np.where(
                    chunk_scores >= similarity_threshold,
                    matched_target_ids,
                    "",
                )
            
            # Format lines and write directly to disk
            lines = [
                f"{s1_id}\t{','.join(candidate for candidate in cands if candidate)}\n"
                for s1_id, cands in zip(chunk_s1_ids, matched_target_ids)
            ]
            f_out.writelines(lines)
            del chunk_indices, matched_target_ids, lines
            
    os.replace(tmp_out, out_tsv_path)
    print(f"Saved {target_name} candidates to {out_tsv_path} in {time.time() - start_time:.2f} seconds.\n", flush=True)

def main(similarity_threshold=None, force_rebuild=False, force_rebuild_index=False, top_k=20):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "datasets", "preprocessed"))
    embed_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "datasets", "dense_embeddings"))
    output_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "output"))
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading Source 1 IDs and memory-mapped Embeddings...", flush=True)
    df_s1 = pd.read_parquet(os.path.join(preprocessed_dir, "train_source1.parquet"), columns=['entity_id'])
    s1_ids = df_s1['entity_id'].values
    del df_s1
    gc.collect()
    
    # Use mmap_mode='r' to prevent loading 3.4GB into RAM all at once
    s1_emb_path = os.path.join(embed_dir, "train_s1_embeddings.npy")
    s1_emb = np.load(s1_emb_path, mmap_mode='r')
    
    # ----- Source 2 -----
    s2_out = os.path.join(output_dir, "candidate_pairs_faiss_s2.tsv")
    s2_chunk_dir = os.path.join(output_dir, f"faiss_s2_chunks_top{top_k}")
    s2_index_path = os.path.join(output_dir, "faiss_index_train_source2.index")
    
    if not force_rebuild and not force_rebuild_index and similarity_threshold is None and os.path.exists(s2_out) and os.path.getsize(s2_out) > 0:
        print("\n✅ Source 2 candidates already exist! Skipping search.", flush=True)
    else:
        print("\nLoading Source 2 IDs...", flush=True)
        df_s2 = pd.read_parquet(os.path.join(preprocessed_dir, "train_source2.parquet"), columns=['entity_id'])
        s2_ids = df_s2['entity_id'].values
        del df_s2
        gc.collect()
        
        s2_emb_path = os.path.join(embed_dir, "train_s2_embeddings.npy")
        run_faiss_search(
            s1_emb, s2_emb_path, s1_ids, s2_ids, "train_source2", s2_out,
            s2_chunk_dir, top_k=top_k, chunk_size=25000,
            similarity_threshold=similarity_threshold,
            index_path=s2_index_path, force_rebuild_index=force_rebuild_index,
        )
        
        del s2_ids
        gc.collect()
        
    # ----- Source 3 -----
    s3_out = os.path.join(output_dir, "candidate_pairs_faiss_s3.tsv")
    s3_chunk_dir = os.path.join(output_dir, f"faiss_s3_chunks_top{top_k}")
    s3_index_path = os.path.join(output_dir, "faiss_index_train_source3.index")
    
    if not force_rebuild and not force_rebuild_index and similarity_threshold is None and os.path.exists(s3_out) and os.path.getsize(s3_out) > 0:
        print("\n✅ Source 3 candidates already exist! Skipping search.", flush=True)
    else:
        print("\nLoading Source 3 IDs...", flush=True)
        df_s3 = pd.read_parquet(os.path.join(preprocessed_dir, "train_source3.parquet"), columns=['entity_id'])
        s3_ids = df_s3['entity_id'].values
        del df_s3
        gc.collect()
        
        s3_emb_path = os.path.join(embed_dir, "train_s3_embeddings.npy")
        run_faiss_search(
            s1_emb, s3_emb_path, s1_ids, s3_ids, "train_source3", s3_out,
            s3_chunk_dir, top_k=top_k, chunk_size=25000,
            similarity_threshold=similarity_threshold,
            index_path=s3_index_path, force_rebuild_index=force_rebuild_index,
        )
        
        del s3_ids
        gc.collect()
    
    # ----- Combine and Save -----
    out_path = os.path.join(output_dir, "candidate_pairs_faiss.tsv")
    if not force_rebuild and not force_rebuild_index and similarity_threshold is None and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"\n✅ {out_path} already finalized. Done!", flush=True)
        return
        
    print("\nMerging FAISS candidates from Source 2 and Source 3...", flush=True)
    s2_candidates_df = pd.read_csv(s2_out, sep='\t')
    s3_candidates_df = pd.read_csv(s3_out, sep='\t')
    
    final_candidates = pd.merge(s2_candidates_df, s3_candidates_df, on='source1_entity_id', how='outer', suffixes=('_s2', '_s3'))
    del s2_candidates_df, s3_candidates_df
    gc.collect()
    
    def combine_lists(row):
        s2_list = str(row['candidate_entity_ids_s2']) if pd.notna(row['candidate_entity_ids_s2']) and row['candidate_entity_ids_s2'] else ""
        s3_list = str(row['candidate_entity_ids_s3']) if pd.notna(row['candidate_entity_ids_s3']) and row['candidate_entity_ids_s3'] else ""
        combined = [x for x in [s2_list, s3_list] if x]
        return ",".join(combined)
        
    final_candidates['candidate_entity_ids'] = final_candidates.apply(combine_lists, axis=1)
    final_candidates = final_candidates[['source1_entity_id', 'candidate_entity_ids']]
    
    final_candidates.to_csv(out_path, sep='\t', index=False)
    print(f"\nSUCCESS! FAISS Candidate pairs saved to {out_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FAISS GPU candidate pairs.")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Keep only candidates with cosine similarity at least this value.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recompute candidates and score checkpoints even when outputs exist.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Recreate and overwrite the saved Source 2 and Source 3 FAISS indexes.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of ranked candidates and scores to retain per query.",
    )
    args = parser.parse_args()
    if args.similarity_threshold is not None and not -1 <= args.similarity_threshold <= 1:
        parser.error("--similarity-threshold must be between -1 and 1")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    main(
        similarity_threshold=args.similarity_threshold,
        force_rebuild=args.rebuild,
        force_rebuild_index=args.rebuild_index,
        top_k=args.top_k,
    )
