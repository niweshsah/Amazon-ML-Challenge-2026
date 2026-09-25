import os
import time
import resource
import pandas as pd
from src.config import Config
from src.preprocessing.cleaner import prepare_dataframe
from src.blocking.tfidf_blocker import TfidfBlocker
# from src.blocking.dense_blocker import DenseBlocker  <-- Plug in later!

def main():
    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    pipeline_started = time.perf_counter()

    def report_step(step_name, started_at):
        memory_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        elapsed = time.perf_counter() - started_at
        print(f"[timing] {step_name}: {elapsed:.2f}s | peak memory {memory_gb:.2f} GB")
    
    print("1. Loading Datasets...")
    step_started = time.perf_counter()
    columns = ["entity_id", "business_name", "business_address", "country"]
    read_options = {"sep": "\t", "usecols": columns, "dtype": "string"}
    s1_df = pd.read_csv(
        os.path.join(cfg.DATA_DIR, f"{cfg.MODE}_source1.tsv"), **read_options
    )
    report_step("loading datasets", step_started)
    
    print("2. Preprocessing Data...")
    step_started = time.perf_counter()
    s1_df = prepare_dataframe(s1_df)
    report_step("preprocessing data", step_started)

    def corpus_chunks():
        for source_number in (2, 3):
            path = os.path.join(
                cfg.DATA_DIR, f"{cfg.MODE}_source{source_number}.tsv"
            )
            for corpus_chunk in pd.read_csv(
                path, chunksize=cfg.CORPUS_CHUNK_SIZE, **read_options
            ):
                yield prepare_dataframe(corpus_chunk)

    corpus_total_rows = 0
    for source_number in (2, 3):
        path = os.path.join(cfg.DATA_DIR, f"{cfg.MODE}_source{source_number}.tsv")
        with open(path, encoding="utf-8") as source_file:
            corpus_total_rows += sum(1 for _ in source_file) - 1
    
    print("3. Initializing Memory-Bounded Blocker...")
    step_started = time.perf_counter()
    blocker = TfidfBlocker(ngram_range=(3, 5), n_features=cfg.HASHING_FEATURES)
    report_step("initializing blocker", step_started)
    
    print("4. Generating Candidate Pairs...")
    step_started = time.perf_counter()
    candidate_df = blocker.generate_candidates(
        queries_df=s1_df, 
        top_k=cfg.BLOCKING_TOP_K, 
        threshold=cfg.BLOCKING_THRESHOLD,
        batch_size=cfg.BLOCKING_BATCH_SIZE,
        corpus_chunks=corpus_chunks(),
        corpus_total_rows=corpus_total_rows,
    )
    report_step("generating candidate pairs", step_started)
    
    step_started = time.perf_counter()
    candidate_file = os.path.join(cfg.OUTPUT_DIR, "candidate_pairs.tsv")
    candidate_df.to_csv(candidate_file, sep="\t", index=False)
    report_step("writing candidate pairs", step_started)
    print(f"Saved: {candidate_file}")
    report_step("total runtime", pipeline_started)

if __name__ == "__main__":
    main()