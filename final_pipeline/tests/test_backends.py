"""Real non-neural backend tests; model fixture tests alone do not cover these paths."""
import json
from pathlib import Path
import numpy as np
import faiss
from entity_resolution.decisions import TreeScorer, train_xgboost
from entity_resolution.data import supervised_pairs
from entity_resolution.retrieval import DenseIndex
from entity_resolution.storage import write_json
from test_pipeline import config
from entity_resolution.pipeline import Pipeline


def test_real_external_memory_xgboost(tmp_path):
    cfg = config(tmp_path)
    pipeline = Pipeline(cfg)
    pipeline.prepare(); pipeline.embed(); pipeline.retrieve(); pipeline.features()
    actual_cfg = dict(cfg, fixture=False, xgboost_trees=4, xgboost_depth=2)
    path = tmp_path/'real-tree.ubj'
    with pipeline.db(pipeline.feature_db) as db:
        train_xgboost(actual_cfg, db, path)
        rows = list(supervised_pairs(db, 'dev'))[:10]
    scorer = TreeScorer(actual_cfg, path)
    scores = scorer.score(rows)
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    assert scorer.model.num_boosted_rounds() <= 4
    assert (tmp_path/'xgboost_training.json').exists()


def test_compressed_index_and_verified_resume(tmp_path):
    faiss.omp_set_num_threads(2)
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(512, 16)).astype('float32')
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(tmp_path/'vectors.npy', vectors)
    ids = [f'S2-{i}' for i in range(len(vectors))]
    write_json(tmp_path/'ids.json', ids)
    entries = [{'vectors': 'vectors.npy', 'ids': 'ids.json', 'rows': len(vectors)}]
    cfg = {'exact_limit': 20, 'pq_m': 4, 'ivf_nlist': 2, 'seed': 42, 'index_train_rows': 512,
           'target_batch_size': 64, 'nprobe': 2}
    path = tmp_path/'dense.index'
    index = DenseIndex(cfg, tmp_path, entries, path, 'key')
    assert isinstance(index.index, faiss.IndexIVFPQ)
    assert index.index.ntotal == 512
    result = index.search(vectors[:2], 3)
    assert len(result)==2 and all(len(r)==3 for r in result)
    assert result[0][0][0] == 'S2-0'
    path.write_bytes(b'interrupted index')
    resumed = DenseIndex(cfg, tmp_path, entries, path, 'key')
    assert resumed.index.ntotal == 512
    assert resumed.search(vectors[:2], 3) == result
