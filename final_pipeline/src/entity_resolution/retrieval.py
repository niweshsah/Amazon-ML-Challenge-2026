"""Resumable embedding shards, exact/IVF-PQ dense search and bounded sparse union."""
from __future__ import annotations
import itertools
import json
import os
from pathlib import Path
import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from .models import NeuralModel, fixture_vectors
from .storage import Cache, atomic_file, connect, digest, fingerprint, write_json


def batches(rows, size):
    iterator = iter(rows)
    while batch := list(itertools.islice(iterator, size)):
        yield batch


def embed(cfg, db, root, key):
    cache = Cache(root)
    model = None if cfg['fixture'] else NeuralModel(cfg, 'biencoder')
    catalog = {}
    for source in (1, 2, 3):
        entries = []
        rows = db.execute('SELECT id,raw FROM records WHERE source=? ORDER BY position', (source,))
        for i, chunk in enumerate(batches(rows, cfg['shard_rows'])):
            vectors = root / f'source{source}-{i:05d}.npy'
            ids = root / f'source{source}-{i:05d}.ids.json'
            shard_key = digest([key, source, i, [r['id'] for r in chunk]])
            if not cache.valid(f'source{source}-{i:05d}', shard_key, [vectors, ids]):
                parts = []
                # Encode incrementally into a memmap, then atomically commit the shard.
                tmp = vectors.with_suffix('.tmp.npy')
                arr = None
                start = 0
                for batch in batches(chunk, cfg['batch_size']):
                    texts = [r['raw'] for r in batch]
                    result = fixture_vectors(texts) if model is None else model.embed(texts)
                    if arr is None:
                        arr = np.lib.format.open_memmap(tmp, mode='w+', dtype='float32', shape=(len(chunk), result.shape[1]))
                    arr[start:start+len(batch)] = result
                    start += len(batch)
                arr.flush()
                del arr
                os.replace(tmp, vectors)
                write_json(ids, [r['id'] for r in chunk])
                cache.commit(f'source{source}-{i:05d}', shard_key, [vectors, ids])
            entries.append({'vectors': vectors.name, 'ids': ids.name, 'rows': len(chunk)})
        catalog[str(source)] = entries
    write_json(root / 'catalog.json', catalog)


class DenseIndex:
    def __init__(self, cfg, root, entries, destination, key):
        import faiss
        faiss.omp_set_num_threads(cfg.get('faiss_threads', 4))
        self.faiss = faiss
        self.ids = []
        for entry in entries:
            self.ids.extend(json.loads((root / entry['ids']).read_text()))
        self.index = None
        if not entries:
            return
        n = len(self.ids)
        dim = np.load(root / entries[0]['vectors'], mmap_mode='r').shape[1]
        cache = Cache(destination.parent)
        if cache.valid(destination.stem, key, [destination]):
            self.index = faiss.read_index(str(destination))
        else:
            if n <= cfg['exact_limit']:
                index = faiss.IndexFlatIP(dim)
            else:
                m = min(cfg['pq_m'], dim)
                while dim % m:
                    m -= 1
                nlist = min(cfg['ivf_nlist'], max(1, n//40))
                # Deterministic reservoir positions, bounded training sample across the full pool.
                positions = np.random.default_rng(cfg['seed']).choice(n, min(n, cfg['index_train_rows']), replace=False)
                positions.sort()
                samples, offset = [], 0
                for e in entries:
                    vectors = np.load(root / e['vectors'], mmap_mode='r')
                    ix = positions[(positions >= offset) & (positions < offset+len(vectors))]-offset
                    if len(ix):
                        samples.append(np.asarray(vectors[ix], dtype=np.float32))
                    offset += len(vectors)
                sample = np.concatenate(samples)
                if len(sample) < 256:
                    raise ValueError('IVF-PQ needs at least 256 training vectors; raise exact_limit')
                index = faiss.IndexIVFPQ(faiss.IndexFlatIP(dim), dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
                index.cp.seed = cfg['seed']
                index.pq.cp.seed = cfg['seed']
                index.train(sample)
            for e in entries:
                arr = np.load(root / e['vectors'], mmap_mode='r')
                for start in range(0, len(arr), cfg['target_batch_size']):
                    index.add(np.asarray(arr[start:start+cfg['target_batch_size']], dtype=np.float32))
            tmp = destination.with_suffix('.tmp.index')
            faiss.write_index(index, str(tmp))
            os.replace(tmp, destination)
            cache.commit(destination.stem, key, [destination])
            self.index = index
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = cfg['nprobe']

    def search(self, queries, k):
        if self.index is None:
            return [[] for _ in queries]
        scores, positions = self.index.search(np.asarray(queries, dtype=np.float32), min(k, len(self.ids)))
        return [[(self.ids[j], float(s), rank+1) for rank, (j, s) in enumerate(zip(ix, ss)) if j >= 0]
                for ix, ss in zip(positions, scores)]


def lexical_text(row):
    return row['normalized'] + ' ' + row['latin']


class SparseIndex:
    def __init__(self, cfg, db, source, root, key):
        self.cfg, self.root = cfg, root
        root.mkdir(parents=True, exist_ok=True)
        model_path = root / 'vectorizer.joblib'
        manifest = root / 'catalog.json'
        cache = Cache(root)
        if not cache.valid('vectorizer', key, [model_path]):
            texts = (lexical_text(r) for r in db.execute('SELECT normalized,latin FROM records WHERE source=? ORDER BY position', (source,)))
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), dtype=np.float32,
                                         max_features=cfg['tfidf_max_features'])
            try:
                vectorizer.fit(texts)
            except ValueError as exc:
                if 'empty vocabulary' not in str(exc):
                    raise
                vectorizer.fit(['___empty___'])
            with atomic_file(model_path, binary=True) as stream:
                joblib.dump(vectorizer, stream)
            cache.commit('vectorizer', key, [model_path])
        self.vectorizer = joblib.load(model_path)
        self.parts = []
        rows = db.execute('SELECT id,normalized,latin FROM records WHERE source=? ORDER BY position', (source,))
        for i, chunk in enumerate(batches(rows, cfg['target_batch_size'])):
            path, ids = root / f'{i:05d}.npz', root / f'{i:05d}.ids.json'
            if not cache.valid(f'part-{i}', key, [path, ids]):
                matrix = self.vectorizer.transform(lexical_text(r) for r in chunk)
                with atomic_file(path, binary=True) as stream:
                    sparse.save_npz(stream, matrix)
                write_json(ids, [r['id'] for r in chunk])
                cache.commit(f'part-{i}', key, [path, ids])
            self.parts.append((path, ids))
        write_json(manifest, {'key': key, 'parts': [(p.name, i.name) for p, i in self.parts]})

    def search(self, texts, k):
        q = self.vectorizer.transform(texts)
        best = [[] for _ in texts]
        for path, ids_path in self.parts:
            targets = sparse.load_npz(path)
            ids = json.loads(ids_path.read_text())
            scores = (q @ targets.T).tocsr()
            # Sparse products are bounded by query_batch_size x target_batch_size.
            for i in range(len(texts)):
                row = scores.getrow(i)
                if len(row.data):
                    indices = np.argsort(-row.data, kind='stable')[:k]
                    candidates = [(ids[row.indices[j]], float(row.data[j])) for j in indices]
                    best[i] = sorted(best[i]+candidates, key=lambda x: (-x[1], x[0]))[:k]
        return [[(tid, score, i+1) for i, (tid, score) in enumerate(row)] for row in best]


def retrieve(cfg, records, embedding_root, destination, key):
    catalog = json.loads((embedding_root / 'catalog.json').read_text())
    tmp = destination.with_suffix('.tmp.sqlite')
    tmp.unlink(missing_ok=True)
    db = connect(tmp)
    db.executescript('''CREATE TABLE pairs(qid TEXT, tid TEXT, source INTEGER,
      dense_score REAL, tfidf_score REAL, dense_rank INTEGER, tfidf_rank INTEGER,
      channels TEXT, features TEXT, PRIMARY KEY(qid,tid));''')
    for source in (2, 3):
        dense = DenseIndex(cfg, embedding_root, catalog[str(source)], destination.parent / f'source{source}.index', digest([key, source, 'dense']))
        lex = SparseIndex(cfg, records, source, destination.parent / f'tfidf-source{source}', digest([key, source, 'tfidf']))
        from functools import lru_cache
        @lru_cache(maxsize=4)
        def target_shard(part):
            return np.load(embedding_root / catalog[str(source)][part]['vectors'], mmap_mode='r')
        for entry in catalog['1']:
            arr = np.load(embedding_root / entry['vectors'], mmap_mode='r')
            ids = json.loads((embedding_root / entry['ids']).read_text())
            for start in range(0, len(ids), cfg['query_batch_size']):
                qids = ids[start:start+cfg['query_batch_size']]
                texts = [lexical_text(records.execute('SELECT normalized,latin FROM records WHERE id=?', (sid,)).fetchone()) for sid in qids]
                dense_rows = dense.search(arr[start:start+len(qids)], cfg['top_k_dense'])
                sparse_rows = lex.search(texts, cfg['top_k_tfidf'])
                for local_ix, (sid, ds, ss) in enumerate(zip(qids, dense_rows, sparse_rows)):
                    union = {}
                    for channel, rows in (('dense', ds), ('tfidf', ss)):
                        for tid, score, rank in rows:
                            item = union.setdefault(tid, [0., 0., None, None, []])
                            ix = 0 if channel == 'dense' else 1
                            item[ix], item[ix+2] = score, rank
                            item[4].append(channel)
                    # Every union member gets both similarities; absent channel rank stays null.
                    # Ground truth is never read by this candidate-generation stage.
                    query_lex = lex.vectorizer.transform([texts[local_ix]])
                    for tid, item in union.items():
                        target = records.execute('SELECT normalized,latin,position FROM records WHERE id=?', (tid,)).fetchone()
                        if item[2] is None:
                            position = target['position']
                            target_vector = target_shard(position//cfg['shard_rows'])[position%cfg['shard_rows']]
                            item[0] = float(np.dot(arr[start+local_ix], target_vector))
                        if item[3] is None:
                            target_lex = lex.vectorizer.transform([lexical_text(target)])
                            item[1] = float(query_lex.multiply(target_lex).sum())
                    db.executemany('INSERT INTO pairs VALUES (?,?,?,?,?,?,?,?,NULL)',
                        [(sid, tid, source, *item[:4], ','.join(item[4])) for tid, item in sorted(union.items())])
                db.commit()
        del dense, lex
    db.close()
    os.replace(tmp, destination)
