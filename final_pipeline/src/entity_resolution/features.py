"""Explicit feature allowlist: IDs, labels and sampling metadata never enter XGBoost."""
from __future__ import annotations
import json
import re
from difflib import SequenceMatcher
import numpy as np
from .data import FIELDS, normalize

FEATURE_NAMES = ['dense_similarity', 'tfidf_similarity'] + [f'{f}_{s}' for f in FIELDS for s in
    ('exact', 'sequence', 'jaccard', 'latin_sequence', 'left_missing', 'right_missing')] + ['numeric_jaccard', 'numeric_exact']


def jaccard(a, b):
    return len(a & b)/len(a | b) if a and b else 0.


def pair_features(pair, left, right):
    from unidecode import unidecode
    values = [pair['dense_score'], pair['tfidf_score']]
    for field in FIELDS:
        a, b = normalize(left.get(field)), normalize(right.get(field))
        values.extend([float(bool(a and b) and a == b), SequenceMatcher(None, a, b).ratio() if a and b else 0.,
                       jaccard(set(a.split()), set(b.split())),
                       SequenceMatcher(None, unidecode(a), unidecode(b)).ratio() if a and b else 0., float(not a), float(not b)])
    a, b = (set(re.findall(r'\d+', row.get('business_address', ''))) for row in (left, right))
    values.extend([jaccard(a, b), float(bool(a and b) and a == b)])
    assert len(values) == len(FEATURE_NAMES)
    return values


def extract(db, batch_size):
    from .retrieval import batches
    # Cursor reads and writes separate tables safely; bounded update batches.
    cursor = db.execute('''SELECT p.*, q.payload left_payload,t.payload right_payload
      FROM pairs p JOIN records q ON q.id=p.qid JOIN records t ON t.id=p.tid ORDER BY p.qid,p.tid''')
    for batch in batches(cursor, batch_size):
        updates = [(json.dumps(pair_features(r, json.loads(r['left_payload']), json.loads(r['right_payload']))), r['qid'], r['tid']) for r in batch]
        db.executemany('UPDATE pairs SET features=? WHERE qid=? AND tid=?', updates)
    db.commit()


def matrix(rows):
    return np.asarray([json.loads(r['features']) for r in rows], dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
