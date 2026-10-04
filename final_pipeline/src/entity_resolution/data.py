"""Raw Unicode serialization, bounded selection, and grouped train splits."""
from __future__ import annotations
import csv
import heapq
import json
import math
import os
import re
import unicodedata
from pathlib import Path
from unidecode import unidecode
from .storage import connect, digest, write_json

FIELDS = ('business_name', 'business_address', 'country')
SERIALIZATION = 'raw-name-address-country-v1'


def normalize(text):
    text = unicodedata.normalize('NFKC', str(text or '')).casefold()
    return ' '.join(''.join(c if c.isalnum() or unicodedata.category(c).startswith('M') else ' ' for c in text).split())


def serialize(row):
    return ' | '.join(f'{label}: {row.get(field, "")}' for label, field in zip(('Name', 'Address', 'Country'), FIELDS) if row.get(field))


def views(row):
    raw = serialize(row)
    norm = normalize(raw)
    return raw, norm, normalize(unidecode(norm))


def read_tsv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream, delimiter='\t')


def rank(entity_id, seed, salt):
    return digest([seed, salt, entity_id])


def select(ids, percent, seed, salt):
    if not 0 < percent <= 100:
        raise ValueError('Percent must be greater than zero and at most 100')
    # Exact ceil percentage, independent of input order; only selected IDs are retained.
    count = sum(1 for _ in ids())
    size = math.ceil(count * percent / 100)
    return set(heapq.nsmallest(size, ids(), key=lambda x: (rank(x, seed, salt), x)))


def split_groups(ids, seed):
    ordered = sorted(ids, key=lambda x: (rank(x, seed, 'split'), x))
    n = len(ordered)
    # Largest remainder gives 70/10/10/10 and keeps tiny samples usable for diagnostics.
    counts = [int(n * p) for p in (.7, .1, .1, .1)]
    for i in sorted(range(4), key=lambda i: -(n * (.7, .1, .1, .1)[i] - counts[i]))[:n-sum(counts)]:
        counts[i] += 1
    groups = {}
    start = 0
    for label, size in zip(('fit', 'dev', 'calibration', 'audit'), counts):
        for sid in ordered[start:start+size]:
            groups[sid] = label
        start += size
    return groups


def prepare(cfg, destination):
    data = Path(cfg['data_dir'])
    mode = cfg['mode']
    files = [data / f'{mode}_source{i}.tsv' for i in (1, 2, 3)]
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f'Required input missing: {path}')
    ids = lambda: (r['entity_id'] for r in read_tsv(files[0]))
    selected = select(ids, cfg['train_percent'], cfg['seed'], 'sample') if mode == 'train' else None
    groups = split_groups(selected, cfg['seed']) if selected is not None else {}
    tmp = destination.with_suffix('.tmp.sqlite')
    tmp.unlink(missing_ok=True)
    db = connect(tmp)
    db.executescript('''
      CREATE TABLE records(id TEXT PRIMARY KEY, source INTEGER, raw TEXT, normalized TEXT, latin TEXT,
        payload TEXT, split TEXT, position INTEGER);
      CREATE INDEX record_source ON records(source, position);
      CREATE TABLE truth(qid TEXT, tid TEXT, PRIMARY KEY(qid,tid));
      CREATE INDEX truth_target ON truth(tid);
    ''')
    if mode == 'train':
        truth_path = data / 'train_ground_truth.tsv'
        truth_groups = set()
        for r in read_tsv(truth_path):
            sid = r['source1_entity_id']
            if sid not in selected:
                continue
            if sid in truth_groups:
                raise ValueError(f'Duplicate ground truth group: {sid}')
            truth_groups.add(sid)
            db.executemany('INSERT INTO truth VALUES (?,?)', ((sid, t.strip()) for t in r['matched_entity_ids'].split(',') if t.strip()))
        if truth_groups != selected:
            raise ValueError('Ground truth must include every selected entity, including singletons')
    mandatory = {r[0] for r in db.execute('SELECT DISTINCT tid FROM truth')}
    found = set()
    counts = {}
    distractor_percent = cfg['target_distractor_percent'] or cfg['train_percent']
    for source, path in enumerate(files, 1):
        distractors = None
        if selected is not None and source > 1:
            remaining = lambda: (r['entity_id'] for r in read_tsv(path) if r['entity_id'] not in mandatory)
            distractors = select(remaining, distractor_percent, cfg['seed'], f'target{source}')
        allowed = mandatory | distractors if distractors is not None else None
        position = 0
        for row in read_tsv(path):
            if not {'entity_id', *FIELDS} <= row.keys():
                raise ValueError(f'Invalid source header: {path}')
            sid = row['entity_id']
            if not sid or any(c in sid for c in ',\t\r\n'):
                raise ValueError('Entity IDs must be nonempty and cannot contain list separators')
            if source == 1 and selected is not None and sid not in selected:
                continue
            if source > 1 and distractors is not None and sid not in allowed:
                continue
            raw, norm, latin = views(row)
            db.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?)',
                       (sid, source, raw, norm, latin, json.dumps(row, ensure_ascii=False), groups.get(sid, 'test'), position))
            found.add(sid) if sid in mandatory else None
            position += 1
        counts[str(source)] = position
        db.commit()
    if mandatory != found:
        raise ValueError(f'Missing positive target records: {len(mandatory-found)}')
    db.close()
    os.replace(tmp, destination)
    write_json(destination.parent / 'splits.json', {'groups': groups, 'counts': counts,
        'scope': 'sampled training target pools' if mode == 'train' else 'complete test target pools',
        'train_percent': cfg['train_percent'], 'target_distractor_percent': distractor_percent})


def supervised_pairs(db, split='fit'):
    # Held-out positive targets are forbidden even as fitting negatives.
    return db.execute('''SELECT p.*, q.raw AS query_text, t.raw AS target_text,
      EXISTS(SELECT 1 FROM truth g WHERE g.qid=p.qid AND g.tid=p.tid) AS label
      FROM pairs p JOIN records q ON q.id=p.qid JOIN records t ON t.id=p.tid
      WHERE q.split=? AND (?!='fit' OR NOT EXISTS(
        SELECT 1 FROM truth g JOIN records h ON h.id=g.qid WHERE g.tid=p.tid AND h.split!='fit'))
      ORDER BY p.qid, p.dense_score DESC, p.tfidf_score DESC, p.tid''', (split, split))
