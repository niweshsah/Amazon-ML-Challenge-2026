"""Atomic stage commits and content-verified resumability."""
from __future__ import annotations
import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def fingerprint(path: Path):
    if not path.exists():
        return {'missing': str(path)}
    if path.is_file():
        return {'path': str(path.resolve()), 'sha256': sha(path)}
    return {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob('*')) if p.is_file()}


@contextmanager
def atomic_file(path: Path, binary=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    try:
        with tmp.open('wb' if binary else 'w', **({} if binary else {'encoding': 'utf-8', 'newline': ''})) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path: Path, value):
    with atomic_file(path) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def connect(path: Path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA temp_store=FILE')
    db.execute('PRAGMA cache_size=-32768')
    return db


class Cache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def valid(self, name, key, paths):
        stamp = self.root / (name + '.manifest.json')
        if not stamp.exists():
            return False
        old = json.loads(stamp.read_text())
        return old.get('key') == key and old.get('outputs') == [fingerprint(p) for p in paths]

    def commit(self, name, key, paths):
        write_json(self.root / (name + '.manifest.json'), {
            'key': key, 'outputs': [fingerprint(p) for p in paths], 'success': True})


def model_fingerprints(cfg, tree):
    """Path-independent model content provenance, including cached base weights."""
    def content(path):
        result = fingerprint(path)
        if Path(path).is_file():
            result.pop('path', None)
        return result
    result = {'tree': content(tree)}
    for kind in ('biencoder', 'crossencoder'):
        checkpoint = Path(cfg[kind+'_checkpoint'])
        result[kind] = content(checkpoint)
        if not cfg['fixture']:
            from huggingface_hub import snapshot_download
            base = Path(cfg[kind+'_base'])
            if not base.is_dir():
                base = Path(snapshot_download(cfg[kind+'_base'], revision=cfg[kind+'_revision'],
                                              local_files_only=cfg['local_files_only']))
            result[kind+'_base'] = content(base)
    return result
