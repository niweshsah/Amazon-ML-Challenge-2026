"""Exact challenge TSVs, disk-backed validation and self-contained packaging."""
from __future__ import annotations
import csv
import json
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from .storage import atomic_file

HEADERS = {'matching_results.tsv': ['source1_entity_id', 'matched_entity_ids'],
           'candidate_pairs.tsv': ['source1_entity_id', 'candidate_entity_ids']}


def write_outputs(db, root):
    with atomic_file(root / 'candidate_pairs.tsv') as candidate, atomic_file(root / 'matching_results.tsv') as matching:
        writers = [csv.writer(s, delimiter='\t', lineterminator='\n') for s in (candidate, matching)]
        writers[0].writerow(HEADERS['candidate_pairs.tsv'])
        writers[1].writerow(HEADERS['matching_results.tsv'])
        for row in db.execute('SELECT id FROM records WHERE source=1 ORDER BY position'):
            sid = row['id']
            candidates = [r[0] for r in db.execute('SELECT tid FROM pairs WHERE qid=? ORDER BY tid', (sid,))]
            matches = [r[0] for r in db.execute('SELECT tid FROM decisions WHERE qid=? AND accepted=1 ORDER BY tid', (sid,))]
            if not set(matches) <= set(candidates):
                raise ValueError('Accepted candidate containment failed')
            writers[0].writerow([sid, ','.join(candidates)])
            writers[1].writerow([sid, ','.join(matches)])


def validate(data_dir, output_dir, mode='test'):
    """No neural dependencies or RAM-sized ID sets required."""
    with tempfile.TemporaryDirectory(prefix='er-validation-') as tmp:
        db = sqlite3.connect(Path(tmp) / 'validation.sqlite')
        db.executescript('CREATE TABLE ids(id TEXT PRIMARY KEY,source INTEGER); CREATE TABLE lists(qid TEXT PRIMARY KEY,candidates TEXT,matches TEXT);')
        for source in (1, 2, 3):
            with (Path(data_dir) / f'{mode}_source{source}.tsv').open(encoding='utf-8-sig', newline='') as input_stream:
                reader = csv.DictReader(input_stream, delimiter='\t')
                for row in reader:
                    sid = row['entity_id']
                    if not sid.startswith(f'S{source}-'):
                        raise ValueError(f'Invalid source prefix: {sid}')
                    db.execute('INSERT INTO ids VALUES (?,?)', (sid, source))
            db.commit()
        for name in ('candidate_pairs.tsv', 'matching_results.tsv'):
            path = Path(output_dir) / name
            with path.open(encoding='utf-8', newline='') as stream:
                reader = csv.reader(stream, delimiter='\t')
                if next(reader, None) != HEADERS[name]:
                    raise ValueError(f'Incorrect header in {name}')
                for row in reader:
                    if len(row) != 2:
                        raise ValueError(f'Expected two tab-separated fields in {name}')
                    sid, value = row
                    if db.execute('SELECT source FROM ids WHERE id=?', (sid,)).fetchone() != (1,):
                        raise ValueError(f'Unknown Source 1 ID: {sid}')
                    ids = value.split(',') if value else []
                    if len(ids) != len(set(ids)) or any(not x for x in ids):
                        raise ValueError(f'Duplicate or empty target IDs: {sid}')
                    for tid in ids:
                        if db.execute('SELECT source FROM ids WHERE id=?', (tid,)).fetchone() not in ((2,), (3,)):
                            raise ValueError(f'Unknown target: {tid}')
                    if name == 'candidate_pairs.tsv':
                        db.execute('INSERT INTO lists(qid,candidates) VALUES (?,?)', (sid, value))
                    else:
                        existing = db.execute('SELECT candidates,matches FROM lists WHERE qid=?', (sid,)).fetchone()
                        if not existing or existing[1] is not None:
                            raise ValueError(f'Missing candidates or duplicate match row: {sid}')
                        if not set(ids) <= set(existing[0].split(',') if existing[0] else []):
                            raise ValueError(f'Matches outside candidate union: {sid}')
                        db.execute('UPDATE lists SET matches=? WHERE qid=?', (value, sid))
                db.commit()
        expected = db.execute('SELECT COUNT(*) FROM ids WHERE source=1').fetchone()[0]
        actual = db.execute('SELECT COUNT(*) FROM lists WHERE matches IS NOT NULL').fetchone()[0]
        if expected != actual:
            raise ValueError(f'Expected {expected} Source 1 rows; got {actual}')
        db.close()
    return {'status': 'PASS', 'source1_rows': actual, 'mode': mode}


def package(cfg, project, destination, model_dir, policy_dir):
    if cfg['mode'] != 'test' or cfg['fixture']:
        raise ValueError('Submission packaging requires complete test inference using trained assets')
    validate(cfg['data_dir'], cfg['output_dir'], 'test')
    from huggingface_hub import snapshot_download
    from .storage import digest, model_fingerprints
    # Build a unique member map, then commit the archive atomically.
    prefix = 'code/business_entity_resolution/'
    members = {}
    overrides = {}
    for name in HEADERS:
        members['output/'+name] = Path(cfg['output_dir'])/name
    members['Documentation_template.md'] = project/'Documentation_template.md'
    for directory in ('src', 'configs', 'scripts'):
        for p in sorted((project/directory).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc'):
                members[prefix+str(p.relative_to(project))] = p
    for name in ('README.md', 'pyproject.toml', 'requirements.txt', 'requirements-gpu.txt', 'uv.lock'):
        members[prefix+name] = project/name
    for p in policy_dir.rglob('*'):
        if p.is_file():
            members[prefix+'models/calibration/'+str(p.relative_to(policy_dir))] = p
    members[prefix+'models/xgboost.ubj'] = model_dir/'xgboost.ubj'
    effective = dict(cfg)
    effective.update(data_dir='data', output_dir='outputs', model_dir='models', policy_dir='models/calibration')
    fingerprints = model_fingerprints(cfg, model_dir/'xgboost.ubj')
    for kind in ('biencoder', 'crossencoder'):
        base = cfg[kind+'_base']
        base_dir = Path(base) if Path(base).is_dir() else Path(snapshot_download(base, revision=cfg[kind+'_revision'], local_files_only=True))
        for p in sorted(base_dir.rglob('*')):
            if p.is_file():
                members[prefix+f'models/base-{kind}/'+str(p.relative_to(base_dir))] = p
        checkpoint = Path(cfg[kind+'_checkpoint'])
        for p in sorted(checkpoint.rglob('*')):
            if p.is_file():
                members[prefix+f'models/{kind}/'+str(p.relative_to(checkpoint))] = p
        effective[kind+'_base'] = f'models/base-{kind}'
        effective[kind+'_checkpoint'] = f'models/{kind}'
        for name in ('metadata.json', 'adapter_config.json'):
            metadata = json.loads((checkpoint/name).read_text())
            if name=='metadata.json':
                metadata['base'] = effective[kind+'_base']
                metadata['tokenizer_base'] = effective[kind+'_base']
            else:
                metadata['base_model_name_or_path'] = effective[kind+'_base']
            encoded = json.dumps(metadata, indent=2).encode()
            overrides[prefix+f'models/{kind}/'+name] = encoded
            import hashlib
            fingerprints[kind][name] = hashlib.sha256(encoded).hexdigest()
    overrides[prefix+'models/calibration/model_fingerprints.json'] = json.dumps(fingerprints, indent=2).encode()
    overrides[prefix+'configs/submission.yaml'] = json.dumps(effective, indent=2).encode()
    with atomic_file(destination, binary=True) as stream:
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name,p in sorted(members.items()):
                if name not in overrides:
                    archive.write(p, name)
            for name,content in sorted(overrides.items()):
                archive.writestr(name, content)
    return {'archive': str(destination), 'members': len(set(members)|set(overrides))}
