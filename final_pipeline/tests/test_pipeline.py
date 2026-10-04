import csv
import importlib.util
import json
import sqlite3
from pathlib import Path
import numpy as np
import pytest
import yaml
from entity_resolution import data, decisions, features, models, retrieval, submission
from entity_resolution.cli import PROJECT, main
from entity_resolution.pipeline import Pipeline
from entity_resolution.storage import Cache, fingerprint, sha

spec = importlib.util.spec_from_file_location('fixture', PROJECT/'scripts/make_fixture.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def config(tmp_path, **overrides):
    fixture.make(tmp_path/'data')
    cfg = yaml.safe_load((PROJECT/'configs/default.yaml').read_text())
    cfg.update(data_dir=str(tmp_path/'data'), output_dir=str(tmp_path/'output'), fixture=True,
               train_percent=100., faiss_threads=1, top_k_dense=8, top_k_tfidf=8, threshold_grid_size=5, shard_rows=23, target_batch_size=31)
    cfg.update(overrides)
    cfg['model_dir'] = str(tmp_path/'output/models')
    cfg['policy_dir'] = str(tmp_path/'output/models/calibration')
    for kind in ('biencoder', 'crossencoder'):
        cfg[kind+'_checkpoint'] = str(tmp_path/f'output/models/{kind}')
    return cfg


def test_sampling_reproducibility():
    ids = [f'S1-{i}' for i in range(1000)]
    chosen = data.select(lambda: iter(ids), 1., 42, 'sample')
    assert len(chosen) == 10
    assert chosen == data.select(lambda: iter(reversed(ids)), 1., 42, 'sample')
    assert chosen != data.select(lambda: iter(ids), 1., 43, 'sample')
    assert len(data.select(lambda: iter(ids), .01, 42, 'sample')) == 1
    for p in (0, -1, 101):
        with pytest.raises(ValueError):
            data.select(lambda: iter(ids), p, 42, 'sample')


def test_unicode_and_features():
    row = dict(entity_id='S1-1', business_name='श्री गणेश', business_address='Étoile 42', country='NeverSeen')
    raw, norm, latin = data.views(row)
    assert row['business_name'] in raw and row['business_name'] in norm
    assert 'etoile' in latin and latin != norm
    result = features.pair_features({'dense_score': .5, 'tfidf_score': .7}, row, row)
    assert len(result) == len(features.FEATURE_NAMES)
    assert result[14] == 1  # country equality, unrestricted labels
    empty = dict(row, business_address='')
    vector = features.pair_features({'dense_score': 0, 'tfidf_score': 0}, empty, empty)
    assert vector[8] == 0 and vector[-1] == 0
    assert not any(x in features.FEATURE_NAMES for x in ('entity_id', 'label', 'split', 'pool_role'))
    np.testing.assert_allclose(np.linalg.norm(models.fixture_vectors([raw]), axis=1), 1)


def test_positive_preservation_and_isolation(tmp_path):
    cfg = config(tmp_path, train_percent=50., target_distractor_percent=2.)
    p = Pipeline(cfg)
    p.prepare()
    with p.db(p.records) as db:
        assert db.execute('SELECT COUNT(*) FROM records WHERE source=1').fetchone()[0] == 50
        assert db.execute('SELECT COUNT(*) FROM truth g LEFT JOIN records t ON t.id=g.tid WHERE t.id IS NULL').fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM records q WHERE q.source=1 AND NOT EXISTS(SELECT 1 FROM truth WHERE qid=q.id)").fetchone()[0] > 0
        counts = dict(db.execute('SELECT split,COUNT(*) FROM records WHERE source=1 GROUP BY split').fetchall())
        assert counts == {'fit': 35, 'dev': 5, 'calibration': 5, 'audit': 5}
        # A held-out true target deliberately appears in a fitting candidate set.
        fit = db.execute("SELECT id FROM records WHERE split='fit' LIMIT 1").fetchone()[0]
        held = db.execute("SELECT g.tid FROM truth g JOIN records q ON q.id=g.qid WHERE q.split='audit' LIMIT 1").fetchone()[0]
        db.executescript('CREATE TABLE pairs(qid TEXT,tid TEXT,features TEXT,dense_score REAL,tfidf_score REAL);')
        db.execute('INSERT INTO pairs(qid,tid,features) VALUES (?,?,?)', (fit, held, '[]'))
        assert list(data.supervised_pairs(db, 'fit')) == []
    groups = data.split_groups([f'S1-{i}' for i in range(100)], 42)
    assert groups == data.split_groups(reversed(list(groups)), 42)


def test_routing_paths_and_boundaries():
    policy = {'lower': .2, 'upper': .8, 'cross_acceptance': .7}
    accept, routed = decisions.route([.1, .9, .5, .5, .2, .8], [1, 0, .75, .6, .7, .7], policy)
    assert accept.tolist() == [False, True, True, False, True, True]
    assert routed.tolist() == [False, False, True, True, True, True]
    with pytest.raises(FileNotFoundError):
        decisions.route([.5], None, policy)
    assert decisions.route([.1, .9], None, policy)[0].tolist() == [False, True]
    assert decisions.entity_metrics(set(), set(), set())['f'] == 1
    assert decisions.entity_metrics(set(), {'a'}, {'a'})['f'] == 0
    assert decisions.entity_metrics({'a', 'b'}, {'a'}, {'a'})['f'] == pytest.approx(1.25/1.5)
    assert decisions.entity_metrics({'a'}, {'a', 'b'}, {'a','b'})['f'] == pytest.approx(1.25/2.25)


def test_stale_cache_and_atomic_failure(tmp_path):
    from entity_resolution.storage import atomic_file
    p = tmp_path/'artifact.json'
    p.write_text('original')
    cache = Cache(tmp_path/'state')
    cache.commit('stage', 'key', [p])
    assert cache.valid('stage', 'key', [p])
    assert not cache.valid('stage', 'changed', [p])
    with pytest.raises(RuntimeError):
        with atomic_file(p) as stream:
            stream.write('broken')
            raise RuntimeError('interrupted')
    assert p.read_text() == 'original'
    assert not p.with_name(p.name+'.tmp').exists()
    p.write_text('corrupted')
    assert not cache.valid('stage', 'key', [p])


def test_end_to_end_resume_and_complete_test(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    pipeline = Pipeline(cfg)
    pipeline.all()
    policy = json.loads((pipeline.policy/'policy.json').read_text())
    assert not set(policy['probability_entities']) & set(policy['policy_entities'])
    evaluation = json.loads((pipeline.root/'evaluation.json').read_text())
    assert evaluation['fixture'] is True
    assert evaluation['entities'] == 10
    assert 'singleton' in evaluation['subgroups']
    before = fingerprint(pipeline.root/'embeddings')
    pipeline.all()
    assert before == fingerprint(pipeline.root/'embeddings')
    # Incomplete embedding shard is rebuilt, unaffected shards are reused.
    shard = sorted((pipeline.root/'embeddings').glob('*.npy'))[0]
    good = sha(shard)
    shard.write_bytes(b'interrupted')
    pipeline.embed()
    assert sha(shard) == good
    # Full test prediction is independent of the training sampling percentage.
    test_cfg = dict(cfg, mode='test', train_percent=.01, output_dir=str(tmp_path/'test-output'))
    test_pipeline = Pipeline(test_cfg)
    test_pipeline.all()
    result = submission.validate(test_cfg['data_dir'], test_cfg['output_dir'])
    assert result['source1_rows'] == 100
    matches = list(data.read_tsv(test_pipeline.root/'matching_results.tsv'))
    candidates = list(data.read_tsv(test_pipeline.root/'candidate_pairs.tsv'))
    assert any(not r['matched_entity_ids'] for r in matches)
    assert any(',' in r['matched_entity_ids'] for r in matches)
    for m,c in zip(matches,candidates):
        assert set(filter(None,m['matched_entity_ids'].split(','))) <= set(c['candidate_entity_ids'].split(','))
    with test_pipeline.db(test_pipeline.prediction_db) as db:
        for row in candidates:
            actual = {r[0] for r in db.execute('SELECT tid FROM pairs WHERE qid=?', (row['source1_entity_id'],))}
            assert actual == set(filter(None,row['candidate_entity_ids'].split(',')))
    # Trained inference cannot silently run with fixture artifacts or missing models.
    (pipeline.models/'crossencoder/fixture.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='fingerprints'):
        test_pipeline.predict()
    assert not test_pipeline.cache.valid('predict', 'wrong-key', [test_pipeline.prediction_db])


def test_validator_rejects_bad_submissions(tmp_path):
    fixture.make(tmp_path/'data', n=2)
    output = tmp_path/'output'
    output.mkdir()
    (output/'candidate_pairs.tsv').write_text('source1_entity_id\tcandidate_entity_ids\nS1-0000\t\nS1-0001\tS2-0001\n')
    path = output/'matching_results.tsv'
    path.write_text('source1_entity_id\tmatched_entity_ids\nS1-0000\t\nS1-0001\tS2-0001\n')
    assert submission.validate(tmp_path/'data', output)['status'] == 'PASS'
    path.write_text('source1_entity_id\tmatched_entity_ids\nS1-0000\t\nS1-0001\tS3-0001\n')
    with pytest.raises(ValueError, match='outside'):
        submission.validate(tmp_path/'data', output)
    path.write_text('source1_entity_id\tmatched_entity_ids\nS1-0000\t\nS1-0001\tS2-0001,S2-0001\n')
    with pytest.raises(ValueError, match='Duplicate'):
        submission.validate(tmp_path/'data', output)


def test_cli_rejects_bad_percentage():
    with pytest.raises(SystemExit):
        main(['prepare','--train-percent','0'])
