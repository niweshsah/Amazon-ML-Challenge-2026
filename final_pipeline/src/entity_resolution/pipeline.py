"""One CLI's transactional stage orchestration."""
from __future__ import annotations
import json
import os
import resource
import shutil
import time
from pathlib import Path
from . import data, decisions, features, retrieval, submission
from .models import train_neural
from .storage import Cache, connect, digest, fingerprint, model_fingerprints, write_json


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg['output_dir'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.models = Path(cfg['model_dir'])
        self.models.mkdir(parents=True, exist_ok=True)
        self.policy = Path(cfg['policy_dir'])
        self.cache = Cache(self.root / 'state')
        self.code = fingerprint(Path(__file__).parent)
        self.records = self.root / 'records.sqlite'
        self.pairs = self.root / 'pairs.sqlite'
        self.feature_db = self.root / 'features.sqlite'
        self.prediction_db = self.root / 'predictions.sqlite'
        self.tree = self.models / 'xgboost.ubj'
        write_json(self.root / 'resolved_config.json', cfg)

    def key(self, stage, deps):
        # Ignore interpreter caches in code provenance.
        code = {k:v for k,v in self.code.items() if '__pycache__' not in k and not k.endswith('.pyc')}
        assets = {}
        if stage in ('embed', 'calibrate', 'predict', 'train-biencoder', 'train-crossencoder') and not self.cfg['fixture']:
            kinds = ('biencoder',) if stage in ('embed', 'train-biencoder') else (('crossencoder',) if stage=='train-crossencoder' else ('biencoder', 'crossencoder'))
            for kind in kinds:
                from huggingface_hub import snapshot_download
                base = Path(self.cfg[kind+'_base'])
                if not base.is_dir():
                    base = Path(snapshot_download(self.cfg[kind+'_base'], revision=self.cfg[kind+'_revision'], local_files_only=self.cfg['local_files_only']))
                assets[kind] = fingerprint(base)
        return digest({'assets': assets, 'stage': stage, 'config': self.cfg, 'code': code, 'dependencies': [fingerprint(p) for p in deps]})

    def run_stage(self, stage, deps, outputs, action):
        for path in deps:
            if not path.exists():
                raise FileNotFoundError(f'{stage} requires {path}; run its prerequisite stage')
        key = self.key(stage, deps)
        if self.cache.valid(stage, key, outputs):
            print(json.dumps({'stage': stage, 'cache': 'verified'}), flush=True)
            return
        started = time.perf_counter()
        action(key)
        self.cache.commit(stage, key, outputs)
        elapsed = time.perf_counter()-started
        write_json(self.root / 'state' / (stage + '.runtime.json'), {'stage': stage, 'seconds': elapsed,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if os.sys.platform=='darwin' else 1024)})
        print(json.dumps({'stage': stage, 'seconds': round(elapsed, 3), 'cache': 'rebuilt'}), flush=True)

    def db(self, path):
        db = connect(path)
        if path != self.records:
            db.execute('ATTACH DATABASE ? AS prepared', (str(self.records),))
        return db

    def prepare(self):
        deps = [Path(self.cfg['data_dir']) / f'{self.cfg["mode"]}_source{i}.tsv' for i in (1, 2, 3)]
        if self.cfg['mode'] == 'train':
            deps.append(Path(self.cfg['data_dir']) / 'train_ground_truth.tsv')
        self.run_stage('prepare', deps, [self.records, self.root/'splits.json'], lambda _: data.prepare(self.cfg, self.records))
        write_json(self.root/'input_fingerprints.json', {p.name: fingerprint(p) for p in deps})

    def train_biencoder(self):
        destination = self.models / 'biencoder'
        def action(_):
            cfg = dict(self.cfg)
            cfg['biencoder_checkpoint'] = None
            tmp = self.models / 'biencoder.training'
            if tmp.exists():
                shutil.rmtree(tmp)
            working = self.root / 'biencoder-training.sqlite'
            shutil.copyfile(self.records, working)
            with self.db(working) as db:
                train_neural(cfg, 'biencoder', db, tmp)
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(tmp, destination)
        self.run_stage('train-biencoder', [self.records], [destination], action)

    def embed(self):
        root = self.root / 'embeddings'
        deps = [self.records]
        if not self.cfg['fixture']:
            deps.append(Path(self.cfg['biencoder_checkpoint']))
        def action(key):
            with self.db(self.records) as db:
                retrieval.embed(self.cfg, db, root, key)
        # Verify all shards, not just the catalog, to detect truncated/corrupt caches.
        self.run_stage('embed', deps, [root], action)

    def retrieve(self):
        def action(key):
            with self.db(self.records) as db:
                retrieval.retrieve(self.cfg, db, self.root/'embeddings', self.pairs, key)
        self.run_stage('retrieve', [self.records, self.root/'embeddings'], [self.pairs], action)

    def features(self):
        def action(_):
            tmp = self.feature_db.with_suffix('.tmp.sqlite')
            shutil.copyfile(self.pairs, tmp)
            with self.db(tmp) as db:
                features.extract(db, self.cfg['batch_size']*16)
            os.replace(tmp, self.feature_db)
        self.run_stage('features', [self.records, self.pairs], [self.feature_db], action)

    def train_xgboost(self):
        def action(_):
            with self.db(self.feature_db) as db:
                decisions.train_xgboost(self.cfg, db, self.tree)
        self.run_stage('train-xgboost', [self.records, self.feature_db], [self.tree], action)

    def train_crossencoder(self):
        destination = self.models / 'crossencoder'
        def action(_):
            cfg = dict(self.cfg)
            cfg['crossencoder_checkpoint'] = None
            tmp = self.models / 'crossencoder.training'
            if tmp.exists():
                shutil.rmtree(tmp)
            working = self.root / 'crossencoder-training.sqlite'
            shutil.copyfile(self.feature_db, working)
            with self.db(working) as db:
                train_neural(cfg, 'crossencoder', db, tmp)
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(tmp, destination)
        self.run_stage('train-crossencoder', [self.records, self.feature_db], [destination], action)

    def calibrate(self):
        def action(_):
            # Calibration scores go into a disposable working database; never mutate a committed feature stage.
            path = self.root / 'calibration.sqlite'
            shutil.copyfile(self.feature_db, path)
            with self.db(path) as db:
                decisions.calibrate(self.cfg, db, self.tree, self.policy)
        deps = [self.records, self.feature_db, self.tree]
        if not self.cfg['fixture']:
            deps.append(Path(self.cfg['crossencoder_checkpoint']))
        self.run_stage('calibrate', deps, [self.policy/'policy.json', self.policy/'calibrators.joblib'], action)
        write_json(self.policy/'model_fingerprints.json', model_fingerprints(self.cfg, self.tree))

    def predict(self):
        def action(_):
            provenance = json.loads((self.policy/'model_fingerprints.json').read_text())
            expected = model_fingerprints(self.cfg, self.tree)
            if provenance != expected:
                raise ValueError('Model fingerprints differ from calibrated policy; rerun calibration')
            tmp = self.prediction_db.with_suffix('.tmp.sqlite')
            shutil.copyfile(self.feature_db, tmp)
            with self.db(tmp) as db:
                decisions.predict_pairs(self.cfg, db, self.tree, self.policy)
                submission.write_outputs(db, self.root)
            os.replace(tmp, self.prediction_db)
        deps = [self.records, self.feature_db, self.tree, self.policy]
        for kind in ('biencoder', 'crossencoder'):
            path = Path(self.cfg[kind+'_checkpoint'])
            if path.exists():
                deps.append(path)
        self.run_stage('predict', deps, [self.prediction_db, self.root/'candidate_pairs.tsv', self.root/'matching_results.tsv'], action)

    def evaluate(self):
        if self.cfg['mode'] != 'train':
            raise ValueError('Test has no labels; evaluate requires --mode train')
        with self.db(self.prediction_db) as db:
            result = decisions.evaluate(db)
        result['fixture'] = self.cfg['fixture']
        result['runtime'] = {p.stem: json.loads(p.read_text()) for p in (self.root/'state').glob('*.runtime.json')}
        write_json(self.root/'evaluation.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    def validate(self):
        if self.cfg['mode'] != 'test':
            # Sampled training validation checks against selected records, not full source coverage.
            with self.db(self.prediction_db) as db:
                for r in db.execute('SELECT qid,tid FROM decisions WHERE accepted=1 EXCEPT SELECT qid,tid FROM pairs'):
                    raise ValueError(f'Match outside candidates: {tuple(r)}')
            print(json.dumps({'status': 'PASS', 'scope': 'sampled training outputs'}))
            return
        print(json.dumps(submission.validate(self.cfg['data_dir'], self.root)))

    def all(self):
        self.prepare()
        if self.cfg['mode'] == 'train' and (Path(self.cfg['biencoder_checkpoint']) == self.models/'biencoder' ):
            self.train_biencoder()
        self.embed(); self.retrieve(); self.features()
        if self.cfg['mode'] == 'train':
            self.train_xgboost()
            if Path(self.cfg['crossencoder_checkpoint']) == self.models/'crossencoder' :
                self.train_crossencoder()
            self.calibrate()
        self.predict()
        if self.cfg['mode'] == 'train':
            self.evaluate()
        self.validate()
