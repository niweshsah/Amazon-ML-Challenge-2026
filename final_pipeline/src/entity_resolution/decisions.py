"""Batched boosted-tree scoring, disjoint calibration halves and macro F0.5 routing."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from .data import rank, supervised_pairs
from .features import FEATURE_NAMES, matrix
from .models import NeuralModel, score_cross
from .retrieval import batches
from .storage import atomic_file, write_json


def train_xgboost(cfg, db, path):
    if cfg['fixture']:
        write_json(path, {'fixture': True, 'features': FEATURE_NAMES})
        return
    import xgboost as xgb

    class PairIterator(xgb.DataIter):
        def __init__(self, split):
            self.split = split
            super().__init__(cache_prefix=str(path.parent / ('xgb-cache-' + split)), release_data=True)
        def reset(self):
            self.iterator = iter(batches(supervised_pairs(db, self.split), cfg['batch_size']*64))
        def next(self, input_data):
            rows = next(self.iterator, None)
            if rows is None:
                return False
            input_data(data=matrix(rows), label=np.asarray([r['label'] for r in rows]), feature_names=FEATURE_NAMES)
            return True

    # External-memory quantile matrices keep the complete feature table off RAM.
    fit_iter, dev_iter = PairIterator('fit'), PairIterator('dev')
    fit = xgb.ExtMemQuantileDMatrix(fit_iter, max_bin=256, nthread=cfg['xgboost_threads'])
    dev = xgb.ExtMemQuantileDMatrix(dev_iter, max_bin=256, ref=fit, nthread=cfg['xgboost_threads'])
    params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'tree_method': 'hist',
              'device': 'cpu', 'nthread': cfg['xgboost_threads'], 'max_depth': cfg['xgboost_depth'], 'seed': cfg['seed'], 'eta': .05}
    history = {}
    model = xgb.train(params, fit, num_boost_round=cfg['xgboost_trees'], evals=[(dev, 'dev')],
                      early_stopping_rounds=30, evals_result=history, verbose_eval=False)
    model = model[:model.best_iteration+1]
    with atomic_file(path, binary=True) as stream:
        stream.write(model.save_raw(raw_format='ubj'))
    write_json(path.parent / 'xgboost_training.json', {'history': history, 'feature_names': FEATURE_NAMES})


class TreeScorer:
    def __init__(self, cfg, path):
        self.fixture = cfg['fixture']
        if not path.exists():
            raise FileNotFoundError(f'Missing XGBoost model: {path}; run train-xgboost')
        if self.fixture:
            if not json.loads(path.read_text()).get('fixture'):
                raise ValueError('Fixture mode requires fixture models')
        else:
            import xgboost as xgb
            self.model = xgb.Booster()
            self.model.load_model(path)
            self.model.set_param({'nthread': cfg['xgboost_threads']})
            if self.model.feature_names != FEATURE_NAMES:
                raise ValueError('XGBoost feature schema mismatch')
    def score(self, rows):
        x = matrix(rows)
        if self.fixture:
            # Only exercises plumbing; no claim of learned quality.
            return np.clip(.1*x[:, 0] + .5*x[:, 3] + .35*x[:, 9] + .05*x[:, 14], 0, 1)
        import xgboost as xgb
        return self.model.predict(xgb.DMatrix(x, feature_names=FEATURE_NAMES))


def route(tree_scores, cross_scores, policy):
    low, high, acceptance = policy['lower'], policy['upper'], policy['cross_acceptance']
    tree_scores = np.asarray(tree_scores)
    routed = (tree_scores >= low) & (tree_scores <= high)
    accepted = tree_scores > high
    if np.any(routed):
        if cross_scores is None:
            raise FileNotFoundError('Routed pairs require a compatible trained cross-encoder and calibration assets')
        accepted |= routed & (np.asarray(cross_scores) >= acceptance)
    return accepted, routed


def entity_metrics(truth, prediction, candidates):
    tp = len(truth & prediction)
    f = float(not prediction) if not truth else (1.25*tp/(.25*len(truth)+len(prediction)) if tp else 0.)
    return {'f': f, 'tp': tp, 'predicted': len(prediction), 'positives': len(truth),
            'candidate_hits': len(truth & candidates)}


def aggregate(items):
    totals = {k: sum(i[k] for i in items) for k in ('tp', 'predicted', 'positives', 'candidate_hits')}
    return {'entities': len(items), 'macro_f05': sum(i['f'] for i in items)/max(1, len(items)),
            'precision': totals['tp']/max(1, totals['predicted']),
            'recall': totals['tp']/max(1, totals['positives']),
            'candidate_pair_recall': totals['candidate_hits']/max(1, totals['positives']), **totals}


def calibration_groups(db, seed):
    ids = [r[0] for r in db.execute("SELECT id FROM records WHERE split='calibration' AND source=1")]
    ids.sort(key=lambda x: rank(x, seed, 'calibration-half'))
    if len(ids) < 2:
        raise ValueError('Need at least two calibration entities; increase --train-percent')
    return set(ids[:len(ids)//2]), set(ids[len(ids)//2:])


def calibrate(cfg, db, tree_path, destination):
    probability_ids, policy_ids = calibration_groups(db, cfg['seed'])
    tree = TreeScorer(cfg, tree_path)
    cross = None if cfg['fixture'] else NeuralModel(cfg, 'crossencoder')
    db.executescript('DROP TABLE IF EXISTS calibration_scores; CREATE TABLE calibration_scores(qid TEXT,tid TEXT,x REAL,c REAL,label INTEGER, PRIMARY KEY(qid,tid));')
    for rows in batches(supervised_pairs(db, 'calibration'), cfg['batch_size']):
        xs, cs = tree.score(rows), score_cross(cfg, rows, cross)
        db.executemany('INSERT INTO calibration_scores VALUES (?,?,?,?,?)',
                       [(r['qid'], r['tid'], float(x), float(c), r['label']) for r, x, c in zip(rows, xs, cs)])
    db.commit()
    rows = [r for r in db.execute('SELECT * FROM calibration_scores') if r['qid'] in probability_ids]
    if not rows or len({r['label'] for r in rows}) < 2:
        raise ValueError('Probability calibration needs positive and negative candidate pairs; increase sample/retrieval limits')
    y = np.array([r['label'] for r in rows])
    calibrators = {key: IsotonicRegression(out_of_bounds='clip').fit([r[key] for r in rows], y) for key in ('x', 'c')}
    destination.mkdir(parents=True, exist_ok=True)
    with atomic_file(destination / 'calibrators.joblib', binary=True) as stream:
        joblib.dump(calibrators, stream)
    grid = np.linspace(0, 1, cfg['threshold_grid_size'])
    grid = np.unique(np.r_[-1e-6, grid, 1+1e-6])
    policies = np.asarray([(lo, hi, c) for lo in grid for hi in grid if lo < hi for c in grid])
    sums, tp_sum, pred_sum, routed_sum = (np.zeros(len(policies)) for _ in range(4))
    for sid in sorted(policy_ids):
        truth = {r[0] for r in db.execute('SELECT tid FROM truth WHERE qid=?', (sid,))}
        rows = list(db.execute('SELECT * FROM calibration_scores WHERE qid=? ORDER BY tid', (sid,)))
        x = calibrators['x'].predict([r['x'] for r in rows]) if rows else np.array([])
        c = calibrators['c'].predict([r['c'] for r in rows]) if rows else np.array([])
        labels = np.array([r['label'] for r in rows])
        # Bounded by one entity's candidates and the threshold grid, never all pairs.
        routed = (x[None, :] >= policies[:, 0, None]) & (x[None, :] <= policies[:, 1, None])
        accepted = (x[None, :] > policies[:, 1, None]) | (routed & (c[None, :] >= policies[:, 2, None]))
        tp = (accepted*labels).sum(1)
        predicted = accepted.sum(1)
        sums += (predicted == 0) if not truth else np.divide(1.25*tp, .25*len(truth)+predicted)
        tp_sum += tp; pred_sum += predicted; routed_sum += routed.sum(1)
    precision = tp_sum/np.maximum(1, pred_sum)
    best = max(range(len(policies)), key=lambda i: (sums[i], precision[i], -routed_sum[i], -i))
    lo, hi, c = policies[best]
    policy = {'lower': float(lo), 'upper': float(hi), 'cross_acceptance': float(c),
              'macro_f05': sums[best]/len(policy_ids), 'precision': float(precision[best]),
              'routed_pairs': int(routed_sum[best]), 'probability_entities': sorted(probability_ids),
              'policy_entities': sorted(policy_ids), 'scope': 'sampled training target pools',
              'fixture': cfg['fixture']}
    write_json(destination / 'policy.json', policy)
    return policy


def predict_pairs(cfg, db, tree_path, policy_dir):
    policy = json.loads((policy_dir / 'policy.json').read_text())
    if policy['fixture'] != cfg['fixture']:
        raise ValueError('Fixture and trained inference artifacts cannot be mixed')
    calibrators = joblib.load(policy_dir / 'calibrators.joblib')
    tree = TreeScorer(cfg, tree_path)
    cross = None
    db.executescript('DROP TABLE IF EXISTS decisions; CREATE TABLE decisions(qid TEXT,tid TEXT,x REAL,c REAL,routed INTEGER,accepted INTEGER,PRIMARY KEY(qid,tid));')
    cursor = db.execute('''SELECT p.*,q.raw query_text,t.raw target_text FROM pairs p
      JOIN records q ON q.id=p.qid JOIN records t ON t.id=p.tid ORDER BY p.qid,p.tid''')
    for rows in batches(cursor, cfg['batch_size']):
        xs = calibrators['x'].predict(tree.score(rows))
        routed = (xs >= policy['lower']) & (xs <= policy['upper'])
        cs = np.zeros(len(rows))
        if routed.any():
            if not cfg['fixture'] and cross is None:
                cross = NeuralModel(cfg, 'crossencoder')
            selected = [r for r, flag in zip(rows, routed) if flag]
            cs[routed] = calibrators['c'].predict(score_cross(cfg, selected, cross))
        accepted, routed = route(xs, cs, policy)
        db.executemany('INSERT INTO decisions VALUES (?,?,?,?,?,?)',
                       [(r['qid'], r['tid'], float(x), float(c) if flag else None, int(flag), int(a))
                        for r, x, c, flag, a in zip(rows, xs, cs, routed, accepted)])
    db.commit()


def evaluate(db, split='audit'):
    groups = defaultdict(list)
    source_items = defaultdict(list)
    for query in db.execute('SELECT * FROM records WHERE source=1 AND split=? ORDER BY position', (split,)):
        sid = query['id']
        truth = {r[0] for r in db.execute('SELECT tid FROM truth WHERE qid=?', (sid,))}
        pairs = list(db.execute('SELECT p.tid,p.source,d.accepted,d.routed FROM pairs p JOIN decisions d USING(qid,tid) WHERE p.qid=?', (sid,)))
        pred = {r['tid'] for r in pairs if r['accepted']}
        cand = {r['tid'] for r in pairs}
        item = entity_metrics(truth, pred, cand)
        item['routed'] = sum(r['routed'] for r in pairs)
        item['pairs'] = len(pairs)
        payload = json.loads(query['payload'])
        tags = ['all', 'country:'+payload['country']]
        if not truth:
            tags.append('singleton')
        if any(not payload[f] for f in ('business_name', 'business_address', 'country')):
            tags.append('missing_field')
        if any(ord(c) > 127 for c in query['raw']):
            tags.append('non_ascii')
        for tag in tags:
            groups[tag].append(item)
        for source in (2, 3):
            tids = {r[0] for r in db.execute('SELECT id FROM records WHERE source=? AND id IN (SELECT tid FROM truth WHERE qid=?)', (source, sid))}
            source_items[source].append(entity_metrics(tids, {r['tid'] for r in pairs if r['source']==source and r['accepted']}, {r['tid'] for r in pairs if r['source']==source}))
    result = aggregate(groups['all'])
    result.update({'scope': 'sampled training target pools', 'split': split,
                   'routed_pairs': sum(i['routed'] for i in groups['all']),
                   'candidate_pairs': sum(i['pairs'] for i in groups['all']),
                   'subgroups': {k: aggregate(v) for k, v in groups.items() if k != 'all'},
                   'target_sources': {str(k): aggregate(v) for k,v in source_items.items()}})
    return result
