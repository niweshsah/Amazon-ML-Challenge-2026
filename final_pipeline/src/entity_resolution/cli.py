from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from .pipeline import Pipeline
from .submission import package

STAGES = ('prepare', 'train-biencoder', 'embed', 'retrieve', 'features', 'train-xgboost',
          'train-crossencoder', 'calibrate', 'evaluate', 'predict', 'validate', 'all', 'package')
PROJECT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Grouped multilingual entity-resolution pipeline')
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/default.yaml')
    for option in ('data-dir', 'output-dir', 'model-dir', 'policy-dir', 'biencoder-checkpoint', 'crossencoder-checkpoint'):
        parser.add_argument('--'+option)
    parser.add_argument('--mode', choices=('train', 'test'))
    parser.add_argument('--train-percent', type=float)
    parser.add_argument('--target-distractor-percent', type=float)
    parser.add_argument('--fixture', action='store_true', default=None)
    parser.add_argument('--set', action='append', default=[], metavar='KEY=YAML_VALUE')
    parser.add_argument('--archive', type=Path, default=Path('team_submission.zip'))
    args = parser.parse_args(argv)
    cfg = yaml.safe_load((PROJECT/'configs/default.yaml').read_text())
    cfg.update(yaml.safe_load(args.config.read_text()))
    for key in ('data_dir', 'output_dir', 'model_dir', 'policy_dir', 'biencoder_checkpoint', 'crossencoder_checkpoint',
                'mode', 'train_percent', 'target_distractor_percent', 'fixture'):
        if getattr(args, key, None) is not None:
            cfg[key] = getattr(args, key)
    for override in args.set:
        key, separator, value = override.partition('=')
        if not separator or key not in cfg:
            parser.error(f'Unknown configuration override: {override}')
        cfg[key] = yaml.safe_load(value)
    for key in ('train_percent', 'target_distractor_percent'):
        if cfg[key] is not None and not 0 < cfg[key] <= 100:
            parser.error(f'{key} must be greater than zero and at most 100')
    for key in ('batch_size', 'shard_rows', 'query_batch_size', 'target_batch_size', 'top_k_dense', 'top_k_tfidf',
                'exact_limit', 'ivf_nlist', 'pq_m', 'nprobe', 'index_train_rows', 'epochs', 'threshold_grid_size'):
        if cfg[key] < 1:
            parser.error(f'{key} must be positive')
    cfg['model_dir'] = str(Path(cfg.get('model_dir') or Path(cfg['output_dir'])/'models').resolve())
    cfg['policy_dir'] = str(Path(cfg.get('policy_dir') or Path(cfg['model_dir'])/'calibration').resolve())
    for kind in ('biencoder', 'crossencoder'):
        cfg[kind+'_checkpoint'] = str(Path(cfg[kind+'_checkpoint'] or Path(cfg['model_dir'])/kind).resolve())
    cfg['data_dir'] = str(Path(cfg['data_dir']).resolve())
    cfg['output_dir'] = str(Path(cfg['output_dir']).resolve())
    if cfg['mode'] != 'train' and args.stage.startswith('train-'):
        parser.error('Training requires --mode train')
    pipeline = Pipeline(cfg)
    if args.stage == 'package':
        print(json.dumps(package(cfg, PROJECT, args.archive, pipeline.models, pipeline.policy)))
    else:
        getattr(pipeline, args.stage.replace('-', '_'))()


if __name__ == '__main__':
    main()
