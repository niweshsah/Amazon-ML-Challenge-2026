"""Check real neural training/loading on CUDA using synthetic matching examples."""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from entity_resolution.cli import PROJECT
from entity_resolution.models import NeuralModel, train_neural
from entity_resolution.pipeline import Pipeline
from make_fixture import make

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path, default=Path('gpu-smoke'))
parser.add_argument('--download', action='store_true')
args = parser.parse_args()
make(args.output_dir/'data', n=40)
cfg = yaml.safe_load((PROJECT/'configs/default.yaml').read_text())
cfg.update(data_dir=str(args.output_dir/'data'), output_dir=str(args.output_dir/'run'), train_percent=100,
           batch_size=2, epochs=1, hard_negatives=2, max_length=64, top_k_dense=8, top_k_tfidf=8,
           threshold_grid_size=5, local_files_only=not args.download, device='cuda',
           model_dir=str(args.output_dir/'models'), policy_dir=str(args.output_dir/'models/calibration'))
for kind in ('biencoder','crossencoder'):
    cfg[kind+'_checkpoint'] = str(args.output_dir/'models'/kind)
p = Pipeline(cfg)
p.all()
encoder = NeuralModel(cfg, 'biencoder')
vectors = encoder.embed(['Name: श्री गणेश | Country: India', 'Name: Étoile | Country: France'])
assert np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-5)
assert NeuralModel(cfg, 'crossencoder').score(['Name: Étoile'], ['Name: Étoile']).shape == (1,)
print(json.dumps({'status': 'PASS', 'scope': 'synthetic CUDA smoke; not challenge quality'}))
