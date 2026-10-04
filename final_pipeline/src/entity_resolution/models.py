"""Compatible CLS LoRA encoder and supervised multilingual reranker."""
from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np
from .data import SERIALIZATION, normalize
from .storage import digest, write_json


def fixture_vectors(texts, dimension=64):
    out = np.zeros((len(texts), dimension), dtype=np.float32)
    for i, text in enumerate(texts):
        text = normalize(text)
        for j in range(max(1, len(text)-2)):
            key = int(digest(text[j:j+3])[:16], 16)
            out[i, key % dimension] += 1 if key & 256 else -1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class NeuralModel:
    def __init__(self, cfg, kind, training=False):
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        from peft import LoraConfig, PeftModel, get_peft_model
        self.cfg, self.kind, self.torch = cfg, kind, torch
        self.device = cfg['device']
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Use the documented GPU environment or --fixture for tests.')
        checkpoint = cfg[kind + '_checkpoint']
        base = cfg[kind + '_base']
        revision = cfg[kind + '_revision']
        kwargs = {'local_files_only': cfg['local_files_only'], 'revision': revision}
        if checkpoint:
            path = Path(checkpoint)
            if not path.is_dir() or not (path / 'metadata.json').exists():
                raise FileNotFoundError(f'Missing compatible {kind} checkpoint: {path}; run train-{kind}')
            metadata = json.loads((path / 'metadata.json').read_text())
            adapter_config = json.loads((path / 'adapter_config.json').read_text())
            if adapter_config['base_model_name_or_path'] != base:
                raise ValueError('LoRA adapter base_model_name_or_path differs from configured base')
            expected = self.metadata(cfg, kind)
            if any(metadata.get(k) != v for k, v in expected.items()):
                raise ValueError(f'Incompatible {kind} base, revision, serialization, pooling or tokenizer metadata')
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        else:
            if not training:
                raise FileNotFoundError(f'{kind} checkpoint required; run train-{kind} or configure its checkpoint')
            self.tokenizer = AutoTokenizer.from_pretrained(base, **kwargs)
        loader = AutoModel if kind == 'biencoder' else AutoModelForSequenceClassification
        self.model = loader.from_pretrained(base, **kwargs, torch_dtype=torch.bfloat16 if self.device.startswith('cuda') else torch.float32, **({'num_labels': 1} if kind == 'crossencoder' else {}))
        if checkpoint:
            self.model = PeftModel.from_pretrained(self.model, checkpoint, is_trainable=training)
        elif training:
            self.model = get_peft_model(self.model, LoraConfig(
                r=cfg['lora_rank'], lora_alpha=2*cfg['lora_rank'], lora_dropout=.05,
                target_modules=['query', 'value'], bias='none',
                task_type='FEATURE_EXTRACTION' if kind == 'biencoder' else 'SEQ_CLS'))
        if len(self.tokenizer) != self.model.get_input_embeddings().num_embeddings:
            raise ValueError('Tokenizer vocabulary and base embedding table do not match')
        if training:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        self.model.to(self.device)
        self.model.train(training)

    @staticmethod
    def metadata(cfg, kind):
        return {'kind': kind, 'base': cfg[kind+'_base'], 'revision': cfg[kind+'_revision'],
                'serialization': SERIALIZATION, 'pooling': 'normalized_cls' if kind == 'biencoder' else 'binary_logit',
                'tokenizer_base': cfg[kind+'_base'], 'max_length': cfg['max_length']}

    def tokens(self, texts, targets=None):
        return self.tokenizer(texts, text_pair=targets, padding=True, truncation=True,
                              max_length=self.cfg['max_length'], return_tensors='pt').to(self.device)

    def vectors(self, texts):
        hidden = self.model(**self.tokens(texts)).last_hidden_state[:, 0].float()
        return self.torch.nn.functional.normalize(hidden, dim=1)

    def logits(self, texts, targets):
        return self.model(**self.tokens(texts, targets)).logits.reshape(-1).float()

    def embed(self, texts):
        self.model.eval()
        with self.torch.no_grad():
            return self.vectors(texts).cpu().numpy()

    def score(self, texts, targets):
        self.model.eval()
        with self.torch.no_grad():
            return self.torch.sigmoid(self.logits(texts, targets)).cpu().numpy()

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        write_json(path / 'metadata.json', self.metadata(self.cfg, self.kind))


def score_cross(cfg, rows, model=None):
    if cfg['fixture']:
        from difflib import SequenceMatcher
        return np.array([SequenceMatcher(None, normalize(r['query_text']), normalize(r['target_text'])).ratio() for r in rows])
    model = model or NeuralModel(cfg, 'crossencoder')
    return model.score([r['query_text'] for r in rows], [r['target_text'] for r in rows])


def train_neural(cfg, kind, db, destination):
    """Train with binary loss; positives are retained even when retrieval missed them.

    Explicit hard negatives avoid false-negative in-batch assumptions for multi-match entities.
    """
    if cfg['fixture']:
        write_json(destination / 'fixture.json', {'kind': kind, 'deterministic_substitute': True})
        return
    import torch
    from .data import supervised_pairs
    from .storage import atomic_file
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    if kind == 'biencoder':
        from .retrieval import SparseIndex
        from .storage import fingerprint
        db.executescript('DROP TABLE IF EXISTS bi_negatives; CREATE TABLE bi_negatives(qid TEXT,tid TEXT,PRIMARY KEY(qid,tid));')
        for source in (2, 3):
            key = digest([cfg, source, [tuple(r) for r in db.execute('SELECT id FROM records ORDER BY id')]])
            lex = SparseIndex(cfg, db, source, Path(cfg['output_dir']) / 'training-blocks' / str(source), key)
            queries = db.execute("SELECT * FROM records WHERE source=1 AND split IN ('fit','dev') ORDER BY id")
            for q in queries:
                ranked = lex.search([q['normalized']+' '+q['latin']], max(50, cfg['hard_negatives']*10))[0]
                negatives = []
                for tid, _, _ in ranked:
                    if db.execute('SELECT 1 FROM truth WHERE qid=? AND tid=?', (q['id'],tid)).fetchone():
                        continue
                    if q['split']=='fit' and db.execute("SELECT 1 FROM truth g JOIN records h ON h.id=g.qid WHERE g.tid=? AND h.split!='fit'", (tid,)).fetchone():
                        continue
                    negatives.append((q['id'],tid))
                    if len(negatives) >= cfg['hard_negatives']:
                        break
                db.executemany('INSERT OR IGNORE INTO bi_negatives VALUES (?,?)', negatives)
        db.commit()
    model = NeuralModel(cfg, kind, training=True)
    optimizer = torch.optim.AdamW([p for p in model.model.parameters() if p.requires_grad], lr=cfg['learning_rate'])

    def examples(split):
        positives = db.execute('''SELECT q.id qid, t.id tid, q.raw query_text, t.raw target_text, 1 label
          FROM truth g JOIN records q ON q.id=g.qid JOIN records t ON t.id=g.tid
          WHERE q.split=? AND (?!='fit' OR NOT EXISTS(SELECT 1 FROM truth z JOIN records h ON h.id=z.qid
            WHERE z.tid=t.id AND h.split!='fit')) ORDER BY q.id,t.id''', (split, split))
        yield from positives
        last, count = None, 0
        if kind == 'crossencoder':
            candidates = supervised_pairs(db, split)
        else:
            candidates = db.execute("""SELECT q.id qid,t.id tid,q.raw query_text,t.raw target_text,0 label
              FROM bi_negatives n JOIN records q ON q.id=n.qid JOIN records t ON t.id=n.tid
              WHERE q.split=? ORDER BY q.id,t.id""", (split,))
        for row in candidates:
            if row['label']:
                continue
            if row['qid'] != last:
                last, count = row['qid'], 0
            if count < cfg['hard_negatives']:
                yield row
                count += 1

    def loss(batch):
        q = [r['query_text'] for r in batch]
        t = [r['target_text'] for r in batch]
        y = torch.tensor([r['label'] for r in batch], dtype=torch.float32, device=model.device)
        logits = model.logits(q, t) if kind == 'crossencoder' else (model.vectors(q)*model.vectors(t)).sum(1)*20-10
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

    db.executescript('DROP TABLE IF EXISTS training_examples; CREATE TABLE training_examples(qid TEXT,tid TEXT,query_text TEXT,target_text TEXT,label INTEGER,split TEXT,PRIMARY KEY(qid,tid,split));')
    for split in ('fit', 'dev'):
        for row in examples(split):
            db.execute('INSERT OR IGNORE INTO training_examples VALUES (?,?,?,?,?,?)',
                       (row['qid'],row['tid'],row['query_text'],row['target_text'],row['label'],split))
    db.commit()
    db.create_function('shuffle_key', 3, lambda q,t,e: digest([cfg['seed'],e,q,t]))
    best = float('inf')
    logs = []
    for epoch in range(cfg['epochs']):
        model.model.train()
        total, n, batch = 0., 0, []
        for row in db.execute("SELECT * FROM training_examples WHERE split='fit' ORDER BY shuffle_key(qid,tid,?)", (epoch,)):
            batch.append(row)
            if len(batch) == cfg['batch_size']:
                optimizer.zero_grad()
                value = loss(batch)
                value.backward()
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.)
                optimizer.step()
                total += value.item()*len(batch)
                n += len(batch)
                batch = []
        if batch:
            optimizer.zero_grad(); value = loss(batch); value.backward(); torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.); optimizer.step()
            total += value.item()*len(batch); n += len(batch)
        if not n:
            raise ValueError('No fitting examples; increase --train-percent or resolve shared held-out targets')
        model.model.eval()
        dev_loss, dev_n, batch = 0., 0, []
        with torch.no_grad():
            for row in db.execute("SELECT * FROM training_examples WHERE split='dev' ORDER BY qid,tid"):
                batch.append(row)
                if len(batch) == cfg['batch_size']:
                    dev_loss += loss(batch).item()*len(batch); dev_n += len(batch); batch = []
            if batch:
                dev_loss += loss(batch).item()*len(batch); dev_n += len(batch)
        if not dev_n:
            raise ValueError('No development examples; increase --train-percent')
        metric = dev_loss/dev_n
        if metric < best:
            best = metric
            model.save(destination)
        logs.append({'epoch': epoch+1, 'train_binary_loss': total/n, 'dev_binary_loss': metric, 'selected': metric == best})
        write_json(destination / 'training_log.json', logs)
