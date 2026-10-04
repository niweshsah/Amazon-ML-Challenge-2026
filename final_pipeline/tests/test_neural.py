"""Optional real LoRA training/loading against locally created tiny XLM-R models."""
import json
import numpy as np
import pytest
from pathlib import Path
from test_pipeline import config
from entity_resolution.pipeline import Pipeline
from entity_resolution.models import NeuralModel


def test_actual_lora_training_and_checkpoint_compatibility(tmp_path):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    pytest.importorskip('peft')
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    torch.set_num_threads(2)
    tokenizer = Tokenizer(WordLevel({'<pad>':0,'<s>':1,'</s>':2,'<unk>':3,'Name':4,'Address':5,'Country':6,
                                   'India':7,'US':8,'bakery':9,'branch':10,':':11,'|':12,'Rue':13,'du':14,'Marché':15}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single='<s> $A </s>', pair='<s> $A </s> $B </s>', special_tokens=[('<s>',1),('</s>',2)])
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='<pad>', bos_token='<s>',
                                                eos_token='</s>', unk_token='<unk>', model_max_length=64)
    cfg = config(tmp_path, fixture=False, device='cpu', epochs=1, batch_size=16, max_length=64, hard_negatives=2,
                 biencoder_revision=None, crossencoder_revision=None, xgboost_trees=3)
    for kind in ('biencoder','crossencoder'):
        base = tmp_path/f'base-{kind}'
        tiny = transformers.XLMRobertaConfig(vocab_size=len(fast), hidden_size=16, num_hidden_layers=1,
                  num_attention_heads=2, intermediate_size=32, max_position_embeddings=128,
                  pad_token_id=0, bos_token_id=1, eos_token_id=2, num_labels=1)
        model = transformers.XLMRobertaModel(tiny) if kind=='biencoder' else transformers.XLMRobertaForSequenceClassification(tiny)
        model.save_pretrained(base)
        fast.save_pretrained(base)
        cfg[kind+'_base'] = str(base)
    pipeline = Pipeline(cfg)
    pipeline.prepare(); pipeline.train_biencoder(); pipeline.embed(); pipeline.retrieve(); pipeline.features()
    pipeline.train_xgboost(); pipeline.train_crossencoder()
    encoder = NeuralModel(cfg,'biencoder')
    encoded = encoder.embed(['Name: bakery branch | Country: India','Name: श्री गणेश | Country: France'])
    assert np.isfinite(encoded).all()
    np.testing.assert_allclose(np.linalg.norm(encoded,axis=1),1., atol=1e-5)
    reranker = NeuralModel(cfg,'crossencoder')
    score = reranker.score(['Name: bakery'],['Name: bakery'])
    assert score.shape==(1,) and 0<=score[0]<=1
    assert encoder.model.peft_config['default'].r==cfg['lora_rank']
    assert (pipeline.models/'biencoder/training_log.json').exists()
    assert (pipeline.models/'crossencoder/training_log.json').exists()
    with pytest.raises(ValueError, match='base_model'):
        NeuralModel(dict(cfg, biencoder_base=cfg['crossencoder_base']), 'biencoder')
    with pytest.raises(ValueError, match='Incompatible'):
        NeuralModel(dict(cfg,max_length=32),'biencoder')
