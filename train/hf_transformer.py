"""
Hugging Face Trainerで使用できるPyTorch Transformerモデル
既存のtransformer.pyのTransformerクラスを継承・拡張
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Any
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput
from transformers.utils import logging

from util.embed_utils import embed_circle

logger = logging.get_logger(__name__)
class TransformerConfig(PretrainedConfig):
    """Transformerモデルの設定クラス"""
    
    model_type = "transformer"
    
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        bias: bool = True,
        vocab_size: int = 1000,
        max_input_len: int = 512,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
        use_positional_embedding: str = "learned",
        init_std: float = 0.02,
        linear_init_type: str = "normal",
        embedding_init_type: str = "normal",
        tie_word_embeddings: bool = False,
        seed: int = 42,
        weight_init: bool = True,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            **kwargs
        )
        
        # エンコーダー専用パラメータが渡された場合は無視（後方互換のため）
        encoder_only_keys = (
            "add_register",
            "embed_type",
            "use_tropical_attention",
            "embed_noiz",
        )
        self._ignored_encoder_only_keys = {}
        for key in encoder_only_keys:
            if key in kwargs:
                self._ignored_encoder_only_keys[key] = kwargs.pop(key)
                logger.debug(f"Ignore encoder-only config key on TransformerConfig: {key}")
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.bias = bias
        self.vocab_size = vocab_size
        self.max_input_len = max_input_len
        self.use_positional_embedding = use_positional_embedding
        self.init_std = init_std
        self.linear_init_type = linear_init_type
        self.embedding_init_type = embedding_init_type
        self.tie_word_embeddings = tie_word_embeddings
        self.seed = seed
        self.weight_init = weight_init
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
