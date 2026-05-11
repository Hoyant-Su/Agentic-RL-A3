from typing import Optional, Tuple
import torch
from megatron.core import ModelParallelConfig
from torch import nn
from transformers import Qwen2Config
from verl.utils.megatron_utils import TransformerConfig, convert_config
from .parallel_attention import ParallelQwen2Attention, ParallelQwen2AttentionRmPad
from .parallel_mlp import ParallelQwen2MLP
from .parallel_rmsnorm import ParallelQwen2RMSNorm

class ParallelQwen2DecoderLayer(nn.Module):
    def __init__(
        self, config: Qwen2Config, megatron_config: ModelParallelConfig, layer_idx: int
    ):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = ParallelQwen2Attention(
            config=config, megatron_config=megatron_config
        )
        self.mlp = ParallelQwen2MLP(config, megatron_config=megatron_config)
        self.input_layernorm = ParallelQwen2RMSNorm(config, megatron_config)
        self.post_attention_layernorm = ParallelQwen2RMSNorm(config, megatron_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = hidden_states
        return outputs

class ParallelQwen2DecoderLayerRmPad(nn.Module):
    def __init__(
        self, config: Qwen2Config, megatron_config: ModelParallelConfig, layer_idx: int
    ):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = ParallelQwen2AttentionRmPad(
            config=config, megatron_config=megatron_config
        )
        self.mlp = ParallelQwen2MLP(config, megatron_config=megatron_config)
        self.input_layernorm = ParallelQwen2RMSNorm(config, megatron_config)
        self.post_attention_layernorm = ParallelQwen2RMSNorm(config, megatron_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: int = None,
        indices: torch.Tensor = None,
        cu_seqlens: int = None,
        max_seqlen_in_batch: int = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            sequence_length=sequence_length,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen_in_batch=max_seqlen_in_batch,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = hidden_states
        return outputs
