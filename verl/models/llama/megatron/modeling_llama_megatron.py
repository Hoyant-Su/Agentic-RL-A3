
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from megatron.core import ModelParallelConfig, mpu, tensor_parallel
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast
from verl.utils.megatron import sequence_parallel as sp_utils
from verl.utils.megatron import tensor_parallel as tp_utils
from verl.utils.megatron_utils import TransformerConfig, convert_config
from .layers import (
    ParallelLlamaDecoderLayer,
    ParallelLlamaDecoderLayerRmPad,
    ParallelLlamaRMSNorm,
)

"""
TODO: 
1. Add weight initialization. Here we need to be careful on TP weight init.
2. Add sequence parallel
3. Load checkpoint from meta LLama pretrained checkpoint
"""

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )

class ParallelLlamaModel(nn.Module):

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        if megatron_config is not None:
            assert embedding_kwargs.get("config", False), (
                "must have ModelParallelConfig"
            )
            tp_utils.update_kwargs_with_config(embedding_kwargs, self.megatron_config)
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            **embedding_kwargs,
        )
        self.layers = nn.ModuleList(
            [
                ParallelLlamaDecoderLayer(config, megatron_config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = ParallelLlamaRMSNorm(config, megatron_config)

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds
    ):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        batch_size, seq_length = input_ids.shape
        inputs_embeds = self.embed_tokens(input_ids)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds
        )
        hidden_states = inputs_embeds
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_outputs
        hidden_states = self.norm(hidden_states)
        return hidden_states

class ParallelLlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.model = ParallelLlamaModel(config, megatron_config=megatron_config)
        self.vocab_size = config.vocab_size
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if megatron_config is not None:
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        self.lm_head = tensor_parallel.ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = outputs
        logits = self.lm_head(hidden_states)[0]
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
        logits = logits.float()
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

class ParallelLlamaModelRmPad(nn.Module):

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        self.megatron_config = megatron_config
        if megatron_config is not None:
            assert embedding_kwargs.get("config", False), (
                "must have ModelParallelConfig"
            )
            tp_utils.update_kwargs_with_config(embedding_kwargs, self.megatron_config)
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            **embedding_kwargs,
        )
        self.layers = nn.ModuleList(
            [
                ParallelLlamaDecoderLayerRmPad(config, megatron_config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = ParallelLlamaRMSNorm(config, megatron_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: int = None,
        indices: torch.Tensor = None,
        cu_seqlens: int = None,
        max_seqlen_in_batch: int = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.transpose(0, 1)
        if self.megatron_config.sequence_parallel:
            inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(
                inputs_embeds
            )
        hidden_states = inputs_embeds
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                sequence_length=sequence_length,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen_in_batch=max_seqlen_in_batch,
            )
            hidden_states = layer_outputs
        hidden_states = self.norm(hidden_states)
        return hidden_states

class ParallelLlamaForCausalLMRmPad(nn.Module):
    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.megatron_config = megatron_config
        self.model = ParallelLlamaModelRmPad(config, megatron_config=megatron_config)
        self.vocab_size = config.vocab_size
        self._init_head(config)

    def _init_head(self, config):
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if self.megatron_config is not None:
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        self.lm_head = tensor_parallel.ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

    def _forward_head(self, hidden_states):
        logits = self.lm_head(hidden_states)[0]
        logits = logits.float()
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        batch_size, sequence_length = input_ids.shape
        input_ids, indices, cu_seqlens, max_seqlen_in_batch, *_ = unpad_input(
            input_ids.unsqueeze(dim=-1), attention_mask
        )
        if self.megatron_config.sequence_parallel:
            input_ids = sp_utils.pad_to_sequence_parallel(input_ids)
        input_ids = input_ids.transpose(0, 1)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            sequence_length=sequence_length,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen_in_batch=max_seqlen_in_batch,
        )
        hidden_states = outputs
        logits = self._forward_head(hidden_states)
        if self.megatron_config.sequence_parallel:
            totol_nnz = cu_seqlens[-1]
            logits = logits[:totol_nnz]
        logits = torch.squeeze(logits, dim=1)
        logits = pad_input(logits, indices, batch_size, seqlen=sequence_length)
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

class ParallelLlamaForValueRmPad(ParallelLlamaForCausalLMRmPad):
    def _init_head(self, config):
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if self.megatron_config is not None:
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        self.lm_head = nn.Linear(
            in_features=config.hidden_size, out_features=1, bias=False
        )
        sp_utils.mark_parameter_as_sequence_parallel(self.lm_head.weight)

    def _forward_head(self, hidden_states):
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        if self.megatron_config.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(
                logits, tensor_parallel_output_grad=False
            )
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output = super().forward(input_ids, attention_mask, position_ids)
        output.logits = torch.squeeze(output.logits, dim=-1)
        return output

"""
Support pipeline parallelism
"""

class ParallelLlamaModelRmPadPP(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        megatron_config: ModelParallelConfig,
        pre_process,
        post_process,
    ):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pre_process = pre_process
        self.post_process = post_process
        self.megatron_config = megatron_config
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        if megatron_config is not None:
            assert embedding_kwargs.get("config", False), (
                "must have ModelParallelConfig"
            )
            tp_utils.update_kwargs_with_config(embedding_kwargs, self.megatron_config)
        if pre_process:
            self.embed_tokens = tensor_parallel.VocabParallelEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                **embedding_kwargs,
            )
        else:
            self.embed_tokens = None
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = megatron_config.pipeline_model_parallel_size
        self.num_layer_per_pp = config.num_hidden_layers // pp_size
        vpp_size = megatron_config.virtual_pipeline_model_parallel_size
        vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
        if vpp_size is not None:
            self.layers = nn.ModuleList()
            self.num_layer_vpp_chunk = self.num_layer_per_pp // vpp_size
            self.num_layer_this_model = self.num_layer_vpp_chunk
            offset = vpp_rank * (config.num_hidden_layers // vpp_size) + (
                pp_rank * self.num_layer_vpp_chunk
            )
        else:
            self.num_layer_this_model = self.num_layer_per_pp
            offset = pp_rank * self.num_layer_per_pp
        self.layers = nn.ModuleList()
        for i in range(self.num_layer_this_model):
            layer = ParallelLlamaDecoderLayerRmPad(
                config, megatron_config, layer_idx=offset + i
            )
            self.layers.add_module(f"{i}", layer)
        if post_process:
            self.norm = ParallelLlamaRMSNorm(config, megatron_config)
        else:
            self.norm = None

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        sequence_length: int = None,
        indices: torch.Tensor = None,
        cu_seqlens: int = None,
        max_seqlen_in_batch: int = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if self.pre_process:
            inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds.transpose(0, 1)
            if self.megatron_config.sequence_parallel:
                inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(
                    inputs_embeds
                )
            hidden_states = inputs_embeds
        else:
            hidden_states = self.input_tensor
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                sequence_length=sequence_length,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen_in_batch=max_seqlen_in_batch,
            )
            hidden_states = layer_outputs
        if self.post_process:
            hidden_states = self.norm(hidden_states)
        return hidden_states

class ParallelLlamaForCausalLMRmPadPP(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        megatron_config: ModelParallelConfig,
        pre_process,
        post_process,
        share_embeddings_and_output_weights=False,
    ):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.megatron_config = megatron_config
        self.model = ParallelLlamaModelRmPadPP(
            config,
            megatron_config=megatron_config,
            pre_process=pre_process,
            post_process=post_process,
        )
        assert share_embeddings_and_output_weights is False, (
            "Llama Model not supports sharing embedding and output weights"
        )
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.vocab_size = config.vocab_size
        self.pre_process = pre_process
        self.post_process = post_process
        if post_process:
            self._init_head(config)

    def set_input_tensor(self, input_tensor):
        assert len(input_tensor) == 1
        self.model.set_input_tensor(input_tensor[0])

    def _init_head(self, config):
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if self.megatron_config is not None:
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        self.lm_head = tensor_parallel.ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )

    def _forward_head(self, hidden_states):
        logits = self.lm_head(hidden_states)[0]
        logits = logits.float()
        return logits

    def forward(
        self,
        *,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        batch_size, sequence_length = input_ids.shape
        input_ids_rmpad, indices, cu_seqlens, max_seqlen_in_batch, *_ = unpad_input(
            input_ids.unsqueeze(dim=-1), attention_mask
        )
        if self.megatron_config.sequence_parallel:
            input_ids_rmpad = sp_utils.pad_to_sequence_parallel(input_ids_rmpad)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
        outputs = self.model(
            input_ids=input_ids_rmpad,
            position_ids=position_ids,
            sequence_length=sequence_length,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen_in_batch=max_seqlen_in_batch,
        )
        if self.post_process:
            hidden_states = outputs
            logits = self._forward_head(hidden_states)
            logits = torch.squeeze(logits, dim=1)
            if self.megatron_config.sequence_parallel:
                totol_nnz = cu_seqlens[-1]
                logits = logits[:totol_nnz]
            logits = pad_input(logits, indices, batch_size, seqlen=sequence_length)
            return CausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )
        else:
            return outputs

class ParallelLlamaForValueRmPadPP(ParallelLlamaForCausalLMRmPadPP):
    def _init_head(self, config):
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        if self.megatron_config is not None:
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        self.lm_head = nn.Linear(
            in_features=config.hidden_size, out_features=1, bias=False
        )
        sp_utils.mark_parameter_as_sequence_parallel(self.lm_head.weight)

    def _forward_head(self, hidden_states):
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        if self.megatron_config.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(
                logits, tensor_parallel_output_grad=False
            )
        return logits

    def forward(
        self,
        *,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        if self.post_process:
            output.logits = torch.squeeze(output.logits, dim=-1)
            return output
        else:
            return output
