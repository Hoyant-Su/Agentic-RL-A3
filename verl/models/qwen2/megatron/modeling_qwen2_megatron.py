
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from megatron.core import ModelParallelConfig, mpu, parallel_state, tensor_parallel
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import CausalLMOutputWithPast
from verl.utils.megatron import sequence_parallel as sp_utils
from verl.utils.megatron import tensor_parallel as tp_utils
from verl.utils.megatron_utils import TransformerConfig, convert_config
from .layers import (
    ParallelQwen2DecoderLayer,
    ParallelQwen2DecoderLayerRmPad,
    ParallelQwen2RMSNorm,
)

"""
TODO: 
1. Add weight initialization. Here we need to be careful on TP weight init.
2. Add sequence parallel
3. Load checkpoint from Qwen2 pretrained checkpoint
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

class ParallelQwen2Model(nn.Module):

    def __init__(self, config: Qwen2Config, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        if megatron_config is not None:
            assert embedding_kwargs.get("config", False), (
                "must have ModelParallelConfig"
            )
            tp_utils.update_kwargs_with_config(embedding_kwargs, megatron_config)
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            **embedding_kwargs,
        )
        self.layers = nn.ModuleList(
            [
                ParallelQwen2DecoderLayer(config, megatron_config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = ParallelQwen2RMSNorm(config, megatron_config)

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

class ParallelQwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.model = ParallelQwen2Model(config, megatron_config=megatron_config)
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

class ParallelQwen2ModelRmPad(nn.Module):

    def __init__(self, config: Qwen2Config, megatron_config: ModelParallelConfig):
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
                ParallelQwen2DecoderLayerRmPad(config, megatron_config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = ParallelQwen2RMSNorm(config, megatron_config)

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

class ParallelQwen2ForCausalLMRmPad(nn.Module):
    def __init__(self, config: Qwen2Config, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.megatron_config = megatron_config
        self.model = ParallelQwen2ModelRmPad(config, megatron_config=megatron_config)
        self.vocab_size = config.vocab_size
        self._init_head(config)

    def _init_head(self, config: Qwen2Config):
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

class ParallelQwen2ForValueRmPad(ParallelQwen2ForCausalLMRmPad):
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

class ParallelQwen2ModelRmPadPP(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
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
            layer = ParallelQwen2DecoderLayerRmPad(
                config, megatron_config, layer_idx=i + offset
            )
            self.layers.add_module(f"{i}", layer)
        if post_process:
            self.norm = ParallelQwen2RMSNorm(config, megatron_config)
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

class ParallelQwen2ForCausalLMRmPadPP(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        megatron_config: ModelParallelConfig,
        pre_process,
        post_process,
        share_embeddings_and_output_weights,
    ):
        super().__init__()
        self.config: TransformerConfig = convert_config(config, megatron_config)
        self.megatron_config = megatron_config
        self.model = ParallelQwen2ModelRmPadPP(
            config,
            megatron_config=megatron_config,
            pre_process=pre_process,
            post_process=post_process,
        )
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.vocab_size = config.vocab_size
        self.pre_process = pre_process
        self.post_process = post_process
        if post_process:
            self._init_head(config)
        if pre_process or post_process:
            self.setup_embeddings_and_output_layer()

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
            skip_weight_param_allocation=self.pre_process
            and self.share_embeddings_and_output_weights,
            **column_kwargs,
        )

    def setup_embeddings_and_output_layer(self) -> None:
        if self.pre_process:
            self.model.embed_tokens.weight.is_embedding_or_output_parameter = True
        if self.post_process and self.lm_head.weight is not None:
            self.lm_head.weight.is_embedding_or_output_parameter = True
        if not self.share_embeddings_and_output_weights:
            return
        if parallel_state.get_pipeline_model_parallel_world_size() == 1:
            self.shared_embedding_or_output_weight().zero_out_wgrad = True
            return
        if (
            parallel_state.is_pipeline_first_stage()
            and self.pre_process
            and not self.post_process
        ):
            self.shared_embedding_or_output_weight().shared_embedding = True
        if self.post_process and not self.pre_process:
            assert not parallel_state.is_pipeline_first_stage()
            self.lm_head.weight.data.fill_(0)
            self.lm_head.weight.shared = True
            self.lm_head.weight.shared_embedding = True
        if (
            torch.distributed.is_initialized()
            and parallel_state.is_rank_in_embedding_group()
        ):
            weight = self.shared_embedding_or_output_weight()
            weight.data = weight.data.cuda()
            torch.distributed.all_reduce(
                weight.data, group=parallel_state.get_embedding_group()
            )

    def shared_embedding_or_output_weight(self) -> torch.Tensor:
        if self.pre_process:
            return self.model.embed_tokens.weight
        elif self.post_process:
            return self.lm_head.weight
        return None

    def _forward_head(self, hidden_states):
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits = self.lm_head(hidden_states, weight=output_weight)[0]
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

class ParallelQwen2ForValueRmPadPP(ParallelQwen2ForCausalLMRmPadPP):
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
