from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_outpus_and_unpad,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
    validate_ulysses_config,
)

def _merge_with_image_features(
    self,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    image_features: torch.Tensor,
):
    image_token_index: int = self.config.media_placeholder_token_id
    batch_size, sequence_length, input_embed_dim = inputs_embeds.shape
    image_feature_nums, image_feature_dim = image_features.shape
    assert image_feature_dim == input_embed_dim
    image_token_nums = (input_ids == image_token_index).sum()
    total_image_token_nums = torch.tensor(
        [image_token_nums], dtype=image_token_nums.dtype, device=input_ids.device
    )
    total_image_token_nums = gather_outpus_and_unpad(
        total_image_token_nums, gather_dim=0
    )
    assert image_feature_nums == total_image_token_nums.sum()
    inputs_embeds = inputs_embeds.reshape(-1, input_embed_dim)
    input_ids = input_ids.flatten()
    sp_image_features = image_features.split(total_image_token_nums.tolist(), dim=0)
    sp_rank = get_ulysses_sequence_parallel_rank()
    image_features = sp_image_features[sp_rank]
    inputs_embeds[input_ids == image_token_index] = image_features
    inputs_embeds = inputs_embeds.reshape(
        (batch_size, sequence_length, input_embed_dim)
    )
    return inputs_embeds

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def _ulysses_flash_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        .transpose(1, 2)
    )
    k_nope, value_states = torch.split(
        kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )
    kv_seq_len = value_states.shape[-2]
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
    kv_seq_len *= ulysses_sp_size
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
    key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
    if self.q_head_dim != self.v_head_dim:
        value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
    if ulysses_sp_size > 1:
        validate_ulysses_config(self.num_heads, ulysses_sp_size)
        num_key_value_groups = (
            self.config.num_attention_heads // self.config.num_key_value_heads
        )
        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)
        full_q_len = query_states.size(2)
        position_ids_list = [
            torch.empty_like(position_ids) for _ in range(ulysses_sp_size)
        ]
        torch.distributed.all_gather(
            position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group()
        )
        position_ids = torch.concat(position_ids_list, dim=-1)
    else:
        full_q_len = q_len
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    dropout_rate = self.attention_dropout if self.training else 0.0
    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        full_q_len,
        dropout=dropout_rate,
        sliding_window=None,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        position_ids=position_ids,
        softmax_scale=self.softmax_scale,
    )
    if ulysses_sp_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)
    if self.q_head_dim != self.v_head_dim:
        attn_output = attn_output[:, :, :, : self.v_head_dim]
    attn_output = attn_output.reshape(
        bsz, q_len, self.num_heads * self.v_head_dim
    ).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None, None
