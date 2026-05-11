import torch
from transformers import PretrainedConfig
from verl.utils.device import get_torch_device

VALID_CONFIG_TYPE = {
    "llama",
    "qwen2",
    "qwen2_moe",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3",
    "qwen3_moe",
    "qwen3_vl",
    "qwen3_vl_moe",
    "deepseek_v3",
    "minicpmv",
    "minicpmo",
    "mistral",
    "gemma3_text",
    "seed_oss",
    "apertus",
    "glm4v",
}

def get_device_flops(unit="T"):

    def unit_convert(number, level):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number
        ptr = 0
        while ptr < len(units) and units[ptr] != level:
            number /= 1000
            ptr += 1
        return number

    device = get_torch_device()
    if device == torch.cpu:
        device_name = "CPU"
    else:
        device_name = get_torch_device().get_device_name()
    flops = float("inf")
    if "CPU" in device_name:
        flops = 448e9
    elif "GB200" in device_name:
        flops = 2.5e15
    elif "B200" in device_name:
        flops = 2.25e15
    elif "MI300X" in device_name:
        flops = 1336e12
    elif "H100" in device_name or "H800" in device_name or "H200" in device_name:
        flops = 989e12
    elif "A100" in device_name or "A800" in device_name:
        flops = 312e12
    elif "L40S" in device_name:
        flops = 362.05e12
    elif "L40" in device_name:
        flops = 181.05e12
    elif "A40" in device_name:
        flops = 149.7e12
    elif "L20" in device_name:
        flops = 119.5e12
    elif "H20" in device_name:
        flops = 148e12
    elif "910B" in device_name:
        flops = 354e12
    elif "Ascend910" in device_name:
        flops = 354e12
    elif "RTX 3070 Ti" in device_name:
        flops = 21.75e12
    flops_unit = unit_convert(flops, unit)
    return flops_unit

class FlopsCounter:

    def __init__(self, config: PretrainedConfig):
        if config.model_type not in VALID_CONFIG_TYPE:
            print(
                f"Only support config type of {VALID_CONFIG_TYPE}, but got {config.model_type}. MFU will always be "
                f"zero."
            )
        self.estimate_func = {
            "qwen2": self._estimate_qwen2_flops,
            "llama": self._estimate_qwen2_flops,
            "qwen2_moe": self._estimate_qwen2_moe_flops,
            "qwen2_vl": self._estimate_qwen2_flops,
            "qwen2_5_vl": self._estimate_qwen2_flops,
            "qwen3": self._estimate_qwen2_flops,
            "qwen3_moe": self._estimate_qwen2_moe_flops,
            "qwen3_vl": self._estimate_qwen2_flops,
            "qwen3_vl_moe": self._estimate_qwen2_moe_flops,
            "deepseek_v3": self._estimate_deepseek_v3_flops,
            "minicpmv": self._estimate_qwen2_flops,
            "minicpmo": self._estimate_qwen2_flops,
            "mistral": self._estimate_qwen2_flops,
            "gemma3_text": self._estimate_gemma3_flops,
            "seed_oss": self._estimate_qwen2_flops,
            "apertus": self._estimate_apertus_flops,
            "glm4v": self._estimate_qwen2_flops,
        }
        self.config = getattr(config, "text_config", config)

    def _estimate_unknown_flops(self, tokens_sum, batch_seqlens, delta_time):
        return 0

    def _estimate_qwen2_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size
        head_dim = getattr(
            self.config,
            "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (
            q_size + k_size + v_size + num_attention_heads * head_dim
        )
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        dense_N_flops = 6 * dense_N * tokens_sum
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = (
            12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers
        )
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_deepseek_v3_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        moe_intermediate_size = self.config.moe_intermediate_size
        num_hidden_layers = self.config.num_hidden_layers
        first_k_dense_replace = self.config.first_k_dense_replace
        num_query_heads = self.config.num_attention_heads
        moe_num_expert = self.config.n_routed_experts
        moe_topk = self.config.num_experts_per_tok
        share_expert_num = self.config.n_shared_experts
        moe_gata_N = hidden_size * moe_num_expert
        moe_expertmlp_N = (
            hidden_size * moe_intermediate_size * (moe_topk + share_expert_num) * 3
        )
        attn_linear_N = 0
        q_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        if self.config.q_lora_rank is None:
            attn_linear_N += hidden_size * num_query_heads * q_head_dim
        else:
            attn_linear_N += hidden_size * self.config.q_lora_rank
            attn_linear_N += num_query_heads * q_head_dim * self.config.q_lora_rank
        attn_linear_N += hidden_size * (
            self.config.kv_lora_rank + self.config.qk_rope_head_dim
        )
        attn_linear_N += (
            num_query_heads
            * (q_head_dim - self.config.qk_rope_head_dim + self.config.v_head_dim)
            * self.config.kv_lora_rank
        )
        attn_linear_N += num_query_heads * self.config.v_head_dim * hidden_size
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        moe_N = (
            (moe_gata_N + moe_expertmlp_N + attn_linear_N)
            * (num_hidden_layers - first_k_dense_replace)
            + (hidden_size * self.config.intermediate_size * 3 + attn_linear_N)
            * first_k_dense_replace
            + emd_and_lm_head_N
        )
        dense_N_flops = 6 * moe_N * tokens_sum
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen * num_hidden_layers
        attn_qkv_flops = 12 * seqlen_square_sum * q_head_dim * num_query_heads
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_qwen2_moe_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        moe_intermediate_size = self.config.moe_intermediate_size
        moe_topk = self.config.num_experts_per_tok
        num_experts = self.config.num_experts
        head_dim = getattr(
            self.config,
            "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim
        moe_mlp_N = (
            hidden_size * moe_topk * moe_intermediate_size * 3
            + hidden_size * num_experts
        )
        attn_linear_N = hidden_size * (
            q_size + k_size + v_size + num_attention_heads * head_dim
        )
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        dense_N = (moe_mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        dense_N_flops = 6 * dense_N * tokens_sum
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = (
            12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers
        )
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_gemma3_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size
        head_dim = getattr(
            self.config,
            "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (
            q_size + k_size + v_size + num_attention_heads * head_dim
        )
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        dense_N_flops = 6 * dense_N * tokens_sum
        seqlen_square_sum = 0
        layer_types = getattr(self.config, "layer_types", None)
        sliding_window = getattr(self.config, "sliding_window", 1024)
        sliding_window_pattern = getattr(self.config, "sliding_window_pattern", 6)
        if (
            layer_types is None
            and sliding_window is not None
            and sliding_window_pattern is not None
        ):
            layer_types = [
                "sliding_attention"
                if bool((i + 1) % sliding_window_pattern)
                else "full_attention"
                for i in range(num_hidden_layers)
            ]
        if layer_types:
            for layer_idx in range(num_hidden_layers):
                is_sliding = False
                if layer_types and layer_idx < len(layer_types):
                    is_sliding = layer_types[layer_idx] == "sliding_attention"
                for seqlen in batch_seqlens:
                    if is_sliding and sliding_window:
                        effective_seqlen = min(seqlen, sliding_window)
                        seqlen_square_sum += seqlen * effective_seqlen
                    else:
                        seqlen_square_sum += seqlen * seqlen
        else:
            for seqlen in batch_seqlens:
                seqlen_square_sum += seqlen * seqlen
            seqlen_square_sum *= num_hidden_layers
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_apertus_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size
        head_dim = getattr(
            self.config,
            "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim
        mlp_N = hidden_size * intermediate_size * 2
        attn_linear_N = hidden_size * (
            q_size + k_size + v_size + num_attention_heads * head_dim
        )
        qk_norm_params_per_layer = hidden_size + num_key_value_heads * head_dim
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        dense_N = (
            mlp_N + attn_linear_N + qk_norm_params_per_layer
        ) * num_hidden_layers + emd_and_lm_head_N
        dense_N_flops = 6 * dense_N * tokens_sum
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = (
            12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers
        )
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def estimate_flops(self, batch_seqlens, delta_time):
        tokens_sum = sum(batch_seqlens)
        func = self.estimate_func.get(
            self.config.model_type, self._estimate_unknown_flops
        )
        estimated_flops = func(tokens_sum, batch_seqlens, delta_time)
        promised_flops = get_device_flops()
        return estimated_flops, promised_flops
