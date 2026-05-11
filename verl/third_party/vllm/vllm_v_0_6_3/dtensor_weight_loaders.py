from typing import Dict
import torch.nn as nn
from torch.distributed._tensor import DTensor
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import is_pp_missing_parameter

def gemma_dtensor_weight_loader(
    actor_weights: Dict, vllm_model: nn.Module
) -> nn.Module:
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        for param_name, shard_name, shard_id in stacked_params_mapping:
            if shard_name not in name:
                continue
            stacked_name = name.replace(shard_name, param_name)
            if stacked_name.endswith(".bias") and stacked_name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(
                param_name=name, loaded_weights=loaded_weight
            )
            param = params_dict[stacked_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            if "lm_head.weight" in name:
                continue
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(
                param_name=name, loaded_weights=loaded_weight
            )
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype))

def gptbigcode_dtensor_load_weights(actor_weights: Dict, vllm_model: nn.Module):
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "lm_head.weight" in name:
            continue
        if ".attn.bias" in name:
            continue
        local_loaded_weight = redistribute_dtensor(
            param_name=name, loaded_weights=loaded_weight
        )
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, local_loaded_weight.to(dtype=param.dtype))

def starcoder2_dtensor_load_weights(actor_weights: Dict, vllm_model: nn.Module):
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    ]
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            local_loaded_weight = redistribute_dtensor(
                param_name=name, loaded_weights=loaded_weight
            )
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            param = params_dict[name]
            local_loaded_weight = redistribute_dtensor(
                param_name=name, loaded_weights=loaded_weight
            )
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype))

def llama_dtensor_weight_loader(
    actor_weights: Dict, vllm_model: nn.Module
) -> nn.Module:
    stacked_params_mapping = [
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
            continue
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(
                param_name=name, loaded_weights=loaded_weight
            )
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(
                param_name=name, loaded_weights=loaded_weight
            )
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight)

def qwen2_dtensor_weight_loader(
    actor_weights: Dict, vllm_model: nn.Module
) -> nn.Module:
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(
                param_name=name, loaded_weights=loaded_weight
            )
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            local_loaded_weight = redistribute_dtensor(
                param_name=name, loaded_weights=loaded_weight
            )
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype))

def qwen2vl_dtensor_weight_loader(
    actor_weights: Dict, vllm_model: nn.Module
) -> nn.Module:
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(
                param_name=name, loaded_weights=loaded_weight
            )
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            local_loaded_weight = redistribute_dtensor(
                param_name=name, loaded_weights=loaded_weight
            )
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype))

def deepseekv2_dtensor_weight_loader(
    actor_weights: Dict, vllm_model: nn.Module
) -> nn.Module:
    stacked_params_mapping = [
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=vllm_model.config.n_routed_experts,
    )
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            if ("mlp.experts." in name) and name not in params_dict:
                continue
            name = name.replace(weight_name, param_name)
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, vllm_model):
                continue
            param = params_dict[name]
            local_loaded_weight = redistribute_dtensor(
                param_name=name, loaded_weights=loaded_weight
            )
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, vllm_model):
                    continue
                param = params_dict[name]
                local_loaded_weight = redistribute_dtensor(
                    param_name=name, loaded_weights=loaded_weight
                )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    local_loaded_weight.to(dtype=param.dtype),
                    weight_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, vllm_model):
                    continue
                param = params_dict[name]
                local_loaded_weight = redistribute_dtensor(
                    param_name=name, loaded_weights=loaded_weight
                )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, local_loaded_weight.to(dtype=param.dtype))

def gpt2_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    pass

def redistribute_dtensor(
    param_name: str, loaded_weights: DTensor, parallelize_plan: Dict = None
):
    param_name = _process_parameter_names(name=param_name)
    if parallelize_plan is not None:
        assert param_name in parallelize_plan, (
            f"param name: {param_name} not in parallelize_plan :{parallelize_plan.keys()}"
        )
        placement = parallelize_plan[param_name]
        local_loaded_weights = loaded_weights.redistribute(
            device_mesh=loaded_weights.device_mesh, placements=placement
        ).to_local()
    else:
        local_loaded_weights = loaded_weights.full_tensor()
    return local_loaded_weights

def _process_parameter_names(name):
    if name.endswith(".weight"):
        name = name[:-7]
    if "model.layers" in name:
        parts = name.split(".")
        name = ".".join(parts[3:])
    elif name.startswith("model."):
        name = name[6:]
    return name

__MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__ = {
    "GPT2LMHeadModel": gpt2_dtensor_weight_loader,
    "LlamaForCausalLM": llama_dtensor_weight_loader,
    "LLaMAForCausalLM": llama_dtensor_weight_loader,
    "MistralForCausalLM": llama_dtensor_weight_loader,
    "InternLMForCausalLM": llama_dtensor_weight_loader,
    "AquilaModel": llama_dtensor_weight_loader,
    "AquilaForCausalLM": llama_dtensor_weight_loader,
    "Phi3ForCausalLM": llama_dtensor_weight_loader,
    "GemmaForCausalLM": gemma_dtensor_weight_loader,
    "Gemma2ForCausalLM": gemma_dtensor_weight_loader,
    "GPTBigCodeForCausalLM": gptbigcode_dtensor_load_weights,
    "Starcoder2ForCausalLM": starcoder2_dtensor_load_weights,
    "Qwen2ForCausalLM": qwen2_dtensor_weight_loader,
    "DeepseekV2ForCausalLM": deepseekv2_dtensor_weight_loader,
    "Qwen2VLForConditionalGeneration": qwen2vl_dtensor_weight_loader,
}

def load_dtensor_weights(actor_weights: Dict, vllm_model: nn.Module):
    weight_loader = _get_model_weight_loader(vllm_model.__class__.__name__)
    weight_loader(actor_weights, vllm_model)
    vllm_model = vllm_model.cuda()

def _get_model_weight_loader(arch: str):
    if arch in __MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__:
        return __MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__[arch]
    raise ValueError(
        f"Model architectures {arch} are not supported for now. Supported architectures: {__MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__.keys()}"
    )

def update_dtensor_weight_loader():
    pass
