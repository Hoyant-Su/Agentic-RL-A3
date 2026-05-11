import importlib
from typing import List, Optional, Type
import torch.nn as nn

_MODELS = {
    "LlamaForCausalLM": (
        "llama",
        (
            "ParallelLlamaForCausalLMRmPadPP",
            "ParallelLlamaForValueRmPadPP",
            "ParallelLlamaForCausalLMRmPad",
        ),
    ),
    "Qwen2ForCausalLM": (
        "qwen2",
        (
            "ParallelQwen2ForCausalLMRmPadPP",
            "ParallelQwen2ForValueRmPadPP",
            "ParallelQwen2ForCausalLMRmPad",
        ),
    ),
    "MistralForCausalLM": (
        "mistral",
        (
            "ParallelMistralForCausalLMRmPadPP",
            "ParallelMistralForValueRmPadPP",
            "ParallelMistralForCausalLMRmPad",
        ),
    ),
}

class ModelRegistry:
    @staticmethod
    def load_model_cls(model_arch: str, value=False) -> Optional[Type[nn.Module]]:
        if model_arch not in _MODELS:
            return None
        megatron = "megatron"
        module_name, model_cls_name = _MODELS[model_arch]
        if not value:
            model_cls_name = model_cls_name[0]
        elif value:
            model_cls_name = model_cls_name[1]
        module = importlib.import_module(
            f"verl.models.{module_name}.{megatron}.modeling_{module_name}_megatron"
        )
        return getattr(module, model_cls_name, None)

    @staticmethod
    def get_supported_archs() -> List[str]:
        return list(_MODELS.keys())
