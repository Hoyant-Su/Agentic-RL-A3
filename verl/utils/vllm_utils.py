SUPPORTED_MOE_MODELS = []
try:
    from vllm.model_executor.models.deepseek_v2 import (
        DeepseekV2ForCausalLM,
        DeepseekV3ForCausalLM,
    )

    SUPPORTED_MOE_MODELS.append(DeepseekV2ForCausalLM)
    SUPPORTED_MOE_MODELS.append(DeepseekV3ForCausalLM)
except ImportError:
    pass
try:
    from vllm.model_executor.models.mixtral import MixtralForCausalLM

    SUPPORTED_MOE_MODELS.append(MixtralForCausalLM)
except ImportError:
    pass
try:
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen2MoeForCausalLM)
except ImportError:
    pass
try:
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeForCausalLM)
except ImportError:
    pass
try:
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeLLMForCausalLM)
except ImportError:
    pass
try:
    from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3NextForCausalLM)
except ImportError:
    pass
try:
    from vllm.model_executor.models.kimi_vl import KimiVLForConditionalGeneration

    SUPPORTED_MOE_MODELS.append(KimiVLForConditionalGeneration)
except ImportError:
    pass
from typing import List
from msgspec import field
from packaging import version as vs
from vllm.lora.models import LoRAModel
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from verl.third_party.vllm import get_version

def patch_vllm_moe_model_weight_loader(model):
    if not SUPPORTED_MOE_MODELS:
        return
    original_model_type = type(model)
    if hasattr(model, "runnable") and "ACLGraphWrapper" in str(original_model_type):
        model = model.runnable
        original_model_type = type(model)
    MLP_ATTR_MAPPING = {}
    try:
        from vllm.model_executor.models.mixtral import MixtralForCausalLM

        MLP_ATTR_MAPPING[MixtralForCausalLM] = "block_sparse_moe"
    except ImportError:
        pass
    DEFAULT_MLP_ATTR = "mlp"
    inner_model = getattr(model, "model", None) or getattr(
        model, "language_model", None
    )
    if inner_model is None:
        raise ValueError(
            "The provided model does not have a valid 'model' or 'language_model' attribute."
        )
    if not isinstance(model, tuple(SUPPORTED_MOE_MODELS)) and not isinstance(
        inner_model, tuple(SUPPORTED_MOE_MODELS)
    ):
        return
    if type(inner_model).__name__ == "Qwen3MoeLLMForCausalLM":
        inner_model = inner_model.model
    for layer_idx, layer in enumerate(inner_model.layers):
        mlp_attr = MLP_ATTR_MAPPING.get(original_model_type, DEFAULT_MLP_ATTR)
        mlp = getattr(layer, mlp_attr, None)
        if not mlp:
            continue
        experts = getattr(mlp, "experts", None)
        if not experts or not hasattr(experts, "weight_loader"):
            continue
        for name, param in mlp.named_parameters():
            if "w13_weight" in name or "w2_weight" in name:
                param.weight_loader = experts.weight_loader

class TensorLoRARequest(LoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)

class VLLMHijack:
    @staticmethod
    def hijack():
        def hijack__load_adapter(self, lora_request: TensorLoRARequest) -> LoRAModel:
            try:
                supported_lora_modules = self._adapter_manager.supported_lora_modules
                packed_modules_mapping = self._adapter_manager.packed_modules_mapping
                expected_lora_modules: List[str] = []
                for module in supported_lora_modules:
                    if module in packed_modules_mapping:
                        expected_lora_modules.extend(packed_modules_mapping[module])
                    else:
                        expected_lora_modules.append(module)
                expected_lora_modules = list(set(expected_lora_modules))
                lora_tensors = None
                from vllm.lora.peft_helper import PEFTHelper

                if isinstance(lora_request, TensorLoRARequest):
                    peft_config = lora_request.peft_config
                    lora_tensors = lora_request.lora_tensors
                    peft_helper = PEFTHelper.from_dict(peft_config)
                else:
                    lora_path = get_adapter_absolute_path(lora_request.lora_path)
                    peft_helper = PEFTHelper.from_local_dir(
                        lora_path, self.max_position_embeddings
                    )
                peft_helper.validate_legal(self.lora_config)
                model = self._adapter_manager.model
                hf_to_vllm_mapper = None
                if (
                    hasattr(model, "hf_to_vllm_mapper")
                    and model.hf_to_vllm_mapper is not None
                ):
                    hf_to_vllm_mapper = model.hf_to_vllm_mapper
                if isinstance(lora_request, TensorLoRARequest):
                    lora = self._lora_model_cls.from_lora_tensors(
                        lora_model_id=lora_request.lora_int_id,
                        tensors=lora_tensors,
                        peft_helper=peft_helper,
                        device="cpu",
                        dtype=self.lora_config.lora_dtype,
                        embeddings=None,
                        target_embedding_padding=self.vocab_size
                        + self.lora_config.lora_extra_vocab_size,
                        embedding_modules=self.embedding_modules,
                        embedding_padding_modules=self.embedding_padding_modules,
                        weights_mapper=hf_to_vllm_mapper,
                    )
                else:
                    lora = self._lora_model_cls.from_local_checkpoint(
                        lora_path,
                        expected_lora_modules,
                        peft_helper=peft_helper,
                        lora_model_id=lora_request.lora_int_id,
                        device="cpu",
                        dtype=self.lora_config.lora_dtype,
                        target_embedding_padding=self.vocab_size
                        + self.lora_config.lora_extra_vocab_size,
                        embedding_modules=self.embedding_modules,
                        embedding_padding_modules=self.embedding_padding_modules,
                        weights_mapper=hf_to_vllm_mapper,
                    )
            except Exception as e:
                raise e
            if lora.extra_vocab_size > self.lora_config.lora_extra_vocab_size:
                raise ValueError(
                    f"LoRA added vocab size {lora.extra_vocab_size} "
                    f"is greater than lora_extra_vocab_size "
                    f"{self.lora_config.lora_extra_vocab_size}."
                )
            return lora

        def do_hijack(target_cls, target_method_name, hooking_method):
            setattr(target_cls, target_method_name, hooking_method)

        do_hijack(LRUCacheWorkerLoRAManager, "_load_adapter", hijack__load_adapter)

def is_version_ge(pkg: str = "vllm", minver: str = "0.7.3"):
    return vs.parse(get_version(pkg)) >= vs.parse(minver)
