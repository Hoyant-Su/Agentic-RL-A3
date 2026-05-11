import enum
import json
from dataclasses import dataclass, field
from typing import List, Optional, Union
import torch
from transformers import PretrainedConfig
from vllm.config import (
    ModelConfig,
    MultiModalConfig,
    _get_and_verify_dtype,
    _get_and_verify_max_len,
    get_served_model_name,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.model_loader import BaseModelLoader
from vllm.transformers_utils.config import get_hf_text_config
from vllm.utils import is_hip, print_warning_once

GPTQMarlinConfig = get_quantization_config("gptq_marlin")
logger = init_logger(__name__)
_GB = 1 << 30

class ModelConfig(ModelConfig):

    def __init__(
        self,
        hf_config: PretrainedConfig,
        tokenizer_mode: str,
        trust_remote_code: bool,
        dtype: Union[str, torch.dtype],
        seed: int,
        revision: Optional[str] = None,
        code_revision: Optional[str] = None,
        rope_scaling: Optional[dict] = None,
        rope_theta: Optional[float] = None,
        tokenizer_revision: Optional[str] = None,
        max_model_len: Optional[int] = None,
        quantization: Optional[str] = None,
        quantization_param_path: Optional[str] = None,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: Optional[int] = None,
        max_logprobs: int = 20,
        disable_sliding_window: bool = False,
        skip_tokenizer_init: bool = False,
        served_model_name: Optional[Union[str, List[str]]] = None,
        multimodal_config: Optional[MultiModalConfig] = None,
    ) -> None:
        self.model = hf_config._name_or_path
        self.tokenizer = hf_config._name_or_path
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.seed = seed
        self.revision = revision
        self.code_revision = code_revision
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        if tokenizer_revision is None:
            self.tokenizer_revision = revision
        else:
            self.tokenizer_revision = tokenizer_revision
        self.quantization = quantization
        self.quantization_param_path = quantization_param_path
        self.enforce_eager = enforce_eager
        if max_context_len_to_capture is not None:
            raise ValueError(
                "`max_context_len_to_capture` is deprecated. Use `max_seq_len_to_capture` instead."
            )
        self.max_seq_len_to_capture = max_seq_len_to_capture
        self.max_logprobs = max_logprobs
        self.disable_sliding_window = disable_sliding_window
        self.skip_tokenizer_init = skip_tokenizer_init
        self.hf_config = hf_config
        self.hf_text_config = get_hf_text_config(hf_config)
        self.dtype = _get_and_verify_dtype(self.hf_text_config, dtype)
        if (
            not self.disable_sliding_window
            and self.hf_text_config.model_type == "gemma2"
            and self.hf_text_config.sliding_window is not None
        ):
            print_warning_once(
                f"Gemma 2 uses sliding window attention for every odd layer, which is currently not supported by vLLM. Disabling sliding window and capping the max length to the sliding window size ({self.hf_text_config.sliding_window})."
            )
            self.disable_sliding_window = True
        self.max_model_len = _get_and_verify_max_len(
            hf_config=self.hf_text_config,
            max_model_len=max_model_len,
            disable_sliding_window=self.disable_sliding_window,
            sliding_window_len=self.get_hf_config_sliding_window(),
        )
        self.served_model_name = get_served_model_name(
            self.model,
            served_model_name,
        )
        self.multimodal_config = multimodal_config
        if not self.skip_tokenizer_init:
            self._verify_tokenizer_mode()
        self._verify_embedding_mode()
        self._verify_quantization()
        self._verify_cuda_graph()

class LoadFormat(str, enum.Enum):
    AUTO = "auto"
    MEGATRON = "megatron"
    HF = "hf"
    DTENSOR = "dtensor"
    DUMMY_HF = "dummy_hf"
    DUMMY_MEGATRON = "dummy_megatron"
    DUMMY_DTENSOR = "dummy_dtensor"

@dataclass
class LoadConfig:

    load_format: Union[str, LoadFormat, BaseModelLoader] = LoadFormat.AUTO
    download_dir: Optional[str] = None
    model_loader_extra_config: Optional[Union[str, dict]] = field(default_factory=dict)
    ignore_patterns: Optional[Union[List[str], str]] = None

    def __post_init__(self):
        model_loader_extra_config = self.model_loader_extra_config or {}
        if isinstance(model_loader_extra_config, str):
            self.model_loader_extra_config = json.loads(model_loader_extra_config)
        self._verify_load_format()
        if self.ignore_patterns is not None and len(self.ignore_patterns) > 0:
            logger.info(
                "Ignoring the following patterns when downloading weights: %s",
                self.ignore_patterns,
            )
        else:
            self.ignore_patterns = ["original/**/*"]

    def _verify_load_format(self) -> None:
        if not isinstance(self.load_format, str):
            return
        load_format = self.load_format.lower()
        self.load_format = LoadFormat(load_format)
        rocm_not_supported_load_format: List[str] = []
        if is_hip() and load_format in rocm_not_supported_load_format:
            rocm_supported_load_format = [
                f
                for f in LoadFormat.__members__
                if (f not in rocm_not_supported_load_format)
            ]
            raise ValueError(
                f"load format '{load_format}' is not supported in ROCm. Supported load formats are {rocm_supported_load_format}"
            )
