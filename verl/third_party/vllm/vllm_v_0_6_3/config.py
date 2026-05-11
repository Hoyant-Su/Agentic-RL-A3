import enum
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union
from transformers import PretrainedConfig
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.utils import is_hip

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.loader import BaseModelLoader
logger = init_logger(__name__)

class LoadFormat(str, enum.Enum):
    AUTO = "auto"
    MEGATRON = "megatron"
    HF = "hf"
    DTENSOR = "dtensor"
    DUMMY_HF = "dummy_hf"
    DUMMY_MEGATRON = "dummy_megatron"
    DUMMY_DTENSOR = "dummy_dtensor"

class ModelConfig(ModelConfig):
    def __init__(self, hf_config: PretrainedConfig, *args, **kwargs) -> None:
        super().__init__(
            model=hf_config._name_or_path,
            tokenizer=hf_config._name_or_path,
            *args,
            **kwargs,
        )
        self.hf_config = hf_config

@dataclass
class LoadConfig:

    load_format: Union[str, LoadFormat, "BaseModelLoader"] = LoadFormat.AUTO
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
