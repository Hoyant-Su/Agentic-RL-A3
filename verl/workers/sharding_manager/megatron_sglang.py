
import logging
import os
import torch
from sglang.srt.entrypoints.engine import Engine
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from verl.protocol import DataProto, all_gather_data_proto
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.megatron_utils import per_tensor_generator
from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))
"""
Megatron Hybrid Engine:
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

class MegatronSGLangShardingManager(BaseShardingManager):
    def __init__(
        self,
        actor_module: nn.ModuleList,
        inference_engine: Engine,
        model_config,
        transformer_config,
        layer_name_mapping,
        weight_converter,
        device_mesh: DeviceMesh | None = None,
    ):
        self.actor_module = actor_module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.transformer_config = transformer_config
        self.layer_name_mapping = layer_name_mapping
        self.weight_converter = weight_converter
        self.device_mesh = device_mesh
        if self.device_mesh is not None:
            self.infer_tp_size = self.device_mesh["tp"].mesh.size()[0]
        else:
            self.infer_tp_size = self.inference_engine._tp_size
        self.torch_random_states = torch.cuda.get_rng_state()
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    @GPUMemoryLogger(role="MegatronSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        per_tensor_param = per_tensor_generator(
            self.actor_module,
            self.model_config,
            self.weight_converter,
            self.transformer_config,
            self.layer_name_mapping,
        )
        self.update_weights(per_tensor_param)
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="MegatronSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
        self.release_memory()
        log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)
        for model in self.actor_module:
            model.train()
        torch.cuda.empty_cache()
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def update_weights(self, params):
        if self.device_mesh["tp"].get_local_rank() == 0:
            self.inference_engine.resume_memory_occupation()
        named_tensors = params
        load_format = None
        for tensor_index, (name, tensor) in enumerate(named_tensors):
            if self.device_mesh["tp"].get_local_rank() == 0:
                self.inference_engine.update_weights_from_tensor(
                    named_tensors=[
                        (
                            name,
                            tensor.detach(),
                        )
                    ],
                    load_format=load_format,
                    flush_cache=False,
                )
            if self.device_mesh["tp"].get_local_rank() == 0:
                self.inference_engine.flush_cache()

    def release_memory(self):
        if self.device_mesh["tp"].get_local_rank() == 0:
            self.inference_engine.release_memory_occupation()

    @GPUMemoryLogger(role="megatron sglang sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        if self.infer_tp_size == 1:
            return data
        all_gather_data_proto(data, self.device_mesh["tp"].get_group())
        return data

    @GPUMemoryLogger(role="megatron sglang sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        if self.infer_tp_size == 1:
            return data
        return data.chunk(chunks=self.infer_tp_size)[
            self.device_mesh["tp"].get_local_rank()
        ]
