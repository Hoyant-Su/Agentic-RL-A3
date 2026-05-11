import os
import socket
from typing import Iterable, List, Optional, Set, Tuple
import torch
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoRAConfig,
    ObservabilityConfig,
    ParallelConfig,
    PromptAdapterConfig,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from .config import LoadConfig, ModelConfig

logger = init_logger(__name__)

class SPMDGPUExecutor(ExecutorBase):

    def __init__(
        self,
        model,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        speculative_config: Optional[SpeculativeConfig],
        prompt_adapter_config: Optional[PromptAdapterConfig],
        observability_config: Optional[ObservabilityConfig],
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.load_config = load_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.speculative_config = speculative_config
        self.prompt_adapter_config = prompt_adapter_config
        self.observability_config = observability_config
        distributed_init_method = initialize_cluster(parallel_config)
        self._init_executor(model, distributed_init_method)

    def _init_executor(self, model, distributed_init_method) -> None:
        assert not self.speculative_config, (
            "Speculative decoding not yet supported for multi-GPU backend."
        )
        self._init_workers_sp(model, distributed_init_method)

    def _init_workers_sp(self, model, distributed_init_method: str):
        from .worker import Worker

        rank = int(os.getenv("RANK"))
        local_rank = int(os.getenv("LOCAL_RANK"))
        print(f"local rank {local_rank}")
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        self.worker = Worker(
            model,
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
            self.cache_config,
            self.load_config,
            local_rank,
            rank,
            distributed_init_method,
            lora_config=self.lora_config,
            speculative_config=None,
            prompt_adapter_config=self.speculative_config,
            is_driver_worker=True,
            model_runner_cls=None,
        )
        self.worker.init_device()
        self.worker.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        num_blocks = self.worker.determine_num_available_blocks()
        num_gpu_blocks = num_blocks[0]
        num_cpu_blocks = num_blocks[1]
        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        logger.info(
            "# GPU blocks: %d, # CPU blocks: %d", num_gpu_blocks, num_cpu_blocks
        )
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        if torch.distributed.get_rank() == 0:
            print(
                f"before init cache memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB"
            )
        self.worker.initialize_cache(
            num_gpu_blocks=num_gpu_blocks, num_cpu_blocks=num_cpu_blocks
        )
        if torch.distributed.get_rank() == 0:
            print(
                f"after init cache memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB"
            )

    def init_cache_engine(self) -> None:
        self.worker._init_cache_engine()

    def free_cache_engine(self) -> None:
        self.worker.free_cache_engine()

    def execute_model(self, execute_model_req) -> List[SamplerOutput]:
        all_outputs = self.worker.execute_model(execute_model_req=execute_model_req)
        return all_outputs

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return self.worker.add_lora(lora_request=lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self.worker.remove_lora(lora_id=lora_id)

    def list_loras(self) -> Set[int]:
        return self.worker.list_loras()

    def check_health(self) -> None:
        return

    from vllm.prompt_adapter.request import PromptAdapterRequest

    def add_prompt_adapter(self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        assert prompt_adapter_request.prompt_adapter_id > 0, (
            "prompt_adapter_id must be greater than 0."
        )
        return self.worker.add_prompt_adapter(prompt_adapter_request)

    def list_prompt_adapters(self) -> Set[int]:
        return self.worker.list_prompt_adapters()

    def pin_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self.worker.pin_lora(lora_id)

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        assert prompt_adapter_id > 0, "prompt_adapter_id must be greater than 0."
        return self.worker.pin_prompt_adapter(prompt_adapter_id)

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        assert prompt_adapter_id > 0, "prompt_adapter_id must be greater than 0."
        return self.worker.remove_prompt_adapter(prompt_adapter_id)

    def offload_model_weights(self) -> None:
        self.worker.offload_model_weights()

    def sync_model_weights(self, actor_weights: Iterable, load_format: str) -> None:
        self.worker.sync_model_weights(
            actor_weights=actor_weights, load_format=load_format
        )

def initialize_cluster(
    parallel_config: ParallelConfig,
    engine_use_ray: bool = False,
    ray_address: Optional[str] = None,
) -> Tuple[str, Optional[None]]:
    distributed_init_method = "env://"
    return distributed_init_method

def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

class SPMDGPUExecutorAsync(SPMDGPUExecutor, ExecutorAsyncBase):
    async def execute_model_async(
        self, execute_model_req: ExecuteModelRequest
    ) -> List[SamplerOutput]:
        raise NotImplementedError

    async def check_health_async(self) -> None:
        self.check_health()
