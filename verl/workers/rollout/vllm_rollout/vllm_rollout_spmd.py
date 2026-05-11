
import logging
import os
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Union
import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf, ListConfig
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.config import CompilationConfig, LoRAConfig
from vllm.lora.request import LoRARequest

try:
    from vllm.config import CompilationMode

    _use_compilation_mode = True
except ImportError:
    from vllm.config import CompilationLevel

    _use_compilation_mode = False
try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ModuleNotFoundError:
    from vllm.v1.worker.worker_base import WorkerWrapperBase
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][
        0
    ]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

def _repeat_interleave(
    value: Union[torch.Tensor, np.ndarray], repeats: int
) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

class vLLMRollout(BaseRollout):
    def __init__(
        self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs
    ):
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), (
            "disable CUDA graph (enforce_eager = False) if free cache engine"
        )
        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)
        if kwargs.get("train_tp") is not None:
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(
                    tensor_model_parallel_size=tensor_parallel_size,
                    num_tp_per_train_tp=num_tp_per_train_tp,
                )
            else:
                vllm_ps.initialize_model_parallel(
                    tensor_model_parallel_size=tensor_parallel_size
                )
        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = (
                    model_hf_config.llm_config.max_position_embeddings
                )
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = (
                    model_hf_config.text_config.max_position_embeddings
                )
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert (
                max_position_embeddings >= config.prompt_length + config.response_length
            ), "model context length should be greater than total sequence length"
        max_model_len = int(
            config.max_model_len or config.prompt_length + config.response_length
        )
        if (
            max_num_batched_tokens < max_model_len
            and self.config.enable_chunked_prefill
        ):
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )
        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = (
            "dummy" if config.load_format.startswith("dummy") else config.load_format
        )
        limit_mm_per_prompt = None
        if config.get("limit_images", None):
            limit_mm_per_prompt = {"image": config.get("limit_images")}
        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        engine_kwargs = {
            key: val for key, val in engine_kwargs.items() if val is not None
        }
        if config.get("limit_images", None):
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}
        compilation_config = {}
        cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
        if not config.enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, ListConfig):
                compilation_args = {"cudagraph_capture_sizes": cudagraph_capture_sizes}
                if _use_compilation_mode:
                    compilation_args["mode"] = CompilationMode.VLLM_COMPILE
                else:
                    compilation_args["level"] = CompilationLevel.PIECEWISE
                compilation_config["compilation_config"] = CompilationConfig(
                    **compilation_args
                )
            else:
                logger.warning(
                    f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}"
                )
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=config.max_num_seqs,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **compilation_config,
            **self.lora_kwargs,
            **engine_kwargs,
        )
        self.inference_engine.sleep(level=1)
        kwargs = dict(
            n=1,
            logprobs=0,
            max_tokens=config.response_length,
        )
        kwargs["detokenize"] = False
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [
                    _pre_process_inputs(self.pad_token_id, idx[i])
                    for i in range(batch_size)
                ],
                dtype=object,
            )
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")
        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"),
                non_tensor_batch.pop("multi_modal_data"),
            ):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids,
                        "multi_modal_data": multi_modal_data,
                    }
                )
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids}
                for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,
            }
        elif is_validate:
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,
            }
        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(
                        lora_name=f"{lora_int_id}",
                        lora_int_id=lora_int_id,
                        lora_path="/simon-stub-path",
                    )
                ] * batch_size
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )
            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    curr_log_prob = []
                    for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                        curr_log_prob.append(logprob[response_ids[i]].logprob)
                    rollout_log_probs.append(curr_log_prob)
            response = pad_2d_list_to_length(
                response, self.pad_token_id, max_length=self.config.response_length
            ).to(idx.device)
            rollout_log_probs = pad_2d_list_to_length(
                rollout_log_probs, -1, max_length=self.config.response_length
            ).to(idx.device)
            rollout_log_probs = rollout_log_probs.to(torch.float32)
            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(
                    attention_mask, self.sampling_params.n
                )
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(
                        non_tensor_batch["tools_kwargs"], self.sampling_params.n
                    )
            seq = torch.cat([idx, response], dim=-1)
        response_length = response.size(1)
        delta_position_id = torch.arange(
            1, response_length + 1, device=position_ids.device
        )
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
                batch_size, position_ids.size(1), -1
            )
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "rollout_log_probs": rollout_log_probs,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

class vLLMAsyncRollout:

    def __init__(self, *args, **kwargs):
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0
        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
