
import logging
import os
from contextlib import contextmanager
from copy import deepcopy
from typing import List
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch import nn
from vllm import SamplingParams
from verl import DataProto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
from vllm.lora.request import LoRARequest

def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][
        0
    ]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

class vLLMRollout(BaseRollout):
    def __init__(
        self,
        actor_module: nn.Module,
        config: DictConfig,
        tokenizer,
        model_hf_config,
        **kwargs,
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
        max_num_batched_tokens = int(self.config.get("max_num_batched_tokens", 8192))
        if kwargs.get("train_tp") is not None:
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp")
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                vllm_ps.initialize_parallel_state(
                    tensor_model_parallel_size=tensor_parallel_size,
                    num_tp_per_train_tp=num_tp_per_train_tp,
                )
        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            assert (
                model_hf_config.max_position_embeddings
                >= config.prompt_length + config.response_length
            ), "model context length should be greater than total sequence length"
        max_model_len = (
            self.config.max_model_len
            if self.config.max_model_len
            else config.prompt_length + config.response_length
        )
        max_model_len = int(max_model_len)
        if (
            max_num_batched_tokens < max_model_len
            and self.config.enable_chunked_prefill
        ):
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        engine_kwargs = {
            key: val for key, val in engine_kwargs.items() if val is not None
        }
        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            **lora_kwargs,
            **engine_kwargs,
        )
        self.inference_engine.offload_model_weights()
        kwargs = dict(
            n=1,
            logprobs=0,
            max_tokens=config.response_length,
        )
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
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
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)
        idx_list = []
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))
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
            output = self.inference_engine.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                lora_request=lora_requests,
                use_tqdm=False,
            )
            response = output[0].to(idx.device)
            log_probs = output[1].to(idx.device)
            if response.shape[1] < self.config.response_length:
                response = pad_sequence_to_length(
                    response, self.config.response_length, self.pad_token_id
                )
                log_probs = pad_sequence_to_length(
                    log_probs, self.config.response_length, self.pad_token_id
                )
            if self.sampling_params.n > 1 and do_sample:
                idx = idx.repeat_interleave(self.sampling_params.n, dim=0)
                attention_mask = attention_mask.repeat_interleave(
                    self.sampling_params.n, dim=0
                )
                position_ids = position_ids.repeat_interleave(
                    self.sampling_params.n, dim=0
                )
                batch_size = batch_size * self.sampling_params.n
            seq = torch.cat([idx, response], dim=-1)
        response_length = response.size(1)
        delta_position_id = torch.arange(
            1, response_length + 1, device=position_ids.device
        )
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
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
                "rollout_log_probs": log_probs,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
        return DataProto(batch=batch)
