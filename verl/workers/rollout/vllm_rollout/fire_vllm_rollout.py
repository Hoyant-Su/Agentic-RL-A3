
from contextlib import contextmanager
from typing import List
import torch
import torch.distributed
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import nn
from vllm import SamplingParams
from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.vllm_rollout.vllm_rollout import vLLMRollout

def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][
        0
    ]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

class FIREvLLMRollout(vLLMRollout):
    def __init__(
        self,
        actor_module: nn.Module,
        config: DictConfig,
        tokenizer,
        model_hf_config,
        **kwargs,
    ):
        super().__init__(actor_module, config, tokenizer, model_hf_config, **kwargs)
        self.use_fire_sampling = config.get("use_fire_sampling", False)
        if self.use_fire_sampling:
            kwargs_0 = kwargs.copy()
            kwargs_0["temperature"] = 30
            kwargs_0["max_tokens"] = 1
            if "top_k" not in kwargs_0 or kwargs_0["top_k"] <= 0:
                kwargs_0["top_k"] = 16
            self.sampling_params.max_tokens = config.response_length - 1
            for k in config.keys():
                if hasattr(SamplingParams(), str(k)):
                    kwargs_0[k] = config.get(k)
            self.sampling_params_0 = SamplingParams(**kwargs_0)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        if self.use_fire_sampling:
            old_sampling_params_args_0 = {}
            if kwargs:
                for key, value in kwargs.items():
                    if hasattr(self.sampling_params_0, key):
                        old_value = getattr(self.sampling_params_0, key)
                        old_sampling_params_args_0[key] = old_value
                        setattr(self.sampling_params_0, key, value)
        yield
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)
        if self.use_fire_sampling:
            for key, value in old_sampling_params_args_0.items():
                setattr(self.sampling_params_0, key, value)

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
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,
            }
        if not self.use_fire_sampling:
            with self.update_sampling_params(**kwargs):
                output = self.inference_engine.generate(
                    prompts=None,
                    sampling_params=self.sampling_params,
                    prompt_token_ids=idx_list,
                    use_tqdm=False,
                )
            response = output[0].to(idx.device)
        else:
            with self.update_sampling_params(**kwargs):
                output_0 = self.inference_engine.generate(
                    prompts=None,
                    sampling_params=self.sampling_params_0,
                    prompt_token_ids=idx_list,
                    use_tqdm=False,
                )
                new_idx_list = []
                for i in range(batch_size):
                    new_idx_list.append(idx_list[i] + output_0[0][i].tolist())
                output = self.inference_engine.generate(
                    prompts=None,
                    sampling_params=self.sampling_params,
                    prompt_token_ids=new_idx_list,
                    use_tqdm=False,
                )
            response = torch.cat([output_0[0], output[0]], dim=1).to(idx.device)
        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(
                response, self.config.response_length, self.pad_token_id
            )
        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
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
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
        return DataProto(batch=batch)
