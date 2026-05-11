from typing import Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import PretrainedConfig, PreTrainedTokenizer, PreTrainedTokenizerFast
from vllm import LLM
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.utils import Counter
from verl.workers.rollout.tokenizer import HybridEngineBaseTokenizer
from .arg_utils import EngineArgs
from .llm_engine_sp import LLMEngine

class LLM(LLM):

    def __init__(
        self,
        model: Union[nn.Module, Dict],
        tokenizer: Union[
            PreTrainedTokenizer, PreTrainedTokenizerFast, HybridEngineBaseTokenizer
        ],
        model_hf_config: PretrainedConfig,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        skip_tokenizer_init: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        load_format="auto",
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        removed_vision_keys = (
            "image_token_id",
            "image_feature_size",
            "image_input_shape",
            "image_input_type",
        )
        if any(k in kwargs for k in removed_vision_keys):
            raise TypeError(
                "There is no need to pass vision-related arguments anymore."
            )
        engine_args = EngineArgs(
            model_hf_config=model_hf_config,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            load_format=load_format,
            **kwargs,
        )
        tokenizer_cls = (
            PreTrainedTokenizer,
            PreTrainedTokenizerFast,
            HybridEngineBaseTokenizer,
        )
        if not isinstance(tokenizer, tokenizer_cls):
            raise ValueError(
                f"Unexpected tokenizer type: {type(tokenizer)}. Must beone of the following: PreTrainedTokenizer, PreTrainedTokenizerFast, verl.workers.rollout.HybridEngineBaseTokenizer"
            )
        self.llm_engine = LLMEngine.from_engine_args(model, tokenizer, engine_args)
        self.request_counter = Counter()

    def init_cache_engine(self):
        self.llm_engine.init_cache_engine()

    def free_cache_engine(self):
        self.llm_engine.free_cache_engine()

    def get_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer

    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.llm_engine.tokenizer = tokenizer

    def _run_engine(
        self, *, use_tqdm: bool
    ) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        outputs = super()._run_engine(use_tqdm=use_tqdm)
        return self._post_process_outputs(outputs)

    def _post_process_outputs(
        self, request_outputs: List[RequestOutput]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_token_ids = []
        logprobs = []
        for request_output in request_outputs:
            outputs = request_output.outputs
            for output in outputs:
                output_token_ids.append(torch.tensor(output.token_ids))
                logprobs_dicts = output.logprobs
                if logprobs_dicts is not None:
                    logprob = []
                    for logprobs_dict, id in zip(logprobs_dicts, output.token_ids):
                        logprob.append(logprobs_dict[id].logprob)
                    logprobs.append(torch.tensor(logprob))
        pad_token_id = (
            self.llm_engine.tokenizer.pad_token_id
            if self.llm_engine.tokenizer.pad_token_id is not None
            else self.llm_engine.tokenizer.eos_token_id
        )
        output_token_ids = pad_sequence(
            output_token_ids, batch_first=True, padding_value=pad_token_id
        )
        if len(logprobs) > 0:
            logprobs = pad_sequence(
                logprobs, batch_first=True, padding_value=pad_token_id
            )
        return output_token_ids, logprobs

    def sync_model_weights(self, actor_weights: Iterable, load_format: str) -> None:
        self.llm_engine.sync_model_weights(
            actor_weights=actor_weights, load_format=load_format
        )

    def offload_model_weights(self) -> None:
        self.llm_engine.offload_model_weights()
