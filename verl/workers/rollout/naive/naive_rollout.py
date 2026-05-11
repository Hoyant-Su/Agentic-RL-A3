
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from verl import DataProto
from verl.utils.torch_functional import logprobs_from_logits
from ..base import BaseRollout

__all__ = ["NaiveRollout"]

class NaiveRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)
        prompt_length = idx.size(1)
        self.module.eval()
        prev_attention_mask = torch.ones(
            size=(batch_size, 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        logits_lst = []
        for _ in range(self.config.response_length):
            idx_cond = idx
            output = self.module(
                input_ids=idx_cond,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            logits = output.logits
            logits = logits[:, -1, :] / self.config.temperature
            if self.config.top_k is not None:
                v, _ = torch.topk(logits, min(self.config.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            if self.config.do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            attention_mask = torch.cat((attention_mask, prev_attention_mask), dim=-1)
            for token_id in eos_token_id:
                prev_attention_mask = torch.logical_and(
                    idx_next != token_id, prev_attention_mask.bool()
                )
            prev_attention_mask.to(attention_mask.dtype)
            position_ids = torch.cat((position_ids, position_ids[:, -1:] + 1), dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)
            logits_lst.append(logits)
        logits = torch.stack(logits_lst, dim=1)
        prompts = idx[:, :prompt_length]
        response = idx[:, prompt_length:]
        log_probs = logprobs_from_logits(logits=logits, labels=response)
        batch = TensorDict(
            {
                "input_ids": prompts,
                "responses": response,
                "sequences": idx,
                "old_log_probs": log_probs,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        self.module.train()
        return DataProto(batch=batch)
