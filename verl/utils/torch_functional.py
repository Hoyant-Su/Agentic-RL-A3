
import math
from contextlib import contextmanager
from typing import Dict, List, Optional, Union
import torch
import torch.distributed
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import PreTrainedTokenizer

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False

def gather_from_labels(data, label):
    output = torch.gather(data, -1, label.unsqueeze(-1)).squeeze(-1)
    return output

def logprobs_from_logits(logits, labels, inplace_backward=True):
    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logits = logits.reshape(-1, last_dim)
        labels = labels.reshape(-1)
        output = logprobs_from_logits_flash_attn(
            logits, labels, inplace_backward=inplace_backward
        )
        output = output.view(*batch_dim)
    else:
        output = logprobs_from_logits_v2(logits, labels)
    return output

def logprobs_from_logits_flash_attn(logits, labels, inplace_backward=True):
    output = cross_entropy_loss(logits, labels, inplace_backward=inplace_backward)
    assert isinstance(output, tuple), (
        "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    )
    return -output[0]

def logprobs_from_logits_naive(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logpy = gather_from_labels(logp, labels)
    return logpy

def logprobs_from_logits_v2(logits: torch.FloatTensor, labels):
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(
            logits, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        logsumexp_values = torch.stack(
            [torch.logsumexp(logit, dim=-1) for logit in logits]
        )
        logprobs_labels = logits_labels - logsumexp_values
    else:
        logprobs_labels = []
        for row_logits, row_labels in zip(logits, labels):
            row_logprobs = F.log_softmax(row_logits, dim=-1)
            row_logprobs_labels = row_logprobs.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            logprobs_labels.append(row_logprobs_labels)
        logprobs_labels = torch.stack(logprobs_labels)
    return logprobs_labels

def clip_by_value(x, tensor_min, tensor_max):
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped

def entropy_from_logits(logits: torch.Tensor):
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy

def masked_sum(values, mask, axis=None):
    return (values * mask).sum(axis=axis)

def masked_mean(values, mask, axis=None):
    mask = mask.to(dtype=values.dtype)
    safe_values = torch.where(mask > 0, values, torch.zeros_like(values))
    return safe_values.sum(axis=axis) / (mask.sum(axis=axis) + 1e-8)

def masked_var(values, mask, unbiased=True):
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        if mask_sum == 1:
            raise ValueError(
                "The sum of the mask is one, which can cause a division by zero."
            )
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance

def masked_whiten(values, mask, shift_mean=True):
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

def get_response_mask(
    response_id: torch.Tensor, eos_token: Union[int, List[int]] = 2, dtype=torch.int64
):
    eos_mask = torch.isin(
        response_id, torch.tensor(eos_token, device=response_id.device)
    ).int()
    return (eos_mask.cumsum(dim=1) - eos_mask).eq(0).to(dtype)

def compute_grad_norm(model: nn.Module):
    total_grad_square = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_square += torch.sum(torch.square(param.grad.detach())).item()
    return total_grad_square

def broadcast_dict_tensor(
    tensors: Union[Dict[str, torch.Tensor], TensorDict], src, group
):
    for key in tensors.sorted_keys:
        torch.distributed.broadcast(tensors[key], src=src, group=group, async_op=False)

def allgather_dict_tensors(
    tensors: Union[Dict[str, torch.Tensor], TensorDict], size, group, dim=0
):
    if isinstance(tensors, TensorDict):
        is_tensor_dict = True
        tensors_as_dict = tensors.to_dict()
    else:
        tensors_as_dict = tensors
        is_tensor_dict = False
    output = {}
    sorted_keys = sorted(tensors_as_dict.keys())
    for key in sorted_keys:
        val = tensors_as_dict[key]
        output[key] = [torch.empty_like(val) for _ in range(size)]
        torch.distributed.all_gather(output[key], val, group=group, async_op=False)
        output[key] = torch.cat(output[key], dim=dim)
    if is_tensor_dict:
        output = TensorDict(source=output, batch_size=tensors.batch_size[0] * size)
    return output

def split_dict_tensor_into_batches(tensors: TensorDict, batch_size) -> List[TensorDict]:
    assert tensors.batch_size[0] % batch_size == 0, (
        f"input data batch size: {tensors.batch_size[0]}, split batch size: {batch_size}"
    )
    return tensors.split(batch_size)

def pad_2d_list_to_length(response, pad_token_id, max_length=None):
    response_length = max(len(sub_list) for sub_list in response)
    target_length = (
        max_length
        if max_length is not None and max_length > response_length
        else response_length
    )
    padded_response = [
        tuple(sub_list) + (pad_token_id,) * (target_length - len(sub_list))
        for sub_list in response
    ]
    tensor = torch.tensor(padded_response)
    return tensor

def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    pad_tuple = (
        (max_seq_len - tensors.shape[-1], 0)
        if left_pad
        else (0, max_seq_len - tensors.shape[-1])
    )
    return F.pad(tensors, pad_tuple, "constant", pad_token_id)

def postprocess_data(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_length: int,
    pad_token_id: int,
    left_pad=True,
    truncation="error",
):
    assert truncation in ["left", "right", "middle", "error"]
    assert input_ids.ndim == 2
    sequence_length = input_ids.shape[-1]
    if sequence_length < max_length:
        input_ids = pad_sequence_to_length(
            input_ids,
            max_seq_len=max_length,
            pad_token_id=pad_token_id,
            left_pad=left_pad,
        )
        attention_mask = pad_sequence_to_length(
            attention_mask, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad
        )
    elif sequence_length > max_length:
        if truncation == "left":
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
        elif truncation == "right":
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        elif truncation == "middle":
            left_half = max_length // 2
            right_half = max_length - left_half
            input_ids = torch.cat(
                [input_ids[:, :left_half], input_ids[:, -right_half:]], dim=-1
            )
            attention_mask = torch.cat(
                [attention_mask[:, :left_half], attention_mask[:, -right_half:]], dim=-1
            )
        elif truncation == "error":
            raise NotImplementedError(
                f"{sequence_length=} is larger than {max_length=}"
            )
        else:
            raise NotImplementedError(f"Unknown truncation method {truncation}")
    return input_ids, attention_mask

def tokenize_and_postprocess_data(
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    pad_token_id: int,
    left_pad=True,
    truncation="error",
):
    input_data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = input_data["input_ids"]
    attention_mask = input_data["attention_mask"]
    return postprocess_data(
        input_ids, attention_mask, max_length, pad_token_id, left_pad, truncation
    )

def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    no_padding_batch = []
    for ids, mask in zip(input_ids, attention_mask):
        no_padding_batch.append((ids[len(ids) - mask.sum() :]).cpu().numpy().tolist())
    return no_padding_batch

def log_probs_from_logits_response(input_ids, logits, response_length):
    response_logits = logits[:, -response_length - 1 : -1]
    response = input_ids[:, -response_length:]
    response_log_prob = logprobs_from_logits(logits=response_logits, labels=response)
    return response_log_prob

def log_probs_from_logits_response_rmpad(
    input_ids, attention_mask, logits_rmpad, response_length
):
    from flash_attn.bert_padding import pad_input, unpad_input

    batch_size, seqlen = input_ids.shape
    input_ids_rmpad, indices, *_ = unpad_input(
        input_ids.unsqueeze(-1), attention_mask=attention_mask
    )
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    full_log_probs_rmpad = logprobs_from_logits(
        logits=logits_rmpad, labels=input_ids_rmpad_rolled
    )
    full_output = pad_input(
        hidden_states=full_log_probs_rmpad.unsqueeze(-1),
        indices=indices,
        batch=batch_size,
        seqlen=seqlen,
    )
    output = full_output.squeeze(-1)[:, -response_length - 1 : -1]
    return output

def log_probs_from_logits_all_rmpad(
    input_ids_rmpad, logits_rmpad, indices, batch_size, seqlen, response_length
):
    from flash_attn.bert_padding import pad_input

    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    full_log_probs_rmpad = logprobs_from_logits(
        logits=logits_rmpad, labels=input_ids_rmpad_rolled
    )
    full_output = pad_input(
        hidden_states=full_log_probs_rmpad.unsqueeze(-1),
        indices=indices,
        batch=batch_size,
        seqlen=seqlen,
    )
    output = full_output.squeeze(-1)[:, -response_length - 1 : -1]
    return output

def post_process_logits(input_ids, logits, temperature, top_k, top_p):
    if temperature != 1.0:
        logits = logits.div_(temperature)
    return logits

"""
Optimizer related
"""

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    min_lr_ratio = 0.0 if min_lr_ratio is None else min_lr_ratio
    assert min_lr_ratio >= 0 and min_lr_ratio <= 1.0
    coef = (1 - min_lr_ratio) * 0.5
    intercept = (1 + min_lr_ratio) * 0.5

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * (
                float(current_step) / float(max(1, num_warmup_steps))
            )
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        x = math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        return max(min_lr_ratio, x * coef + intercept)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds):
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        ).to(inputs_embeds.device)
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )
    return combined_attention_mask

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )

def get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

def get_wsd_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    stable_ratio: float = 0.9,
):
    remaining_steps = max(0, num_training_steps - num_warmup_steps)
    num_stable_steps = int(remaining_steps * stable_ratio)
    num_decay_steps = remaining_steps - num_stable_steps

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step < num_warmup_steps + num_stable_steps:
            return 1.0
        if current_step < num_training_steps:
            progress = float(
                current_step - num_warmup_steps - num_stable_steps
            ) / float(max(1, num_decay_steps))
            value = max(
                0.0,
                0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
            )
            return (1.0 - min_lr_ratio) * value + min_lr_ratio
        return min_lr_ratio

    return LambdaLR(optimizer, lr_lambda, last_epoch)

@contextmanager
def check_cuda_is_available():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be initialized before importing this module.")
    yield

def distributed_mean_max_min_std(
    local_tensor, compute_max=True, compute_min=True, compute_std=True
):
    local_sum = torch.sum(local_tensor)
    local_num = torch.tensor(torch.numel(local_tensor), device="cuda")
    torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(local_num, op=torch.distributed.ReduceOp.SUM)
    global_mean = local_sum / local_num
    if compute_max:
        local_max = torch.max(local_tensor)
        torch.distributed.all_reduce(local_max, op=torch.distributed.ReduceOp.MAX)
    else:
        local_max = None
    if compute_min:
        local_min = torch.min(local_tensor)
        torch.distributed.all_reduce(local_min, op=torch.distributed.ReduceOp.MIN)
    else:
        local_min = None
    if compute_std:
        square_diff = torch.sum(torch.pow(local_tensor - global_mean, 2))
        torch.distributed.all_reduce(square_diff, op=torch.distributed.ReduceOp.SUM)
        global_std = torch.sqrt(square_diff / (local_num - 1))
    else:
        global_std = None
    return global_mean, local_max, local_min, global_std

def distributed_masked_mean(local_tensor, local_mask):
    local_tensor = local_tensor * local_mask
    local_sum = torch.sum(local_tensor)
    local_num = torch.sum(local_mask)
    torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(local_num, op=torch.distributed.ReduceOp.SUM)
    global_mean = local_sum / local_num
    return global_mean
