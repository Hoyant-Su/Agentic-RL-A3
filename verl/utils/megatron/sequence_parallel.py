import torch
import torch.nn.functional as F
from megatron.core import parallel_state as mpu

def mark_parameter_as_sequence_parallel(parameter):
    parameter.sequence_parallel = True

def is_sequence_parallel_param(param):
    return hasattr(param, "sequence_parallel") and param.sequence_parallel

def pad_to_sequence_parallel(unpad_tokens: torch.Tensor):
    total_nnz = unpad_tokens.shape[0]
    sp_world_size = mpu.get_tensor_model_parallel_world_size()
    pad_size = (
        0
        if total_nnz % sp_world_size == 0
        else sp_world_size - total_nnz % sp_world_size
    )
    if pad_size > 0:
        if unpad_tokens.ndim == 1:
            unpad_tokens = F.pad(unpad_tokens, (0, pad_size))
        elif unpad_tokens.ndim == 2:
            unpad_tokens = F.pad(unpad_tokens, (0, 0, 0, pad_size))
        else:
            raise NotImplementedError(
                f"Padding dim {unpad_tokens.ndim()} is not supported"
            )
    return unpad_tokens
