
from typing import Dict, List, Optional
import torch
from torch import nn

class MemoryBuffer:

    def __init__(
        self,
        numel: int,
        numel_padded: int,
        dtype: torch.dtype,
        source: Optional[torch.Tensor] = None,
    ):
        self.numel = numel
        self.numel_padded = numel_padded
        self.dtype = dtype
        if source is not None:
            self.data = source
        else:
            self.data = torch.zeros(
                self.numel_padded, dtype=self.dtype, device="cuda", requires_grad=False
            )

    def zero(self):
        self.data.zero_()

    def get(self, shape, start_index):
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, "requested tensor is out of the buffer range."
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor

def calc_padded_numel(shape: torch.Size, dtype: torch.dtype):
    align_numel = 128 // torch.finfo(dtype).bits
    numel = shape.numel()
    return (numel + align_numel - 1) // align_numel * align_numel

def get_weight_buffer_meta_from_module(module: nn.Module) -> Dict[str, Dict]:
    weight_buffer_meta = {}
    for name, param in sorted(module.named_parameters()):
        weight_buffer_meta[name] = {"shape": param.shape, "dtype": param.dtype}
    return weight_buffer_meta

def build_memory_buffer(
    weight_buffer_meta: Dict[str, Dict],
) -> Dict[torch.dtype, MemoryBuffer]:
    memory_buffers = {}
    total_numel_map = {}
    for name, meta_info in sorted(weight_buffer_meta.items()):
        shape = meta_info["shape"]
        dtype = meta_info["dtype"]
        assert isinstance(shape, torch.Size)
        assert isinstance(dtype, torch.dtype)
        if dtype not in total_numel_map:
            total_numel_map[dtype] = 0
        total_numel_map[dtype] += calc_padded_numel(shape, dtype)
    for dtype, total_numel in total_numel_map.items():
        memory_buffers[dtype] = MemoryBuffer(total_numel, total_numel, dtype)
    return memory_buffers

def build_memory_reference_from_module(
    module: torch.nn.Module,
    memory_buffers: Dict[torch.dtype, MemoryBuffer],
    maintain_weight=True,
):
    start_index = {}
    for dtype in memory_buffers:
        start_index[dtype] = 0
    for name, param in sorted(module.named_parameters()):
        memory_buffer = memory_buffers[param.dtype]
        buffer = memory_buffer.get(
            shape=param.shape, start_index=start_index[param.dtype]
        )
        start_index[param.dtype] += calc_padded_numel(param.shape, param.dtype)
        if maintain_weight:
            buffer.copy_(param.data)
        param.data = buffer

def build_memory_reference(
    weight_buffer_meta: Dict[str, Dict], memory_buffers: Dict[torch.dtype, MemoryBuffer]
):
    start_idx = {}
    weight_buffers = {}
    for dtype in memory_buffers:
        start_idx[dtype] = 0
    for name, meta_info in sorted(weight_buffer_meta.items()):
        shape = meta_info["shape"]
        dtype = meta_info["dtype"]
        buffer = memory_buffers[dtype].get(shape, start_index=start_idx[dtype])
        start_idx[dtype] += calc_padded_numel(shape, dtype)
        weight_buffers[name] = buffer
    return weight_buffers

class MemoryBufferModuleWrapper:

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.weight_buffer_meta = get_weight_buffer_meta_from_module(self.module)
        self.memory_buffers = build_memory_buffer(self.weight_buffer_meta)
        build_memory_reference_from_module(self.module, self.memory_buffers)

    def get_memory_buffers(self):
        return self.memory_buffers

    def get_weight_buffer_meta(self):
        return self.weight_buffer_meta

class MegatronMemoryBufferForRollout:

    def __init__(self, transform_memory_param_fn):
        self._memory_buffers = []
        self._weight_buffers = []
        self._named_parameters = {}
        self.transform_memory_param_fn = transform_memory_param_fn

    def initialize_weight_buffer(self, weight_buffer_meta_pp: List[Dict[str, Dict]]):
        self.weight_buffer_meta_pp = weight_buffer_meta_pp
        for weight_buffer_meta in self.weight_buffer_meta_pp:
            memory_buffer = build_memory_buffer(weight_buffer_meta)
            self._memory_buffers.append(memory_buffer)
            self._weight_buffers.append(None)

    def build_memory_reference(self):
        for i, weight_buffer_meta in enumerate(self.weight_buffer_meta_pp):
            self._weight_buffers[i] = build_memory_reference(
                weight_buffer_meta, self._memory_buffers[i]
            )
        self._named_parameters = self.transform_memory_param_fn(self._weight_buffers)

    @property
    def named_parameters(self):
        return self._named_parameters

    @property
    def weight_buffers(self):
        return self._weight_buffers

    @property
    def memory_buffers(self):
        return self._memory_buffers
