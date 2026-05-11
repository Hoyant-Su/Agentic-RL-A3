import logging
import torch

logger = logging.getLogger(__name__)

def is_torch_npu_available() -> bool:
    try:
        import torch_npu

        return torch.npu.is_available()
    except ImportError:
        return False

is_cuda_available = torch.cuda.is_available()
is_npu_available = is_torch_npu_available()

def get_visible_devices_keyword() -> str:
    return "CUDA_VISIBLE_DEVICES" if is_cuda_available else "ASCEND_RT_VISIBLE_DEVICES"

def get_device_name() -> str:
    if is_cuda_available:
        device = "cuda"
    elif is_npu_available:
        device = "npu"
    else:
        device = "cpu"
    return device

def get_torch_device() -> any:
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(
            f"Device namespace '{device_name}' not found in torch, try to load torch.cuda."
        )
        return torch.cuda

def get_device_id() -> int:
    return get_torch_device().current_device()

def get_nccl_backend() -> str:
    if is_cuda_available:
        return "nccl"
    elif is_npu_available:
        return "hccl"
    else:
        raise RuntimeError(
            f"No available nccl backend found on device type {get_device_name()}."
        )

def set_expandable_segments(enable: bool) -> None:
    if is_cuda_available:
        torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")
