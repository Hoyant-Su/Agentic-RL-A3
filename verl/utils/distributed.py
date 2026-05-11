
import os
from verl.utils.device import is_cuda_available, get_torch_device

def initialize_global_process_group(timeout_second=36000):
    from datetime import timedelta
    import torch.distributed

    torch.distributed.init_process_group(
        "nccl" if is_cuda_available else "hccl",
        timeout=timedelta(seconds=timeout_second),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.distributed.is_initialized():
        get_torch_device().set_device(local_rank)
    return local_rank, rank, world_size
