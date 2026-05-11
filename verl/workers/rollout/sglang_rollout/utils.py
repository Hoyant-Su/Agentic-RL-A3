import pickle
from typing import Any, List, Optional
import numpy as np
import torch
import torch.distributed as dist

def broadcast_pyobj(
    data: List[Any],
    rank: int,
    dist_group: Optional[torch.distributed.ProcessGroup] = None,
    src: int = 0,
    force_cpu_device: bool = False,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not force_cpu_device else "cpu"
    )
    if rank == src:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long, device=device)
            dist.broadcast(tensor_size, src=src, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)
            tensor_data = torch.ByteTensor(
                np.frombuffer(serialized_data, dtype=np.uint8)
            ).to(device)
            tensor_size = torch.tensor([size], dtype=torch.long, device=device)
            dist.broadcast(tensor_size, src=src, group=dist_group)
            dist.broadcast(tensor_data, src=src, group=dist_group)
        return data
    else:
        tensor_size = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(tensor_size, src=src, group=dist_group)
        size = tensor_size.item()
        if size == 0:
            return []
        tensor_data = torch.empty(size, dtype=torch.uint8, device=device)
        dist.broadcast(tensor_data, src=src, group=dist_group)
        serialized_data = bytes(tensor_data.cpu().numpy())
        data = pickle.loads(serialized_data)
        return data
