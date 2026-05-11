import logging
from typing import Tuple
import logging
from typing import Tuple
import datetime
import inspect
from typing import Any
import torch
import torch.distributed as dist
from verl.utils.logger.aggregate_logger import DecoratorLoggerBase
from verl.utils.device import get_torch_device

def _get_current_mem_info(unit: str = "GB", precision: int = 2) -> Tuple[str]:
    assert unit in ["GB", "MB", "KB"]
    divisor = 1024**3 if unit == "GB" else 1024**2 if unit == "MB" else 1024
    mem_allocated = get_torch_device().memory_allocated()
    mem_reserved = get_torch_device().memory_reserved()
    mem_free, mem_total = get_torch_device().mem_get_info()
    mem_used = mem_total - mem_free
    mem_allocated = f"{mem_allocated / divisor:.{precision}f}"
    mem_reserved = f"{mem_reserved / divisor:.{precision}f}"
    mem_used = f"{mem_used / divisor:.{precision}f}"
    mem_total = f"{mem_total / divisor:.{precision}f}"
    return mem_allocated, mem_reserved, mem_used, mem_total

def log_gpu_memory_usage(
    head: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0
):
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = f"{head}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, device memory used/total (GB): {mem_used}/{mem_total}"
        if logger is None:
            print(message)
        else:
            logger.log(msg=message, level=level)

class GPUMemoryLogger(DecoratorLoggerBase):

    def __init__(
        self,
        role: str,
        logger: logging.Logger = None,
        level=logging.DEBUG,
        log_only_rank_0: bool = True,
    ):
        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
        else:
            rank = 0
        super().__init__(role, logger, level, rank, log_only_rank_0)

    def __call__(self, decorated_function: callable):
        def f(*args, **kwargs):
            return self.log(decorated_function, *args, **kwargs)

        return f

    def log(self, func, *args, **kwargs):
        name = func.__name__
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = f"Before {name}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, device memory used/total (GB): {mem_used}/{mem_total}"
        self.logging_function(message)
        output = func(*args, **kwargs)
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = f"After {name}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, device memory used/total (GB): {mem_used}/{mem_total}"
        self.logging_function(message)
        return output

def log_print(ctn: Any):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    frame = inspect.currentframe().f_back
    function_name = frame.f_code.co_name
    line_number = frame.f_lineno
    file_name = frame.f_code.co_filename.split("/")[-1]
    print(f"[{file_name}:{line_number}:{function_name}]: {ctn}")
