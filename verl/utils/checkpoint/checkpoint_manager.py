import os
import random
import shutil
import tempfile
from typing import Optional, Union
import numpy as np
import torch
import torch.distributed
from filelock import FileLock
from transformers import PreTrainedTokenizer, ProcessorMixin
from verl.utils.device import is_cuda_available, is_npu_available

class BaseCheckpointManager:

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        checkpoint_contents: Optional[list] = None,
    ):
        if checkpoint_contents is None:
            checkpoint_contents = ["model", "optimizer", "extra"]
        self.previous_global_step = None
        self.previous_saved_paths = []
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.processing_class = processing_class
        self.checkpoint_contents = checkpoint_contents
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def load_checkpoint(
        self, local_path: str, hdfs_path: str = None, del_local_after_load: bool = False
    ):
        raise NotImplementedError

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str = None,
        global_step: int = 0,
        max_ckpt_to_keep: int = None,
    ):
        raise NotImplementedError

    @staticmethod
    def checkpath(local_path: str, hdfs_path: str):
        assert local_path is not None or hdfs_path is not None, (
            "local_path and hdfs_path cannot be both None"
        )
        return (
            local_path is not None,
            local_path if local_path is not None else hdfs_path,
        )

    def remove_previous_save_local_path(self, path):
        if isinstance(path, str):
            path = [path]
        for p in path:
            abs_path = os.path.abspath(p)
            print(f"Checkpoint manager remove previous save local path: {abs_path}")
            if not os.path.exists(abs_path):
                continue
            shutil.rmtree(abs_path, ignore_errors=True)

    @staticmethod
    def local_mkdir(path):
        if not os.path.isabs(path):
            working_dir = os.getcwd()
            path = os.path.join(working_dir, path)
        lock_filename = f"ckpt_{hash(path) & 0xFFFFFFFF:08x}.lock"
        lock_path = os.path.join(tempfile.gettempdir(), lock_filename)
        try:
            with FileLock(lock_path, timeout=60):
                os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to acquire lock for {path}: {e}")
            os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_rng_state():
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        if is_cuda_available:
            rng_state["cuda"] = torch.cuda.get_rng_state()
        elif is_npu_available:
            rng_state["npu"] = torch.npu.get_rng_state()
        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])
        if is_cuda_available:
            torch.cuda.set_rng_state(rng_state["cuda"])
        elif is_npu_available:
            torch.npu.set_rng_state(rng_state["npu"])

def find_latest_ckpt_path(path, directory_format="global_step_{}"):
    if path is None:
        return None
    tracker_file = get_checkpoint_tracker_filename(path)
    if not os.path.exists(tracker_file):
        print("Checkpoint tracker file does not exist: %s", tracker_file)
        return None
    with open(tracker_file, "rb") as f:
        iteration = int(f.read().decode())
    ckpt_path = os.path.join(path, directory_format.format(iteration))
    if not os.path.exists(ckpt_path):
        print("Checkpoint does not exist: %s", ckpt_path)
        return None
    print("Found checkpoint: %s", ckpt_path)
    return ckpt_path

def get_checkpoint_tracker_filename(root_path: str):
    return os.path.join(root_path, "latest_checkpointed_iteration.txt")
