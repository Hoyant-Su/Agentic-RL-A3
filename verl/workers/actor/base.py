
from abc import ABC, abstractmethod
from typing import Dict
import torch
from verl import DataProto

__all__ = ["BasePPOActor"]

class BasePPOActor(ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        pass

    @abstractmethod
    def update_policy(self, data: DataProto) -> Dict:
        pass
