
from abc import ABC, abstractmethod
from verl import DataProto

class BasePPORewardModel(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def compute_reward(self, data: DataProto) -> DataProto:
        pass
