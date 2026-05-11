from abc import ABC, abstractmethod
from verl import DataProto

__all__ = ["BaseRollout"]

class BaseRollout(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        pass
