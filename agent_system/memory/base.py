from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseMemory(ABC):

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    @abstractmethod
    def reset(self, batch_size: int):
        pass

    @abstractmethod
    def store(self, record: Dict[str, List[Any]]):
        pass

    @abstractmethod
    def fetch(self, step: int):
        pass
