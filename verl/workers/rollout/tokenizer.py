
from abc import ABC, abstractmethod
from typing import Dict, List, Union
import numpy as np
import torch

__all__ = ["HybridEngineBaseTokenizer"]

class HybridEngineBaseTokenizer(ABC):

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def pad_token_id(self):
        pass

    @property
    @abstractmethod
    def eos_token_id(self):
        pass

    @property
    @abstractmethod
    def all_special_ids(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def all_special_tokens(self) -> List[str]:
        pass

    @abstractmethod
    def encode(self, text):
        pass

    @abstractmethod
    def decode(
        self,
        token_ids: Union[int, List[int], np.ndarray, torch.Tensor],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        pass

    @abstractmethod
    def get_added_vocab(self) -> Dict[str, int]:
        pass

    @abstractmethod
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        pass

    @property
    def is_fast(self):
        return False
