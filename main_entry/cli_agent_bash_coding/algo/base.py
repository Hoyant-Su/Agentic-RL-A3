from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class BashCodingAlgorithm:
    name: str
    compute_advantage: Callable | None = None
    compute_policy_loss: Callable | None = None
