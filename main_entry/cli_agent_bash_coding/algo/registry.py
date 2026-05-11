from main_entry.cli_agent_bash_coding.algo.base import BashCodingAlgorithm
from main_entry.cli_agent_bash_coding.algo.A3_algo import (
    compute_policy_loss_A3_algo,
    compute_A3_algo_advantage,
)

_ALGORITHMS = {
    "A3_algo": BashCodingAlgorithm(
        name="A3_algo",
        compute_advantage=compute_A3_algo_advantage,
        compute_policy_loss=compute_policy_loss_A3_algo,
    ),
    "hgpo": BashCodingAlgorithm(
        name="hgpo",
    ),
}


def get_bash_coding_algorithm(name: str) -> BashCodingAlgorithm | None:
    key = str(name).strip().lower()
    if key == "":
        return None
    for reg_key, algo in _ALGORITHMS.items():
        if str(reg_key).lower() == key:
            return algo
    return None


def list_bash_coding_algorithms() -> tuple[str, ...]:
    return tuple(_ALGORITHMS.keys())
