from typing import Any

class RetroAgentLiteReward:
    def __init__(self) -> None:
        return None

    def is_enabled(self) -> bool:
        return False

    def apply(
        self,
        *,
        data_source: str,
        prompt_text: str,
        response_text: str,
        base_score: float,
        reward_extra_info: dict[str, list[Any]],
    ) -> float:
        _ = data_source
        _ = prompt_text
        _ = response_text
        _ = base_score
        reward_extra_info.setdefault("retroagent_bonus", []).append(0.0)
        reward_extra_info.setdefault("retroagent_numerical_bonus", []).append(0.0)
        reward_extra_info.setdefault("retroagent_language_bonus", []).append(0.0)
        reward_extra_info.setdefault("retroagent_memory_size", []).append(0.0)
        reward_extra_info.setdefault("retroagent_retrieved_count", []).append(0.0)
        return 0.0
