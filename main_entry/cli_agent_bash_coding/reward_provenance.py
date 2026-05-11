from typing import Dict

REWARD_PROVENANCE_KEYS = (
    "progress_gain_coef",
    "answer_reward",
    "exec_error_penalty",
    "use_model_evidence_gain",
)

def zero_reward_provenance() -> Dict[str, float]:
    return {key: 0.0 for key in REWARD_PROVENANCE_KEYS}

def add_reward_provenance(
    target: Dict[str, float], delta: Dict[str, float]
) -> Dict[str, float]:
    for key in REWARD_PROVENANCE_KEYS:
        target[key] = float(target.get(key, 0.0)) + float(delta.get(key, 0.0))
    return target
