import math
from typing import Any
import numpy as np
from .memory import MemoryEntry

def rank_memory_entries(
    *,
    entries: list[MemoryEntry],
    relevances: list[float],
    top_k: int,
    retrieve_mode: str,
    retrieve_type: str,
    alpha: float,
    temperature: float,
    ucb_scale: float,
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    total_counts = max(sum(max(entry.count, 1) for entry in entries), 1)
    for entry, relevance in zip(entries, relevances):
        if relevance < similarity_threshold:
            continue
        if retrieve_mode != "both" and entry.attempt_type != retrieve_mode:
            continue
        utility = float(entry.utility_score)
        count = max(int(entry.count), 1)
        if retrieve_type == "relevance_only":
            score = float(relevance)
        elif retrieve_type == "softmax":
            score = alpha * float(relevance) + (1.0 - alpha) * utility
        else:
            exploration_bonus = ucb_scale * math.sqrt(
                math.log(total_counts) / float(count)
            )
            score = alpha * float(relevance) + (1.0 - alpha) * (
                utility + exploration_bonus
            )
        filtered.append(
            {
                "entry_id": entry.entry_id,
                "reflection": entry.reflection,
                "attempt_type": entry.attempt_type,
                "utility_score": utility,
                "relevance": float(relevance),
                "score": float(score),
            }
        )
    if not filtered:
        return []
    filtered.sort(key=lambda item: item["score"], reverse=True)
    if retrieve_type != "softmax" or len(filtered) <= top_k:
        return filtered[:top_k]
    scores = np.array([item["score"] for item in filtered], dtype=np.float64)
    temp = max(float(temperature), 1e-5)
    shifted = scores - np.max(scores)
    probs = np.exp(shifted / temp)
    probs = probs / np.sum(probs)
    choice = np.random.choice(len(filtered), size=top_k, replace=False, p=probs)
    return [filtered[idx] for idx in choice]
