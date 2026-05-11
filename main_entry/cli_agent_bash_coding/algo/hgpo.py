import os
from collections import defaultdict
from difflib import SequenceMatcher
import numpy as np
import torch
from main_entry.cli_agent_bash_coding.algo.stable_ops import build_response_mask
from main_entry.cli_agent_bash_coding.algo.stable_ops import get_turn_discounted_returns
from main_entry.cli_agent_bash_coding.tooling.semantic_similarity.semantic_similarity import (
    semantic_similarity_batch_strict,
)

def _to_hashable(value):
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(value.flatten())
    if isinstance(value, (list, tuple)):
        return tuple(_to_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _to_hashable(item)) for key, item in value.items()))
    raise TypeError(f"Unsupported type: {type(value)}")

def _history_sequence_to_text(sequence) -> str:
    parts = []
    for idx, item in enumerate(sequence, start=1):
        parts.append(f"[STEP {idx}]\n{str(item)}")
    return "\n\n".join(parts)

def _build_exact_history_clusters(traj_histories, history_length: int):
    clusters = defaultdict(list)
    max_k = history_length + 1
    for traj_idx, history in enumerate(traj_histories):
        for step_idx in range(len(history)):
            upper = min(max_k, step_idx + 1)
            for k in range(1, upper + 1):
                sequence = history[step_idx - k + 1 : step_idx + 1]
                clusters[(k, tuple(sequence))].append((traj_idx, step_idx))
    return clusters

def _build_similarity_history_clusters(
    traj_histories,
    *,
    history_length: int,
    backend: str,
    api_base_url: str,
    similarity_thresh: float,
    similarity_batch_size: int,
):
    max_k = history_length + 1
    history_items_by_k = defaultdict(list)
    for traj_idx, history in enumerate(traj_histories):
        for step_idx in range(len(history)):
            upper = min(max_k, step_idx + 1)
            for k in range(1, upper + 1):
                sequence = history[step_idx - k + 1 : step_idx + 1]
                history_items_by_k[k].append((tuple(sequence), traj_idx, step_idx))
    clusters = defaultdict(list)
    for k, items in history_items_by_k.items():
        exact_groups = defaultdict(list)
        sequence_text = {}
        for sequence, traj_idx, step_idx in items:
            exact_groups[sequence].append((traj_idx, step_idx))
            if sequence not in sequence_text:
                sequence_text[sequence] = _history_sequence_to_text(sequence)
        unique_sequences = list(exact_groups.keys())
        if len(unique_sequences) == 1:
            clusters[(k, 0)] = list(exact_groups[unique_sequences[0]])
            continue
        parent = list(range(len(unique_sequences)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x

        text_pairs = []
        pair_indices = []
        for left in range(len(unique_sequences)):
            for right in range(left + 1, len(unique_sequences)):
                pair_indices.append((left, right))
                text_pairs.append(
                    (
                        sequence_text[unique_sequences[left]],
                        sequence_text[unique_sequences[right]],
                    )
                )
        if text_pairs:
            if backend == "llm":
                scores = semantic_similarity_batch_strict(
                    text_pairs,
                    api_base_url=api_base_url,
                    batch_size=similarity_batch_size,
                )
            elif backend == "sequence_matcher":
                scores = [SequenceMatcher(None, a, b).ratio() for a, b in text_pairs]
            else:
                raise ValueError(f"Invalid similarity backend: {backend}")
            for (left, right), score in zip(pair_indices, scores):
                if score >= similarity_thresh:
                    union(left, right)
        merged_members = defaultdict(list)
        for idx, sequence in enumerate(unique_sequences):
            merged_members[find(idx)].extend(exact_groups[sequence])
        for cluster_id, members in merged_members.items():
            clusters[(k, cluster_id)] = members
    return clusters

def _compute_trajectory_advantages(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float,
    remove_std: bool,
    compute_mean_std_cross_steps: bool,
) -> torch.Tensor:
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        for i in range(scores.shape[0]):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx, score_list in id2score.items():
            if len(score_list) == 1:
                id2mean[idx] = torch.tensor(
                    0.0, device=scores.device, dtype=scores.dtype
                )
                id2std[idx] = torch.tensor(
                    1.0, device=scores.device, dtype=scores.dtype
                )
                continue
            stacked = torch.stack(
                [
                    score.to(device=scores.device, dtype=scores.dtype)
                    for score in score_list
                ]
            )
            id2mean[idx] = stacked.mean()
            id2std[idx] = stacked.std(unbiased=False)
        for i in range(scores.shape[0]):
            centered = scores[i] - id2mean[index[i]]
            if remove_std:
                scores[i] = centered
            else:
                scores[i] = centered / (id2std[index[i]] + epsilon)
    return scores.unsqueeze(-1).tile([1, response_length]) * response_mask

def _aggregate_items(
    items, *, device: torch.device, epsilon: float, length_weight_alpha: float
) -> torch.Tensor:
    if not items:
        return torch.zeros((), device=device, dtype=torch.float32)
    ks = []
    advantages = []
    for k, value in items:
        ks.append(k)
        advantages.append(value.to(torch.float32))
    advantages = torch.stack(advantages, dim=0)
    ks_tensor = torch.tensor(ks, device=device, dtype=torch.float32)
    valid_mask = advantages != 0
    if not valid_mask.any():
        return torch.zeros((), device=device, dtype=torch.float32)
    valid_advantages = advantages[valid_mask]
    valid_lengths = ks_tensor[valid_mask] + 1
    weights = valid_lengths.pow(length_weight_alpha)
    weights = weights / (weights.sum() + epsilon)
    return (weights * valid_advantages).sum()

def _estimate_hgpo_advantages(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    index: np.ndarray,
    traj_index: np.ndarray,
    history_length: int,
    epsilon: float,
    hgpo_mode: str,
    hgpo_length_weight_alpha: float,
    hgpo_base_group: bool,
    compute_mean_std_cross_steps: bool,
    enable_similarity_sequence_matcher: bool,
    enable_similarity_llm: bool,
    similarity_thresh: float,
    similarity_batch_size: int,
) -> torch.Tensor:
    device = response_mask.device
    response_length = response_mask.shape[-1]
    step_rewards = step_rewards.detach().to(device=device, dtype=torch.float32)
    all_step_advantages = torch.zeros(
        anchor_obs.shape[0], device=device, dtype=torch.float32
    )
    if enable_similarity_sequence_matcher and enable_similarity_llm:
        raise ValueError(
            "enable_similarity_sequence_matcher and enable_similarity_llm cannot both be True"
        )
    use_similarity = enable_similarity_sequence_matcher or enable_similarity_llm
    similarity_url = os.environ.get(
        "BASH_CODING_SEMANTIC_SIMILARITY_URL", "http://127.0.0.1:30003"
    )
    base_advantages = None
    if hgpo_base_group:
        base_advantages = _compute_trajectory_advantages(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
            epsilon=epsilon,
            remove_std=(hgpo_mode == "mean_norm"),
            compute_mean_std_cross_steps=compute_mean_std_cross_steps,
        )
        mask_sum = response_mask.sum(dim=1).clamp(min=1e-8)
        base_advantages = (base_advantages * response_mask).sum(dim=1) / mask_sum
    for group_id in np.unique(index):
        group_indices = np.flatnonzero(index == group_id)
        group_obs = anchor_obs[group_indices]
        group_traj_ids = traj_index[group_indices]
        group_index_tensor = torch.as_tensor(
            group_indices, device=device, dtype=torch.long
        )
        group_rewards = step_rewards.index_select(0, group_index_tensor)
        unique_trajs, inverse = np.unique(group_traj_ids, return_inverse=True)
        traj_positions = [[] for _ in range(len(unique_trajs))]
        for pos, traj_id in enumerate(inverse):
            traj_positions[traj_id].append(pos)
        traj_positions = [
            np.asarray(pos_list, dtype=np.int64) for pos_list in traj_positions
        ]
        traj_obs = [group_obs[pos_list] for pos_list in traj_positions]
        traj_rewards = [
            group_rewards.index_select(
                0, torch.as_tensor(pos_list, device=device, dtype=torch.long)
            )
            for pos_list in traj_positions
        ]
        traj_global_indices = [group_indices[pos_list] for pos_list in traj_positions]
        traj_histories = [
            [_to_hashable(obs) for obs in obs_list] for obs_list in traj_obs
        ]
        if use_similarity:
            backend = "llm" if enable_similarity_llm else "sequence_matcher"
            clusters = _build_similarity_history_clusters(
                traj_histories,
                history_length=history_length,
                backend=backend,
                api_base_url=similarity_url,
                similarity_thresh=similarity_thresh,
                similarity_batch_size=similarity_batch_size,
            )
        else:
            clusters = _build_exact_history_clusters(
                traj_histories,
                history_length=history_length,
            )
        per_step_items = defaultdict(list)
        for (k, _), members in clusters.items():
            if len(members) <= 1:
                continue
            rewards = torch.stack(
                [traj_rewards[traj_idx][step_idx] for traj_idx, step_idx in members],
                dim=0,
            )
            mean = rewards.mean()
            std = rewards.std(unbiased=False)
            if hgpo_mode == "mean_std_norm":
                normalized = (rewards - mean) / (std + epsilon)
            elif hgpo_mode == "mean_norm":
                normalized = rewards - mean
            else:
                raise ValueError(f"Invalid hgpo_mode: {hgpo_mode}")
            for pos, (traj_idx, step_idx) in enumerate(members):
                per_step_items[(traj_idx, step_idx)].append((k, normalized[pos]))
        for traj_idx, obs_list in enumerate(traj_obs):
            for step_idx in range(len(obs_list)):
                global_idx = int(traj_global_indices[traj_idx][step_idx])
                items = per_step_items.get((traj_idx, step_idx), [])
                if hgpo_base_group and base_advantages is not None:
                    items = [(0, base_advantages[global_idx])] + items
                all_step_advantages[global_idx] = _aggregate_items(
                    items,
                    device=device,
                    epsilon=epsilon,
                    length_weight_alpha=hgpo_length_weight_alpha,
                )
    return all_step_advantages.unsqueeze(-1).expand(-1, response_length) * response_mask

def compute_hgpo_advantage(
    data,
    *,
    multi_turn: bool,
    history_length: int,
    hgpo_mode: str = "mean_std_norm",
    hgpo_length_weight_alpha: float = 1.0,
    hgpo_base_group: bool = False,
    gamma: float = 1.0,
    epsilon: float = 1e-6,
    compute_mean_std_cross_steps: bool = True,
    enable_similarity_sequence_matcher: bool = False,
    enable_similarity_llm: bool = False,
    hgpo_similarity_thresh: float = 0.95,
    hgpo_similarity_batch_size: int = 32,
):
    response_mask = build_response_mask(data, multi_turn=multi_turn)
    step_rewards = get_turn_discounted_returns(data, gamma=gamma)
    advantages = _estimate_hgpo_advantages(
        token_level_rewards=data.batch["token_level_rewards"],
        step_rewards=step_rewards,
        response_mask=response_mask,
        anchor_obs=np.asarray(data.non_tensor_batch["anchor_obs"], dtype=object),
        index=np.asarray(data.non_tensor_batch["uid"], dtype=object),
        traj_index=np.asarray(data.non_tensor_batch["traj_uid"], dtype=object),
        history_length=history_length,
        epsilon=epsilon,
        hgpo_mode=hgpo_mode,
        hgpo_length_weight_alpha=hgpo_length_weight_alpha,
        hgpo_base_group=hgpo_base_group,
        compute_mean_std_cross_steps=compute_mean_std_cross_steps,
        enable_similarity_sequence_matcher=enable_similarity_sequence_matcher,
        enable_similarity_llm=enable_similarity_llm,
        similarity_thresh=hgpo_similarity_thresh,
        similarity_batch_size=hgpo_similarity_batch_size,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = advantages
    return data
