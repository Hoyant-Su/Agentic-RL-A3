import os
import uuid
from collections import Counter, defaultdict
import numpy as np
import torch
from verl import DataProto
from difflib import SequenceMatcher
from typing import Sequence, List, Dict, Any
from main_entry.cli_agent_bash_coding.tooling.semantic_similarity.semantic_similarity import (
    semantic_similarity_batch_strict,
)


def to_hashable(x):
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def summarize_group_size(group_size: list):
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts)
    summary = {}
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0
        summary[size] = (cnt, prop)
    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size, (cnt, prop) in summary.items():
        if prop:
            print(f"{size:>4} | {cnt:>5} | {prop:>9.2%}")

def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError(
            "Only text-based observations are supported for similarity-based GiGPO in this version."
        )
    return SequenceMatcher(None, a, b).ratio() >= threshold

def compute_step_discounted_returns(batch: DataProto, gamma: float):
    rewards = batch.non_tensor_batch["rewards"].astype(np.float32)
    traj_uids = batch.non_tensor_batch["traj_uid"]
    active_masks = batch.non_tensor_batch["active_masks"].astype(np.float32)
    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)
    for uid in unique_traj_uids:
        traj_indices = np.where(traj_uids == uid)[0]
        traj_rewards = rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]
        assert traj_active_masks.all(), (
            "active_masks should be all 1s for the same trajectory"
        )
        traj_returns = np.zeros_like(traj_rewards)
        running_return = 0
        for t in reversed(range(len(traj_rewards))):
            running_return = traj_rewards[t] + gamma * running_return
            traj_returns[t] = running_return
        returns_by_traj[uid] = traj_returns
    all_returns = np.zeros_like(rewards)
    for i, uid in enumerate(traj_uids):
        traj_indices = np.where(traj_uids == uid)[0]
        idx_in_traj = np.where(traj_indices == i)[0][0]
        all_returns[i] = returns_by_traj[uid][idx_in_traj]
    all_returns = torch.tensor(
        all_returns, dtype=torch.float32, device=batch.batch["input_ids"].device
    )
    return all_returns

def compute_gigpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.array,
    index: np.array,
    traj_index: np.array,
    epsilon: float = 1e-6,
    step_advantage_w: float = 1.0,
    mode: str = "mean_norm",
    enable_similarity_sequence_matcher: bool = False,
    enable_similarity_llm: bool = False,
    similarity_thresh: float = 0.95,
    compute_mean_std_cross_steps: bool = True,
):
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    episode_advantages = episode_norm_reward(
        token_level_rewards,
        response_mask,
        index,
        traj_index,
        epsilon,
        remove_std,
        compute_mean_std_cross_steps,
    )
    step_group_uids = build_step_group(
        anchor_obs,
        index,
        enable_similarity_sequence_matcher=enable_similarity_sequence_matcher,
        enable_similarity_llm=enable_similarity_llm,
        similarity_thresh=similarity_thresh,
    )
    step_advantages = step_norm_reward(
        step_rewards, response_mask, step_group_uids, epsilon, remove_std
    )
    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores

def episode_norm_reward(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.array,
    traj_index: np.array,
    epsilon: float = 1e-6,
    remove_std: bool = True,
    compute_mean_std_cross_steps: bool = True,
):
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx, score_list in id2score.items():
            stacked = torch.stack(
                [
                    score.detach().to(device=scores.device, dtype=scores.dtype)
                    for score in score_list
                ]
            )
            id2mean[idx] = stacked.mean()
            if stacked.numel() == 1:
                id2std[idx] = torch.tensor(
                    1.0, device=scores.device, dtype=scores.dtype
                )
            else:
                id2std[idx] = stacked.std(unbiased=False)
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (
                    id2std[index[i]] + epsilon
                )
        episode_advantages = (
            scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        )
    return episode_advantages

def build_step_group(
    anchor_obs: np.array,
    index: np.array,
    *,
    enable_similarity_sequence_matcher: bool = False,
    enable_similarity_llm: bool = False,
    similarity_thresh: float = 0.95,
    summarize: bool = False,
):
    if enable_similarity_sequence_matcher and enable_similarity_llm:
        raise ValueError(
            "enable_similarity_sequence_matcher and enable_similarity_llm cannot both be True"
        )
    use_similarity = enable_similarity_sequence_matcher or enable_similarity_llm
    if use_similarity:
        assert 0.0 < similarity_thresh < 1.0, (
            "When enabling similarity-based step-level group, similarity_thresh should be in (0, 1)"
        )
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    unique_indices = np.unique(index)
    group_size: List[int] = []
    for idx in unique_indices:
        if not use_similarity:
            indices = np.where(index == idx)[0]
            obs_group = anchor_obs[indices]
            clusters = defaultdict(list)
            for i, obs in enumerate(obs_group):
                clusters[to_hashable(obs)].append(indices[i])
            for obs, original_indices in clusters.items():
                uid = str(uuid.uuid4())
                group_size.append(len(original_indices))
                for original_idx in original_indices:
                    step_group_uids[original_idx] = uid
        else:
            locs = np.where(index == idx)[0]
            obs_group = anchor_obs[locs]
            clusters: List[Dict[str, Any]] = []
            api_base_url = os.environ.get(
                "BASH_CODING_SEMANTIC_SIMILARITY_URL", "http://127.0.0.1:30003"
            )
            for obs, loc in zip(obs_group, locs):
                placed = False
                if enable_similarity_llm and len(clusters) > 0:
                    pairs = [(obs, cluster["rep"]) for cluster in clusters]
                    scores = semantic_similarity_batch_strict(
                        pairs, api_base_url=api_base_url, batch_size=len(pairs)
                    )
                    for cluster, score in zip(clusters, scores):
                        if score >= similarity_thresh:
                            cluster["locs"].append(loc)
                            placed = True
                            break
                elif enable_similarity_sequence_matcher and len(clusters) > 0:
                    for cluster in clusters:
                        if are_similar(obs, cluster["rep"], similarity_thresh):
                            cluster["locs"].append(loc)
                            placed = True
                            break
                if not placed:
                    clusters.append({"rep": obs, "locs": [loc]})
            for cluster in clusters:
                uid = str(uuid.uuid4())
                group_size.append(len(cluster["locs"]))
                for loc in cluster["locs"]:
                    step_group_uids[loc] = uid
    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(
            f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}"
        )
    if summarize:
        summarize_group_size(group_size)
    print(f"Avg size of step-level group: {np.mean(group_size)}")
    return step_group_uids

def step_norm_reward(
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.array,
    epsilon: float = 1e-6,
    remove_std: bool = True,
):
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx, score_list in id2score.items():
            stacked = torch.stack(
                [
                    score.detach().to(device=scores.device, dtype=scores.dtype)
                    for score in score_list
                ]
            )
            id2mean[idx] = stacked.mean()
            if stacked.numel() == 1:
                id2std[idx] = torch.tensor(
                    1.0, device=scores.device, dtype=scores.dtype
                )
            else:
                id2std[idx] = stacked.std(unbiased=False)
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (
                    id2std[index[i]] + epsilon
                )
        step_advantages = (
            scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        )
    return step_advantages
