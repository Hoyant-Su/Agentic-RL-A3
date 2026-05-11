from collections import defaultdict
import numpy as np
import torch
import verl.utils.torch_functional as verl_F

def build_response_mask(data, multi_turn: bool) -> torch.Tensor:
    response_length = data.batch["responses"].size(1)
    if multi_turn:
        return data.batch["loss_mask"][:, -response_length:]
    return data.batch["attention_mask"][:, -response_length:]

def get_episode_scores(data) -> torch.Tensor:
    device = data.batch["token_level_rewards"].device
    dtype = data.batch["token_level_rewards"].dtype
    scores = np.asarray(data.non_tensor_batch["episode_rewards"], dtype=np.float32)
    return torch.from_numpy(scores).to(device=device, dtype=dtype)

def get_turn_discounted_returns(data, gamma: float) -> torch.Tensor:
    device = data.batch["token_level_rewards"].device
    dtype = data.batch["token_level_rewards"].dtype
    rewards = np.asarray(data.non_tensor_batch["rewards"], dtype=np.float32)
    traj_uids = np.asarray(data.non_tensor_batch["traj_uid"], dtype=object)
    turn_idxs = np.asarray(data.non_tensor_batch["turn_idx"], dtype=np.int64)
    discounted_returns = np.zeros_like(rewards, dtype=np.float32)
    for traj_uid in np.unique(traj_uids):
        traj_indices = np.where(traj_uids == traj_uid)[0]
        ordered_indices = traj_indices[np.argsort(turn_idxs[traj_indices])]
        running_return = 0.0
        for idx in ordered_indices[::-1]:
            running_return = rewards[idx] + gamma * running_return
            discounted_returns[idx] = running_return
    return torch.from_numpy(discounted_returns).to(device=device, dtype=dtype)

def build_group_indices(keys) -> list[list[int]]:
    grouped = defaultdict(list)
    for idx, key in enumerate(keys):
        grouped[key].append(idx)
    return list(grouped.values())

def build_prompt_turn_groups(data) -> tuple[list[list[int]], list[list[int]]]:
    prompt_groups = build_group_indices(list(data.non_tensor_batch["uid"]))
    turn_groups = build_group_indices(
        [
            (uid, int(turn_idx))
            for uid, turn_idx in zip(
                data.non_tensor_batch["uid"], data.non_tensor_batch["turn_idx"]
            )
        ]
    )
    return prompt_groups, turn_groups

def sequence_mean(values: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    return torch.sum(values * response_mask, dim=-1) / lengths

def group_median_mad(
    scores: torch.Tensor,
    group_indices: list[list[int]],
    epsilon: float = 1e-6,
    min_scale: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = torch.empty_like(scores)
    scales = torch.empty_like(scores)
    for indices in group_indices:
        group_scores = scores[indices]
        center = group_scores.median()
        abs_dev = torch.abs(group_scores - center)
        scale = torch.clamp(abs_dev.median() * 1.4826, min=min_scale) + epsilon
        centers[indices] = center
        scales[indices] = scale
    return centers, scales

def group_variance(
    scores: torch.Tensor,
    group_indices: list[list[int]],
    epsilon: float = 1e-6,
    min_var: float = 0.05,
) -> torch.Tensor:
    variances = torch.empty_like(scores)
    for indices in group_indices:
        group_scores = scores[indices]
        if group_scores.numel() <= 1:
            variance = torch.tensor(min_var, device=scores.device, dtype=scores.dtype)
        else:
            variance = torch.clamp(group_scores.var(unbiased=False), min=min_var)
        variances[indices] = variance + epsilon
    return variances

def leave_one_out_mean_std(
    scores: torch.Tensor,
    group_indices: list[list[int]],
    epsilon: float = 1e-6,
    min_scale: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    baselines = torch.empty_like(scores)
    scales = torch.empty_like(scores)
    for indices in group_indices:
        group_scores = scores[indices]
        group_std = (
            torch.clamp(group_scores.std(unbiased=False), min=min_scale) + epsilon
        )
        scales[indices] = group_std
        if group_scores.numel() == 1:
            baselines[indices] = group_scores[0]
            continue
        total = group_scores.sum()
        denom = group_scores.numel() - 1
        baselines[indices] = (total - group_scores) / denom
    return baselines, scales

def centered_ranks(
    scores: torch.Tensor, group_indices: list[list[int]]
) -> torch.Tensor:
    ranks = torch.empty_like(scores)
    for indices in group_indices:
        group_scores = scores[indices]
        if group_scores.numel() == 1:
            ranks[indices] = 0.0
            continue
        unique_scores, inverse, counts = torch.unique(
            group_scores,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        del unique_scores
        cumulative = torch.cumsum(counts, dim=0)
        start = cumulative - counts
        average_rank = (
            start.to(scores.dtype) + cumulative.to(scores.dtype) - 1.0
        ) * 0.5
        if group_scores.numel() == 1:
            scaled = torch.zeros_like(group_scores, dtype=scores.dtype)
        else:
            scaled = average_rank[inverse] / float(group_scores.numel() - 1)
        ranks[indices] = scaled * 2.0 - 1.0
    return ranks

def smooth_pairwise_utility(
    scores: torch.Tensor,
    group_indices: list[list[int]],
    temperature: float = 1.0,
) -> torch.Tensor:
    utilities = torch.empty_like(scores)
    temperature = max(float(temperature), 1e-6)
    for indices in group_indices:
        group_scores = scores[indices]
        if group_scores.numel() <= 1:
            utilities[indices] = 0.0
            continue
        pairwise_diff = (
            group_scores.unsqueeze(1) - group_scores.unsqueeze(0)
        ) / temperature
        pairwise_pref = torch.tanh(pairwise_diff)
        utilities[indices] = pairwise_pref.sum(dim=1) / float(group_scores.numel() - 1)
    return utilities

def smooth_bounded_score(values: torch.Tensor, bound: float) -> torch.Tensor:
    return bound * torch.tanh(values / bound)

def conservative_fusion(
    prompt_signal: torch.Tensor,
    turn_signal: torch.Tensor,
    turn_weight: float,
    disagreement_penalty: float,
) -> torch.Tensor:
    mixed = turn_weight * turn_signal + (1.0 - turn_weight) * prompt_signal
    return mixed - disagreement_penalty * torch.abs(turn_signal - prompt_signal)

def risk_balanced_fusion(
    prompt_signal: torch.Tensor,
    turn_signal: torch.Tensor,
    prompt_variance: torch.Tensor,
    turn_variance: torch.Tensor,
    turn_weight: float,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    prompt_precision = torch.reciprocal(prompt_variance + epsilon)
    turn_precision = torch.reciprocal(turn_variance + epsilon)
    prompt_weight = (1.0 - turn_weight) * prompt_precision
    turn_weight_tensor = turn_weight * turn_precision
    normalizer = prompt_weight + turn_weight_tensor + epsilon
    return (
        prompt_weight * prompt_signal + turn_weight_tensor * turn_signal
    ) / normalizer

def compute_sequence_trust_region_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    ratio_cap: float,
    trust_coeff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if cliprange_low is None:
        cliprange_low = cliprange if cliprange is not None else ratio_cap
    if cliprange_high is None:
        cliprange_high = cliprange if cliprange is not None else ratio_cap
    negative_approx_kl = log_prob - old_log_prob
    seq_delta = sequence_mean(negative_approx_kl, response_mask)
    seq_adv = sequence_mean(advantages, response_mask)
    seq_ratio = torch.exp(torch.clamp(seq_delta, min=-ratio_cap, max=ratio_cap))
    pg_losses1 = -seq_adv * seq_ratio
    pg_losses2 = -seq_adv * torch.clamp(
        seq_ratio, 1 - cliprange_low, 1 + cliprange_high
    )
    pg_losses = torch.maximum(pg_losses1, pg_losses2) + trust_coeff * seq_delta.square()
    pg_loss = torch.mean(pg_losses)
    pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses1).float())
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
