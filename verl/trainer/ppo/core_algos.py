
from collections import defaultdict
import numpy as np
import torch
import verl.utils.torch_functional as verl_F

class AdaptiveKLController:

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

class FixedKLController:

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass

def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, (
            f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        )
        return AdaptiveKLController(
            init_kl_coef=kl_ctrl.kl_coef,
            target_kl=kl_ctrl.target_kl,
            horizon=kl_ctrl.horizon,
        )
    else:
        raise NotImplementedError

def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]
        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns

def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
    compute_mean_std_cross_steps: bool = True,
):
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
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (
                    id2std[index[i]] + epsilon
                )
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask
    return scores, scores

def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    compute_mean_std_cross_steps: bool = True,
):
    scores = token_level_rewards.sum(dim=-1)
    advantages = torch.zeros_like(scores)
    id2scores = defaultdict(list)
    id2indices = defaultdict(list)
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage
    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages

def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    compute_mean_std_cross_steps: bool = True,
):
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask
    return scores, scores

def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    compute_mean_std_cross_steps: bool = True,
):
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[
                    index[i]
                ] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask
    return scores, scores

def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor
):
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0
        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            running_return = running_return * response_mask[:, t]
        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask
    return advantages, returns

def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
):
    with torch.no_grad():
        returns = (
            (token_level_rewards * response_mask)
            .flip(dims=[-1])
            .cumsum(dim=-1)
            .flip(dims=[-1])
        )
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask
    return advantages, returns

def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio

def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(
            loss_mask, dim=-1
        )
        loss = torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")
    return loss

def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower

def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "seq-mean-token-mean",
):
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    negative_approx_kl = log_prob - old_log_prob
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = (
        torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths
    )
    log_seq_importance_ratio = (
        log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    )
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)
    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(
        seq_importance_ratio, 1 - cliprange_low, 1 + cliprange_high
    )
    pg_losses = torch.maximum(pg_losses1, pg_losses2)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean"
    )
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower

def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    token_entropy = verl_F.entropy_from_logits(logits)
    entropy_loss = agg_loss(
        loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )
    return entropy_loss

def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    vpredclipped = verl_F.clip_by_value(
        vpreds, values - cliprange_value, values + cliprange_value
    )
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(
        loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
    )
    vf_clipfrac = verl_F.masked_mean(
        torch.gt(vf_losses2, vf_losses1).float(), response_mask
    )
    return vf_loss, vf_clipfrac

def kl_penalty(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob
    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()
    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)
    if kl_penalty == "full":
        raise NotImplementedError
    raise NotImplementedError

def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):

    @torch.no_grad()
    def compute_weights(
        scores: torch.Tensor, reweight_method: str, weight_pow: float
    ) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where(
                (scores == max_score) | (scores == min_score), 1.0, 0.0
            )
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)
    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)
    resampled_batch = {
        key: tensor[sample_indices] for key, tensor in data.batch.items()
    }
    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]
    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value
    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info
    return resampled_data
