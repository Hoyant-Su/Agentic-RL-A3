
from __future__ import annotations
import json
import re
from collections import defaultdict
import numpy as np
import torch
import verl.utils.torch_functional as verl_F
from main_entry.cli_agent_bash_coding.algo.stable_ops import build_group_indices
from main_entry.cli_agent_bash_coding.algo.stable_ops import build_response_mask
from main_entry.cli_agent_bash_coding.algo.stable_ops import get_episode_scores
from main_entry.cli_agent_bash_coding.algo.stable_ops import group_median_mad
from main_entry.cli_agent_bash_coding.algo.stable_ops import smooth_bounded_score
from main_entry.cli_agent_bash_coding.tooling.action_space_similarity import (
    BashIntentActionSpace,
)


def compute_policy_loss_gspo_sequence(
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
    log_seq_importance_ratio = torch.clamp(negative_approx_kl_seq, max=10.0)
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)
    seq_advantages = torch.sum(advantages * response_mask, dim=-1) / seq_lengths
    pg_losses1 = -seq_advantages * seq_importance_ratio
    pg_losses2 = -seq_advantages * torch.clamp(
        seq_importance_ratio, 1 - cliprange_low, 1 + cliprange_high
    )
    pg_losses = torch.maximum(pg_losses1, pg_losses2)
    pg_loss = torch.mean(pg_losses)
    pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses1).float())
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def _build_payload_action_keys(
    responses: torch.Tensor, response_mask: torch.Tensor
) -> list[tuple[int, ...]]:
    action_keys: list[tuple[int, ...]] = []
    payload_mask = response_mask > 0
    for idx in range(responses.size(0)):
        action_keys.append(
            tuple(int(x) for x in responses[idx][payload_mask[idx]].tolist())
        )
    return action_keys


def _compute_prompt_backbone_advantage(
    episode_scores: torch.Tensor,
    prompt_ids: np.ndarray,
    traj_ids: np.ndarray,
    *,
    epsilon: float,
) -> torch.Tensor:
    traj_groups = build_group_indices(list(traj_ids))
    traj_representatives = [group[0] for group in traj_groups]
    rep_tensor = torch.as_tensor(
        traj_representatives, device=episode_scores.device, dtype=torch.long
    )
    rep_scores = episode_scores.index_select(0, rep_tensor)
    rep_prompt_ids = [prompt_ids[idx] for idx in traj_representatives]
    rep_prompt_groups = build_group_indices(rep_prompt_ids)
    rep_centers, rep_scales = group_median_mad(
        rep_scores, rep_prompt_groups, epsilon=epsilon
    )
    rep_adv = smooth_bounded_score((rep_scores - rep_centers) / rep_scales, bound=1.0)
    backbone_adv = torch.empty_like(episode_scores)
    for local_idx, traj_indices in enumerate(traj_groups):
        backbone_adv[traj_indices] = rep_adv[local_idx]
    return backbone_adv


def _compute_tree_decision_return(
    episode_scores: torch.Tensor,
    traj_ids: np.ndarray,
    turn_idxs: np.ndarray,
    state_keys: list[object],
    action_keys: list[tuple[int, ...]],
    *,
    epsilon: float,
    support_prior: float,
    decision_gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_groups = build_group_indices(state_keys)
    local_gain = torch.zeros_like(episode_scores)
    local_support = torch.zeros_like(episode_scores)
    for state_indices in state_groups:
        if len(state_indices) <= 1:
            continue
        parent_value = episode_scores[state_indices].mean()
        state_action_groups = build_group_indices(
            [action_keys[idx] for idx in state_indices]
        )
        if len(state_action_groups) <= 1:
            continue
        for local_group in state_action_groups:
            branch_indices = [state_indices[local_idx] for local_idx in local_group]
            child_value = episode_scores[branch_indices].mean()
            branch_count = float(len(branch_indices))
            shrink = branch_count / (branch_count + support_prior + epsilon)
            local_gain[branch_indices] = shrink * (child_value - parent_value)
            local_support[branch_indices] = shrink
    tree_return = torch.zeros_like(episode_scores)
    tree_support = torch.zeros_like(episode_scores)
    traj_groups = build_group_indices(list(traj_ids))
    for traj_indices in traj_groups:
        ordered_indices = sorted(traj_indices, key=lambda idx: int(turn_idxs[idx]))
        running_gain = torch.tensor(
            0.0, device=episode_scores.device, dtype=episode_scores.dtype
        )
        running_support = torch.tensor(
            0.0, device=episode_scores.device, dtype=episode_scores.dtype
        )
        for idx in reversed(ordered_indices):
            running_gain = local_gain[idx] + decision_gamma * running_gain
            running_support = local_support[idx] + decision_gamma * running_support
            tree_return[idx] = running_gain
            tree_support[idx] = running_support
    return tree_return, tree_support


def _trie_residual_backbone_style(
    trie_res: torch.Tensor,
    prompt_ids: np.ndarray,
    traj_ids: np.ndarray,
    *,
    epsilon: float,
) -> torch.Tensor:
    traj_groups = build_group_indices(list(traj_ids))
    rep_rows = [group[0] for group in traj_groups]
    rep_tensor = torch.as_tensor(rep_rows, device=trie_res.device, dtype=torch.long)
    rep_vals = trie_res.index_select(0, rep_tensor)
    rep_uids = [prompt_ids[i] for i in rep_rows]
    uid_groups = build_group_indices(rep_uids)
    centers, scales = group_median_mad(rep_vals, uid_groups, epsilon=epsilon)
    rep_bounded = smooth_bounded_score((rep_vals - centers) / scales, bound=1.0)
    out = torch.empty_like(trie_res)
    for k, group in enumerate(traj_groups):
        out[group] = rep_bounded[k]
    return out


class _SingleLinkageUnionFind:
    __slots__ = ("p", "r")

    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True


def _adaptive_traj_bucket_k(m: int, tau: int) -> int:
    tau = max(1, int(tau))
    if m <= 1:
        return 1
    return max(1, min(m, (m + tau - 1) // tau))


def _single_linkage_k_labels(D: np.ndarray, k: int) -> np.ndarray:
    m = D.shape[0]
    assert D.shape == (m, m)
    k = int(k)
    if m == 0:
        return np.array([], dtype=np.int64)
    if k >= m:
        return np.arange(m, dtype=np.int64)
    if k <= 1:
        return np.zeros(m, dtype=np.int64)
    edges = [(float(D[i, j]), i, j) for i in range(m) for j in range(i + 1, m)]
    edges.sort(key=lambda x: x[0])
    uf = _SingleLinkageUnionFind(m)
    comp = m
    ei = 0
    while comp > k and ei < len(edges):
        _, i, j = edges[ei]
        if uf.union(i, j):
            comp -= 1
        ei += 1
    roots: dict[int, int] = {}
    out = np.empty(m, dtype=np.int64)
    nlab = 0
    for i in range(m):
        r = uf.find(i)
        if r not in roots:
            roots[r] = nlab
            nlab += 1
        out[i] = roots[r]
    return out


_CODE_BLOCK = re.compile(r"<code>\s*([\s\S]*?)\s*</code>", flags=re.IGNORECASE)
_RESIDUAL_EPS = 1e-9


def _stable_bucket_key(uid: object, tuid: object, k: int) -> tuple[str, str, int]:
    return (str(uid), str(tuid), int(k))

class _UnionFind:
    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

def _g_bucket_weighted_hamming(
    v_a: tuple[int, ...],
    v_b: tuple[int, ...],
    w_scope: np.ndarray,
) -> float:
    acc = 0.0
    for si in range(w_scope.shape[0]):
        acc += float(w_scope[si]) * float(v_a[si] != v_b[si])
    return acc

def _time_weights_from_root(num_past: int, decay: float) -> np.ndarray:
    raw = np.array([decay**j for j in range(num_past)], dtype=np.float64)
    return raw / (raw.sum() + _RESIDUAL_EPS)

def _same_tree_state_g_bucket(
    idx_i: int,
    idx_j: int,
    traj_turn: dict[tuple[str, str], dict[int, str]],
    vec_key: dict[tuple[str, str, int], tuple[int, ...]],
    prompt_ids: np.ndarray,
    traj_ids: np.ndarray,
    turn_idxs: np.ndarray,
    w_scope: np.ndarray,
    time_decay: float,
    dissim_tau: float,
) -> bool:
    if prompt_ids[idx_i] != prompt_ids[idx_j] or int(turn_idxs[idx_i]) != int(
        turn_idxs[idx_j]
    ):
        return False
    uid = prompt_ids[idx_i]
    k = int(turn_idxs[idx_i])
    tuid_i = traj_ids[idx_i]
    tuid_j = traj_ids[idx_j]
    keys_i = set(traj_turn[(uid, tuid_i)].keys())
    keys_j = set(traj_turn[(uid, tuid_j)].keys())
    pre_i = {t for t in keys_i if t < k}
    pre_j = {t for t in keys_j if t < k}
    if pre_i != pre_j:
        return False
    ordered = sorted(pre_i)
    m = len(ordered)
    if m == 0:
        return True
    tw = _time_weights_from_root(m, time_decay)
    acc = 0.0
    for j, t in enumerate(ordered):
        v_i = vec_key[_stable_bucket_key(uid, tuid_i, t)]
        v_j = vec_key[_stable_bucket_key(uid, tuid_j, t)]
        acc += tw[j] * _g_bucket_weighted_hamming(v_i, v_j, w_scope)
    return acc <= dissim_tau

def _same_tree_action_g_bucket(
    idx_i: int,
    idx_j: int,
    vec_key: dict[tuple[str, str, int], tuple[int, ...]],
    prompt_ids: np.ndarray,
    traj_ids: np.ndarray,
    turn_idxs: np.ndarray,
    w_scope: np.ndarray,
    dissim_tau: float,
) -> bool:
    if prompt_ids[idx_i] != prompt_ids[idx_j] or int(turn_idxs[idx_i]) != int(
        turn_idxs[idx_j]
    ):
        return False
    uid = prompt_ids[idx_i]
    k = int(turn_idxs[idx_i])
    tuid_i = traj_ids[idx_i]
    tuid_j = traj_ids[idx_j]
    v_i = vec_key[_stable_bucket_key(uid, tuid_i, k)]
    v_j = vec_key[_stable_bucket_key(uid, tuid_j, k)]
    return _g_bucket_weighted_hamming(v_i, v_j, w_scope) <= dissim_tau

def _tree_state_action_keys_g_bucket(
    n: int,
    traj_turn: dict[tuple[str, str], dict[int, str]],
    vec_key: dict[tuple[str, str, int], tuple[int, ...]],
    prompt_ids: np.ndarray,
    traj_ids: np.ndarray,
    turn_idxs: np.ndarray,
    w_intent_scopes: tuple[float, ...],
    time_decay: float,
    dissim_tau: float,
) -> tuple[list[object], list[tuple[int, ...]]]:
    w_scope = np.asarray(w_intent_scopes, dtype=np.float64)
    w_scope = w_scope / (w_scope.sum() + _RESIDUAL_EPS)
    uf_s = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if _same_tree_state_g_bucket(
                i,
                j,
                traj_turn,
                vec_key,
                prompt_ids,
                traj_ids,
                turn_idxs,
                w_scope,
                time_decay,
                dissim_tau,
            ):
                uf_s.union(i, j)
    state_root = [uf_s.find(i) for i in range(n)]
    state_keys: list[object] = [
        (prompt_ids[i], int(turn_idxs[i]), state_root[i]) for i in range(n)
    ]
    buckets: dict[tuple[object, int, int], list[int]] = defaultdict(list)
    for i in range(n):
        buckets[(prompt_ids[i], int(turn_idxs[i]), state_root[i])].append(i)
    action_tuple_keys: list[tuple[int, ...]] = [tuple() for _ in range(n)]
    for _key, inds in buckets.items():
        uf_a = _UnionFind(len(inds))
        m = len(inds)
        for ii in range(m):
            for jj in range(ii + 1, m):
                a = inds[ii]
                b = inds[jj]
                if _same_tree_action_g_bucket(
                    a, b, vec_key, prompt_ids, traj_ids, turn_idxs, w_scope, dissim_tau
                ):
                    uf_a.union(ii, jj)
        local = [uf_a.find(ii) for ii in range(m)]
        for ii, row in enumerate(inds):
            action_tuple_keys[row] = (local[ii],)
    return state_keys, action_tuple_keys

def _traj_turn_bash_map(
    prompt_ids: np.ndarray,
    traj_ids: np.ndarray,
    turn_idxs: np.ndarray,
    bash_texts: list[str],
) -> dict[tuple[str, str], dict[int, str]]:
    out: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    for i in range(len(prompt_ids)):
        out[(str(prompt_ids[i]), str(traj_ids[i]))][int(turn_idxs[i])] = bash_texts[i]
    return dict(out)

def _bash_snippet_for_scope(turn_bash: dict[int, str], k: int, scope: int) -> str:
    if not turn_bash:
        return ""
    ordered_turns = sorted(turn_bash)
    if scope == -1:
        return "\n".join(turn_bash[t] for t in ordered_turns)
    keys_le_k = [t for t in ordered_turns if t <= k]
    if scope == 1:
        return turn_bash.get(k, "")
    if not keys_le_k:
        return ""
    tail = keys_le_k[-min(scope, len(keys_le_k)) :]
    return "\n".join(turn_bash[t] for t in tail)

def _cluster_labels_for_strings(
    shells: list[str], space: BashIntentActionSpace, trie_min_count: int
) -> list[int]:
    mloc = len(shells)
    if mloc == 0:
        return []
    if mloc == 1:
        return [0]
    d = np.asarray(space.distance_matrix(shells), dtype=np.float64)
    k_star = _adaptive_traj_bucket_k(mloc, trie_min_count)
    labels = _single_linkage_k_labels(d, k_star)
    return [int(x) for x in labels]

def _multi_scale_turn_vectors_and_residual(
    episode_scores: torch.Tensor,
    prompt_ids: np.ndarray,
    traj_ids: np.ndarray,
    turn_idxs: np.ndarray,
    bash_texts: list[str],
    space: BashIntentActionSpace,
    trie_min_count: int,
    intent_scopes: tuple[int, ...],
    w_intent_scopes: tuple[float, ...],
) -> tuple[torch.Tensor, dict[str, float], dict[tuple[str, str, int], tuple[int, ...]]]:
    traj_turn = _traj_turn_bash_map(prompt_ids, traj_ids, turn_idxs, bash_texts)
    traj_score: dict[tuple[str, str], float] = {}
    for i in range(len(prompt_ids)):
        traj_score[(prompt_ids[i], traj_ids[i])] = float(episode_scores[i].item())
    uid_to_tuids: dict[object, list[object]] = defaultdict(list)
    for uid, tuid in traj_turn:
        uid_to_tuids[uid].append(tuid)
    for uid in uid_to_tuids:
        uid_to_tuids[uid] = list({t for t in uid_to_tuids[uid]})
    G = len(intent_scopes)
    assert len(w_intent_scopes) == G
    w = np.asarray(w_intent_scopes, dtype=np.float64)
    w = w / (w.sum() + 1e-12)
    vec_key: dict[tuple[str, str, int], tuple[int, ...]] = {}
    loo_cache: dict[tuple[object, int, int], dict[object, float]] = {}
    per_si_cell_stats: list[list[tuple[int, int]]] = [[] for _ in range(G)]
    n_clustering_cells = 0
    n_distance_pairs = 0
    for uid, tuids in uid_to_tuids.items():
        turn_set: set[int] = set()
        for tuid in tuids:
            turn_set |= set(traj_turn[(uid, tuid)].keys())
        for k in sorted(turn_set):
            tuids_here = [t for t in tuids if k in traj_turn[(uid, t)]]
            if not tuids_here:
                continue
            per_scope_label: list[dict[object, int]] = [{} for _ in range(G)]
            for si, scope in enumerate(intent_scopes):
                n_clustering_cells += 1
                shells = [
                    _bash_snippet_for_scope(traj_turn[(uid, tuid)], k, scope)
                    for tuid in tuids_here
                ]
                mloc = len(shells)
                if mloc >= 2:
                    np_add = mloc * (mloc - 1) // 2
                    n_distance_pairs += np_add
                labs = _cluster_labels_for_strings(shells, space, trie_min_count)
                per_si_cell_stats[si].append((len(tuids_here), len(set(labs))))
                for j, tuid in enumerate(tuids_here):
                    per_scope_label[si][tuid] = labs[j]
                loo_mean: dict[object, float] = {}
                if len(tuids_here) <= 1:
                    tuid = tuids_here[0]
                    loo_mean[tuid] = traj_score[(uid, tuid)]
                else:
                    uid_c: dict[int, list[tuple[object, float]]] = defaultdict(list)
                    for j, tuid in enumerate(tuids_here):
                        uid_c[labs[j]].append((tuid, traj_score[(uid, tuid)]))
                    for _c, members in uid_c.items():
                        s = len(members)
                        total = sum(sc for _, sc in members)
                        for tuid, sc in members:
                            if s <= 1:
                                loo_mean[tuid] = sc
                            else:
                                loo_mean[tuid] = (total - sc) / (s - 1)
                loo_cache[(uid, k, si)] = loo_mean
            for tuid in tuids_here:
                vec_key[_stable_bucket_key(uid, tuid, k)] = tuple(
                    per_scope_label[si][tuid] for si in range(G)
                )
    trie_res = torch.zeros_like(episode_scores)
    for idx in range(len(prompt_ids)):
        uid = prompt_ids[idx]
        tuid = traj_ids[idx]
        k = int(turn_idxs[idx])
        R = float(episode_scores[idx].item())
        acc = 0.0
        for si in range(G):
            lm = loo_cache.get((uid, k, si), {}).get(tuid, R)
            acc += w[si] * (R - lm)
        trie_res[idx] = acc
    merged_stats: dict[str, float] = {
        "actor/A3_algo_ms_G": float(G),
        "actor/A3_algo_ms_n_clustering_cells": float(n_clustering_cells),
        "actor/A3_algo_ms_n_distance_pairs": float(n_distance_pairs),
    }
    n_cells_per_si = len(per_si_cell_stats[0]) if G else 0
    merged_stats["actor/A3_algo_ms_n_uid_turn_cells"] = float(n_cells_per_si)
    for si in range(G):
        pairs = per_si_cell_stats[si]
        if not pairs:
            continue
        n_trajs = [p[0] for p in pairs]
        n_uni = [p[1] for p in pairs]
        merged_stats[f"actor/A3_algo_ms_si{si}_mean_trajs_per_cell"] = float(
            np.mean(n_trajs)
        )
        merged_stats[f"actor/A3_algo_ms_si{si}_mean_unique_clusters_per_cell"] = float(
            np.mean(n_uni)
        )
        merged_stats[f"actor/A3_algo_ms_si{si}_frac_cells_ge2_clusters"] = float(
            np.mean([1.0 if u >= 2 else 0.0 for u in n_uni])
        )
        merged_stats[f"actor/A3_algo_ms_si{si}_max_unique_clusters_in_cell"] = float(
            max(n_uni)
        )
    return trie_res, merged_stats, vec_key

def _decode_row_text(
    responses: torch.Tensor, response_mask: torch.Tensor, row: int, tokenizer
) -> str:
    row_ids = responses[row]
    m = response_mask[row] > 0
    ids = row_ids[m].tolist()
    return tokenizer.decode(ids, skip_special_tokens=True)

def _extract_bash_from_model_text(text: str) -> str:
    m = _CODE_BLOCK.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()[:8192]

def compute_A3_algo_advantage(
    data,
    *,
    multi_turn: bool,
    epsilon: float,
    adv_bound: float,
    trie_min_count: int,
    w_trie: float,
    support_prior: float,
    decision_gamma: float,
    w_action_cluster: float,
    tokenizer,
    intent_scopes,
    w_intent_scopes,
    A3_algo_tree_intent_match_tau: float,
    A3_algo_tree_bucket_time_decay: float = 0.85,
):
    if isinstance(intent_scopes, str):
        intent_scopes = json.loads(intent_scopes)
    if isinstance(w_intent_scopes, str):
        w_intent_scopes = json.loads(w_intent_scopes)
    intent_scopes = tuple(int(x) for x in intent_scopes)
    w_intent_scopes = tuple(float(x) for x in w_intent_scopes)
    A3_algo_tree_intent_match_tau = float(A3_algo_tree_intent_match_tau)
    A3_algo_tree_bucket_time_decay = float(A3_algo_tree_bucket_time_decay)
    if tokenizer is None:
        raise ValueError(
            "A3_algo requires a tokenizer to decode responses into bash for BashIntent clustering and tree grouping; "
            "got tokenizer=None."
        )
    response_mask = build_response_mask(data, multi_turn=multi_turn)
    episode_scores = get_episode_scores(data)
    prompt_ids = np.asarray(
        [str(x) for x in data.non_tensor_batch["uid"]], dtype=object
    )
    traj_ids = np.asarray(
        [str(x) for x in data.non_tensor_batch["traj_uid"]], dtype=object
    )
    turn_idxs = np.asarray(data.non_tensor_batch["turn_idx"], dtype=np.int64)
    action_keys = _build_payload_action_keys(data.batch["responses"], response_mask)
    n = len(prompt_ids)
    responses = data.batch["responses"]
    bash_texts = [
        _extract_bash_from_model_text(
            _decode_row_text(responses, response_mask, i, tokenizer)
        )
        for i in range(n)
    ]
    space = BashIntentActionSpace()
    backbone_adv = _compute_prompt_backbone_advantage(
        episode_scores,
        prompt_ids=prompt_ids,
        traj_ids=traj_ids,
        epsilon=epsilon,
    )
    trie_res, trie_dbg, vec_key = _multi_scale_turn_vectors_and_residual(
        episode_scores,
        prompt_ids,
        traj_ids,
        turn_idxs,
        bash_texts,
        space,
        trie_min_count,
        intent_scopes=intent_scopes,
        w_intent_scopes=w_intent_scopes,
    )
    trie_bounded = _trie_residual_backbone_style(
        trie_res,
        prompt_ids,
        traj_ids,
        epsilon=epsilon,
    )
    G = len(intent_scopes)
    cluster_action_keys: list[tuple[int, ...]] = []
    cluster_labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        key = _stable_bucket_key(prompt_ids[i], traj_ids[i], int(turn_idxs[i]))
        vec = vec_key.get(key, tuple(0 for _ in range(G)))
        cluster_action_keys.append(tuple(int(x) for x in vec))
        cluster_labels[i] = int(vec[0]) if vec else 0
    tree_tau = A3_algo_tree_intent_match_tau
    traj_turn_tree = _traj_turn_bash_map(prompt_ids, traj_ids, turn_idxs, bash_texts)
    tree_state_keys, tree_action_keys = _tree_state_action_keys_g_bucket(
        n,
        traj_turn_tree,
        vec_key,
        prompt_ids,
        traj_ids,
        turn_idxs,
        w_intent_scopes,
        A3_algo_tree_bucket_time_decay,
        tree_tau,
    )
    exact_tree_return, exact_tree_support = _compute_tree_decision_return(
        episode_scores=episode_scores,
        traj_ids=traj_ids,
        turn_idxs=turn_idxs,
        state_keys=tree_state_keys,
        action_keys=tree_action_keys,
        epsilon=epsilon,
        support_prior=support_prior,
        decision_gamma=decision_gamma,
    )
    tree_gate = exact_tree_support / (exact_tree_support + support_prior + epsilon)
    tree_component = tree_gate * exact_tree_return
    w_trie_f = float(w_trie)
    w_tree_f = float(w_action_cluster)
    mae_bb = backbone_adv.abs().mean() + _RESIDUAL_EPS
    mae_tr = trie_bounded.abs().mean() + _RESIDUAL_EPS
    mae_tc = tree_component.abs().mean() + _RESIDUAL_EPS
    bb_n = backbone_adv / mae_bb
    tr_n = trie_bounded / mae_tr
    tc_n = tree_component / mae_tc
    contrib_trie = w_trie_f * tr_n
    contrib_tree = w_tree_f * tc_n
    pre_tanh = bb_n + contrib_trie + contrib_tree
    scalar_adv = smooth_bounded_score(pre_tanh, bound=adv_bound)
    token_adv = scalar_adv.unsqueeze(-1) * response_mask
    traj_groups = build_group_indices(list(traj_ids))
    rep_rows = [group[0] for group in traj_groups]
    rep_tensor = torch.as_tensor(rep_rows, device=scalar_adv.device, dtype=torch.long)
    delta_tb = (
        (
            trie_bounded.index_select(0, rep_tensor)
            - backbone_adv.index_select(0, rep_tensor)
        )
        .abs()
        .mean()
    )
    n_hit = sum(
        1
        for i in range(n)
        if _stable_bucket_key(prompt_ids[i], traj_ids[i], int(turn_idxs[i])) in vec_key
    )
    merged_metrics = {
        "actor/A3_algo_tree_intent_match_tau": float(tree_tau),
        "actor/A3_algo_tree_bucket_time_decay": float(A3_algo_tree_bucket_time_decay),
        "actor/A3_algo_tree_g_bucket_grouping": 1.0,
        "actor/A3_algo_vec_key_row_hit_frac": float(n_hit) / float(max(n, 1)),
        "actor/A3_algo_backbone_abs_mean": backbone_adv.abs().mean().item(),
        "actor/A3_algo_merge_mae_backbone": mae_bb.item(),
        "actor/A3_algo_merge_mae_trie": mae_tr.item(),
        "actor/A3_algo_merge_mae_tree": mae_tc.item(),
        "actor/A3_algo_trie_bounded_abs_mean": trie_bounded.abs().mean().item(),
        "actor/A3_algo_w_trie": float(w_trie),
        "actor/A3_algo_w_action_cluster": float(w_action_cluster),
        "actor/A3_algo_support_prior": float(support_prior),
        "actor/A3_algo_decision_gamma": float(decision_gamma),
        "actor/A3_algo_tree_return_abs_mean": exact_tree_return.abs().mean().item(),
        "actor/A3_algo_tree_component_abs_mean": tree_component.abs().mean().item(),
        "actor/A3_algo_tree_gate_mean": tree_gate.mean().item(),
        "actor/A3_algo_final_abs_mean": scalar_adv.abs().mean().item(),
        "actor/A3_algo_final_nonzero_ratio": (scalar_adv.abs() > epsilon)
        .to(scalar_adv.dtype)
        .mean()
        .item(),
        "actor/A3_algo_dbg_mean_abs_trie_minus_backbone_at_traj": delta_tb.item(),
        "actor/A3_algo_design_abs_mean_w_trie_trie": contrib_trie.abs().mean().item(),
        "actor/A3_algo_design_abs_mean_w_tree_tree": contrib_tree.abs().mean().item(),
        "actor/A3_algo_design_abs_mean_pre_tanh_sum": pre_tanh.abs().mean().item(),
        "actor/A3_algo_design_pre_tanh_over_adv_bound": (
            pre_tanh.abs().mean() / adv_bound
        ).item(),
        "actor/A3_algo_design_unique_cluster_labels_dim0": float(
            len(np.unique(cluster_labels))
        ),
        "actor/A3_algo_unique_intent_vectors_in_batch": float(
            len(set(cluster_action_keys))
        ),
        "actor/A3_algo_intent_G": float(G),
    }
    bb_denom = bb_n.abs().mean().item() + 1e-12
    merged_metrics["actor/A3_algo_design_ratio_abs_w_trie_term_to_backbone"] = (
        merged_metrics["actor/A3_algo_design_abs_mean_w_trie_trie"] / bb_denom
    )
    merged_metrics["actor/A3_algo_design_ratio_abs_w_tree_term_to_backbone"] = (
        merged_metrics["actor/A3_algo_design_abs_mean_w_tree_tree"] / bb_denom
    )
    merged_metrics.update(trie_dbg)
    data.meta_info["algorithm_debug_metrics"] = merged_metrics
    data.batch.set_("advantages", token_adv)
    data.batch.set_("returns", token_adv)
    return data

def compute_policy_loss_A3_algo(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange,
    cliprange_low,
    cliprange_high,
    clip_ratio_c,
    loss_agg_mode: str,
):
    return compute_policy_loss_gspo_sequence(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        cliprange=cliprange,
        cliprange_low=cliprange_low,
        cliprange_high=cliprange_high,
        clip_ratio_c=clip_ratio_c,
        loss_agg_mode=loss_agg_mode,
    )
