from collections import Counter
from functools import partial
from typing import List, Tuple, Dict, Any, Set
import numpy as np
import torch
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
import os
from agent_system.memory import SimpleMemory as BaselineSimpleMemory
from omegaconf import OmegaConf
from main_entry.cli_agent_bash_coding.baseline_methods import BaselineMethodManager
from main_entry.cli_agent_bash_coding.action_schema import uses_commit_action_schema
from main_entry.cli_agent_bash_coding.env_package.bash_coding import (
    build_bash_coding_envs,
    bash_coding_projection,
)
from main_entry.cli_agent_bash_coding.env_package.bash_coding.projection import (
    ANSWER_PREFIX,
)
from main_entry.cli_agent_bash_coding.harness import build_bash_coding_harness
from main_entry.cli_agent_bash_coding.memory import SimpleMemory as ResearchSimpleMemory
from main_entry.cli_agent_bash_coding.tooling.action_space_similarity.bash_intent_action_space import (
    BashIntentActionSpace,
)
from main_entry.cli_agent_bash_coding.tooling.semantic_similarity.semantic_similarity import (
    semantic_similarity_batch,
)

class BashCodingEnvironmentManager(EnvironmentManagerBase):

    def __init__(self, envs, projection_f, config):
        enabled = str(os.environ.get("BASH_CODING_ENABLE", "0")).strip() == "1"
        self.memory = ResearchSimpleMemory() if enabled else BaselineSimpleMemory()
        self.method_manager = BaselineMethodManager.from_config(config)
        self.harness = build_bash_coding_harness(
            OmegaConf.to_container(config.env, resolve=True)
        )
        self.sample_kwargs_list: List[Dict[str, Any]] = []
        self.semantic_similarity_url = os.environ.get(
            "BASH_CODING_SEMANTIC_SIMILARITY_URL", "http://127.0.0.1:30003"
        )
        self.semantic_similarity_batch_size = int(
            os.environ.get("BASH_CODING_SEMANTIC_SIMILARITY_BATCH_SIZE", "32")
        )
        self._sigma_space: BashIntentActionSpace | None = (
            BashIntentActionSpace()
            if (
                getattr(self.harness, "sigma_reject_enabled", False)
                or getattr(self.harness, "sigma_cmi_enabled", False)
                or getattr(self.harness, "sigma_tcm_enabled", False)
                or getattr(self.harness, "sigma_antimode_enabled", False)
                or getattr(self.harness, "sigma_cohort_enabled", False)
                or getattr(self.harness, "sigma_witness_enabled", False)
                or getattr(self.harness, "sigma_dual_enabled", False)
                or getattr(self.harness, "sigma_concentration_enabled", False)
            )
            else None
        )
        self._sigma_seen: List[Set[Tuple[str, ...]]] = []
        self._sigma_counts: List[Counter] = []
        self._sigma_concentration_tau: float = float(
            getattr(config.env, "sigma_concentration_tau", 0.8)
        )
        super().__init__(envs, projection_f, config)

    @staticmethod
    def _expand_kwargs_for_processes(
        kwargs, total_processes: int, group_n: int
    ) -> List[Dict[str, Any]]:
        if isinstance(kwargs, np.ndarray):
            kwargs_list = kwargs.tolist()
        else:
            kwargs_list = list(kwargs)
        if len(kwargs_list) * group_n == total_processes and group_n > 1:
            expanded_kwargs: List[Dict[str, Any]] = []
            for item in kwargs_list:
                expanded_kwargs.extend([item] * group_n)
            return expanded_kwargs
        return kwargs_list

    def _apply_string_semantic_progress(
        self,
        actions: List[str],
        rewards: np.ndarray,
        infos: List[Dict[str, Any]],
        indices: List[int],
    ) -> np.ndarray:
        reward_array = np.asarray(rewards, dtype=np.float32).copy()
        if not bool(int(getattr(self.config.env, "use_model_evidence_gain", 0))):
            return reward_array
        score_requests: List[tuple[str, str]] = []
        score_targets: List[int] = []
        for local_idx, action in enumerate(actions):
            if not action.startswith(ANSWER_PREFIX):
                continue
            if not bool(infos[local_idx].get("won", False)):
                continue
            global_idx = indices[local_idx]
            sample_kwargs = self.sample_kwargs_list[global_idx]
            reward_spec = sample_kwargs["reward_spec"]
            if reward_spec.get("type") != "string":
                continue
            answer_text = action[len(ANSWER_PREFIX) :].strip()
            expected = str(reward_spec["expected"])
            score_requests.append((answer_text, expected))
            score_targets.append(local_idx)
        raw_scores = semantic_similarity_batch(
            score_requests,
            api_base_url=self.semantic_similarity_url,
            batch_size=self.semantic_similarity_batch_size,
        )
        for local_idx, score in zip(score_targets, raw_scores):
            bonus = float(self.config.env.progress_gain_coef) * float(score)
            reward_array[local_idx] += bonus
            infos[local_idx]["semantic_similarity"] = float(score)
            provenance = infos[local_idx]["reward_provenance"]
            provenance["progress_gain_coef"] = (
                float(provenance["progress_gain_coef"]) + bonus
            )
            infos[local_idx]["reward_provenance"] = provenance
            infos[local_idx]["reward_total"] = (
                float(infos[local_idx]["reward_total"]) + bonus
            )
        return reward_array

    def sigma_reject_active(self) -> bool:
        return self._sigma_space is not None

    def would_be_sigma_repeat(
        self,
        text_actions: List[str],
        active_masks: np.ndarray | None = None,
    ) -> List[bool]:
        if self._sigma_space is None:
            return [False] * len(text_actions)
        projected, _ = self.projection_f(list(text_actions))
        result = [False] * len(text_actions)
        for i, action_text in enumerate(projected):
            if active_masks is not None and not bool(active_masks[i]):
                continue
            if i >= len(self._sigma_seen):
                continue
            sig = tuple(self._sigma_space.intent_signature(str(action_text)))
            if sig in self._sigma_seen[i]:
                result[i] = True
        return result

    def _commit_sigma_seen(self, actions: List[str]) -> None:
        for i, action_text in enumerate(actions):
            if i >= len(self._sigma_seen):
                continue
            sig = tuple(self._sigma_space.intent_signature(str(action_text)))
            self._sigma_seen[i].add(sig)
            self._sigma_counts[i][sig] += 1

    def sigma_exit_active(self) -> bool:
        return bool(getattr(self.harness, "sigma_exit_enabled", False))

    def needs_exit_resample(
        self,
        text_actions: List[str],
        active_masks: np.ndarray | None = None,
    ) -> List[bool]:
        if not self.sigma_exit_active():
            return [False] * len(text_actions)
        projected, _ = self.projection_f(list(text_actions))
        result = [False] * len(text_actions)
        for i, action_text in enumerate(projected):
            if active_masks is not None and not bool(active_masks[i]):
                continue
            if not str(action_text).startswith(ANSWER_PREFIX):
                result[i] = True
        return result

    def sigma_cmi_active(self) -> bool:
        return bool(getattr(self.harness, "sigma_cmi_enabled", False))

    @staticmethod
    def mean_rollout_logprob(rollout_log_probs: torch.Tensor) -> np.ndarray:
        rlp = rollout_log_probs.detach().to(torch.float32)
        mask = (rlp != -1.0).to(torch.float32)
        denom = mask.sum(dim=-1).clamp(min=1.0)
        mean_logp = (rlp * mask).sum(dim=-1) / denom
        return mean_logp.cpu().numpy()

    def sigma_cmi_decide(
        self,
        text_actions_a: List[str],
        text_actions_b: List[str],
        mean_logp_a: np.ndarray,
        mean_logp_b: np.ndarray,
        active_masks: np.ndarray,
        beta: float,
    ) -> List[bool]:
        n = len(text_actions_a)
        if not self.sigma_cmi_active():
            return [False] * n
        projected_a, _ = self.projection_f(list(text_actions_a))
        projected_b, _ = self.projection_f(list(text_actions_b))
        result = [False] * n
        for i in range(n):
            if active_masks is not None and not bool(active_masks[i]):
                continue
            if i >= len(self._sigma_counts):
                continue
            s_a = tuple(self._sigma_space.intent_signature(str(projected_a[i])))
            s_b = tuple(self._sigma_space.intent_signature(str(projected_b[i])))
            if s_a == s_b:
                continue
            counts = self._sigma_counts[i]
            score_a = float(mean_logp_a[i]) - beta * float(counts[s_a])
            score_b = float(mean_logp_b[i]) - beta * float(counts[s_b])
            result[i] = score_b > score_a
        return result

    def sigma_tcm_active(self) -> bool:
        return bool(getattr(self.harness, "sigma_tcm_enabled", False))

    def sigma_tcm_decide(
        self,
        candidate_texts: List[List[str]],
        active_masks: np.ndarray,
    ) -> List[int]:
        n = len(candidate_texts[0])
        result = [0] * n
        if not self.sigma_tcm_active():
            return result
        k = len(candidate_texts)
        majority_threshold = (k // 2) + 1
        projected_per_round = [
            self.projection_f(list(round_texts))[0] for round_texts in candidate_texts
        ]
        for i in range(n):
            if active_masks is not None and not bool(active_masks[i]):
                continue
            sigs: List[Tuple[str, ...]] = [
                tuple(
                    self._sigma_space.intent_signature(str(projected_per_round[j][i]))
                )
                for j in range(k)
            ]
            vote = Counter(sigs)
            mode_sig, mode_count = vote.most_common(1)[0]
            if mode_count < majority_threshold:
                continue
            result[i] = next(j for j in range(k) if sigs[j] == mode_sig)
        return result

    _TERMINAL_SELECT_MODES: Tuple[Tuple[str, str], ...] = (
        ("antimode", "sigma_antimode_enabled"),
        ("cohort", "sigma_cohort_enabled"),
        ("witness", "sigma_witness_enabled"),
        ("dual", "sigma_dual_enabled"),
        ("concentration", "sigma_concentration_enabled"),
    )

    def sigma_terminal_select_active(self) -> bool:
        return any(
            getattr(self.harness, flag, False)
            for _, flag in self._TERMINAL_SELECT_MODES
        )

    def sigma_terminal_select_mode(self) -> str:
        for mode, flag in self._TERMINAL_SELECT_MODES:
            if getattr(self.harness, flag, False):
                return mode
        return ""

    def _terminal_project_sigs(
        self,
        candidate_texts: List[List[str]],
    ) -> List[List[Tuple[str, ...]]]:
        return [
            [
                tuple(self._sigma_space.intent_signature(str(a)))
                for a in self.projection_f(list(round_texts))[0]
            ]
            for round_texts in candidate_texts
        ]

    def _decide_antimode(self, sigs_per_round, active_masks, n, k) -> List[int]:
        result = [0] * n
        majority_threshold = (k // 2) + 1
        for i in range(n):
            if active_masks is not None and not bool(active_masks[i]):
                continue
            sigs = [sigs_per_round[j][i] for j in range(k)]
            ctr = Counter(sigs)
            mode_sig, mode_count = ctr.most_common(1)[0]
            if mode_count < majority_threshold:
                continue
            minorities = [j for j in range(k) if sigs[j] != mode_sig]
            if not minorities:
                continue
            result[i] = minorities[0]
        return result

    def _decide_cohort(self, sigs_per_round, active_masks, n, k) -> List[int]:
        result = [0] * n
        for i in range(n):
            if active_masks is not None and not bool(active_masks[i]):
                continue
            sigs_i = [list(sigs_per_round[j][i]) for j in range(k)]
            best_j, best_score = 0, -1.0
            for j in range(k):
                nearest = min(
                    BashIntentActionSpace._norm_dist(sigs_i[j], sigs_i[l])
                    for l in range(k)
                    if l != j
                )
                if nearest > best_score:
                    best_j, best_score = j, nearest
            result[i] = best_j
        return result

    def _decide_witness(self, sigs_per_round, active_masks, n, k) -> List[int]:
        result = [0] * n
        for i in range(n):
            if active_masks is not None and not bool(active_masks[i]):
                continue
            if i >= len(self._sigma_counts):
                continue
            counts_i = self._sigma_counts[i]
            sigs = [sigs_per_round[j][i] for j in range(k)]
            best_j, best_count = 0, counts_i[sigs[0]]
            for j in range(1, k):
                c = counts_i[sigs[j]]
                if c < best_count:
                    best_j, best_count = j, c
            result[i] = best_j
        return result

    def _decide_dual(self, sigs_per_round, active_masks, n, k) -> List[int]:
        result = [0] * n
        majority_threshold = (k // 2) + 1
        for i in range(n):
            if active_masks is not None and not bool(active_masks[i]):
                continue
            if i >= len(self._sigma_counts):
                continue
            sigs = [sigs_per_round[j][i] for j in range(k)]
            ctr = Counter(sigs)
            mode_sig, mode_count = ctr.most_common(1)[0]
            if mode_count < majority_threshold:
                continue
            if self._sigma_counts[i][mode_sig] < 1:
                continue
            minorities = [j for j in range(k) if sigs[j] != mode_sig]
            if not minorities:
                continue
            result[i] = minorities[0]
        return result

    def _decide_concentration(self, sigs_per_round, active_masks, n, k) -> List[int]:
        result = [0] * n
        tau = float(getattr(self, "_sigma_concentration_tau", 0.8))
        for i in range(n):
            if active_masks is not None and not bool(active_masks[i]):
                continue
            sigs_i = [list(sigs_per_round[j][i]) for j in range(k)]
            dmat = [
                [
                    BashIntentActionSpace._norm_dist(sigs_i[j], sigs_i[l])
                    if j != l
                    else 0.0
                    for l in range(k)
                ]
                for j in range(k)
            ]
            pair_sum = 0.0
            pair_n = 0
            for j in range(k):
                for l in range(j + 1, k):
                    pair_sum += dmat[j][l]
                    pair_n += 1
            d_mean = pair_sum / pair_n if pair_n > 0 else 0.0
            if d_mean <= tau:
                best_j, best = 0, float("inf")
                for j in range(k):
                    s = sum(dmat[j])
                    if s < best:
                        best_j, best = j, s
                result[i] = best_j
            else:
                best_j, best = 0, -1.0
                for j in range(k):
                    nearest = min(dmat[j][l] for l in range(k) if l != j)
                    if nearest > best:
                        best_j, best = j, nearest
                result[i] = best_j
        return result

    def sigma_terminal_decide(
        self,
        candidate_texts: List[List[str]],
        active_masks: np.ndarray,
    ) -> List[int]:
        n = len(candidate_texts[0])
        if not self.sigma_terminal_select_active():
            return [0] * n
        k = len(candidate_texts)
        sigs_per_round = self._terminal_project_sigs(candidate_texts)
        mode = self.sigma_terminal_select_mode()
        if mode == "antimode":
            return self._decide_antimode(sigs_per_round, active_masks, n, k)
        if mode == "cohort":
            return self._decide_cohort(sigs_per_round, active_masks, n, k)
        if mode == "witness":
            return self._decide_witness(sigs_per_round, active_masks, n, k)
        if mode == "dual":
            return self._decide_dual(sigs_per_round, active_masks, n, k)
        if mode == "concentration":
            return self._decide_concentration(sigs_per_round, active_masks, n, k)
        return [0] * n

    def reset(self, kwargs=None) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        group_n = int(getattr(self.envs, "group_n", 1))
        self.sample_kwargs_list = self._expand_kwargs_for_processes(
            kwargs, len(obs), group_n
        )
        if self._sigma_space is not None:
            self._sigma_seen = [set() for _ in range(len(obs))]
            self._sigma_counts = [Counter() for _ in range(len(obs))]
        self.tasks = []
        for i, info in enumerate(infos):
            task = info.get("task") if isinstance(info, dict) else None
            self.tasks.append(task if task else "Complete the bash coding task")
        self.memory.reset(batch_size=len(obs))
        self._last_obs = list(obs)
        self.method_manager.prepare_batch(
            self.tasks,
            is_eval=not bool(getattr(self.envs, "is_train", True)),
            group_n=group_n,
        )
        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy(),
        }
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        if self._sigma_space is not None:
            self._commit_sigma_seen(actions)
        self.harness.apply_history_commit_decisions(
            self.memory, infos, next_text_obs=list(next_obs)
        )
        history_actions = list(text_actions)
        self.memory.store(
            {
                "observation": self._last_obs,
                "observation_for_history": self._last_obs,
                "action": history_actions,
            }
        )
        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy(),
        }
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
        self._last_obs = list(next_obs)
        rewards = to_numpy(rewards)
        rewards = self._apply_string_semantic_progress(
            actions, rewards, infos, list(range(len(actions)))
        )
        dones = to_numpy(dones)
        return next_observations, rewards, dones, infos

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        return self.harness.build_text_obs(self, text_obs, init=init)

    def methods_enabled(self) -> bool:
        return self.method_manager.enabled()

    def apply_baseline_methods_post_rollout(
        self, **kwargs
    ) -> dict[str, list[dict[str, Any]]]:
        group_n = int(getattr(self.envs, "group_n", 1))
        return self.method_manager.apply_post_rollout(
            tasks=self.tasks,
            group_n=group_n,
            **kwargs,
        )

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item["active_masks"]:
                info = total_infos[batch_idx][i]
                won_value = float(info.get("won", False))
                success["success_rate"].append(won_value)
                return

    def export_branch_states(self, indices):
        return self.envs.export_states(list(indices))

    def restore_branch_states(self, indices, states):
        self.envs.import_states(list(indices), list(states))

    def branch_step_from_states(self, indices, states, text_actions):
        actions, valids = self.projection_f(list(text_actions))
        next_obs, rewards, dones, infos = self.envs.branch_step(
            list(indices), list(states), list(actions)
        )
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
        rewards = to_numpy(rewards)
        rewards = self._apply_string_semantic_progress(
            actions, rewards, infos, indices=list(indices)
        )
        return next_obs, rewards, dones, infos

def _build_bash_coding_env_kwargs(config) -> Dict[str, Any]:
    env_kwargs = OmegaConf.to_container(config.env, resolve=True)
    env_kwargs = dict(env_kwargs)
    env_kwargs.pop("env_name", None)
    env_kwargs.pop("seed", None)
    env_kwargs.pop("rollout", None)
    env_kwargs.pop("resources_per_worker", None)
    return env_kwargs

def make_envs(config):
    if "bash_coding" not in config.env.env_name.lower():
        raise ValueError(
            "cli_agent_bash_coding.make_envs only supports bash_coding env_name; "
            f"got {config.env.env_name!r}"
        )
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(
        config.env.resources_per_worker, resolve=True
    )
    env_kwargs = _build_bash_coding_env_kwargs(config)
    train_envs = build_bash_coding_envs(
        seed=config.env.seed,
        env_num=config.data.train_batch_size,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=True,
        **env_kwargs,
    )
    val_envs = build_bash_coding_envs(
        seed=config.env.seed + 1000,
        env_num=config.data.val_batch_size,
        group_n=1,
        resources_per_worker=resources_per_worker,
        is_train=False,
        **env_kwargs,
    )
    projection_f = partial(
        bash_coding_projection,
        enable_commit=uses_commit_action_schema(env_kwargs["bash_coding_harness"]),
    )
    envs = BashCodingEnvironmentManager(train_envs, projection_f, config)
    val_envs = BashCodingEnvironmentManager(val_envs, projection_f, config)
    return envs, val_envs
