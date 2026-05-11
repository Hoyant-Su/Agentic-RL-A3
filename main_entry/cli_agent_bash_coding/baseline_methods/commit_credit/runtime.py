from dataclasses import dataclass
from typing import Any
from omegaconf import OmegaConf
from main_entry.cli_agent_bash_coding.action_schema import parse_bash_coding_action

def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

@dataclass(frozen=True)
class CommitCreditConfig:
    enable: bool
    coef: float
    clip_value: float

    @classmethod
    def from_config(cls, config) -> "CommitCreditConfig":
        enable = _as_bool(
            OmegaConf.select(config, "reward.commit_credit_enable", default=False)
        )
        coef = float(OmegaConf.select(config, "reward.commit_credit_coef", default=0.2))
        clip_value = float(
            OmegaConf.select(config, "reward.commit_credit_clip_value", default=1.0)
        )
        return cls(enable=enable, coef=coef, clip_value=clip_value)

class CommitCreditRuntime:
    method_name = "commit_credit"

    def __init__(self, config: CommitCreditConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config) -> "CommitCreditRuntime":
        return cls(CommitCreditConfig.from_config(config))

    def is_enabled(self) -> bool:
        return self.config.enable

    @staticmethod
    def _build_empty_results(total_batch_list) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for traj_steps in total_batch_list:
            step_count = len(traj_steps)
            results.append(
                {
                    "commit_credit_bonus": [0.0 for _ in range(step_count)],
                    "commit_credit_branch_gap": [0.0 for _ in range(step_count)],
                    "commit_credit_v_keep": [0.0 for _ in range(step_count)],
                    "commit_credit_v_rollback": [0.0 for _ in range(step_count)],
                    "commit_credit_effective_decision": [
                        0.0 for _ in range(step_count)
                    ],
                }
            )
        return results

    @staticmethod
    def _extract_step_commit(step_data, tokenizer) -> str:
        response_ids = step_data.get("responses", None)
        if response_ids is None:
            return ""
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        parsed = parse_bash_coding_action(response_text, enable_commit=True)
        if parsed is None or getattr(parsed, "commit", None) is None:
            return ""
        return str(parsed.commit).strip().lower()

    @staticmethod
    def _prev_step_reward(traj_steps: list, step_idx: int) -> float:
        if step_idx <= 0:
            return 0.0
        prev = traj_steps[step_idx - 1]
        return float(prev.get("step_reward_total", 0.0))

    @staticmethod
    def _traj_step_reward_min_max(traj_steps: list) -> tuple[float, float]:
        vals = [float(s.get("step_reward_total", 0.0)) for s in traj_steps]
        if not vals:
            return 0.0, 0.0
        return min(vals), max(vals)

    @staticmethod
    def _relative_quality_u(r_prev: float, s_min: float, s_max: float) -> float:
        span = s_max - s_min
        if span <= 1e-12:
            return 0.5
        return (r_prev - s_min) / span

    def apply_post_rollout(
        self,
        *,
        tasks: list[str],
        total_batch_list,
        total_infos,
        final_infos: list[dict[str, Any]],
        episode_rewards,
        group_n: int,
        tokenizer,
        actor_rollout_wg,
        meta_info,
        generate_freeform_texts,
    ) -> list[dict[str, Any]]:
        _ = tasks
        _ = total_infos
        _ = final_infos
        _ = episode_rewards
        _ = group_n
        _ = actor_rollout_wg
        _ = meta_info
        _ = generate_freeform_texts
        results = self._build_empty_results(total_batch_list)
        if not self.is_enabled():
            return results
        coef = self.config.coef
        clip_v = self.config.clip_value
        for traj_idx, traj_steps in enumerate(total_batch_list):
            s_min, s_max = self._traj_step_reward_min_max(traj_steps)
            for step_idx, step_data in enumerate(traj_steps):
                commit = self._extract_step_commit(step_data, tokenizer)
                if commit not in {"keep", "rollback"}:
                    continue
                info = total_infos[traj_idx][step_idx]
                if not bool(info.get("cib_prev_decision_effective", False)):
                    continue
                results[traj_idx]["commit_credit_effective_decision"][step_idx] = 1.0
                if step_idx == 0:
                    continue
                r_prev = self._prev_step_reward(traj_steps, step_idx)
                u = self._relative_quality_u(r_prev, s_min, s_max)
                align_keep = 2.0 * u - 1.0
                align_rollback = 1.0 - 2.0 * u
                raw = coef * align_keep if commit == "keep" else coef * align_rollback
                if clip_v > 0:
                    raw = max(min(raw, clip_v), -clip_v)
                results[traj_idx]["commit_credit_bonus"][step_idx] = float(raw)
                results[traj_idx]["commit_credit_branch_gap"][step_idx] = float(u)
                results[traj_idx]["commit_credit_v_keep"][step_idx] = (
                    float(align_keep) if commit == "keep" else 0.0
                )
                results[traj_idx]["commit_credit_v_rollback"][step_idx] = (
                    float(align_rollback) if commit == "rollback" else 0.0
                )
        return results
