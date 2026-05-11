from dataclasses import dataclass
from typing import Any, Dict, Tuple
import torch
from main_entry.cli_agent_bash_coding.action_schema import get_bash_coding_action_spans
from main_entry.cli_agent_bash_coding.action_schema import parse_bash_coding_action
from main_entry.cli_agent_bash_coding.reward_provenance import (
    REWARD_PROVENANCE_KEYS,
    zero_reward_provenance,
)
from verl import DataProto

@dataclass(frozen=True)
class _ValueSpan:
    full_response: Tuple[int, int] | None
    plan: Tuple[int, int] | None
    payload: Tuple[int, int] | None
    commit: Tuple[int, int] | None = None

class SegmentedEpisodeRewardManager:
    def __init__(
        self,
        tokenizer,
        num_examine: int,
        normalize_by_length: bool = False,
        enable_commit: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = int(num_examine)
        self.normalize_by_length = bool(normalize_by_length)
        self.enable_commit = bool(enable_commit)

    @staticmethod
    def _full_response_span(valid_response_length: int) -> Tuple[int, int] | None:
        if valid_response_length <= 0:
            return None
        return 0, valid_response_length

    def _char_span_to_token_span(
        self, text: str, span: Tuple[int, int] | None
    ) -> Tuple[int, int] | None:
        if span is None:
            return None
        start_char, end_char = span
        start_tok = len(
            self.tokenizer.encode(text[:start_char], add_special_tokens=False)
        )
        end_tok = len(self.tokenizer.encode(text[:end_char], add_special_tokens=False))
        if end_tok <= start_tok:
            return None
        return start_tok, end_tok

    def _value_spans(self, response_str: str, valid_response_length: int) -> _ValueSpan:
        full_span = self._full_response_span(valid_response_length)
        if (
            parse_bash_coding_action(response_str, enable_commit=self.enable_commit)
            is None
        ):
            return _ValueSpan(
                full_response=full_span, plan=None, payload=None, commit=None
            )
        char_spans = get_bash_coding_action_spans(
            response_str, enable_commit=self.enable_commit
        )
        if char_spans is None:
            return _ValueSpan(
                full_response=full_span, plan=None, payload=None, commit=None
            )
        plan_span = self._char_span_to_token_span(response_str, char_spans.plan)
        payload_span = self._char_span_to_token_span(response_str, char_spans.payload)
        commit_span = None
        if self.enable_commit and char_spans.commit is not None:
            commit_span = self._char_span_to_token_span(response_str, char_spans.commit)
        return _ValueSpan(
            full_response=full_span,
            plan=plan_span,
            payload=payload_span,
            commit=commit_span,
        )

    @staticmethod
    def _assign_uniform(
        reward_row: torch.Tensor, span: Tuple[int, int], value: float
    ) -> None:
        s, e = span
        n = e - s
        reward_row[s:e] += float(value) / float(n)

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {
                    "reward_tensor": data.batch["rm_scores"],
                    "reward_extra_info": {},
                }
            return data.batch["rm_scores"]
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print_data_sources: Dict[str, int] = {}
        extra: Dict[str, list[Any]] = {
            "base_score": [],
        }
        for key in REWARD_PROVENANCE_KEYS:
            extra.setdefault(key, [])
        for i in range(len(data)):
            item = data[i]
            prompt_ids = item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(
                item.batch["attention_mask"][:prompt_length].sum().item()
            )
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = item.batch["responses"]
            valid_response_length = int(
                item.batch["attention_mask"][prompt_length:].sum().item()
            )
            valid_response_ids = response_ids[:valid_response_length]
            prompt_str = self.tokenizer.decode(
                valid_prompt_ids, skip_special_tokens=False
            )
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=True
            )
            data_source = item.non_tensor_batch["data_source"]
            episode_rewards = item.non_tensor_batch["episode_rewards"]
            episode_lengths = item.non_tensor_batch["episode_lengths"]
            score = (
                (episode_rewards / episode_lengths)
                if self.normalize_by_length
                else episode_rewards
            )
            score_f = float(score)
            spans = self._value_spans(response_str, valid_response_length)
            target_span = (
                spans.payload if spans.payload is not None else spans.full_response
            )
            self._assign_uniform(reward_tensor[i], target_span, score_f)
            extra["base_score"].append(score_f)
            precomputed_retro = item.non_tensor_batch.get("retroagent_bonus", None)
            if precomputed_retro is None:
                retroagent_bonus_to_assign = 0.0
                extra.setdefault("retroagent_bonus", []).append(0.0)
                extra.setdefault("retroagent_numerical_bonus", []).append(0.0)
                extra.setdefault("retroagent_language_bonus", []).append(0.0)
                extra.setdefault("retroagent_memory_size", []).append(0.0)
                extra.setdefault("retroagent_retrieved_count", []).append(0.0)
                extra.setdefault("retroagent_current_phi", []).append(0.0)
                extra.setdefault("retroagent_previous_phi", []).append(0.0)
                extra.setdefault("retroagent_raw_improvement", []).append(0.0)
            else:
                retroagent_bonus_to_assign = 0.0
                extra.setdefault("retroagent_bonus", []).append(
                    float(precomputed_retro)
                )
                extra.setdefault("retroagent_numerical_bonus", []).append(
                    float(item.non_tensor_batch.get("retroagent_numerical_bonus", 0.0))
                )
                extra.setdefault("retroagent_language_bonus", []).append(
                    float(item.non_tensor_batch.get("retroagent_language_bonus", 0.0))
                )
                extra.setdefault("retroagent_memory_size", []).append(
                    float(item.non_tensor_batch.get("retroagent_memory_size", 0.0))
                )
                extra.setdefault("retroagent_retrieved_count", []).append(
                    float(item.non_tensor_batch.get("retroagent_retrieved_count", 0.0))
                )
                extra.setdefault("retroagent_current_phi", []).append(
                    float(item.non_tensor_batch.get("retroagent_current_phi", 0.0))
                )
                extra.setdefault("retroagent_previous_phi", []).append(
                    float(item.non_tensor_batch.get("retroagent_previous_phi", 0.0))
                )
                extra.setdefault("retroagent_raw_improvement", []).append(
                    float(item.non_tensor_batch.get("retroagent_raw_improvement", 0.0))
                )
            if retroagent_bonus_to_assign != 0.0:
                self._assign_uniform(
                    reward_tensor[i], target_span, retroagent_bonus_to_assign
                )
            commit_credit_bonus = float(
                item.non_tensor_batch.get("commit_credit_bonus", 0.0)
            )
            extra.setdefault("commit_credit_bonus", []).append(commit_credit_bonus)
            extra.setdefault("commit_credit_branch_gap", []).append(
                float(item.non_tensor_batch.get("commit_credit_branch_gap", 0.0))
            )
            extra.setdefault("commit_credit_v_keep", []).append(
                float(item.non_tensor_batch.get("commit_credit_v_keep", 0.0))
            )
            extra.setdefault("commit_credit_v_rollback", []).append(
                float(item.non_tensor_batch.get("commit_credit_v_rollback", 0.0))
            )
            extra.setdefault("commit_credit_effective_decision", []).append(
                float(
                    item.non_tensor_batch.get("commit_credit_effective_decision", 0.0)
                )
            )
            if commit_credit_bonus != 0.0:
                if spans is not None and spans.commit is not None:
                    self._assign_uniform(
                        reward_tensor[i], spans.commit, commit_credit_bonus
                    )
                elif spans is not None and spans.full_response is not None:
                    self._assign_uniform(
                        reward_tensor[i], spans.full_response, commit_credit_bonus
                    )
            episode_reward_provenance = item.non_tensor_batch.get(
                "episode_reward_provenance", None
            )
            if not isinstance(episode_reward_provenance, dict):
                episode_reward_provenance = zero_reward_provenance()
            for key in REWARD_PROVENANCE_KEYS:
                component = float(episode_reward_provenance.get(key, 0.0))
                extra[key].append(component)
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"[{data_source}][prompt]", prompt_str)
                print(f"[{data_source}][response]", response_str)
                print(f"[{data_source}][score]", score_f)
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": extra}
        return reward_tensor
