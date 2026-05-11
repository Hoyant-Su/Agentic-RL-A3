from dataclasses import dataclass
from typing import Any
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from .memory import RetroAgentMemory
from .reflection import build_reflection_prompt
from .reflection import ParsedReflection
from .reflection import parse_reflection
from .retrieval import rank_memory_entries

def _cfg_select(config, key: str):
    value = OmegaConf.select(config, key, default=None)
    if value is None:
        raise KeyError(f"Missing RetroAgent config: {key}")
    return value

def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

@dataclass(frozen=True)
class RetroAgentConfig:
    enable: bool
    memory_path: str
    top_k: int
    store_threshold: float
    numerical_reward_coef: float
    language_reward_coef: float
    similarity_threshold: float
    max_entries_per_task: int
    reflection_max_tokens: int
    retrieval_type: str
    retrieve_mode: str
    alpha: float
    beta: float
    temperature: float
    ucb_scale: float
    group_relative_intrinsic_rewards: bool
    potential_based_on_binary_success: bool
    full_group_memory: bool
    single_reflection_per_group: bool
    group_outperformance: bool
    embedding_model_path: str
    reflection_max_prompt_tokens: int
    reflection_hard_truncate_tokens: int
    reflection_keep_last_k_steps: int
    reflection_max_obs_chars_per_step: int
    reflection_max_feedback_chars_per_step: int
    reflection_max_changed_files: int

    @classmethod
    def from_config(cls, config) -> "RetroAgentConfig":
        return cls(
            enable=_as_bool(_cfg_select(config, "reward.retroagent_enable")),
            memory_path=str(_cfg_select(config, "reward.retroagent_memory_path")),
            top_k=int(_cfg_select(config, "reward.retroagent_top_k")),
            store_threshold=float(
                _cfg_select(config, "reward.retroagent_store_threshold")
            ),
            numerical_reward_coef=float(
                _cfg_select(config, "reward.retroagent_numerical_reward_coef")
            ),
            language_reward_coef=float(
                _cfg_select(config, "reward.retroagent_language_reward_coef")
            ),
            similarity_threshold=float(
                _cfg_select(config, "reward.retroagent_similarity_threshold")
            ),
            max_entries_per_task=int(
                _cfg_select(config, "reward.retroagent_max_memory_per_task")
            ),
            reflection_max_tokens=int(
                _cfg_select(config, "reward.retroagent_reflection_max_tokens")
            ),
            retrieval_type=str(_cfg_select(config, "reward.retroagent_retrieval_type")),
            retrieve_mode=str(_cfg_select(config, "reward.retroagent_retrieve_mode")),
            alpha=float(_cfg_select(config, "reward.retroagent_alpha")),
            beta=float(_cfg_select(config, "reward.retroagent_beta")),
            temperature=float(_cfg_select(config, "reward.retroagent_temperature")),
            ucb_scale=float(_cfg_select(config, "reward.retroagent_ucb_scale")),
            group_relative_intrinsic_rewards=_as_bool(
                _cfg_select(
                    config, "reward.retroagent_group_relative_intrinsic_rewards"
                )
            ),
            potential_based_on_binary_success=_as_bool(
                _cfg_select(
                    config, "reward.retroagent_potential_based_on_binary_success"
                )
            ),
            full_group_memory=_as_bool(
                _cfg_select(config, "reward.retroagent_full_group_memory")
            ),
            single_reflection_per_group=_as_bool(
                _cfg_select(config, "reward.retroagent_single_reflection_per_group")
            ),
            group_outperformance=_as_bool(
                _cfg_select(config, "reward.retroagent_group_outperformance")
            ),
            embedding_model_path=str(
                _cfg_select(config, "reward.retroagent_embedding_model_path")
            ),
            reflection_max_prompt_tokens=int(
                _cfg_select(config, "reward.retroagent_reflection_max_prompt_tokens")
            ),
            reflection_hard_truncate_tokens=int(
                _cfg_select(config, "reward.retroagent_reflection_hard_truncate_tokens")
            ),
            reflection_keep_last_k_steps=int(
                _cfg_select(config, "reward.retroagent_reflection_keep_last_k_steps")
            ),
            reflection_max_obs_chars_per_step=int(
                _cfg_select(
                    config, "reward.retroagent_reflection_max_obs_chars_per_step"
                )
            ),
            reflection_max_feedback_chars_per_step=int(
                _cfg_select(
                    config, "reward.retroagent_reflection_max_feedback_chars_per_step"
                )
            ),
            reflection_max_changed_files=int(
                _cfg_select(config, "reward.retroagent_reflection_max_changed_files")
            ),
        )

class RetroAgentRuntime:
    method_name = "retroagent"

    def __init__(self, config: RetroAgentConfig) -> None:
        self.config = config
        self.memory = RetroAgentMemory(
            config.memory_path,
            max_entries_per_task=config.max_entries_per_task,
            utility_beta=config.beta,
        )
        self.embedding_model = SentenceTransformer(config.embedding_model_path)
        self._memory_entry_ids: tuple[str, ...] = ()
        self._memory_task_embeddings = None
        self.task_potential_history: dict[str, float] = {}
        self.current_reflections: list[str] = []
        self.retrieved_entries: list[list[dict[str, Any]]] = []
        self.current_retrieval_groups: list[str] = []
        self.batch_previous_potentials: list[float] = []

    @classmethod
    def from_config(cls, config) -> "RetroAgentRuntime":
        return cls(RetroAgentConfig.from_config(config))

    def is_enabled(self) -> bool:
        return self.config.enable

    def build_prompt_context(self, index: int) -> str:
        if not self.is_enabled():
            return ""
        if index >= len(self.current_reflections):
            return ""
        return self.current_reflections[index]

    def _group_split_index(self, group_n: int, is_eval: bool) -> int:
        if is_eval or self.config.full_group_memory:
            return 0
        if self.config.single_reflection_per_group:
            return max(group_n - 1, 0)
        return group_n // 2

    def _refresh_memory_embeddings(self, entries) -> None:
        entry_ids = tuple(entry.entry_id for entry in entries)
        if entry_ids == self._memory_entry_ids:
            return
        self._memory_entry_ids = entry_ids
        if not entries:
            self._memory_task_embeddings = None
            return
        task_descriptions = [entry.task_description for entry in entries]
        self._memory_task_embeddings = self.embedding_model.encode(
            task_descriptions, convert_to_tensor=True
        )

    def _retrieve_for_task(
        self, task_description: str, top_k: int
    ) -> list[dict[str, Any]]:
        entries = self.memory.entries
        if not entries:
            return []
        self._refresh_memory_embeddings(entries)
        query_embedding = self.embedding_model.encode(
            task_description, convert_to_tensor=True
        )
        cos_scores = util.cos_sim(query_embedding, self._memory_task_embeddings)[0]
        relevances = [float(score) for score in cos_scores]
        return rank_memory_entries(
            entries=entries,
            relevances=relevances,
            top_k=top_k,
            retrieve_mode=self.config.retrieve_mode,
            retrieve_type=self.config.retrieval_type,
            alpha=self.config.alpha,
            temperature=self.config.temperature,
            ucb_scale=self.config.ucb_scale,
            similarity_threshold=self.config.similarity_threshold,
        )

    @staticmethod
    def _format_reflection_block(items: list[dict[str, Any]]) -> str:
        if not items:
            return ""
        lines = ["Past reflections from similar shell coding tasks:"]
        for idx, item in enumerate(items, start=1):
            prefix = item["attempt_type"].upper()
            lines.append(f"{idx}. [{prefix}] {item['reflection']}")
        lines.append(
            "Use these lessons only when they fit the current workspace state."
        )
        return "\n".join(lines)

    def prepare_batch(self, tasks: list[str], *, is_eval: bool, group_n: int) -> None:
        self.current_reflections = []
        self.retrieved_entries = []
        self.current_retrieval_groups = []
        self.batch_previous_potentials = []
        if not self.is_enabled():
            self.current_reflections = ["" for _ in tasks]
            self.retrieved_entries = [[] for _ in tasks]
            self.current_retrieval_groups = ["disabled" for _ in tasks]
            self.batch_previous_potentials = [0.0 for _ in tasks]
            return
        split_index = self._group_split_index(group_n, is_eval)
        retrieval_cache: dict[str, list[dict[str, Any]]] = {}
        train_top_k = self.config.top_k if is_eval else min(self.config.top_k, 1)
        for idx, task in enumerate(tasks):
            self.batch_previous_potentials.append(
                self.task_potential_history.get(task, 0.0)
            )
            retrieval_group = "control"
            should_retrieve = False
            if is_eval:
                retrieval_group = "eval_retrieval"
                should_retrieve = True
            elif idx % group_n >= split_index:
                retrieval_group = "experiment"
                should_retrieve = True
            if not should_retrieve:
                self.current_reflections.append("")
                self.retrieved_entries.append([])
                self.current_retrieval_groups.append(retrieval_group)
                continue
            if task not in retrieval_cache:
                retrieval_cache[task] = self._retrieve_for_task(task, top_k=train_top_k)
            retrieved = retrieval_cache[task]
            self.current_reflections.append(self._format_reflection_block(retrieved))
            self.retrieved_entries.append(retrieved)
            self.current_retrieval_groups.append(retrieval_group)

    @staticmethod
    def _clip_text_tail(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        text = str(text).strip()
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    def _keep_step_indices(self, traj_steps, traj_infos, tokenizer) -> list[int]:
        active_indices = [
            idx for idx, step in enumerate(traj_steps) if step.get("active_masks", True)
        ]
        if not active_indices:
            return []
        keep = {active_indices[-1]}
        for idx in active_indices[-self.config.reflection_keep_last_k_steps :]:
            keep.add(idx)
        for idx in active_indices:
            info = traj_infos[idx]
            response_text = tokenizer.decode(
                traj_steps[idx]["responses"], skip_special_tokens=True
            ).strip()
            if "submit_answer" in response_text:
                keep.add(idx)
            if info.get("changed_files", []):
                keep.add(idx)
            if info.get("error", None) is not None:
                keep.add(idx)
            returncode = info.get("returncode", 0)
            if returncode not in (0, None):
                keep.add(idx)
        return sorted(keep)

    def build_trajectory_texts(
        self, total_batch_list, total_infos, tokenizer
    ) -> list[str]:
        trajectory_texts: list[str] = []
        for traj_idx, traj_steps in enumerate(total_batch_list):
            lines = []
            keep_indices = self._keep_step_indices(
                traj_steps, total_infos[traj_idx], tokenizer
            )
            step_num = 1
            for step_idx in keep_indices:
                step_data = traj_steps[step_idx]
                info = total_infos[traj_idx][step_idx]
                raw_obs = self._clip_text_tail(
                    step_data.get("anchor_obs", ""),
                    self.config.reflection_max_obs_chars_per_step,
                )
                response_text = tokenizer.decode(
                    step_data["responses"], skip_special_tokens=True
                ).strip()
                env_feedback = self._clip_text_tail(
                    info.get("stdout", "") or info.get("stderr", "") or "",
                    self.config.reflection_max_feedback_chars_per_step,
                )
                changed_files = list(info.get("changed_files", []))[
                    : self.config.reflection_max_changed_files
                ]
                lines.append(f"[Observation {step_num}] {raw_obs}")
                lines.append(f"[Action {step_num}] {response_text}")
                if env_feedback:
                    lines.append(f"[Feedback {step_num}] {env_feedback}")
                if changed_files:
                    lines.append(
                        f"[Changed Files {step_num}] {', '.join(str(item) for item in changed_files)}"
                    )
                step_num += 1
            trajectory_texts.append("\n".join(lines))
        return trajectory_texts

    def update_retrieval_utility(
        self, final_infos: list[dict[str, Any]], *, group_n: int
    ) -> None:
        if not self.is_enabled():
            return
        batch_size = len(final_infos)
        if batch_size == 0:
            return
        if group_n <= 1:
            for idx, info in enumerate(final_infos):
                score = 1.0 if bool(info.get("won", False)) else 0.0
                for item in self.retrieved_entries[idx]:
                    self.memory.update_utility(item["entry_id"], score)
            return
        for start in range(0, batch_size, group_n):
            end = min(start + group_n, batch_size)
            control_wins = 0
            experiment_wins = 0
            for idx in range(start, end):
                won = bool(final_infos[idx].get("won", False))
                group = self.current_retrieval_groups[idx]
                if group == "control":
                    control_wins += int(won)
                elif group in {"experiment", "eval_retrieval"}:
                    experiment_wins += int(won)
            group_outperformed = experiment_wins > control_wins
            for idx in range(start, end):
                if self.current_retrieval_groups[idx] == "control":
                    continue
                won = bool(final_infos[idx].get("won", False))
                if self.config.group_outperformance:
                    score = 1.0 if won and group_outperformed else 0.0
                else:
                    score = 1.0 if won else 0.0
                for item in self.retrieved_entries[idx]:
                    self.memory.update_utility(item["entry_id"], score)

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
        **_,
    ) -> list[dict[str, Any]] | None:
        if not self.is_enabled():
            return None
        self.update_retrieval_utility(final_infos, group_n=group_n)
        trajectory_texts = self.build_trajectory_texts(
            total_batch_list, total_infos, tokenizer
        )
        prompts = [
            build_reflection_prompt(task, trajectory_text, bool(info.get("won", False)))
            for task, trajectory_text, info in zip(tasks, trajectory_texts, final_infos)
        ]
        reflection_outputs = generate_freeform_texts(
            prompts=prompts,
            actor_rollout_wg=actor_rollout_wg,
            meta_info=meta_info,
            max_prompt_tokens=self.config.reflection_max_prompt_tokens,
            hard_truncate_tokens=self.config.reflection_hard_truncate_tokens,
            data_source=f"{self.method_name}_reflection",
        )
        results = self.apply_reflections(
            tasks=tasks,
            trajectory_texts=trajectory_texts,
            reflection_outputs=reflection_outputs,
            final_infos=final_infos,
            group_n=group_n,
        )
        for idx, stats in enumerate(results):
            episode_rewards[idx] += float(stats["retroagent_bonus"])
        return results

    def apply_reflections(
        self,
        *,
        tasks: list[str],
        trajectory_texts: list[str],
        reflection_outputs: list[str],
        final_infos: list[dict[str, Any]],
        group_n: int,
    ) -> list[dict[str, Any]]:
        parsed_reflections: list[ParsedReflection | None] = [
            parse_reflection(text) for text in reflection_outputs
        ]
        raw_improvements: list[float] = []
        numerical_bonuses: list[float] = []
        language_bonuses: list[float] = []
        current_scores: list[float] = []
        results: list[dict[str, Any]] = []
        for idx, task in enumerate(tasks):
            actual_success = bool(final_infos[idx].get("won", False))
            parsed = parsed_reflections[idx]
            if self.config.potential_based_on_binary_success:
                current_phi = 1.0 if actual_success else 0.0
            else:
                current_phi = 1.0 if actual_success else 0.0
                if parsed is not None and parsed.total_subtasks > 0:
                    current_phi = max(current_phi, parsed.completed_ratio)
            previous_phi = self.batch_previous_potentials[idx]
            improvement = current_phi - previous_phi
            raw_improvements.append(improvement)
            current_scores.append(current_phi)
            language_signal = 0.0
            if actual_success and self.retrieved_entries[idx]:
                language_signal = max(
                    item["relevance"] * max(item["utility_score"], 0.0)
                    for item in self.retrieved_entries[idx]
                )
            language_bonuses.append(self.config.language_reward_coef * language_signal)
            lesson_text = ""
            reflection_consistent = False
            task_success_prediction = None
            if parsed is not None:
                lesson_text = parsed.lesson_text
                task_success_prediction = parsed.task_success
                reflection_consistent = parsed.task_success is None or (
                    parsed.task_success == actual_success
                )
            if (
                lesson_text
                and reflection_consistent
                and current_phi >= self.config.store_threshold
            ):
                self.memory.add(
                    task_description=task,
                    reflection_text=lesson_text,
                    trajectory=trajectory_texts[idx],
                    initial_score=0.5,
                    attempt_type="success" if actual_success else "failure",
                    current_progress_ratio=0.0,
                )
                self._memory_entry_ids = ()
                self._memory_task_embeddings = None
            results.append(
                {
                    "retroagent_reflection": reflection_outputs[idx],
                    "retroagent_reflection_valid": float(parsed is not None),
                    "retroagent_reflection_consistent": float(reflection_consistent),
                    "retroagent_predicted_success": (
                        float(task_success_prediction)
                        if task_success_prediction is not None
                        else -1.0
                    ),
                    "retroagent_current_phi": float(current_phi),
                    "retroagent_previous_phi": float(previous_phi),
                    "retroagent_raw_improvement": float(improvement),
                }
            )
        centered_improvements = list(raw_improvements)
        if self.config.group_relative_intrinsic_rewards and group_n > 0:
            for start in range(0, len(tasks), group_n):
                end = min(start + group_n, len(tasks))
                group_values = raw_improvements[start:end]
                group_mean = sum(group_values) / float(len(group_values))
                for idx in range(start, end):
                    centered_improvements[idx] = raw_improvements[idx] - group_mean
        for idx in range(len(tasks)):
            numerical_bonus = (
                self.config.numerical_reward_coef * centered_improvements[idx]
            )
            numerical_bonuses.append(numerical_bonus)
            total_bonus = numerical_bonus + language_bonuses[idx]
            results[idx]["retroagent_numerical_bonus"] = float(numerical_bonus)
            results[idx]["retroagent_language_bonus"] = float(language_bonuses[idx])
            results[idx]["retroagent_bonus"] = float(total_bonus)
            results[idx]["retroagent_retrieval_group"] = self.current_retrieval_groups[
                idx
            ]
            results[idx]["retroagent_memory_size"] = float(len(self.memory.entries))
            results[idx]["retroagent_retrieved_count"] = float(
                len(self.retrieved_entries[idx])
            )
        for start in range(0, len(tasks), max(group_n, 1)):
            end = min(start + max(group_n, 1), len(tasks))
            task = tasks[start]
            group_success_rate = sum(
                bool(final_infos[idx].get("won", False)) for idx in range(start, end)
            ) / float(end - start)
            self.task_potential_history[task] = max(
                self.task_potential_history.get(task, 0.0), group_success_rate
            )
        return results
