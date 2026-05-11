from typing import Any
from main_entry.cli_agent_bash_coding.prompts.bash_coding import BASH_CODING_TEMPLATE
from main_entry.cli_agent_bash_coding.prompts.bash_coding import (
    BASH_CODING_TEMPLATE_NO_HIS,
)

class BashCodingHarnessBase:
    name = "base"
    sigma_reject_enabled: bool = False
    sigma_exit_enabled: bool = False
    sigma_cmi_enabled: bool = False
    sigma_tcm_enabled: bool = False
    sigma_antimode_enabled: bool = False
    sigma_cohort_enabled: bool = False
    sigma_witness_enabled: bool = False
    sigma_dual_enabled: bool = False
    sigma_concentration_enabled: bool = False

    def uses_commit_action_schema(self) -> bool:
        return False

    def history_action_key(self) -> str:
        return "action"

    def history_obs_key(self) -> str:
        return "observation"

    def prompt_extra_rules(self) -> str:
        return "Use the standard schema for the current harness."

    def prompt_valid_tagged_formats(self) -> str:
        return (
            "- <name>submit_code</name><plan>...</plan><code>...</code>\n"
            "- <name>submit_answer</name><plan>...</plan><answer>...</answer>"
        )

    def build_reset_obs(self, worker, base_obs: str, info: dict[str, Any]) -> str:
        _ = worker
        _ = info
        return base_obs

    def prepare_projected_action(self, worker, action: str) -> str:
        _ = worker
        return action

    def build_step_obs(
        self,
        worker,
        *,
        output_text_truncated: str,
        diff_text_truncated: str,
        trunc_meta: dict[str, Any],
    ) -> str:
        _ = worker
        _ = diff_text_truncated
        _ = trunc_meta
        return output_text_truncated

    def build_step_info(self, worker) -> dict[str, Any]:
        _ = worker
        return {}

    def inject_method_context(self, obs_text: str, reflection_ctx: str) -> str:
        if not reflection_ctx:
            return obs_text
        return obs_text.replace(
            "\nCurrent observation:",
            f"\n\n{reflection_ctx}\n\nCurrent observation:",
            1,
        )

    def apply_history_commit_decisions(
        self,
        memory,
        infos: list[dict[str, Any]],
        next_text_obs: list[str] | None = None,
    ) -> None:
        _ = memory
        _ = infos
        _ = next_text_obs

    def build_text_obs(
        self, env_manager, text_obs: list[str], init: bool = False
    ) -> list[str]:
        postprocess_text_obs: list[str] = []
        memory_ctx = []
        if not init and env_manager.config.env.history_length > 0:
            memory_ctx, _ = env_manager.memory.fetch(
                env_manager.config.env.history_length,
                obs_key=self.history_obs_key(),
                action_key=self.history_action_key(),
            )
        for i in range(len(text_obs)):
            task_description = (
                env_manager.tasks[i]
                if i < len(env_manager.tasks)
                else "Complete the task"
            )
            total_steps = int(getattr(env_manager.config.env, "max_steps", 0))
            if init or env_manager.config.env.history_length <= 0:
                obs_i = BASH_CODING_TEMPLATE_NO_HIS.format(
                    task_description=task_description,
                    total_steps=total_steps,
                    remaining_steps=total_steps,
                    current_observation=text_obs[i],
                    extra_rules=self.prompt_extra_rules(),
                    valid_tagged_formats=self.prompt_valid_tagged_formats(),
                )
            else:
                action_history = memory_ctx[i] if i < len(memory_ctx) else ""
                step_count = len(env_manager.memory[i])
                obs_i = BASH_CODING_TEMPLATE.format(
                    task_description=task_description,
                    step_count=step_count,
                    history_length=env_manager.config.env.history_length,
                    action_history=action_history,
                    current_step=step_count + 1,
                    total_steps=total_steps,
                    remaining_steps=total_steps - step_count,
                    current_observation=text_obs[i],
                    extra_rules=self.prompt_extra_rules(),
                    valid_tagged_formats=self.prompt_valid_tagged_formats(),
                )
            reflection_ctx = env_manager.method_manager.build_prompt_context(i)
            postprocess_text_obs.append(
                self.inject_method_context(obs_i, reflection_ctx)
            )
        return postprocess_text_obs

    def preserve_query_and_history(
        self,
        collector,
        obs_content: str,
        raw_observation: str,
        apply_chat_template_kwargs: dict[str, Any],
    ) -> tuple[str, bool] | None:
        _ = raw_observation
        history_start = obs_content.find(collector.HISTORY_PREFIX)
        if history_start < 0:
            return None
        history_body_start = history_start + len(collector.HISTORY_PREFIX)
        history_end = obs_content.find(collector.HISTORY_SUFFIX, history_body_start)
        if history_end < 0:
            return None
        prefix = obs_content[:history_body_start]
        history = obs_content[history_body_start:history_end]
        suffix = obs_content[history_end:]
        max_len = int(collector.config.data.max_prompt_length)
        base_len = collector._chat_token_len(
            prefix + suffix, apply_chat_template_kwargs
        )
        if base_len >= max_len:
            return None
        history_budget = max_len - base_len
        history_ids = collector.tokenizer.encode(history, add_special_tokens=False)
        if len(history_ids) <= history_budget:
            return obs_content, False
        clipped_history = collector.tokenizer.decode(
            history_ids[-history_budget:], skip_special_tokens=False
        )
        return prefix + clipped_history + suffix, True
