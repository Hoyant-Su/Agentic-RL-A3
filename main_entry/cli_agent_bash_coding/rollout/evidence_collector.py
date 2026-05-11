import uuid
from typing import Any, Dict
import numpy as np
import torch
import verl.utils.torch_functional as verl_F
from omegaconf import OmegaConf
from agent_system.environments import EnvironmentManagerBase
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from agent_system.multi_turn_rollout.utils import to_list_of_dict, torch_to_numpy
from main_entry.cli_agent_bash_coding.action_schema import get_bash_coding_action_regex
from main_entry.cli_agent_bash_coding.action_schema import get_bash_coding_action_spans
from main_entry.cli_agent_bash_coding.harness import build_bash_coding_harness
from main_entry.cli_agent_bash_coding.reward_provenance import (
    add_reward_provenance,
    zero_reward_provenance,
)
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask

class BashCodingTrajectoryCollector(TrajectoryCollector):
    HISTORY_PREFIX = "Here is the history records:\n"
    HISTORY_SUFFIX = "\n\nYou are now at step "

    def __init__(self, config, tokenizer, processor=None):
        super().__init__(config=config, tokenizer=tokenizer, processor=processor)
        self.harness = build_bash_coding_harness(
            OmegaConf.to_container(config.env, resolve=True)
        )
        self.enable_evidence_gain = bool(config.env.use_model_evidence_gain)
        self._validation_dataset_dump: list[str] = []
        self._validation_prompt_dump: list[str] = []
        if self.enable_evidence_gain:
            self.evidence_gain_coef = float(config.env.progress_gain_coef)
        else:
            self.evidence_gain_coef = None

    def generate_action_batch(
        self, prompt_batch: DataProto, actor_rollout_wg
    ) -> tuple[DataProto, list[str]]:
        prompt_batch.meta_info = dict(prompt_batch.meta_info)
        prompt_batch.meta_info["regex"] = get_bash_coding_action_regex(
            enable_commit=self.harness.uses_commit_action_schema()
        )
        prompt_batch_padded, pad_size = pad_dataproto_to_divisor(
            prompt_batch, actor_rollout_wg.world_size
        )
        batch_output_padded = actor_rollout_wg.generate_sequences(prompt_batch_padded)
        batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)
        action_texts = self.tokenizer.batch_decode(
            batch_output.batch["responses"], skip_special_tokens=True
        )
        return batch_output, action_texts

    @staticmethod
    def _copy_reward_provenance(value: Dict[str, Any] | None) -> Dict[str, float]:
        provenance = zero_reward_provenance()
        if isinstance(value, dict):
            add_reward_provenance(provenance, value)
        return provenance

    def _preserve_query_and_history(
        self,
        obs_content: str,
        raw_observation: str,
        apply_chat_template_kwargs: Dict[str, Any],
    ) -> tuple[str, bool]:
        harness_result = self.harness.preserve_query_and_history(
            self,
            obs_content,
            raw_observation,
            apply_chat_template_kwargs,
        )
        if harness_result is not None:
            return harness_result
        return super()._preserve_query_and_history(
            obs_content, raw_observation, apply_chat_template_kwargs
        )

    def _mask_plan_tokens(self, batch: DataProto, text_actions: list[str]) -> None:
        loss_mask = batch.batch.get("loss_mask", None)
        responses = batch.batch.get("responses", None)
        if loss_mask is None or responses is None:
            return
        response_len = responses.size(1)
        for i, text in enumerate(text_actions):
            spans = get_bash_coding_action_spans(
                text, enable_commit=self.harness.uses_commit_action_schema()
            )
            if spans is None:
                continue
            prefix_len = len(
                self.tokenizer.encode(
                    text[: spans.payload[0]], add_special_tokens=False
                )
            )
            payload_len = len(
                self.tokenizer.encode(
                    text[spans.payload[0] : spans.payload[1]], add_special_tokens=False
                )
            )
            row_mask = torch.zeros(
                response_len, dtype=loss_mask.dtype, device=loss_mask.device
            )
            payload_end = min(prefix_len + payload_len, response_len)
            if prefix_len < payload_end:
                row_mask[prefix_len:payload_end] = 1
            if spans.commit is not None:
                commit_prefix_len = len(
                    self.tokenizer.encode(
                        text[: spans.commit[0]], add_special_tokens=False
                    )
                )
                commit_len = len(
                    self.tokenizer.encode(
                        text[spans.commit[0] : spans.commit[1]],
                        add_special_tokens=False,
                    )
                )
                commit_end = min(commit_prefix_len + commit_len, response_len)
                if commit_prefix_len < commit_end:
                    row_mask[commit_prefix_len:commit_end] = 1
            loss_mask[i, -response_len:] = row_mask

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        apply_chat_template_kwargs = self.config.data.get(
            "apply_chat_template_kwargs", {}
        )
        obs_texts = obs.get("text", None)
        obs_anchors = obs.get("anchor", None)
        obs_text = obs_texts[item] if obs_texts is not None else ""
        obs_anchor = obs_anchors[item] if obs_anchors is not None else ""
        raw_observation = obs_anchor if isinstance(obs_anchor, str) else ""
        obs_content, obs_truncated = self._preserve_query_and_history(
            obs_text,
            raw_observation,
            apply_chat_template_kwargs,
        )
        chat = [{"content": obs_content, "role": "user"}]
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs,
        )
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        raw_prompt_ids = self.tokenizer.encode(
            prompt_with_chat_template, add_special_tokens=False
        )
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = (
                    raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
                )
            else:
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}."
                )
        row_dict = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "model_input_text": prompt_with_chat_template,
            "anchor_obs": raw_observation,
            "index": item,
            "data_source": gen_batch.non_tensor_batch["data_source"][item],
            "dataset": str(gen_batch.non_tensor_batch["dataset"][item]),
            "obs_left_truncated": bool(obs_truncated),
        }
        if self.config.data.get("return_raw_chat", False):
            row_dict["raw_prompt"] = chat
        extra_info_arr = gen_batch.non_tensor_batch.get("extra_info", None)
        if extra_info_arr is not None:
            extra_info = extra_info_arr[item]
            row_dict["extra_info"] = extra_info
            if isinstance(extra_info, dict):
                row_dict["sample_id_base"] = str(extra_info.get("id", item))
            else:
                row_dict["sample_id_base"] = str(extra_info)
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto,
        obs: Dict,
    ) -> DataProto:
        batch_size = len(gen_batch.batch["input_ids"])
        processed_samples = []
        for item in range(batch_size):
            processed_samples.append(
                self.preprocess_single_sample(
                    item=item,
                    gen_batch=gen_batch,
                    obs=obs,
                )
            )
        batch = collate_fn(processed_samples)
        meta_info = dict(gen_batch.meta_info)
        return DataProto.from_single_dict(data=batch, meta_info=meta_info)

    def _build_freeform_prompt_batch(
        self,
        prompts: list[str],
        meta_info: Dict[str, Any],
        *,
        max_prompt_tokens: int,
        hard_truncate_tokens: int,
        data_source: str,
    ) -> DataProto:
        apply_chat_template_kwargs = self.config.data.get(
            "apply_chat_template_kwargs", {}
        )
        effective_max_prompt_tokens = min(max_prompt_tokens, hard_truncate_tokens)
        rows = []
        for idx, prompt in enumerate(prompts):
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                [{"content": prompt, "role": "user"}],
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs,
            )
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=self.tokenizer,
                max_length=effective_max_prompt_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.truncation,
            )
            position_ids = compute_position_id_with_mask(attention_mask)
            raw_prompt_ids = self.tokenizer.encode(
                prompt_with_chat_template, add_special_tokens=False
            )
            if len(raw_prompt_ids) > hard_truncate_tokens:
                raw_prompt_ids = raw_prompt_ids[-hard_truncate_tokens:]
            rows.append(
                {
                    "input_ids": input_ids[0],
                    "attention_mask": attention_mask[0],
                    "position_ids": position_ids[0],
                    "raw_prompt_ids": raw_prompt_ids,
                    "index": idx,
                    "data_source": data_source,
                }
            )
        prompt_batch = DataProto.from_single_dict(
            data=collate_fn(rows), meta_info=dict(meta_info)
        )
        prompt_batch.meta_info.pop("regex", None)
        return prompt_batch

    def _generate_freeform_texts(
        self,
        *,
        prompts: list[str],
        actor_rollout_wg,
        meta_info: Dict[str, Any],
        max_prompt_tokens: int,
        hard_truncate_tokens: int,
        data_source: str,
    ) -> list[str]:
        prompt_batch = self._build_freeform_prompt_batch(
            prompts,
            meta_info,
            max_prompt_tokens=max_prompt_tokens,
            hard_truncate_tokens=hard_truncate_tokens,
            data_source=data_source,
        )
        prompt_batch_padded, pad_size = pad_dataproto_to_divisor(
            prompt_batch, actor_rollout_wg.world_size
        )
        batch_output_padded = actor_rollout_wg.generate_sequences(prompt_batch_padded)
        batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)
        return self.tokenizer.batch_decode(
            batch_output.batch["responses"], skip_special_tokens=True
        )

    def _extract_final_infos(
        self, total_batch_list, total_infos
    ) -> list[dict[str, Any]]:
        final_infos: list[dict[str, Any]] = []
        for traj_idx, traj_steps in enumerate(total_batch_list):
            selected = total_infos[traj_idx][-1]
            for step_idx in reversed(range(len(traj_steps))):
                if traj_steps[step_idx].get("active_masks", True):
                    selected = total_infos[traj_idx][step_idx]
                    break
            final_infos.append(selected)
        return final_infos

    def _apply_baseline_methods(
        self,
        *,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs,
        total_batch_list,
        total_infos,
        episode_rewards: np.ndarray,
    ) -> dict[str, list[dict[str, Any]]]:
        final_infos = self._extract_final_infos(total_batch_list, total_infos)
        return envs.apply_baseline_methods_post_rollout(
            actor_rollout_wg=actor_rollout_wg,
            total_batch_list=total_batch_list,
            total_infos=total_infos,
            final_infos=final_infos,
            episode_rewards=episode_rewards,
            tokenizer=self.tokenizer,
            meta_info=gen_batch.meta_info,
            generate_freeform_texts=self._generate_freeform_texts,
        )

    def multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs,
        is_train: bool = True,
    ) -> DataProto:
        gen_batch_output = super().multi_turn_loop(
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            is_train=is_train,
        )
        if not is_train:
            dataset_arr = gen_batch_output.non_tensor_batch["dataset"]
            self._validation_dataset_dump.extend(
                str(item) for item in dataset_arr.tolist()
            )
            prompt_arr = gen_batch_output.non_tensor_batch.get("model_input_text", None)
            if prompt_arr is not None:
                self._validation_prompt_dump.extend(
                    str(item) for item in prompt_arr.tolist()
                )
        return gen_batch_output

    def vanilla_multi_turn_loop(
        self, gen_batch: DataProto, actor_rollout_wg, envs: EnvironmentManagerBase
    ):
        batch_size = len(gen_batch.batch)
        env_capacity = getattr(getattr(envs, "envs", None), "num_processes", 0)
        if env_capacity > 0 and batch_size > env_capacity:
            total_batch_list = []
            total_episode_rewards = []
            total_episode_lengths = []
            total_success = []
            total_traj_uid = []
            total_tool_callings = []
            for start in range(0, batch_size, env_capacity):
                end = min(start + env_capacity, batch_size)
                chunk_results = self.vanilla_multi_turn_loop(
                    gen_batch=gen_batch[start:end],
                    actor_rollout_wg=actor_rollout_wg,
                    envs=envs,
                )
                (
                    batch_list,
                    episode_rewards,
                    episode_lengths,
                    success,
                    traj_uid,
                    tool_callings,
                ) = chunk_results
                total_batch_list += batch_list
                total_episode_rewards.append(episode_rewards)
                total_episode_lengths.append(episode_lengths)
                total_success.append(success)
                total_traj_uid.append(traj_uid)
                total_tool_callings.append(tool_callings)
            merged_success = {
                key: np.concatenate([success[key] for success in total_success], axis=0)
                for key in total_success[0].keys()
            }
            return (
                total_batch_list,
                np.concatenate(total_episode_rewards, axis=0),
                np.concatenate(total_episode_lengths, axis=0),
                merged_success,
                np.concatenate(total_traj_uid, axis=0),
                np.concatenate(total_tool_callings, axis=0),
            )
        obs, infos = envs.reset(
            kwargs=gen_batch.non_tensor_batch.get("env_kwargs", None)
        )
        length_obs = len(obs["text"]) if obs["text"] is not None else len(obs["image"])
        assert len(gen_batch.batch) == length_obs
        if self.config.env.rollout.n > 0:
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array(
                [uid for _ in range(len(gen_batch.batch))], dtype=object
            )
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array(
            [str(uuid.uuid4()) for _ in range(batch_size)], dtype=object
        )
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        episode_reward_provenance = [
            zero_reward_provenance() for _ in range(batch_size)
        ]
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            prompt_batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            batch_output, text_actions = self.generate_action_batch(
                prompt_batch, actor_rollout_wg
            )
            self._mask_plan_tokens(batch_output, text_actions)
            batch = batch_output
            batch.non_tensor_batch["uid"] = uid_batch
            batch.non_tensor_batch["traj_uid"] = traj_uid
            next_obs, rewards, dones, infos = envs.step(text_actions)
            rewards = torch_to_numpy(rewards)
            dones = torch_to_numpy(dones)
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                dones = dones.squeeze(1)
            if "is_action_valid" in infos[0]:
                batch.non_tensor_batch["is_action_valid"] = np.array(
                    [info["is_action_valid"] for info in infos], dtype=bool
                )
            else:
                batch.non_tensor_batch["is_action_valid"] = np.ones(
                    batch_size, dtype=bool
                )
            if "tool_calling" in infos[0]:
                tool_callings[active_masks] += np.array(
                    [info["tool_calling"] for info in infos], dtype=np.float32
                )[active_masks]
            episode_rewards[active_masks] += rewards[active_masks]
            episode_lengths[active_masks] += 1
            step_reward_provenance = []
            step_reward_total = []
            for i, info in enumerate(infos):
                provenance = self._copy_reward_provenance(
                    info.get("reward_provenance", None)
                )
                info["reward_provenance"] = provenance
                info["reward_total"] = float(rewards[i])
                step_reward_provenance.append(provenance)
                step_reward_total.append(float(info["reward_total"]))
                if active_masks[i]:
                    add_reward_provenance(episode_reward_provenance[i], provenance)
            assert len(rewards) == batch_size
            batch.non_tensor_batch["rewards"] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch["active_masks"] = torch_to_numpy(
                active_masks, is_object=True
            )
            batch.non_tensor_batch["step_reward_provenance"] = np.array(
                step_reward_provenance, dtype=object
            )
            batch.non_tensor_batch["step_reward_total"] = np.array(
                step_reward_total, dtype=np.float32
            )
            batch_list: list[dict] = to_list_of_dict(batch)
            for i in range(batch_size):
                batch_list[i]["step_reward_provenance"] = step_reward_provenance[i]
                batch_list[i]["step_reward_total"] = float(step_reward_total[i])
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])
            is_done = np.logical_or(is_done, dones)
            obs = next_obs
            if is_done.all():
                break
        method_stats: dict[str, list[dict[str, Any]]] = {}
        if getattr(envs, "methods_enabled", lambda: False)() and bool(
            getattr(getattr(envs, "envs", None), "is_train", True)
        ):
            method_stats = self._apply_baseline_methods(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                total_batch_list=total_batch_list,
                total_infos=total_infos,
                episode_rewards=episode_rewards,
            )
        for i in range(batch_size):
            traj_steps = total_batch_list[i]
            traj_len = len(traj_steps)
            for step_idx, step_data in enumerate(traj_steps):
                step_data["episode_reward_provenance"] = dict(
                    episode_reward_provenance[i]
                )
                for stats in method_stats.values():
                    for key, value in stats[i].items():
                        if (
                            isinstance(value, np.ndarray)
                            and value.ndim == 1
                            and len(value) == traj_len
                        ):
                            step_data[key] = (
                                value[step_idx].item()
                                if isinstance(value[step_idx], np.generic)
                                else value[step_idx]
                            )
                            continue
                        if isinstance(value, (list, tuple)) and len(value) == traj_len:
                            step_data[key] = value[step_idx]
                            continue
                        step_data[key] = value
        success: Dict[str, np.ndarray] = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )
        return (
            total_batch_list,
            episode_rewards,
            episode_lengths,
            success,
            traj_uid,
            tool_callings,
        )
