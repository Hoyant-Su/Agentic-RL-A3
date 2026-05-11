import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import (
    process_image,
    to_list_of_dict,
    torch_to_numpy,
    filter_group_data,
)
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict, Any
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self._obs_trunc_notice = (
            "[OBS_TRUNCATED_LEFT] Earlier observation content is omitted due to context limit. "
            "Task/query and executed-step history are preserved.\n"
        )
        self._context_trunc_notice = (
            "[CONTEXT_TRUNCATED] Observation and part of instruction suffix are omitted due to context limit. "
            "Task/query and executed-step history are prioritized.\n"
        )

    def _chat_token_len(
        self, content: str, apply_chat_template_kwargs: Dict[str, Any]
    ) -> int:
        chat = [{"content": content, "role": "user"}]
        token_ids = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
        return len(token_ids)

    def _split_obs_content(self, text: str, raw_observation: str):
        if not raw_observation:
            return None
        obs_start = text.rfind(raw_observation)
        if obs_start < 0:
            return None
        obs_end = obs_start + len(raw_observation)
        return text[:obs_start], text[obs_start:obs_end], text[obs_end:]

    def _preserve_query_and_history(
        self,
        obs_content: str,
        raw_observation: str,
        apply_chat_template_kwargs: Dict[str, Any],
    ) -> tuple[str, bool]:
        split = self._split_obs_content(obs_content, raw_observation)
        if split is None:
            return obs_content, False
        prefix, observation, suffix = split
        max_len = int(self.config.data.max_prompt_length)
        base_len = self._chat_token_len(prefix + suffix, apply_chat_template_kwargs)
        if base_len >= max_len:
            notice_ids = self.tokenizer.encode(
                self._context_trunc_notice, add_special_tokens=False
            )
            keep = max_len - len(notice_ids)
            if keep <= 0:
                return obs_content, False
            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            kept_prefix = self.tokenizer.decode(
                prefix_ids[:keep], skip_special_tokens=False
            )
            return kept_prefix + "\n" + self._context_trunc_notice, True
        obs_budget = max_len - base_len
        obs_ids = self.tokenizer.encode(observation, add_special_tokens=False)
        if len(obs_ids) <= obs_budget:
            return obs_content, False
        notice_ids = self.tokenizer.encode(
            self._obs_trunc_notice, add_special_tokens=False
        )
        if obs_budget > len(notice_ids):
            keep = obs_budget - len(notice_ids)
            clipped = self.tokenizer.decode(obs_ids[-keep:], skip_special_tokens=False)
            new_observation = self._obs_trunc_notice + clipped
        else:
            clipped = self.tokenizer.decode(
                obs_ids[-obs_budget:], skip_special_tokens=False
            )
            new_observation = clipped
        return prefix + new_observation + suffix, True

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        raw_prompt = gen_batch.non_tensor_batch["raw_prompt"][item]
        data_source = gen_batch.non_tensor_batch["data_source"][item]
        apply_chat_template_kwargs = self.config.data.get(
            "apply_chat_template_kwargs", {}
        )
        obs_texts = obs.get("text", None)
        obs_images = obs.get("image", None)
        obs_anchors = obs.get("anchor", None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None
        _obs_anchor = (
            torch_to_numpy(obs_anchor, is_object=True)
            if isinstance(obs_anchor, torch.Tensor)
            else obs_anchor
        )
        obs_content = ""
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")
        raw_observation = _obs_anchor if isinstance(_obs_anchor, str) else ""
        obs_content, obs_truncated = self._preserve_query_and_history(
            obs_content, raw_observation, apply_chat_template_kwargs
        )
        chat = np.array(
            [
                {
                    "content": obs_content,
                    "role": "user",
                }
            ]
        )
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs,
        )
        row_dict = {}
        if is_multi_modal:
            raw_prompt = prompt_with_chat_template.replace(
                "<image>", "<|vision_start|><|image_pad|><|vision_end|>"
            )
            row_dict["multi_modal_data"] = {"image": [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(
                row_dict["multi_modal_data"]["image"], return_tensors="pt"
            )
            image_grid_thw = image_inputs["image_grid_thw"]
            row_dict["multi_modal_inputs"] = {
                key: val for key, val in image_inputs.items()
            }
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while "<image>" in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        "<image>",
                        "<|vision_start|>"
                        + "<|placeholder|>"
                        * (image_grid_thw[index].prod() // merge_length)
                        + "<|vision_end|>",
                        1,
                    )
                    index += 1
                prompt_with_chat_template = prompt_with_chat_template.replace(
                    "<|placeholder|>", self.processor.image_token
                )
        else:
            raw_prompt = prompt_with_chat_template
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.truncation,
        )
        if is_multi_modal:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
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
            elif self.config.data.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}."
                )
        row_dict.update(
            {
                "input_ids": input_ids[0],
                "attention_mask": attention_mask[0],
                "position_ids": position_ids[0],
                "raw_prompt_ids": raw_prompt_ids,
                "anchor_obs": _obs_anchor,
                "index": item,
                "data_source": data_source,
                "obs_left_truncated": bool(obs_truncated),
            }
        )
        if self.config.data.get("return_raw_chat", False):
            row_dict["raw_prompt"] = chat.tolist()
        extra_info_arr = gen_batch.non_tensor_batch.get("extra_info", None)
        if extra_info_arr is not None:
            extra_info = extra_info_arr[item]
            row_dict["extra_info"] = extra_info
            if isinstance(extra_info, dict):
                base_id = extra_info.get("id", str(item))
                row_dict["sample_id_base"] = str(base_id)
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
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        batch = collate_fn(processed_samples)
        new_batch = DataProto.from_single_dict(
            data=batch, meta_info=gen_batch.meta_info
        )
        return new_batch

    def gather_rollout_data(
        self,
        total_batch_list: List[List[Dict]],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        success: Dict[str, np.ndarray],
        traj_uid: np.ndarray,
        tool_callings: np.ndarray,
    ) -> DataProto:
        batch_size = len(total_batch_list)
        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        effective_batch = []
        for bs in range(batch_size):
            for turn_idx, data in enumerate(total_batch_list[bs]):
                assert traj_uid[bs] == data["traj_uid"], (
                    "data is not from the same trajectory"
                )
                if data["active_masks"]:
                    data["turn_idx"] = int(turn_idx)
                    base_id = data.get("sample_id_base", None)
                    if base_id is not None:
                        data["sample_id"] = f"{str(base_id)}_idx_{int(turn_idx)}"
                    elif "sample_id" in data:
                        sid = str(data.get("sample_id", ""))
                        data["sample_id"] = (
                            f"{sid}_idx_{int(turn_idx)}"
                            if sid and "_idx" not in sid
                            else sid
                        )
                    data["episode_rewards"] = episode_rewards[bs]
                    data["episode_lengths"] = episode_lengths[bs]
                    data["tool_callings"] = tool_callings[bs]
                    for key, value in success_rate.items():
                        data[key] = value
                    effective_batch.append(data)
        gen_batch_output = DataProto.from_single_dict(data=collate_fn(effective_batch))
        return gen_batch_output

    def vanilla_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
    ) -> DataProto:
        batch_size = len(gen_batch.batch)
        obs, infos = envs.reset(
            kwargs=gen_batch.non_tensor_batch.get("env_kwargs", None)
        )
        lenght_obs = len(obs["text"]) if obs["text"] is not None else len(obs["image"])
        assert len(gen_batch.batch) == lenght_obs, (
            f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        )
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
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            batch_input.meta_info = gen_batch.meta_info
            batch_input_padded, pad_size = pad_dataproto_to_divisor(
                batch_input, actor_rollout_wg.world_size
            )
            batch_output_padded = actor_rollout_wg.generate_sequences(
                batch_input_padded
            )
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)
            batch.non_tensor_batch["uid"] = uid_batch
            batch.non_tensor_batch["traj_uid"] = traj_uid
            batch = batch.union(batch_output)
            text_actions = self.tokenizer.batch_decode(
                batch.batch["responses"], skip_special_tokens=True
            )
            if envs.sigma_reject_active():
                sigma_reject_retries = int(
                    getattr(self.config.env, "sigma_reject_retries", 1)
                )
                for _sigma_retry in range(sigma_reject_retries):
                    needs_regen = envs.would_be_sigma_repeat(text_actions, active_masks)
                    if not any(needs_regen):
                        break
                    regen_padded, regen_pad_size = pad_dataproto_to_divisor(
                        batch_input, actor_rollout_wg.world_size
                    )
                    regen_output = unpad_dataproto(
                        actor_rollout_wg.generate_sequences(regen_padded),
                        pad_size=regen_pad_size,
                    )
                    for key, val in regen_output.batch.items():
                        if key in batch.batch:
                            for i, r in enumerate(needs_regen):
                                if r:
                                    batch.batch[key][i] = val[i]
                    for key, val in regen_output.non_tensor_batch.items():
                        if key in batch.non_tensor_batch:
                            for i, r in enumerate(needs_regen):
                                if r:
                                    batch.non_tensor_batch[key][i] = val[i]
                    text_actions = self.tokenizer.batch_decode(
                        batch.batch["responses"], skip_special_tokens=True
                    )
            if envs.sigma_cmi_active():
                sigma_cmi_beta = float(getattr(self.config.env, "sigma_cmi_beta", 0.5))
                mean_logp_a = envs.mean_rollout_logprob(
                    batch.batch["rollout_log_probs"]
                )
                regen_padded, regen_pad_size = pad_dataproto_to_divisor(
                    batch_input, actor_rollout_wg.world_size
                )
                regen_output = unpad_dataproto(
                    actor_rollout_wg.generate_sequences(regen_padded),
                    pad_size=regen_pad_size,
                )
                text_actions_b = self.tokenizer.batch_decode(
                    regen_output.batch["responses"], skip_special_tokens=True
                )
                mean_logp_b = envs.mean_rollout_logprob(
                    regen_output.batch["rollout_log_probs"]
                )
                commit_b = envs.sigma_cmi_decide(
                    text_actions_a=text_actions,
                    text_actions_b=text_actions_b,
                    mean_logp_a=mean_logp_a,
                    mean_logp_b=mean_logp_b,
                    active_masks=active_masks,
                    beta=sigma_cmi_beta,
                )
                if any(commit_b):
                    for key, val in regen_output.batch.items():
                        if key in batch.batch:
                            for i, c in enumerate(commit_b):
                                if c:
                                    batch.batch[key][i] = val[i]
                    for key, val in regen_output.non_tensor_batch.items():
                        if key in batch.non_tensor_batch:
                            for i, c in enumerate(commit_b):
                                if c:
                                    batch.non_tensor_batch[key][i] = val[i]
                    text_actions = self.tokenizer.batch_decode(
                        batch.batch["responses"], skip_special_tokens=True
                    )
            if envs.sigma_exit_active() and _step == self.config.env.max_steps - 1:
                sigma_exit_retries = int(
                    getattr(self.config.env, "sigma_exit_retries", 2)
                )
                for _exit_retry in range(sigma_exit_retries):
                    needs_regen = envs.needs_exit_resample(text_actions, active_masks)
                    if not any(needs_regen):
                        break
                    regen_padded, regen_pad_size = pad_dataproto_to_divisor(
                        batch_input, actor_rollout_wg.world_size
                    )
                    regen_output = unpad_dataproto(
                        actor_rollout_wg.generate_sequences(regen_padded),
                        pad_size=regen_pad_size,
                    )
                    resample_texts = self.tokenizer.batch_decode(
                        regen_output.batch["responses"], skip_special_tokens=True
                    )
                    promote_mask = envs.needs_exit_resample(
                        resample_texts,
                        np.logical_and(
                            active_masks,
                            np.array([bool(r) for r in needs_regen], dtype=bool),
                        ),
                    )
                    commit_mask = [
                        bool(needs_regen[i]) and not bool(promote_mask[i])
                        for i in range(len(needs_regen))
                    ]
                    if not any(commit_mask):
                        continue
                    for key, val in regen_output.batch.items():
                        if key in batch.batch:
                            for i, c in enumerate(commit_mask):
                                if c:
                                    batch.batch[key][i] = val[i]
                    for key, val in regen_output.non_tensor_batch.items():
                        if key in batch.non_tensor_batch:
                            for i, c in enumerate(commit_mask):
                                if c:
                                    batch.non_tensor_batch[key][i] = val[i]
                    text_actions = self.tokenizer.batch_decode(
                        batch.batch["responses"], skip_special_tokens=True
                    )
            if (
                envs.sigma_tcm_active()
                and _step == self.config.env.max_steps - 1
                and active_masks.any()
            ):
                sigma_tcm_k = int(getattr(self.config.env, "sigma_tcm_k", 3))
                extra_rounds = []
                for _ in range(sigma_tcm_k - 1):
                    regen_padded, regen_pad_size = pad_dataproto_to_divisor(
                        batch_input, actor_rollout_wg.world_size
                    )
                    regen_output = unpad_dataproto(
                        actor_rollout_wg.generate_sequences(regen_padded),
                        pad_size=regen_pad_size,
                    )
                    regen_texts = self.tokenizer.batch_decode(
                        regen_output.batch["responses"], skip_special_tokens=True
                    )
                    extra_rounds.append((regen_output, regen_texts))
                candidate_texts = [text_actions] + [
                    texts for (_, texts) in extra_rounds
                ]
                commit_j = envs.sigma_tcm_decide(candidate_texts, active_masks)
                for i, j in enumerate(commit_j):
                    if j == 0:
                        continue
                    regen_output, _ = extra_rounds[j - 1]
                    for key, val in regen_output.batch.items():
                        if key in batch.batch:
                            batch.batch[key][i] = val[i]
                    for key, val in regen_output.non_tensor_batch.items():
                        if key in batch.non_tensor_batch:
                            batch.non_tensor_batch[key][i] = val[i]
                if any(j != 0 for j in commit_j):
                    text_actions = self.tokenizer.batch_decode(
                        batch.batch["responses"], skip_special_tokens=True
                    )
            if (
                envs.sigma_terminal_select_active()
                and _step == self.config.env.max_steps - 1
                and active_masks.any()
            ):
                sigma_terminal_k = int(getattr(self.config.env, "sigma_terminal_k", 3))
                extra_rounds = []
                for _ in range(sigma_terminal_k - 1):
                    regen_padded, regen_pad_size = pad_dataproto_to_divisor(
                        batch_input, actor_rollout_wg.world_size
                    )
                    regen_output = unpad_dataproto(
                        actor_rollout_wg.generate_sequences(regen_padded),
                        pad_size=regen_pad_size,
                    )
                    regen_texts = self.tokenizer.batch_decode(
                        regen_output.batch["responses"], skip_special_tokens=True
                    )
                    extra_rounds.append((regen_output, regen_texts))
                candidate_texts = [text_actions] + [
                    texts for (_, texts) in extra_rounds
                ]
                commit_j = envs.sigma_terminal_decide(candidate_texts, active_masks)
                for i, j in enumerate(commit_j):
                    if j == 0:
                        continue
                    regen_output, _ = extra_rounds[j - 1]
                    for key, val in regen_output.batch.items():
                        if key in batch.batch:
                            batch.batch[key][i] = val[i]
                    for key, val in regen_output.non_tensor_batch.items():
                        if key in batch.non_tensor_batch:
                            batch.non_tensor_batch[key][i] = val[i]
                if any(j != 0 for j in commit_j):
                    text_actions = self.tokenizer.batch_decode(
                        batch.batch["responses"], skip_special_tokens=True
                    )
            next_obs, rewards, dones, infos = envs.step(text_actions)
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
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1
            assert len(rewards) == batch_size, (
                f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            )
            batch.non_tensor_batch["rewards"] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch["active_masks"] = torch_to_numpy(
                active_masks, is_object=True
            )
            batch_list: list[dict] = to_list_of_dict(batch)
            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])
            is_done = np.logical_or(is_done, dones)
            obs = next_obs
            if is_done.all():
                break
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

    def dynamic_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
    ) -> DataProto:
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches
        while (
            len(total_batch_list)
            < self.config.data.train_batch_size * self.config.env.rollout.n
            and try_count < max_try_count
        ):
            if len(total_batch_list) > 0:
                print(
                    f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})"
                )
            try_count += 1
            (
                batch_list,
                episode_rewards,
                episode_lengths,
                success,
                traj_uid,
                tool_callings,
            ) = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
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
            ) = filter_group_data(
                batch_list=batch_list,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                success=success,
                traj_uid=traj_uid,
                tool_callings=tool_callings,
                config=self.config,
                last_try=(try_count == max_try_count),
            )
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)
        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {
            key: np.concatenate([success[key] for success in total_success], axis=0)
            for key in total_success[0].keys()
        }
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)
        return (
            total_batch_list,
            total_episode_rewards,
            total_episode_lengths,
            total_success,
            total_traj_uid,
            total_tool_callings,
        )

    def multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        is_train: bool = True,
    ) -> DataProto:
        if is_train:
            gen_batch = gen_batch.repeat(
                repeat_times=self.config.env.rollout.n, interleave=True
            )
        if self.config.algorithm.filter_groups.enable and is_train:
            (
                total_batch_list,
                total_episode_rewards,
                total_episode_lengths,
                total_success,
                total_traj_uid,
                totoal_tool_callings,
            ) = self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            (
                total_batch_list,
                total_episode_rewards,
                total_episode_lengths,
                total_success,
                total_traj_uid,
                totoal_tool_callings,
            ) = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        return gen_batch_output
