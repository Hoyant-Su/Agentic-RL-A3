from verl import DataProto
import torch
import numpy as np
import re

class EpisodeRewardManager:

    def __init__(self, tokenizer, num_examine, normalize_by_length=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.normalize_by_length = normalize_by_length

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print_data_sources = {}
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][
                :prompt_length
            ].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]
            prompt_str = self.tokenizer.decode(
                valid_prompt_ids, skip_special_tokens=False
            )
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=False
            )
            data_source = data_item.non_tensor_batch["data_source"]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            multi_modal_inputs = data_item.non_tensor_batch.get(
                "multi_modal_inputs", None
            )
            if multi_modal_inputs is not None:
                pixel_values = multi_modal_inputs["pixel_values"]
                image_grid_thw = multi_modal_inputs["image_grid_thw"]
            episode_rewards = data_item.non_tensor_batch["episode_rewards"]
            episode_lengths = data_item.non_tensor_batch["episode_lengths"]
            if self.normalize_by_length:
                score = episode_rewards / episode_lengths
            else:
                score = episode_rewards
            reward_tensor[i, valid_response_length - 1] = torch.tensor(
                score, dtype=torch.float32, device=prompt_ids.device
            )
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if (
                already_print_data_sources[data_source] < self.num_examine
                and np.random.random() < 0.1
            ):
                already_print_data_sources[data_source] += 1
                print(f"[{data_source}][prompt]", prompt_str)
                if data_source == "bash_coding":
                    try:
                        response_str = re.sub(
                            r"<think>.*?</think>",
                            "",
                            response_str,
                            flags=re.IGNORECASE | re.DOTALL,
                        )
                        response_str = re.sub(
                            r"<think>.*$",
                            "",
                            response_str,
                            flags=re.IGNORECASE | re.DOTALL,
                        )
                    except Exception:
                        pass
                print(f"[{data_source}][response]", response_str)
                print(f"[{data_source}][score]", score)
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {},
            }
        else:
            return reward_tensor
