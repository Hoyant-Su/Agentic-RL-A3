from typing import List, Tuple, Dict, Union, Any
import os
from collections import defaultdict

import numpy as np
import torch

def to_numpy(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, (int, float, bool, Tuple, List)):
        data = np.array(data)
    else:
        raise ValueError(f"Unsupported type: {type(data)})")
    return data

class EnvironmentManagerBase:
    def __init__(self, envs, projection_f, config):
        self.envs = envs
        self.projection_f = projection_f
        self.config = config

    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        return {"text": None, "image": obs, "anchor": None}, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        next_observations = {"text": None, "image": next_obs, "anchor": None}
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
    ) -> List[str]:
        pass

    def close(self) -> None:
        self.envs.close()

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        total_infos = kwargs["total_infos"]
        total_batch_list = kwargs["total_batch_list"]
        batch_size = len(total_batch_list)
        success = defaultdict(list)
        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)
        assert len(success["success_rate"]) == batch_size
        return {key: np.array(value) for key, value in success.items()}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item["active_masks"]:
                info = total_infos[batch_idx][i]
                won_value = float(info["won"])
                success["success_rate"].append(won_value)
                return

    def save_image(self, image, step):
        path = os.path.join(
            os.path.dirname(__file__), os.path.join("images", self.config.env.env_name)
        )
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f"step{step}.png")
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type: {type(image)})")
        if len(image.shape) == 4:
            image = image[0]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1.0:
            image = image * 255
        image = image.astype(np.uint8)
        from PIL import Image

        image = Image.fromarray(image)
        image.save(path)
