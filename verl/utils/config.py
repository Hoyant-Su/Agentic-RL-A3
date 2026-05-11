from typing import Dict
from omegaconf import DictConfig

def update_dict_with_config(dictionary: Dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)
