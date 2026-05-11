import importlib.util
from pathlib import Path
import sys
import types
from typing import Any
import numpy as np
import torch
from verl import DataProto

RSTAR_ROOT = Path(__file__).resolve().parents[5] / "baseline_methods" / "rStar"
RSTAR_DOWNSAMPLE_DIR = RSTAR_ROOT / "rstar2_agent" / "down_sample"
RSTAR_PACKAGE_NAME = "main_entry_rstar_down_sample"

def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def _load_rstar_downsample_impl():
    package = sys.modules.get(RSTAR_PACKAGE_NAME)
    if package is None:
        package = types.ModuleType(RSTAR_PACKAGE_NAME)
        package.__path__ = [str(RSTAR_DOWNSAMPLE_DIR)]
        sys.modules[RSTAR_PACKAGE_NAME] = package
    utils_module_name = f"{RSTAR_PACKAGE_NAME}.utils"
    reject_module_name = f"{RSTAR_PACKAGE_NAME}.reject_sampling"
    roc_module_name = f"{RSTAR_PACKAGE_NAME}.roc"
    if utils_module_name not in sys.modules:
        _load_module(utils_module_name, RSTAR_DOWNSAMPLE_DIR / "utils.py")
    if reject_module_name not in sys.modules:
        _load_module(reject_module_name, RSTAR_DOWNSAMPLE_DIR / "reject_sampling.py")
    if roc_module_name not in sys.modules:
        _load_module(roc_module_name, RSTAR_DOWNSAMPLE_DIR / "roc.py")
    reject_module = sys.modules[reject_module_name]
    roc_module = sys.modules[roc_module_name]
    return reject_module.reject_equal_reward, roc_module.resample_of_correct

def _filter_extra_info(
    extra_info: dict[str, list[Any]], keep_indices: np.ndarray
) -> dict[str, list[Any]]:
    filtered: dict[str, list[Any]] = {}
    for key, values in extra_info.items():
        filtered[key] = [values[int(i)] for i in keep_indices]
    return filtered

def apply_rstar_resampling(
    data: DataProto,
    reward_tensor: torch.Tensor,
    reward_extra_info: dict[str, list[Any]],
    tokenizer,
    reward_config: dict[str, Any],
    num_trainer_replicas: int,
) -> tuple[torch.Tensor, dict[str, list[Any]], dict[str, float], str | None]:
    reject_equal_reward, resample_of_correct = _load_rstar_downsample_impl()
    metrics: dict[str, float] = {}
    original_batch_size = len(data)
    data.non_tensor_batch["_rstar_row_id"] = np.arange(len(data), dtype=np.int64)
    data.batch["token_level_scores"] = reward_tensor
    do_reject_equal_reward = bool(reward_config.get("rstar_reject_equal_reward", 1))
    data_after_reject, reject_metrics = reject_equal_reward(
        data,
        do_sample=do_reject_equal_reward,
        world_size=num_trainer_replicas,
    )
    metrics.update({key: float(value) for key, value in reject_metrics.items()})
    if data_after_reject is None:
        data.non_tensor_batch.pop("_rstar_row_id")
        metrics["rstar/empty_after_reject_equal_reward"] = 1.0
        metrics["rstar/skipped_training_step"] = 1.0
        return reward_tensor, reward_extra_info, metrics, "reject_equal_reward"
    do_resample_of_correct = bool(
        reward_config.get("rstar_roc_error_ratio", 1)
    ) or bool(reward_config.get("rstar_roc_answer_format", 1))
    roc_config = {
        "roc_error_ratio": bool(reward_config.get("rstar_roc_error_ratio", 1)),
        "roc_answer_format": bool(reward_config.get("rstar_roc_answer_format", 1)),
        "min_zero_reward_trace_num": int(
            reward_config.get("rstar_min_zero_reward_trace_num", 2)
        ),
        "min_non_zero_reward_trace_num": int(
            reward_config.get("rstar_min_non_zero_reward_trace_num", 2)
        ),
        "down_sample_to_n": int(reward_config.get("rstar_downsample_to_n", 0)),
    }
    data_after_resample, roc_metrics = resample_of_correct(
        data_after_reject,
        tokenizer=tokenizer,
        config=roc_config,
        do_sample=do_resample_of_correct,
        world_size=num_trainer_replicas,
    )
    metrics.update({key: float(value) for key, value in roc_metrics.items()})
    if data_after_resample is None:
        data.non_tensor_batch.pop("_rstar_row_id")
        metrics["rstar/empty_after_resample_of_correct"] = 1.0
        metrics["rstar/skipped_training_step"] = 1.0
        return reward_tensor, reward_extra_info, metrics, "resample_of_correct"
    keep_indices = np.asarray(
        data_after_resample.non_tensor_batch["_rstar_row_id"], dtype=np.int64
    )
    data_after_resample.non_tensor_batch.pop("_rstar_row_id")
    data.batch = data_after_resample.batch
    data.non_tensor_batch = data_after_resample.non_tensor_batch
    keep_tensor = torch.as_tensor(
        keep_indices, device=reward_tensor.device, dtype=torch.long
    )
    reward_tensor = reward_tensor.index_select(0, keep_tensor)
    reward_extra_info = _filter_extra_info(reward_extra_info, keep_indices)
    metrics["rstar/after_resampling_trace_num"] = float(len(keep_indices))
    metrics["rstar/before_resampling_trace_num"] = float(original_batch_size)
    metrics["rstar/skipped_training_step"] = 0.0
    return reward_tensor, reward_extra_info, metrics, None
