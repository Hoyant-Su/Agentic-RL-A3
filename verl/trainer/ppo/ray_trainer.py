
import json
import os
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Tuple, Type
import numpy as np
import ray
import torch
from datasets import Dataset as HFDataset
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
from main_entry.cli_agent_bash_coding.algo import core_gigpo
from main_entry.cli_agent_bash_coding.action_schema import (
    extract_plan_from_bash_coding_action,
)
from main_entry.cli_agent_bash_coding.action_schema import uses_commit_action_schema
from main_entry.cli_agent_bash_coding.algo.hgpo import compute_hgpo_advantage
from main_entry.cli_agent_bash_coding.reward import apply_rstar_resampling
from main_entry.cli_agent_bash_coding.reward_provenance import REWARD_PROVENANCE_KEYS
from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch

WorkerType = Type[Worker]

class Role(Enum):

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6

class AdvantageEstimator(str, Enum):

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"
    GiGPO = "gigpo"
    HGPO = "hgpo"

def _is_enabled_flag(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

def _hgpo_enabled(config) -> bool:
    return _is_enabled_flag(config.algorithm.enable_hgpo) or _is_enabled_flag(
        config.actor_rollout_ref.actor.enable_hgpo
    )


@dataclass
class ResourcePoolManager:

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=resource_pool_name,
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool
        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        return sum(
            [
                n_gpus
                for process_on_nodes in self.resource_pool_spec.values()
                for n_gpus in process_on_nodes
            ]
        )

    def _check_resource_available(self):
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0)
            if "GPU" in node_info
            else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [
                n_gpus
                for process_on_nodes in self.resource_pool_spec.values()
                for n_gpus in process_on_nodes
            ]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )

def apply_kl_penalty(
    data: DataProto,
    kl_ctrl: core_algos.AdaptiveKLController,
    kl_penalty="kl",
    multi_turn=False,
):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )
    kld = kld * response_mask
    beta = kl_ctrl.value
    token_level_rewards = token_level_scores - beta * kld
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards
    metrics = {
        "actor/reward_kl_penalty": current_kl,
        "actor/reward_kl_penalty_coeff": beta,
    }
    return data, metrics

def apply_invalid_action_penalty(data: DataProto, invalid_action_penalty_coef=float):
    reward_tensor = data.batch["token_level_scores"]
    if "step_rewards" in data.batch.keys():
        step_rewards = data.batch["step_rewards"]
    for i in range(len(data)):
        data_item = data[i]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        action_valids = data_item.non_tensor_batch["is_action_valid"].astype(np.float32)
        action_invalids = torch.tensor(
            1 - action_valids, dtype=torch.float32, device=prompt_ids.device
        ).squeeze(0)
        reward_tensor[i, valid_response_length - 1] -= (
            invalid_action_penalty_coef * action_invalids
        )
        if "step_rewards" in data.batch.keys():
            step_rewards[i] -= invalid_action_penalty_coef * action_invalids
    valid_action_ratio = np.mean(
        data.non_tensor_batch["is_action_valid"].astype(np.float32)
    ).item()
    metrics = {"episode/valid_action_ratio": valid_action_ratio}
    return data, metrics

def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]

def _resolve_gigpo_obs_similarity_flags(config) -> Tuple[bool, bool]:
    sm = bool(
        OmegaConf.select(
            config, "algorithm.gigpo.enable_similarity_sequence_matcher", default=False
        )
    )
    llm = bool(
        OmegaConf.select(config, "algorithm.gigpo.enable_similarity_llm", default=False)
    )
    if not sm and not llm:
        legacy = OmegaConf.select(
            config, "algorithm.gigpo.enable_similarity", default=None
        )
        if legacy is not None and bool(legacy):
            sm = True
    if sm and llm:
        raise ValueError(
            "algorithm.gigpo: enable_similarity_sequence_matcher and enable_similarity_llm cannot both be True"
        )
    return sm, llm

def _resolve_hgpo_obs_similarity_flags(config) -> Tuple[bool, bool]:
    sm = bool(
        OmegaConf.select(
            config, "algorithm.hgpo.enable_similarity_sequence_matcher", default=False
        )
    )
    llm = bool(
        OmegaConf.select(config, "algorithm.hgpo.enable_similarity_llm", default=False)
    )
    if not sm and not llm:
        legacy = OmegaConf.select(
            config, "algorithm.hgpo.enable_similarity", default=None
        )
        if legacy is not None and bool(legacy):
            llm = True
    if sm and llm:
        raise ValueError(
            "algorithm.hgpo: enable_similarity_sequence_matcher and enable_similarity_llm cannot both be True"
        )
    return sm, llm

def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    multi_turn=False,
    norm_adv_by_std_in_grpo=True,
    compute_mean_std_cross_steps=True,
    step_advantage_w=None,
    gigpo_mode=None,
    gigpo_enable_similarity_sequence_matcher=False,
    gigpo_enable_similarity_llm=False,
    gigpo_similarity_thresh=None,
    hgpo_mode=None,
    hgpo_length_weight_alpha=None,
    hgpo_base_group=None,
    hgpo_enable_similarity_sequence_matcher=False,
    hgpo_enable_similarity_llm=False,
    hgpo_similarity_thresh=None,
    hgpo_similarity_batch_size=None,
    **kwargs,
):
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = grpo_calculation_mask.size(1)
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch["traj_uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            compute_mean_std_cross_steps=compute_mean_std_cross_steps,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch["traj_uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            compute_mean_std_cross_steps=compute_mean_std_cross_steps,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = (
            core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=data.batch["response_mask"],
                index=data.non_tensor_batch["uid"],
                traj_index=data.non_tensor_batch["traj_uid"],
                compute_mean_std_cross_steps=compute_mean_std_cross_steps,
            )
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch["traj_uid"],
            compute_mean_std_cross_steps=compute_mean_std_cross_steps,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GiGPO:
        _gigpo_t = (
            0.95 if gigpo_similarity_thresh is None else float(gigpo_similarity_thresh)
        )
        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            step_rewards=data.batch["step_rewards"],
            response_mask=data.batch["response_mask"],
            anchor_obs=data.non_tensor_batch["anchor_obs"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch["traj_uid"],
            step_advantage_w=step_advantage_w,
            mode=gigpo_mode,
            enable_similarity_sequence_matcher=gigpo_enable_similarity_sequence_matcher,
            enable_similarity_llm=gigpo_enable_similarity_llm,
            similarity_thresh=_gigpo_t,
            compute_mean_std_cross_steps=compute_mean_std_cross_steps,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.HGPO:
        _hgpo_t = (
            0.95 if hgpo_similarity_thresh is None else float(hgpo_similarity_thresh)
        )
        _hgpo_bs = (
            32
            if hgpo_similarity_batch_size is None
            else int(hgpo_similarity_batch_size)
        )
        data = compute_hgpo_advantage(
            data,
            multi_turn=multi_turn,
            history_length=kwargs["history_length"],
            hgpo_mode=hgpo_mode,
            hgpo_length_weight_alpha=hgpo_length_weight_alpha,
            hgpo_base_group=hgpo_base_group,
            gamma=gamma,
            epsilon=kwargs["epsilon"],
            compute_mean_std_cross_steps=compute_mean_std_cross_steps,
            enable_similarity_sequence_matcher=hgpo_enable_similarity_sequence_matcher,
            enable_similarity_llm=hgpo_enable_similarity_llm,
            hgpo_similarity_thresh=_hgpo_t,
            hgpo_similarity_batch_size=_hgpo_bs,
        )
    else:
        raise NotImplementedError
    return data

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last

def _bash_coding_enabled() -> bool:
    return str(os.environ.get("BASH_CODING_ENABLE", "0")).strip() == "1"

def _as_path_list(data_files) -> list[str]:
    if isinstance(data_files, (list, tuple)):
        return [os.path.abspath(str(path)) for path in data_files]
    return [os.path.abspath(str(data_files))]

def _infer_dataset_name_from_parquet(parquet_path: str) -> str:
    return os.path.basename(os.path.dirname(os.path.abspath(parquet_path)))

def _extract_sample_id(example: dict) -> str:
    extra_info = example.get("extra_info", None)
    if isinstance(extra_info, dict) and extra_info.get("id", None) is not None:
        return str(extra_info["id"])
    if example.get("id", None) is not None:
        return str(example["id"])
    raise ValueError("Each validation sample must contain id in extra_info.id or id.")

class RayPPOTrainer:

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.traj_collector = traj_collector
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(
                config.algorithm.kl_ctrl
            )
        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif (
            self.config.algorithm.adv_estimator
            in [
                AdvantageEstimator.GRPO,
                AdvantageEstimator.GRPO_PASSK,
                AdvantageEstimator.REINFORCE_PLUS_PLUS,
                AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO,
                AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
                AdvantageEstimator.GiGPO,
                AdvantageEstimator.HGPO,
            ]
            or _hgpo_enabled(config)
        ):
            self.use_critic = False
        else:
            raise NotImplementedError
        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self.val_sample_dataset_map = None if _bash_coding_enabled() else {}
        self.kl_oscillation_stop = self._build_kl_oscillation_stop_config()
        self.kl_oscillation_stop_state = {
            "values": [],
            "trigger_count": 0,
        }

    def _build_val_sample_dataset_map(self) -> dict[str, str]:
        sample_to_dataset: dict[str, str] = {}
        for parquet_path in _as_path_list(self.config.data.val_files):
            dataset_name = _infer_dataset_name_from_parquet(parquet_path)
            dataset = HFDataset.from_parquet(parquet_path)
            for idx in range(len(dataset)):
                sample_id = _extract_sample_id(dataset[idx])
                existing = sample_to_dataset.get(sample_id, None)
                if existing is not None and existing != dataset_name:
                    raise ValueError(
                        f"Duplicate validation sample id across datasets: {sample_id}"
                    )
                sample_to_dataset[sample_id] = dataset_name
        return sample_to_dataset

    def _get_val_sample_dataset_map(self) -> dict[str, str]:
        if self.val_sample_dataset_map is None:
            self.val_sample_dataset_map = self._build_val_sample_dataset_map()
        return self.val_sample_dataset_map

    def _build_kl_oscillation_stop_config(self) -> dict:
        cfg = OmegaConf.select(self.config, "trainer.kl_oscillation_stop", default=None)
        if cfg is None:
            return {
                "enabled": False,
                "metric_key": "",
                "baseline_steps": 1,
                "window_size": 1,
                "ratio": 1.0,
                "patience": 1,
                "min_steps": 1,
            }
        required_keys = (
            "enable",
            "metric_key",
            "baseline_steps",
            "window_size",
            "ratio",
            "patience",
            "min_steps",
        )
        missing_keys = [key for key in required_keys if cfg.get(key, None) is None]
        if missing_keys:
            raise ValueError(
                f"Missing trainer.kl_oscillation_stop fields: {missing_keys}"
            )
        baseline_steps = int(cfg["baseline_steps"])
        window_size = int(cfg["window_size"])
        patience = int(cfg["patience"])
        min_steps = int(cfg["min_steps"])
        metric_key = str(cfg["metric_key"])
        return {
            "enabled": bool(cfg["enable"]),
            "metric_key": metric_key,
            "baseline_steps": baseline_steps,
            "window_size": window_size,
            "ratio": float(cfg["ratio"]),
            "patience": patience,
            "min_steps": min_steps,
        }

    def _mean_abs_delta(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        deltas = [abs(values[idx] - values[idx - 1]) for idx in range(1, len(values))]
        return float(sum(deltas) / len(deltas))

    def _value_range(self, values: list[float]) -> float:
        if not values:
            return 0.0
        return float(max(values) - min(values))

    def _update_kl_oscillation_stop(
        self, metrics: dict[str, float]
    ) -> tuple[bool, dict[str, float]]:
        cfg = self.kl_oscillation_stop
        if not cfg["enabled"]:
            return False, {}
        metric_key = cfg["metric_key"]
        if metric_key not in metrics:
            raise ValueError(
                f"KL oscillation stop metric '{metric_key}' not found in metrics."
            )
        values = self.kl_oscillation_stop_state["values"]
        values.append(float(metrics[metric_key]))
        baseline_values = values[: cfg["baseline_steps"]]
        recent_values = values[-cfg["window_size"] :]
        baseline_delta_mean = self._mean_abs_delta(baseline_values)
        recent_delta_mean = self._mean_abs_delta(recent_values)
        baseline_range = self._value_range(baseline_values)
        recent_range = self._value_range(recent_values)
        delta_ratio = (
            recent_delta_mean / baseline_delta_mean
            if baseline_delta_mean > 0
            else float("inf")
            if recent_delta_mean > 0
            else 1.0
        )
        range_ratio = (
            recent_range / baseline_range
            if baseline_range > 0
            else float("inf")
            if recent_range > 0
            else 1.0
        )
        stop_metrics = {
            "training/kl_oscillation_metric": 1.0
            if metric_key == "actor/kl_loss"
            else 2.0,
            "training/kl_oscillation_baseline_delta_mean": baseline_delta_mean,
            "training/kl_oscillation_recent_delta_mean": recent_delta_mean,
            "training/kl_oscillation_baseline_range": baseline_range,
            "training/kl_oscillation_recent_range": recent_range,
            "training/kl_oscillation_delta_ratio": delta_ratio,
            "training/kl_oscillation_range_ratio": range_ratio,
            "training/kl_oscillation_trigger_count": float(
                self.kl_oscillation_stop_state["trigger_count"]
            ),
        }
        if len(values) < max(cfg["baseline_steps"], cfg["min_steps"]):
            return False, stop_metrics
        use_range_only = bool(cfg.get("range_only", False))
        triggered = (
            range_ratio >= cfg["ratio"]
            if use_range_only
            else (delta_ratio >= cfg["ratio"] and range_ratio >= cfg["ratio"])
        )
        self.kl_oscillation_stop_state["trigger_count"] = (
            self.kl_oscillation_stop_state["trigger_count"] + 1 if triggered else 0
        )
        stop_metrics["training/kl_oscillation_trigger_count"] = float(
            self.kl_oscillation_stop_state["trigger_count"]
        )
        stop_metrics["training/kl_oscillation_stop"] = float(
            self.kl_oscillation_stop_state["trigger_count"] >= cfg["patience"]
        )
        return self.kl_oscillation_stop_state["trigger_count"] >= cfg[
            "patience"
        ], stop_metrics

    def _load_completed_validation_rollouts(
        self, jsonl_path: str
    ) -> set[tuple[str, int]]:
        completed: set[tuple[str, int]] = set()
        if not os.path.exists(jsonl_path):
            return completed
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                sample_id_base = str(obj.get("sample_id_base", "")).strip()
                rollout_idx = obj.get("rollout_idx", None)
                if sample_id_base and rollout_idx is not None:
                    completed.add((sample_id_base, int(rollout_idx)))
        return completed

    def _extract_validation_sample_id_base(self, raw_line: str) -> str:
        key = '"sample_id_base"'
        pos = raw_line.find(key)
        if pos < 0:
            return ""
        tail = raw_line[pos + len(key) :]
        quote_pos = tail.find('"')
        if quote_pos < 0:
            return ""
        tail = tail[quote_pos + 1 :]
        end_pos = tail.find('"')
        if end_pos < 0:
            return ""
        return tail[:end_pos].strip()

    def _sanitize_incomplete_validation_output(self, jsonl_path: str) -> None:
        if not os.path.exists(jsonl_path):
            return
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        last_idx = -1
        for idx in range(len(lines) - 1, -1, -1):
            if lines[idx].strip():
                last_idx = idx
                break
        if last_idx < 0:
            return
        last_raw = lines[last_idx].strip()
        if last_raw.endswith("}"):
            return
        sample_id_base = self._extract_validation_sample_id_base(last_raw)
        if not sample_id_base:
            for idx in range(last_idx - 1, -1, -1):
                raw = lines[idx].strip()
                if not raw:
                    continue
                sample_id_base = str(json.loads(raw)["sample_id_base"]).strip()
                break
        kept_lines: list[str] = []
        for idx, line in enumerate(lines):
            raw = line.strip()
            if not raw or idx == last_idx:
                continue
            if str(json.loads(raw)["sample_id_base"]).strip() == sample_id_base:
                continue
            kept_lines.append(raw)
        rewritten = "\n".join(kept_lines)
        if rewritten:
            rewritten += "\n"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(rewritten)

    def _pending_val_indices(
        self, completed: set[tuple[str, int]], rollout_n: int
    ) -> list[int]:
        pending = []
        for idx in range(len(self.val_dataset)):
            sample_id = _extract_sample_id(self.val_dataset[idx])
            if all(
                (sample_id, rollout_idx) in completed
                for rollout_idx in range(rollout_n)
            ):
                continue
            pending.append(idx)
        return pending

    def _use_validation_resume(self) -> bool:
        return (
            _bash_coding_enabled()
            and bool(self.config.trainer.get("val_only", False))
            and bool(self.config.trainer.get("validation_data_dir", None))
        )

    def _compute_bash_coding_data_metrics(self, batch: DataProto) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        traj_uid = batch.non_tensor_batch.get("traj_uid", None)
        unique_idx = None
        if traj_uid is not None:
            _, unique_idx = np.unique(traj_uid, return_index=True)
            unique_idx = np.sort(unique_idx)
        for key in REWARD_PROVENANCE_KEYS:
            values = batch.non_tensor_batch.get(key, None)
            if values is None:
                continue
            component = np.asarray(values, dtype=np.float32)
            if unique_idx is not None:
                component = component[unique_idx]
            if component.size == 0:
                continue
            metrics[f"reward_components/{key}/mean"] = float(component.mean())
            metrics[f"reward_components/{key}/max"] = float(component.max())
            metrics[f"reward_components/{key}/min"] = float(component.min())
            metrics[f"reward_components/{key}/nonzero_ratio"] = float(
                np.count_nonzero(component) / component.size
            )
        responses = batch.batch.get("responses", None)
        attention_mask = batch.batch.get("attention_mask", None)
        if responses is None or attention_mask is None:
            return metrics
        response_mask = attention_mask[:, -responses.size(1) :]
        plan_lens: list[int] = []
        missing = 0
        enable_commit = uses_commit_action_schema(
            str(OmegaConf.select(self.config, "env.bash_coding_harness", default=""))
        )
        for i in range(responses.size(0)):
            row_ids = responses[i][response_mask[i].bool()].tolist()
            response_text = self.tokenizer.decode(row_ids, skip_special_tokens=True)
            plan_text = extract_plan_from_bash_coding_action(
                response_text, enable_commit=enable_commit
            )
            if plan_text is None:
                plan_lens.append(0)
                missing += 1
            else:
                plan_lens.append(
                    len(self.tokenizer.encode(plan_text, add_special_tokens=False))
                )
        if plan_lens:
            plan_lens_arr = np.asarray(plan_lens, dtype=np.float32)
            metrics["plan_length/mean"] = float(plan_lens_arr.mean())
            metrics["plan_length/max"] = float(plan_lens_arr.max())
            metrics["plan_length/min"] = float(plan_lens_arr.min())
            metrics["plan_length/missing_ratio"] = float(missing / len(plan_lens))
        rstar_metrics = batch.meta_info.get("rstar_metrics", None)
        if isinstance(rstar_metrics, dict):
            for key, value in rstar_metrics.items():
                metrics[key] = float(value)
        return metrics

    def _validate_config(self):
        config = self.config
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        real_train_batch_size = (
            config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        )
        assert real_train_batch_size % n_gpus == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
        )

        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }
            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"
                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )
                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'"
                        + "is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )
            if self.use_reference_policy:
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )
        if self.use_critic and not config.critic.use_dynamic_bsz:
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size,
                config.critic.ppo_micro_batch_size_per_gpu,
                "critic",
            )
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size,
                config.reward_model.micro_batch_size_per_gpu,
                "reward_model",
            )
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get(
                "ulysses_sequence_parallel_size", 1
            )
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert (
                    config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size
                    >= n_gpus
                )
        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"
        if (
            config.algorithm.use_kl_in_reward
            and config.actor_rollout_ref.actor.use_kl_loss
        ):
            print("NOTICE: You have both enabled in-reward kl and kl loss.")
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert (
                    config.critic.ppo_mini_batch_size
                    % config.critic.ppo_micro_batch_size
                    == 0
                )
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )
        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )
        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert (
                config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None
            ), (
                "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            )
            assert (
                config.algorithm.adv_estimator
                in [
                    AdvantageEstimator.GRPO,
                    AdvantageEstimator.HGPO,
                ]
                or _hgpo_enabled(config)
            ), "only GRPO and HGPO are tested for multi-turn with tool"
        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn
        self.collate_fn = collate_fn
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get(
                "gen_batch_size", self.config.data.train_batch_size
            ),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )
        self.val_dataloader = self._build_eval_dataloader(self.val_dataset)
        self.mid_val_dataset = None
        self.mid_val_dataloader = None
        if self.config.trainer.get("enable_mid_val", False) and self.config.data.get(
            "mid_val_files", None
        ):
            self.mid_val_dataset = create_rl_dataset(
                self.config.data.mid_val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
            )
            self.mid_val_dataloader = self._build_eval_dataloader(self.mid_val_dataset)
        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"
        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}"
        )
        if self.mid_val_dataloader is not None:
            print(
                f"Size of mid-val dataloader: {len(self.mid_val_dataloader)} "
                f"(dataset size={len(self.mid_val_dataset)})"
            )
        total_training_steps = (
            len(self.train_dataloader) * self.config.trainer.total_epochs
        )
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                        total_training_steps
                    )
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(
                f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}"
            )

    def _build_eval_dataloader(self, dataset):
        val_batch_size = self.config.data.val_batch_size
        if val_batch_size is None:
            val_batch_size = len(dataset)
        return StatefulDataLoader(
            dataset=dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def _dump_generations(
        self, inputs, outputs, scores, reward_extra_infos_dict, dump_path
    ):
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")
        n = len(inputs)
        base_data = {}
        identity_first = [
            "dataset",
            "sample_id",
            "sample_id_base",
            "turn_idx",
            "traj_uid",
            "sample_turn_id",
            "rollout_idx",
        ]
        for k in identity_first:
            v = reward_extra_infos_dict.get(k, None)
            if v is not None and len(v) == n:
                base_data[k] = v
        base_data.update(
            {
                "input": inputs,
                "output": outputs,
                "score": scores,
                "step": [self.global_steps] * n,
            }
        )
        for k, v in reward_extra_infos_dict.items():
            if k in base_data:
                continue
            if len(v) == n:
                base_data[k] = v
        file_mode = "a" if self.config.trainer.get("val_only", False) else "w"
        with open(filename, file_mode, encoding="utf-8") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0:
            return
        import numpy as np

        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        samples = samples[:generations_to_log]
        self.validation_generations_logger.log(
            self.config.trainer.logger, samples, self.global_steps
        )

    def _validate(self):
        if not self._use_validation_resume():
            return self._validate_impl()
        validation_data_dir = self.config.trainer.get("validation_data_dir", None)
        dump_file = os.path.join(validation_data_dir, f"{self.global_steps}.jsonl")
        rollout_n = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
        self._sanitize_incomplete_validation_output(dump_file)
        completed = self._load_completed_validation_rollouts(dump_file)
        pending_indices = self._pending_val_indices(completed, rollout_n)
        skipped = len(self.val_dataset) - len(pending_indices)
        print(f"Validation resume: skipped {skipped} completed samples")
        if not pending_indices:
            return {}
        original_val_dataset = self.val_dataset
        original_val_dataloader = self.val_dataloader
        metric_sums: dict[str, float] = {}
        metric_weights: dict[str, int] = {}
        flush_size = 100
        try:
            for start in range(0, len(pending_indices), flush_size):
                subset = torch.utils.data.Subset(
                    original_val_dataset, pending_indices[start : start + flush_size]
                )
                self.val_dataset = subset
                self.val_dataloader = self._build_eval_dataloader(subset)
                if hasattr(self.traj_collector, "_validation_dataset_dump"):
                    self.traj_collector._validation_dataset_dump = []
                if hasattr(self.traj_collector, "_validation_prompt_dump"):
                    self.traj_collector._validation_prompt_dump = []
                metrics = self._validate_impl()
                weight = len(subset)
                for key, value in metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * weight
                    metric_weights[key] = metric_weights.get(key, 0) + weight
        finally:
            self.val_dataset = original_val_dataset
            self.val_dataloader = original_val_dataloader
            if hasattr(self.traj_collector, "_validation_dataset_dump"):
                self.traj_collector._validation_dataset_dump = []
            if hasattr(self.traj_collector, "_validation_prompt_dump"):
                self.traj_collector._validation_prompt_dump = []
        return {
            key: metric_sums[key] / metric_weights[key]
            for key in metric_sums
            if metric_weights[key] > 0
        }

    def _run_mid_validation(self):
        if self.mid_val_dataset is None or self.mid_val_dataloader is None:
            return {}
        original_val_dataset = self.val_dataset
        original_val_dataloader = self.val_dataloader
        original_validation_data_dir = self.config.trainer.get(
            "validation_data_dir", None
        )
        original_val_rollout_n = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
        try:
            self.val_dataset = self.mid_val_dataset
            self.val_dataloader = self.mid_val_dataloader
            with open_dict(self.config):
                self.config.trainer.validation_data_dir = self.config.trainer.get(
                    "mid_val_data_dir", None
                )
                self.config.actor_rollout_ref.rollout.val_kwargs.n = int(
                    self.config.trainer.get("mid_val_rollout_n", 1)
                )
            metrics = self._validate_impl()
        finally:
            self.val_dataset = original_val_dataset
            self.val_dataloader = original_val_dataloader
            with open_dict(self.config):
                self.config.trainer.validation_data_dir = original_validation_data_dir
                self.config.actor_rollout_ref.rollout.val_kwargs.n = (
                    original_val_rollout_n
                )
        renamed_metrics = {}
        for key, value in metrics.items():
            if key.startswith("val/"):
                renamed_metrics[f"mid_val/{key[len('val/') :]}"] = value
            else:
                renamed_metrics[f"mid_val/{key}"] = value
        return renamed_metrics

    def _validate_impl(self):
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_ids = []
        sample_datasets = []
        turn_idxs = []
        traj_uids_for_dump = []
        sample_id_bases = []
        rollout_idxs = []
        if hasattr(self.traj_collector, "_validation_dataset_dump"):
            self.traj_collector._validation_dataset_dump = []
        if hasattr(self.traj_collector, "_validation_prompt_dump"):
            self.traj_collector._validation_prompt_dump = []
        val_progress = tqdm(self.val_dataloader, desc="Validation", leave=True)
        for test_data in val_progress:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )
            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
            ):
                return {}
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source", "dataset"]
            if "extra_info" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("extra_info")
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")
            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                gen_batch=test_gen_batch,
                actor_rollout_wg=self.actor_rollout_wg,
                envs=self.val_envs,
                is_train=False,
            )
            del test_batch
            test_batch = test_output_gen_batch
            input_ids = test_output_gen_batch.batch["input_ids"]
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids
            ]
            sample_inputs.extend(input_texts)
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]
            sample_outputs.extend(output_texts)
            bs = len(output_texts)
            sample_ids.extend(
                [str(x) for x in test_output_gen_batch.non_tensor_batch["sample_id"]]
            )
            sample_id_bases.extend(
                [
                    str(x)
                    for x in test_output_gen_batch.non_tensor_batch["sample_id_base"]
                ]
            )
            sample_datasets.extend(
                [str(x) for x in test_gen_batch.non_tensor_batch["dataset"]]
            )
            turn_idxs.extend(
                [int(x) for x in test_output_gen_batch.non_tensor_batch["turn_idx"]]
            )
            current_traj_uids = [
                str(x) for x in test_output_gen_batch.non_tensor_batch["traj_uid"]
            ]
            traj_uids_for_dump.extend(current_traj_uids)
            rollout_counter = {}
            rollout_index_by_pair = {}
            for base, traj in zip(
                test_output_gen_batch.non_tensor_batch["sample_id_base"],
                current_traj_uids,
            ):
                base_key = str(base)
                pair_key = (base_key, str(traj))
                if pair_key not in rollout_index_by_pair:
                    rollout_index_by_pair[pair_key] = rollout_counter.get(base_key, 0)
                    rollout_counter[base_key] = rollout_index_by_pair[pair_key] + 1
                rollout_idxs.append(rollout_index_by_pair[pair_key])
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(
                test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]
                )
            )
            tool_calling_list.append(
                test_output_gen_batch.non_tensor_batch["tool_callings"]
            )
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch["traj_uid"])
            for k in test_batch.non_tensor_batch.keys():
                if "success_rate" in k:
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                    for i in range(1, len(test_batch.non_tensor_batch[k])):
                        assert (
                            test_batch.non_tensor_batch[k][0]
                            == test_batch.non_tensor_batch[k][i]
                        ), (
                            f"not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}"
                        )
        self._maybe_log_val_generations(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )
        validation_data_dir = self.config.trainer.get("validation_data_dir", None)
        if validation_data_dir:
            reward_extra_infos_dict = {}
            dataset_dump = getattr(
                self.traj_collector, "_validation_dataset_dump", None
            )
            prompt_dump = getattr(self.traj_collector, "_validation_prompt_dump", None)
            dataset_values = None
            if len(sample_datasets) == len(sample_outputs) and any(
                item != "" for item in sample_datasets
            ):
                dataset_values = sample_datasets
            elif dataset_dump is not None and len(dataset_dump) == len(sample_outputs):
                dataset_values = [str(item) for item in dataset_dump]
            if dataset_values is not None and len(dataset_values) == len(
                sample_outputs
            ):
                reward_extra_infos_dict["dataset"] = dataset_values
            if len(sample_ids) == len(sample_outputs):
                reward_extra_infos_dict["sample_id"] = sample_ids
            if len(sample_id_bases) == len(sample_outputs):
                reward_extra_infos_dict["sample_id_base"] = sample_id_bases
            if len(rollout_idxs) == len(sample_outputs):
                reward_extra_infos_dict["rollout_idx"] = rollout_idxs
            if len(turn_idxs) == len(sample_outputs):
                reward_extra_infos_dict["turn_idx"] = turn_idxs
                if len(sample_ids) == len(sample_outputs):
                    reward_extra_infos_dict["sample_turn_id"] = [
                        f"{sid}_{t}" if sid != "" and t is not None else ""
                        for sid, t in zip(sample_ids, turn_idxs)
                    ]
            if len(traj_uids_for_dump) == len(sample_outputs):
                reward_extra_infos_dict["traj_uid"] = traj_uids_for_dump
            dumped_inputs = sample_inputs
            if prompt_dump is not None and len(prompt_dump) == len(sample_outputs):
                dumped_inputs = [str(item) for item in prompt_dump]
            if hasattr(self.traj_collector, "_validation_dataset_dump"):
                self.traj_collector._validation_dataset_dump = []
            if hasattr(self.traj_collector, "_validation_prompt_dump"):
                self.traj_collector._validation_prompt_dump = []
            self._dump_generations(
                inputs=dumped_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=validation_data_dir,
            )
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
        data_source_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]
        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())
        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f"val/{data_source}/test_score"] = np.mean(rewards)
        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f"val/{data_source}/tool_call_count/mean"] = np.mean(tool_calls)
        for k, v in success_rate.items():
            metric_dict[f"val/{k}"] = v
        return metric_dict

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.ActorRollout
            )
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = (
                actor_rollout_cls
            )
        else:
            raise NotImplementedError
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.critic
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.RewardModel
            )
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls
        all_wg = {}
        wg_kwargs = {}
        if (
            OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout")
            is not None
        ):
            wg_kwargs["ray_wait_register_center_timeout"] = (
                self.config.trainer.ray_wait_register_center_timeout
            )
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()
        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir,
                f"global_step_{self.global_steps}",
                "actor",
            )
        )
        remove_previous_ckpt_in_save = self.config.trainer.get(
            "remove_previous_ckpt_in_save", False
        )
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None)
            if not remove_previous_ckpt_in_save
            else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None)
            if not remove_previous_ckpt_in_save
            else 1
        )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
        )
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"global_step_{self.global_steps}",
                    "critic",
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.global_steps,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), (
                    "resume ckpt must be str type"
                )
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")
        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        self.actor_rollout_wg.load_checkpoint(
            actor_path,
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(
                dataloader_local_path, weights_only=False
            )
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(
                f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch"
            )

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = (
            batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()
        )
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        global_idx = torch.tensor(
            [j for partition in global_partition_lst for j in partition]
        )
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst,
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        metrics.update(global_balance_stats)

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        reward_config = OmegaConf.select(self.config, "reward", default={})
        rstar_enabled = bool(reward_config.get("rstar_enable", 0))
        rstar_reward_config = (
            OmegaConf.to_container(reward_config, resolve=True) if rstar_enabled else {}
        )
        num_trainer_replicas = int(self.config.trainer.n_gpus_per_node) * int(
            self.config.trainer.nnodes
        )
        self.global_steps = 0
        self._load_checkpoint()
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
            position=0,
            leave=True,
        )
        self.global_steps += 1
        last_val_metrics = None
        actual_batch_size = self.config.data.get(
            "gen_batch_size", self.config.data.train_batch_size
        )
        dataset_size = len(self.train_dataset)
        batches_per_epoch = dataset_size // actual_batch_size
        if batches_per_epoch == 0:
            batches_per_epoch = 1
        for epoch in range(self.config.trainer.total_epochs):
            batch_progress_bar = tqdm(
                total=batches_per_epoch, desc="Batch Progress", position=1, leave=False
            )
            hgpo_enabled = _hgpo_enabled(self.config)
            for batch_idx, batch_dict in enumerate(self.train_dataloader):
                metrics = {}
                timing_raw = {}
                should_stop = False
                rstar_skip_reason = None
                rstar_metrics = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = [
                    "raw_prompt_ids",
                    "data_source",
                    "dataset",
                ]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                is_last_step = self.global_steps >= self.total_training_steps
                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = (
                                self.actor_rollout_wg.generate_sequences(
                                    gen_baseline_batch
                                )
                            )
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            del gen_baseline_batch, gen_baseline_output
                    del batch
                    batch = gen_batch_output
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                        step_rewards_tensor = (
                            core_gigpo.compute_step_discounted_returns(
                                batch=batch, gamma=self.config.algorithm.gamma
                            )
                        )
                        batch.batch["step_rewards"] = step_rewards_tensor
                    if not hgpo_enabled:
                        batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch and not hgpo_enabled:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()
                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                batch, self.config, self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(
                                batch, self.reward_fn
                            )
                            if rstar_enabled:
                                (
                                    reward_tensor,
                                    reward_extra_infos_dict,
                                    rstar_metrics,
                                    rstar_skip_reason,
                                ) = apply_rstar_resampling(
                                    data=batch,
                                    reward_tensor=reward_tensor,
                                    reward_extra_info=reward_extra_infos_dict or {},
                                    tokenizer=self.tokenizer,
                                    reward_config=rstar_reward_config,
                                    num_trainer_replicas=num_trainer_replicas,
                                )
                                batch.meta_info["rstar_metrics"] = rstar_metrics
                    if rstar_skip_reason is not None:
                        metrics.update(
                            {key: float(value) for key, value in rstar_metrics.items()}
                        )
                        metrics.update(
                            {
                                "training/global_step": self.global_steps,
                                "training/epoch": epoch,
                            }
                        )
                        metrics.update(
                            compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                        )
                        logger.log(data=metrics, step=self.global_steps)
                        print(
                            f"[rstar] Skip training step {self.global_steps}: empty batch after {rstar_skip_reason}."
                        )
                        batch_progress_bar.update(1)
                        batch_progress_bar.set_postfix(
                            {
                                "step": f"{batch_idx + 1}/{batches_per_epoch}",
                                "skip": "rstar",
                            }
                        )
                        continue
                    if not hgpo_enabled:
                        with _timer("old_log_prob", timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = (
                                self.config.actor_rollout_ref.actor.loss_agg_mode
                            )
                            entropy_loss = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=loss_agg_mode,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy_loss": entropy_loss.detach().item()
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                actor_old_log_probs = batch.batch["old_log_probs"]
                                attention_mask = batch.batch["attention_mask"]
                                responses = batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]
                                rollout_probs = torch.exp(rollout_old_log_probs)
                                actor_probs = torch.exp(actor_old_log_probs)
                                rollout_probs_diff = torch.abs(
                                    rollout_probs - actor_probs
                                )
                                rollout_probs_diff = torch.masked_select(
                                    rollout_probs_diff, response_mask.bool()
                                )
                                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                                rollout_probs_diff_std = torch.std(rollout_probs_diff)
                                metrics.update(
                                    {
                                        "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                        "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                        "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                    }
                                )
                        if self.use_reference_policy:
                            with _timer("ref", timing_raw):
                                if not self.ref_in_actor:
                                    ref_log_prob = (
                                        self.ref_policy_wg.compute_ref_log_prob(batch)
                                    )
                                else:
                                    ref_log_prob = (
                                        self.actor_rollout_wg.compute_ref_log_prob(
                                            batch
                                        )
                                    )
                                batch = batch.union(ref_log_prob)
                        if self.use_critic:
                            with _timer("values", timing_raw):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)
                    else:
                        if (
                            self.use_reference_policy
                            and self.config.algorithm.use_kl_in_reward
                        ):
                            with _timer("ref", timing_raw):
                                if not self.ref_in_actor:
                                    ref_log_prob = (
                                        self.ref_policy_wg.compute_ref_log_prob(batch)
                                    )
                                else:
                                    ref_log_prob = (
                                        self.actor_rollout_wg.compute_ref_log_prob(
                                            batch
                                        )
                                    )
                                batch = batch.union(ref_log_prob)
                    with _timer("adv", timing_raw):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(
                                future_reward
                            )
                            if rstar_enabled:
                                (
                                    reward_tensor,
                                    reward_extra_infos_dict,
                                    rstar_metrics,
                                    rstar_skip_reason,
                                ) = apply_rstar_resampling(
                                    data=batch,
                                    reward_tensor=reward_tensor,
                                    reward_extra_info=reward_extra_infos_dict or {},
                                    tokenizer=self.tokenizer,
                                    reward_config=rstar_reward_config,
                                    num_trainer_replicas=num_trainer_replicas,
                                )
                                batch.meta_info["rstar_metrics"] = rstar_metrics
                        if rstar_skip_reason is None:
                            batch.batch["token_level_scores"] = reward_tensor
                            print(f"{list(reward_extra_infos_dict.keys())=}")
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update(
                                    {
                                        k: np.array(v)
                                        for k, v in reward_extra_infos_dict.items()
                                    }
                                )
                            if self.config.actor_rollout_ref.actor.get(
                                "use_invalid_action_penalty", True
                            ):
                                batch, invalid_metrics = apply_invalid_action_penalty(
                                    batch,
                                    invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                )
                                metrics.update(invalid_metrics)
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch,
                                    kl_ctrl=self.kl_ctrl_in_reward,
                                    kl_penalty=self.config.algorithm.kl_penalty,
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch[
                                    "token_level_scores"
                                ]
                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )
                            if hgpo_enabled:
                                effective_adv_estimator = AdvantageEstimator.HGPO
                            else:
                                effective_adv_estimator = (
                                    self.config.algorithm.adv_estimator
                                )
                            advantage_kwargs = {
                                "data": batch,
                                "adv_estimator": effective_adv_estimator,
                                "gamma": self.config.algorithm.gamma,
                                "lam": self.config.algorithm.lam,
                                "num_repeat": self.config.actor_rollout_ref.rollout.n,
                                "norm_adv_by_std_in_grpo": norm_adv_by_std_in_grpo,
                                "compute_mean_std_cross_steps": self.config.algorithm.get(
                                    "compute_mean_std_cross_steps", True
                                ),
                                "multi_turn": self.config.actor_rollout_ref.rollout.multi_turn.enable,
                                "use_pf_ppo": self.config.algorithm.use_pf_ppo,
                                "pf_ppo_reweight_method": self.config.algorithm.pf_ppo.reweight_method,
                                "pf_ppo_weight_pow": self.config.algorithm.pf_ppo.weight_pow,
                            }
                            if effective_adv_estimator == AdvantageEstimator.GiGPO:
                                _gigpo_sm, _gigpo_llm = (
                                    _resolve_gigpo_obs_similarity_flags(self.config)
                                )
                                advantage_kwargs.update(
                                    {
                                        "step_advantage_w": self.config.algorithm.gigpo.step_advantage_w,
                                        "gigpo_mode": self.config.algorithm.gigpo.mode,
                                        "gigpo_enable_similarity_sequence_matcher": _gigpo_sm,
                                        "gigpo_enable_similarity_llm": _gigpo_llm,
                                        "gigpo_similarity_thresh": float(
                                            OmegaConf.select(
                                                self.config,
                                                "algorithm.gigpo.similarity_thresh",
                                                default=0.95,
                                            )
                                        ),
                                    }
                                )
                            elif effective_adv_estimator == AdvantageEstimator.HGPO:
                                _hgpo_sm, _hgpo_llm = (
                                    _resolve_hgpo_obs_similarity_flags(self.config)
                                )
                                advantage_kwargs.update(
                                    {
                                        "hgpo_mode": self.config.algorithm.hgpo.mode,
                                        "hgpo_length_weight_alpha": self.config.algorithm.hgpo.length_weight_alpha,
                                        "hgpo_base_group": self.config.algorithm.hgpo.base_group,
                                        "history_length": self.config.env.history_length,
                                        "epsilon": self.config.algorithm.hgpo.epsilon,
                                        "hgpo_enable_similarity_sequence_matcher": _hgpo_sm,
                                        "hgpo_enable_similarity_llm": _hgpo_llm,
                                        "hgpo_similarity_thresh": float(
                                            OmegaConf.select(
                                                self.config,
                                                "algorithm.hgpo.similarity_thresh",
                                                default=0.95,
                                            )
                                        ),
                                        "hgpo_similarity_batch_size": int(
                                            OmegaConf.select(
                                                self.config,
                                                "algorithm.hgpo.similarity_batch_size",
                                                default=32,
                                            )
                                        ),
                                    }
                                )
                            batch = compute_advantage(**advantage_kwargs)
                    if hgpo_enabled:
                        batch = adjust_batch(self.config, batch)
                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)
                        with _timer("old_log_prob", timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = (
                                self.config.actor_rollout_ref.actor.loss_agg_mode
                            )
                            entropy_loss = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=loss_agg_mode,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy_loss": entropy_loss.detach().item()
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                actor_old_log_probs = batch.batch["old_log_probs"]
                                attention_mask = batch.batch["attention_mask"]
                                responses = batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]
                                rollout_probs = torch.exp(rollout_old_log_probs)
                                actor_probs = torch.exp(actor_old_log_probs)
                                rollout_probs_diff = torch.abs(
                                    rollout_probs - actor_probs
                                )
                                rollout_probs_diff = torch.masked_select(
                                    rollout_probs_diff, response_mask.bool()
                                )
                                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                                rollout_probs_diff_std = torch.std(rollout_probs_diff)
                                metrics.update(
                                    {
                                        "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                        "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                        "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                    }
                                )
                        if (
                            self.use_reference_policy
                            and not self.config.algorithm.use_kl_in_reward
                        ):
                            with _timer("ref", timing_raw):
                                if not self.ref_in_actor:
                                    ref_log_prob = (
                                        self.ref_policy_wg.compute_ref_log_prob(batch)
                                    )
                                else:
                                    ref_log_prob = (
                                        self.actor_rollout_wg.compute_ref_log_prob(
                                            batch
                                        )
                                    )
                                batch = batch.union(ref_log_prob)
                    if rstar_skip_reason is not None:
                        metrics.update(
                            {key: float(value) for key, value in rstar_metrics.items()}
                        )
                        metrics.update(
                            {
                                "training/global_step": self.global_steps,
                                "training/epoch": epoch,
                            }
                        )
                        metrics.update(
                            compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                        )
                        logger.log(data=metrics, step=self.global_steps)
                        print(
                            f"[rstar] Skip training step {self.global_steps}: empty batch after {rstar_skip_reason}."
                        )
                        batch_progress_bar.update(1)
                        batch_progress_bar.set_postfix(
                            {
                                "step": f"{batch_idx + 1}/{batches_per_epoch}",
                                "skip": "rstar",
                            }
                        )
                        continue
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = (
                                self.config.actor_rollout_ref.rollout.multi_turn.enable
                            )
                            batch.meta_info["training_global_step"] = int(
                                self.global_steps
                            )
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)
                        should_stop, kl_stop_metrics = self._update_kl_oscillation_stop(
                            metrics
                        )
                        metrics.update(kl_stop_metrics)
                        if should_stop:
                            print(
                                f"Early stop triggered at step {self.global_steps}: "
                                f"delta_ratio={metrics['training/kl_oscillation_delta_ratio']:.4f}, "
                                f"range_ratio={metrics['training/kl_oscillation_range_ratio']:.4f}"
                            )
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(
                                batch.batch["prompts"], skip_special_tokens=True
                            )
                            outputs = self.tokenizer.batch_decode(
                                batch.batch["responses"], skip_special_tokens=True
                            )
                            scores = (
                                batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            )
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            should_stop
                            or is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                        )
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if should_stop or is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.get("enable_mid_val", False)
                        and (
                            should_stop
                            or is_last_step
                            or self.global_steps
                            % int(self.config.trainer.get("mid_val_freq", 5))
                            == 0
                        )
                    ):
                        with _timer("mid_testing", timing_raw):
                            mid_val_metrics = self._run_mid_validation()
                        metrics.update(mid_val_metrics)
                    if should_stop or (
                        self.config.trainer.save_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                        )
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                if _bash_coding_enabled():
                    metrics.update(self._compute_bash_coding_data_metrics(batch))
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )
                logger.log(data=metrics, step=self.global_steps)
                batch_progress_bar.update(1)
                batch_progress_bar.set_postfix(
                    {"step": f"{batch_idx + 1}/{batches_per_epoch}"}
                )
                progress_bar.update(1)
                self.global_steps += 1
                if should_stop or is_last_step:
                    batch_progress_bar.close()
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
            batch_progress_bar.close()
