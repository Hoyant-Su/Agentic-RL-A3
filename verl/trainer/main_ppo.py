
import json
import os
import numpy as np
import torch
import hydra
import ray
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)

def _bash_coding_enabled() -> bool:
    return str(os.environ.get("BASH_CODING_ENABLE", "0")).strip() == "1"

class CoupledReward:
    def __init__(
        self, base_reward_fn, actor_rollout_wg, coupling_reward_coef: float, tokenizer
    ):
        self.base_reward_fn = base_reward_fn
        self.actor_rollout_wg = actor_rollout_wg
        self.coupling_reward_coef = float(coupling_reward_coef)
        self.tokenizer = tokenizer
        self._obs_marker_ids = tokenizer.encode(
            "Current observation:", add_special_tokens=False
        )
        self._obj_marker_ids = tokenizer.encode("Objective:", add_special_tokens=False)

    @staticmethod
    def _mean_logp(
        old_log_probs: torch.Tensor, response_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = response_mask.to(dtype=old_log_probs.dtype)
        denom = torch.clamp(mask.sum(dim=-1), min=1.0)
        return (old_log_probs * mask).sum(dim=-1) / denom

    @staticmethod
    def _find_subseq(hay: list[int], needle: list[int]) -> int:
        for i in range(0, len(hay) - len(needle) + 1):
            if hay[i : i + len(needle)] == needle:
                return i
        return -1

    def _corrupt_observation_batch_for_logp(self, batch: DataProto) -> DataProto:
        responses = batch.batch["responses"]
        input_ids = batch.batch["input_ids"]
        attention_mask = batch.batch["attention_mask"]
        position_ids = batch.batch["position_ids"]
        response_len = responses.size(1)
        prompt_len = input_ids.size(1) - response_len
        neg_input_ids = input_ids.clone()
        prompt_ids = neg_input_ids[:, :prompt_len]
        for i in range(prompt_ids.size(0)):
            ids = prompt_ids[i].tolist()
            obs_k = self._find_subseq(ids, self._obs_marker_ids)
            obj_k = self._find_subseq(ids, self._obj_marker_ids)
            if obs_k < 0 or obj_k < 0:
                continue
            obs_start = obs_k + len(self._obs_marker_ids)
            obs_end = obj_k
            if obs_end <= obs_start:
                continue
            span = neg_input_ids[i, obs_start:obs_end]
            perm = torch.randperm(span.numel(), device=span.device)
            neg_input_ids[i, obs_start:obs_end] = span[perm]
        return DataProto.from_dict(
            tensors={
                "responses": responses,
                "input_ids": neg_input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            meta_info=batch.meta_info,
        )

    def __call__(self, data: DataProto, return_dict: bool = False):
        base = self.base_reward_fn(data, return_dict=True)
        reward_tensor = base["reward_tensor"]
        if self.coupling_reward_coef <= 0:
            return base if return_dict else reward_tensor
        pos_lp = self.actor_rollout_wg.compute_log_prob(data).batch["old_log_probs"]
        neg_data = self._corrupt_observation_batch_for_logp(data)
        neg_lp = self.actor_rollout_wg.compute_log_prob(neg_data).batch["old_log_probs"]
        response_mask = data.batch.get("response_mask", None)
        if response_mask is None:
            response_len = data.batch["responses"].size(1)
            response_mask = data.batch["attention_mask"][:, -response_len:]
        coupling = self.coupling_reward_coef * (
            self._mean_logp(pos_lp, response_mask)
            - self._mean_logp(neg_lp, response_mask)
        )
        valids = data.non_tensor_batch.get("is_action_valid", None)
        if valids is not None:
            if isinstance(valids, np.ndarray):
                valids = valids.tolist()
            if isinstance(valids, (bool, np.bool_)) or not isinstance(
                valids, (list, tuple)
            ):
                valids = [valids] * reward_tensor.size(0)
            valid_mask = torch.tensor(
                [1.0 if bool(x) else 0.0 for x in valids],
                device=reward_tensor.device,
                dtype=reward_tensor.dtype,
            )
            coupling = coupling.to(dtype=reward_tensor.dtype) * valid_mask
        last = torch.clamp(response_mask.to(dtype=torch.int64).sum(dim=-1) - 1, min=0)
        reward_tensor[
            torch.arange(reward_tensor.size(0), device=reward_tensor.device), last
        ] += coupling.to(reward_tensor.dtype)
        base["reward_tensor"] = reward_tensor
        base["reward_extra_info"]["coupling_reward_coef"] = (
            coupling.detach().cpu().tolist()
        )
        return base if return_dict else reward_tensor

def run_ppo(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create(
            {**ray_init_kwargs, "runtime_env": runtime_env}
        )
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        bash_coding_enabled = _bash_coding_enabled()
        if bash_coding_enabled:
            os.environ["BASH_CODING_ENV_CONFIG_JSON"] = json.dumps(
                OmegaConf.to_container(config.env, resolve=True),
                ensure_ascii=False,
            )
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        if bash_coding_enabled:
            from main_entry.cli_agent_bash_coding.env_manager import make_envs

            envs, val_envs = make_envs(config)
        else:
            from agent_system.environments import make_envs

            envs, val_envs = make_envs(config)
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(
            local_path, trust_remote_code=trust_remote_code, use_fast=True
        )
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError(
                        "PPO LoRA is not supported before vllm 0.7.3"
                    )
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import (
                ActorRolloutRefWorker,
                AsyncActorRolloutRefWorker,
                CriticWorker,
            )

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import (
                ActorRolloutRefWorker,
                CriticWorker,
            )

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id
        if (
            config.algorithm.use_kl_in_reward
            or config.actor_rollout_ref.actor.use_kl_loss
        ):
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id
        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode" and not bash_coding_enabled:
            from agent_system.reward_manager import EpisodeRewardManager

            reward_manager_cls = EpisodeRewardManager
        elif reward_manager_name == "episode" and bash_coding_enabled:
            from main_entry.cli_agent_bash_coding.reward import (
                SegmentedEpisodeRewardManager,
            )

            reward_manager_cls = SegmentedEpisodeRewardManager
        else:
            raise NotImplementedError
        enable_commit = (
            str(OmegaConf.select(config, "env.bash_coding_harness", default=""))
            .strip()
            .lower()
            == "commit_if_better"
        )
        reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=0,
            normalize_by_length=False,
            enable_commit=enable_commit,
        )
        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=1,
            normalize_by_length=False,
            enable_commit=enable_commit,
        )
        if (
            bash_coding_enabled
            and bool(OmegaConf.select(config, "reward.rstar_enable", default=0))
            and bool(
                OmegaConf.select(
                    config, "reward_model.launch_reward_fn_async", default=False
                )
            )
        ):
            raise NotImplementedError(
                "rStar-style resampling does not support async reward functions."
            )
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )
        assert config.actor_rollout_ref.rollout.n == 1, (
            "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"
        )
        if bash_coding_enabled:
            from main_entry.cli_agent_bash_coding.rollout import (
                BashCodingTrajectoryCollector,
            )

            traj_collector = BashCodingTrajectoryCollector(
                config=config, tokenizer=tokenizer, processor=processor
            )
        else:
            from agent_system.multi_turn_rollout import TrajectoryCollector

            traj_collector = TrajectoryCollector(
                config=config, tokenizer=tokenizer, processor=processor
            )
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        coupling_coef = float(
            OmegaConf.select(config, "algorithm.coupling_reward_coef", default=0.0)
        )
        if bash_coding_enabled and coupling_coef > 0:
            trainer.reward_fn = CoupledReward(
                trainer.reward_fn, trainer.actor_rollout_wg, coupling_coef, tokenizer
            )
        trainer.fit()

def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    from torch.utils.data import Dataset
    from verl.utils.dataset.rl_dataset import RLHFDataset

    if (
        "custom_cls" in data_config
        and data_config.custom_cls.get("path", None) is not None
    ):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(
            data_config.custom_cls.path, data_config.custom_cls.name
        )
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )
    return dataset

def create_rl_sampler(data_config, dataset):
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(
            data_source=dataset, generator=train_dataloader_generator
        )
    else:
        sampler = SequentialSampler(data_source=dataset)
    return sampler

if __name__ == "__main__":
    main()
