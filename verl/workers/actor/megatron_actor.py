
import copy
import itertools
import logging
import os
from functools import partial
from typing import Dict, Iterable
import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf
from torch import nn
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.debug.profile import Profiler
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.tensor_parallel import (
    vocab_parallel_entropy,
    vocab_parallel_log_probs_from_logits,
)
from verl.utils.megatron_utils import get_model_config
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import broadcast_dict_tensor
from verl.workers.actor import BasePPOActor

__all__ = ["MegatronPPOActor"]
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class MegatronPPOActor(BasePPOActor):
    def __init__(
        self,
        config,
        model_config,
        hf_config,
        tf_config,
        actor_module: nn.ModuleList,
        actor_optimizer: DistributedOptimizer,
    ):
        super().__init__(config)
        self._validate_config(config)
        self.model_config = model_config
        self.hf_config = hf_config
        self.tf_config = tf_config
        self.actor_module = actor_module
        self.actor_optimizer: DistributedOptimizer = actor_optimizer
        self.prof = Profiler(self.config.profile)
        self.optimizer_step_args = OmegaConf.create(
            {
                "skip_grad": None,
                "overlap_dp_param_comm": False,
                "overlap_dp_grad_comm": False,
                "gradient_accumulation_steps": 1,
                "sequence_parallel": self.tf_config.sequence_parallel,
                "DDP_impl": "local",
                "layernorm_allreduce_bucket_threshold": 0,
                "pipeline_model_parallel_split_rank": None,
                "reduce_grads_use_alltoall": False,
            }
        )
        config = get_model_config(self.actor_module[0])
        print(config)
        config.finalize_model_grads_func = finalize_model_grads

    def _validate_config(self, config) -> None:
        assert config.get("ulysses_sequence_parallel_size", 1) == 1
        if config.get("shuffle", False):
            assert config.data_loader_seed is not None, (
                "If shuffle dataloader, seed must be manually set"
            )
        if config.megatron.tensor_model_parallel_size == 1:
            print("[Warining] Because actor tp size == 1, set sp to False")
            config.megatron.sequence_parallel = False
        self.config = config

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def compute_log_prob(
        self, data: DataProto, calculate_entropy=False
    ) -> torch.Tensor:
        data.batch = data.batch.contiguous()
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)
        micro_batch_size = data.meta_info.get("micro_batch_size", None)
        max_token_len = data.meta_info.get("max_token_len", None)
        assert micro_batch_size is not None, (
            "micro batch size is needed for forward compute"
        )
        if use_dynamic_bsz:
            assert max_token_len is not None, (
                "max_token_len must be set when use_dynamic_bsz is True"
            )
            max_token_len = max_token_len * self.config.megatron.context_parallel_size

        def compute_logprobs_fn(output, data, use_dynamic_bsz=False, indices=None):
            response = data["responses"]
            response_length = response.size(1)
            log_probs = output["log_probs"][:, -response_length - 1 : -1].contiguous()
            return {"log_probs": log_probs}

        recompute_old_log_prob = self.config.get("recompute_old_log_prob", True)
        entropys = torch.Tensor()
        if recompute_old_log_prob:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            batch = data.select(batch_keys=select_keys).batch
            input_ids = batch["input_ids"]
            batch_size = input_ids.size(0)
            response = batch["responses"]
            response_length = response.size(1)
            with torch.no_grad():
                output = self.forward_backward_batch(
                    data,
                    forward_only=True,
                    post_process_fn=compute_logprobs_fn,
                    calculate_entropy=calculate_entropy,
                    use_dynamic_bsz=use_dynamic_bsz,
                    micro_batch_size=micro_batch_size,
                    max_token_len=max_token_len,
                )
                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    if calculate_entropy:
                        log_probs = [o[0]["log_probs"] for o in output["output"]]
                    else:
                        log_probs = [o["log_probs"] for o in output["output"]]
                    log_probs = torch.cat(log_probs, dim=0).to(torch.float32)
                    if use_dynamic_bsz:
                        indices = output["indices"]
                        indices = list(itertools.chain.from_iterable(indices))
                        assert len(indices) == log_probs.size(0), (
                            f"{len(indices)} vs. {log_probs.size()}"
                        )
                        revert_indices = torch.tensor(
                            get_reverse_idx(indices), dtype=torch.long
                        )
                        log_probs = log_probs[revert_indices]
                else:
                    log_probs = torch.empty(
                        size=(batch_size, response_length),
                        dtype=torch.float32,
                        device=input_ids.device,
                    )
                torch.distributed.broadcast(
                    tensor=log_probs,
                    src=mpu.get_pipeline_model_parallel_last_rank(),
                    group=mpu.get_pipeline_model_parallel_group(),
                    async_op=False,
                )
                if calculate_entropy:
                    if mpu.is_pipeline_last_stage(ignore_virtual=True):
                        entropys = torch.cat([o[1] for o in output["output"]], dim=0)
                        entropys = entropys.to(torch.float32)
                        if use_dynamic_bsz:
                            indices = output["indices"]
                            indices = list(itertools.chain.from_iterable(indices))
                            assert len(indices) == entropys.size(0), (
                                f"{len(indices)} vs. {entropys.size()}"
                            )
                            revert_indices = torch.tensor(
                                get_reverse_idx(indices), dtype=torch.long
                            )
                            entropys = entropys[revert_indices]
                    else:
                        entropys = torch.empty(
                            size=(batch_size, response_length),
                            dtype=torch.float32,
                            device=input_ids.device,
                        )
                    torch.distributed.broadcast(
                        tensor=entropys,
                        src=mpu.get_pipeline_model_parallel_last_rank(),
                        group=mpu.get_pipeline_model_parallel_group(),
                        async_op=False,
                    )
        torch.cuda.empty_cache()
        return log_probs, entropys

    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        data = data.select(batch_keys=select_keys)
        return data.make_iterator(
            mini_batch_size=self.config.ppo_mini_batch_size,
            epochs=self.config.ppo_epochs,
            seed=self.config.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.shuffle},
        )

    def forward_backward_batch(
        self,
        data: DataProto,
        forward_only=False,
        post_process_fn=None,
        calculate_entropy=False,
        use_dynamic_bsz=False,
        micro_batch_size=None,
        max_token_len=None,
        mini_batch_size=None,
    ):
        mini_batch = data
        broadcast_dict_tensor(
            mini_batch.batch,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        mini_batch.batch["attention_mask"] = mini_batch.batch["attention_mask"].to(bool)
        indices = None
        if use_dynamic_bsz:
            assert max_token_len is not None, (
                "max_token_len must be set when use_dynamic_bsz is True"
            )
            vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            if vpp_size is not None and vpp_size > 1:
                microbatch_group_size_per_vp_stage = (
                    self.tf_config.microbatch_group_size_per_vp_stage
                )
                micro_batches, indices = rearrange_micro_batches(
                    batch=mini_batch.batch,
                    num_batches_divided_by=microbatch_group_size_per_vp_stage,
                    max_token_len=max_token_len,
                )
                assert (
                    len(micro_batches)
                    % self.tf_config.microbatch_group_size_per_vp_stage
                    == 0
                ), (
                    f"micro_batches {micro_batches} must be divisible by microbatch_group_size_per_vp_stage {microbatch_group_size_per_vp_stage} for megatron backend"
                )
            else:
                micro_batches, indices = rearrange_micro_batches(
                    batch=mini_batch.batch, max_token_len=max_token_len
                )
            total_seqlen = max_token_len
        else:
            assert micro_batch_size is not None, (
                "micro_batch_size is needed to be passed in when not using dynamic batch size"
            )
            micro_batches = mini_batch.batch.split(micro_batch_size)
            seq_len = micro_batches[0]["input_ids"].shape[1]
            total_seqlen = micro_batch_size * seq_len
        n_micro_batch = len(micro_batches)
        forward_backward_func = get_forward_backward_func()

        def loss_func(output, data, meta_info):
            device = output["log_probs"].device
            metrics = {}
            if forward_only:
                if post_process_fn is None:
                    pass
                else:
                    stats = post_process_fn(output, data)
                    metrics.update(stats)
                if not calculate_entropy:
                    return torch.tensor(1.0, device=device), metrics
            responses = data["responses"]
            response_length = responses.size(1)
            attention_mask = data["attention_mask"]
            response_mask = attention_mask[:, -response_length:]
            loss_agg_mode = self.config.loss_agg_mode
            log_prob = output["log_probs"][:, -response_length - 1 : -1].contiguous()
            ret_entropy = None
            if not forward_only:
                old_log_prob = data["old_log_probs"]
                advantages = data["advantages"]
                clip_ratio = meta_info["clip_ratio"]
                clip_ratio_low = (
                    self.config.clip_ratio_low
                    if self.config.clip_ratio_low is not None
                    else clip_ratio
                )
                clip_ratio_high = (
                    self.config.clip_ratio_high
                    if self.config.clip_ratio_high is not None
                    else clip_ratio
                )
                clip_ratio_c = meta_info["clip_ratio_c"]
                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    response_mask=response_mask,
                    cliprange=clip_ratio,
                    cliprange_low=clip_ratio_low,
                    cliprange_high=clip_ratio_high,
                    clip_ratio_c=clip_ratio_c,
                    loss_agg_mode=loss_agg_mode,
                )
                policy_loss = pg_loss
            if calculate_entropy:
                entropy = output["entropy"][:, -response_length - 1 : -1].contiguous()
                if not forward_only:
                    entropy_loss = agg_loss(
                        loss_mat=entropy,
                        loss_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                    )
                    entropy_coeff = meta_info["entropy_coeff"]
                    policy_loss = pg_loss - entropy_coeff * entropy_loss
                else:
                    ret_entropy = entropy
            stats = {}
            if forward_only:
                policy_loss = torch.tensor(1.0, device=device)
            else:
                if self.config.use_kl_loss:
                    ref_log_prob = data["ref_log_prob"]
                    kld = kl_penalty(
                        logprob=log_prob,
                        ref_logprob=ref_log_prob,
                        kl_penalty=self.config.kl_loss_type,
                    )
                    kl_loss = agg_loss(
                        loss_mat=kld,
                        loss_mask=response_mask,
                        loss_agg_mode=self.config.loss_agg_mode,
                    )
                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics["actor/kl_loss"] = kl_loss.detach().item()
                    metrics["actor/kl_coef"] = self.config.kl_loss_coef
                stats.update(
                    {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                )
            append_to_dict(metrics, stats)
            return policy_loss, [metrics, ret_entropy]

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            position_ids = batch["position_ids"]
            responses = batch["responses"]
            response_length = responses.size(1)
            label = copy.deepcopy(position_ids)
            label[:, -response_length - 1 : -1] = responses
            label_mask = copy.deepcopy(attention_mask)
            label_mask[:, : -response_length - 1] = False
            label_mask[:, -1] = False

            def logits_processor(logits, label, label_mask):
                assert logits.shape[:2] == label.shape[:2]
                assert label.shape == label_mask.shape
                ret = {}
                if calculate_entropy:
                    entropy = vocab_parallel_entropy(logits)
                    ret["entropy"] = entropy
                log_probs = vocab_parallel_log_probs_from_logits(logits, label)
                log_probs = log_probs.masked_fill(~label_mask, 0.0)
                ret["log_probs"] = log_probs
                return ret

            logits_processor_args = {"label": label, "label_mask": label_mask}
            from verl.models.mcore import get_mcore_forward_fn

            forward_fn = get_mcore_forward_fn(self.hf_config)
            output = forward_fn(
                model,
                input_ids,
                attention_mask,
                position_ids,
                sequence_parallel=self.tf_config.sequence_parallel,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
            )
            if forward_only:
                meta_info = None
            else:
                clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                meta_info = {
                    "clip_ratio": self.config.clip_ratio,
                    "entropy_coeff": self.config.entropy_coeff,
                    "clip_ratio_c": clip_ratio_c,
                }
            return output, partial(loss_func, data=batch, meta_info=meta_info)

        batch_generator = make_batch_generator(
            micro_batches, vpp_size=len(self.actor_module)
        )
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,
                micro_batch_size=1,
                forward_only=forward_only,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,
                micro_batch_size=1,
                forward_only=forward_only,
            )
        losses_reduced = {"output": losses_reduced}
        if use_dynamic_bsz:
            losses_reduced["indices"] = indices
        return losses_reduced

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def update_policy(self, dataloader: Iterable[DataProto]) -> Dict:
        metrics = {}
        self.prof.start()
        for data in dataloader:
            self.actor_optimizer.zero_grad()
            for chunk in self.actor_module:
                chunk.zero_grad_buffer()
            calculate_entropy = self.config.entropy_coeff != 0
            if data.meta_info.get("micro_batch_size", None) is not None:
                micro_batch_size = data.meta_info["micro_batch_size"]
            else:
                micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
            max_token_len = None
            if self.config.use_dynamic_bsz:
                max_token_len = (
                    self.config.ppo_max_token_len_per_gpu
                    * self.config.megatron.context_parallel_size
                )
            metric_micro_batch = self.forward_backward_batch(
                data,
                calculate_entropy=calculate_entropy,
                use_dynamic_bsz=self.config.use_dynamic_bsz,
                micro_batch_size=micro_batch_size,
                max_token_len=max_token_len,
                mini_batch_size=self.config.ppo_mini_batch_size,
            )
            metric_micro_batch = metric_micro_batch["output"]
            for metric in metric_micro_batch:
                append_to_dict(metrics, metric[0])
            update_successful, grad_norm, num_zeros_in_grad = (
                self.actor_optimizer.step()
            )
            learning_rate = self.actor_optimizer.param_groups[-1]["lr"]
            data = {"actor/grad_norm": grad_norm, "actor/lr": learning_rate}
            append_to_dict(metrics, data)
            if update_successful:
                pass
            else:
                raise NotImplementedError
            self.prof.step()
        self.prof.stop_and_save()
        self.prof.stop_trace()
        torch.cuda.empty_cache()
        return metrics
