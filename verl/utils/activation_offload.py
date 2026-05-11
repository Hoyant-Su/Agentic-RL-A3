
from __future__ import annotations
import functools
import logging
import os
from typing import Any, Optional
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from verl.utils.fsdp_utils import FSDPModule as FSDP2

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def _get_unique_tensor_key(tensor):
    key = (tensor.untyped_storage().data_ptr() + tensor.storage_offset(), tensor.dtype)
    return key

class FSDPParameterFilter:
    def __init__(self):
        self.model_parameters_storage = set()

    def __call__(self, tensor):
        return tensor.untyped_storage().data_ptr() not in self.model_parameters_storage

    def update_model_parameters(self, model):
        new_storage = set()
        for p in model.parameters():
            new_storage.add(p.data.untyped_storage().data_ptr())
        self.model_parameters_storage = new_storage

class CpuOffloadHookWithOffloadHandler:

    def __init__(
        self,
        offload_handler: OffloadHandler,
        handler_extra_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if handler_extra_kwargs is None:
            handler_extra_kwargs = {}
        self.offload_handler: OffloadHandler = offload_handler
        self.handler_extra_kwargs: dict[str, Any] = handler_extra_kwargs
        self.inside_context = False

    def __enter__(self):
        self.inside_context = True
        torch._C._autograd._push_saved_tensors_default_hooks(
            self.on_save_for_backward, self.on_get_saved_tensor
        )

    def __exit__(self, *args: Any):
        self.inside_context = False
        torch._C._autograd._pop_saved_tensors_default_hooks()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        retrieve_identifier = self.offload_handler.tensor_push(
            tensor, **self.handler_extra_kwargs
        )
        return retrieve_identifier

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        tensor = self.offload_handler.tensor_pop(
            saved_state, **self.handler_extra_kwargs
        )
        return tensor

class OffloadHandler:

    def __init__(self) -> None:
        pass

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        raise NotImplementedError(
            "`tensor_push is not implented in OffloadHandler class. Inherit this class and implement your custom tensor_push."
        )

    def tensor_pop(self, tensor_tag: Any, **kwargs):
        raise NotImplementedError(
            "`tensor_pop is not implented in OffloadHandler class. Inherit this class and implement your custom tensor_pop."
        )

class GroupCommitFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler):
        cpu_offload_handler.on_group_commit_forward()
        ctx.cpu_offload_handler = cpu_offload_handler
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_commit_backward()
        return grad_output, None

group_prefetch_offload_commit = GroupCommitFunction.apply

class SynchronizedGroupOffloadHandler(OffloadHandler):

    def __init__(
        self, num_offload_group, tensor_need_offloading_checker=(lambda _: True)
    ) -> None:
        super().__init__()
        self.num_offload_group = num_offload_group
        self.tensor_need_offloading_checker = tensor_need_offloading_checker
        self.groupid_reset()

    def groupid_reset(self):
        self.current_group, self.tensor_count_current_group = (0, 0)
        self.torch_tensor_count = 0
        self.tensor_tag_to_state = {}

    def on_group_commit_forward(self):
        self.current_group += 1
        self.tensor_count_current_group = 0

    def on_group_commit_backward(self):
        self.current_group -= 1
        assert self.current_group >= 0

    @staticmethod
    def offload(src_tensor, pin_memory=True):
        cpu_backup = torch.empty(
            src_tensor.size(),
            dtype=src_tensor.dtype,
            layout=src_tensor.layout,
            device="cpu",
            pin_memory=pin_memory,
        )
        cpu_backup.copy_(src_tensor, non_blocking=True)
        state = (src_tensor.device, cpu_backup)
        return state

    @staticmethod
    def reload(state, non_blocking=None):
        dev, cpu_backup = state
        if non_blocking is None:
            non_blocking = cpu_backup.is_pinned()
        return cpu_backup.to(dev, non_blocking=non_blocking)

    def tensor_push(self, tensor: torch.Tensor, **kwargs):
        tensor_tag = (self.current_group, self.tensor_count_current_group)
        self.tensor_count_current_group += 1
        assert tensor_tag not in self.tensor_tag_to_state
        if (
            self.current_group < self.num_offload_group
            and self.tensor_need_offloading_checker(tensor)
        ):
            state = SynchronizedGroupOffloadHandler.offload(tensor)
            self.tensor_tag_to_state[tensor_tag] = state
        else:
            self.tensor_tag_to_state[tensor_tag] = tensor
        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        assert tensor_tag in self.tensor_tag_to_state
        state = self.tensor_tag_to_state.pop(tensor_tag)
        if isinstance(state, tuple):
            tensor = SynchronizedGroupOffloadHandler.reload(state)
        else:
            tensor = state
        return tensor

class AsyncDoubleBufferGroupOffloadHandler(SynchronizedGroupOffloadHandler):

    def __init__(
        self,
        num_offload_group,
        num_model_group,
        tensor_need_offloading_checker=(lambda t: True),
    ) -> None:
        super().__init__(
            num_offload_group=num_offload_group,
            tensor_need_offloading_checker=tensor_need_offloading_checker,
        )
        self.num_layers = num_model_group
        self.tensor_tag_to_buf = {}
        self.offloaded_group_count = 0
        self.layer_window_map = {}
        self.group_offload_mapping = {}
        constant = 0
        for i in range(self.num_offload_group):
            self.layer_window_map[i] = (
                (self.num_layers // self.num_offload_group) * (i + 1)
            ) - 1
            if i < (self.num_layers % self.num_offload_group):
                self.layer_window_map[i] += i + 1
                constant = i + 1
            else:
                self.layer_window_map[i] += constant
        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        torch_stray_tensor = isinstance(
            tensor,
            (
                torch._subclasses.fake_tensor.FakeTensor,
                torch._subclasses.functional_tensor.FunctionalTensor,
            ),
        )
        need_offload = not torch_stray_tensor
        need_offload = need_offload and self.tensor_need_offloading_checker(tensor)
        if need_offload:
            tensor_tag = (self.current_group, self.tensor_count_current_group)
            self.tensor_count_current_group += 1
            assert tensor_tag not in self.tensor_tag_to_state
            self.tensor_tag_to_state[tensor_tag] = tensor
            if self.current_group < self.num_offload_group:
                self.tensor_tag_to_buf[tensor_tag] = tensor
        else:
            tensor_tag = tensor
        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        if isinstance(tensor_tag, torch.Tensor):
            return tensor_tag
        assert tensor_tag in self.tensor_tag_to_state
        tensor = self.tensor_tag_to_state.pop(tensor_tag)
        self.tensor_tag_to_buf.pop(tensor_tag, None)
        assert not isinstance(tensor, tuple)
        return tensor

    def bulk_offload_group(self, group_to_offload):
        offload_mapping = {}
        offload_size = 0
        with torch.cuda.stream(self.d2h_stream):
            for tensor_tag, state in self.tensor_tag_to_state.items():
                group_id, _ = tensor_tag
                if group_id == group_to_offload:
                    assert not isinstance(state, tuple)
                    key = _get_unique_tensor_key(state)
                    if key not in offload_mapping:
                        offload_mapping[key] = state
                    self.tensor_tag_to_state[tensor_tag] = (key, state.shape)
            for key, tensor in offload_mapping.items():
                state = SynchronizedGroupOffloadHandler.offload(tensor)
                offload_size += tensor.numel() * tensor.element_size()
                offload_mapping[key] = state
            self.group_offload_mapping[group_to_offload] = offload_mapping

    def synchronize_on_group_commit_forward(self, current_group):
        if current_group == 0:
            self.d2h_stream.wait_stream(torch.cuda.current_stream())
            self.bulk_offload_group(current_group)
        if self.layer_window_map[self.offloaded_group_count] == current_group:
            self.d2h_stream.wait_stream(torch.cuda.current_stream())
            torch.cuda.current_stream().wait_stream(self.d2h_stream)
            for tensor_tag, _ in self.tensor_tag_to_buf.items():
                if tensor_tag[0] == self.offloaded_group_count:
                    self.tensor_tag_to_buf[tensor_tag] = None
            if self.offloaded_group_count < (self.num_offload_group - 1):
                self.bulk_offload_group(self.offloaded_group_count + 1)
            self.offloaded_group_count += 1

    def on_group_commit_forward(self):
        self.synchronize_on_group_commit_forward(self.current_group)
        super().on_group_commit_forward()

    @torch.no_grad
    def bulk_reload_group(self, group_to_reload):
        assert group_to_reload < self.num_offload_group
        with torch.cuda.stream(self.h2d_stream):
            offload_mapping = self.group_offload_mapping.pop(group_to_reload)
            assert offload_mapping is not None
            for key, state in offload_mapping.items():
                offload_mapping[key] = SynchronizedGroupOffloadHandler.reload(state)
            for tensor_label, state in self.tensor_tag_to_state.items():
                group_id, _ = tensor_label
                if group_id == group_to_reload and not isinstance(state, torch.Tensor):
                    assert isinstance(state, tuple), f"{group_id} {state}"
                    key, shape = state
                    recovered_tensor = offload_mapping[key].view(shape)
                    self.tensor_tag_to_state[tensor_label] = recovered_tensor

    def on_group_commit_backward(self):
        self.current_group -= 1
        assert self.current_group >= 0
        if self.layer_window_map[self.offloaded_group_count - 1] == self.current_group:
            self.h2d_stream.wait_stream(torch.cuda.current_stream())
            torch.cuda.current_stream().wait_stream(self.h2d_stream)
            self.bulk_reload_group(self.offloaded_group_count - 1)
            self.offloaded_group_count -= 1 if self.offloaded_group_count > 1 else 0
        if self.current_group == 0:
            torch.cuda.current_stream().wait_stream(self.h2d_stream)
            self.offloaded_group_count = 0

def get_activation_offload_context(
    num_layers: int = 1,
    model_layers: int = 1,
    tensor_need_offloading_checker=(lambda t: True),
):
    cpu_offload_handler = AsyncDoubleBufferGroupOffloadHandler(
        num_offload_group=num_layers,
        num_model_group=model_layers,
        tensor_need_offloading_checker=tensor_need_offloading_checker,
    )

    def group_prefetch_offload_commit_async(tensor):
        return group_prefetch_offload_commit(tensor, cpu_offload_handler)

    return (
        CpuOffloadHookWithOffloadHandler(offload_handler=cpu_offload_handler),
        group_prefetch_offload_commit_async,
    )

class ActivationHandler:
    def __init__(self, offload_ctx, sync_func, tensor_filter, enable_ckpt):
        self._offload_ctx = offload_ctx
        self._sync_func = sync_func
        self._enable_ckpt = enable_ckpt
        self._tensor_filter = tensor_filter
        if enable_ckpt:
            self.checkpoint_fn = functools.partial(
                torch.utils.checkpoint.checkpoint,
                use_reentrant=True,
            )

    def pre_forward(self, module):
        if module.training:
            self._offload_ctx.__enter__()
            self._tensor_filter.update_model_parameters(module)

    def post_forward(self, module):
        if module.training:
            self._offload_ctx.__exit__(None, None, None)

    def _pack_kwargs(self, *args, **kwargs):
        kwarg_keys = []
        flat_args = list(args)
        for k, v in kwargs.items():
            kwarg_keys.append(k)
            flat_args.append(v)
        return tuple(flat_args), tuple(kwarg_keys)

    def _unpack_kwargs(self, flat_args, kwarg_keys):
        assert len(kwarg_keys) <= len(flat_args), (
            f"too many keys {len(kwarg_keys)} vs. {len(flat_args)}"
        )
        if len(kwarg_keys) == 0:
            return flat_args, {}
        args = flat_args[: -len(kwarg_keys)]
        kwargs = dict(zip(kwarg_keys, flat_args[-len(kwarg_keys) :]))
        return args, kwargs

    def _ckpt_forward(self, forward_method, *args, **kwargs):
        flat_args, kwarg_keys = self._pack_kwargs(*args, **kwargs)

        def my_function(*inputs):
            nonlocal forward_method, kwarg_keys
            unpacked_args, unpacked_kwargs = self._unpack_kwargs(inputs, kwarg_keys)
            return forward_method(*unpacked_args, **unpacked_kwargs)

        return self.checkpoint_fn(
            my_function,
            *flat_args,
        )

    def forward(self, module, forward_method, *args, **kwargs):
        if not module.training:
            return forward_method(*args, **kwargs)
        if not self._enable_ckpt:
            ret = forward_method(*args, **kwargs)
        else:
            ret = self._ckpt_forward(forward_method, *args, **kwargs)
        binded_tensor = ret
        if isinstance(ret, tuple):
            binded_tensor = ret[0]
        binded_tensor = self._sync_func(binded_tensor)
        final_ret = binded_tensor
        if isinstance(ret, tuple):
            final_ret = (final_ret,) + ret[1:]
        return final_ret

    def wrap_module_forward_method(self, module):
        orig_method = module.forward
        handler = self

        @functools.wraps(orig_method)
        def wrapped_method(model_self, *args, **kwargs):
            nonlocal handler
            handler.pre_forward(model_self)
            out = handler.forward(model_self, orig_method, *args, **kwargs)
            handler.post_forward(model_self)
            return out

        module.forward = wrapped_method.__get__(module, type(module))

def enable_activation_offloading(model, strategy, enable_ckpt=False):
    assert strategy == "fsdp" or strategy == "fsdp2", (
        "activation offloading only supports fsdp strategy"
    )
    layers = []

    def get_layers(module):
        for name, child in module.named_children():
            if not isinstance(child, (FSDP, FSDP2)):
                get_layers(child)
            else:
                wrapped_module = child
                if isinstance(child, FSDP):
                    wrapped_module = child._fsdp_wrapped_module
                if not isinstance(wrapped_module, torch.nn.Embedding):
                    layers.append(child)

    get_layers(model)
    if len(layers) < 3:
        logger.warning(
            f"Find only {len(layers)} fsdp layers, not neccessary to enable async activation offloading"
        )
        return
    tensor_filter = FSDPParameterFilter()
    context, sync_func = get_activation_offload_context(
        len(layers) - 1, len(layers), tensor_filter
    )
    if enable_ckpt:
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing_disable"):
                module.gradient_checkpointing_disable()
    handler = ActivationHandler(context, sync_func, tensor_filter, enable_ckpt)
    for layer in layers:
        module = layer
        if isinstance(layer, FSDP):
            module = module._fsdp_wrapped_module
        handler.wrap_module_forward_method(module)
