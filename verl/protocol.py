
import contextlib
import copy
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import ray
import tensordict
import torch
import torch.distributed
from packaging import version
from tensordict import TensorDict
from torch.utils.data import DataLoader
from verl.utils.device import get_torch_device
from verl.utils.py_functional import union_two_dict
from verl.utils.torch_functional import allgather_dict_tensors

__all__ = ["DataProto", "union_tensor_dict"]
with contextlib.suppress(Exception):
    tensordict.set_lazy_legacy(False).set()

class _DataProtoConfigMeta(type):
    _config = {}
    auto_padding_key = "_verl_auto_padding"

    @property
    def auto_padding(cls):
        enabled_by_env = os.getenv("VERL_AUTO_PADDING", "FALSE").upper() in [
            "TRUE",
            "1",
        ]
        return enabled_by_env or cls._config.get(cls.auto_padding_key, False)

    @auto_padding.setter
    def auto_padding(cls, enabled: bool):
        assert isinstance(enabled, bool), (
            f"enabled must be a boolean, got {enabled} as {type(enabled)}"
        )
        cls._config[cls.auto_padding_key] = enabled

class DataProtoConfig(metaclass=_DataProtoConfigMeta):
    pass

_padding_size_key = "_padding_size_key_x123d"

def pad_dataproto_to_divisor(data: "DataProto", size_divisor: int):
    assert isinstance(data, DataProto), "data must be a DataProto"
    if len(data) % size_divisor != 0:
        pad_size = size_divisor - len(data) % size_divisor
        padding_protos = []
        remaining_pad = pad_size
        while remaining_pad > 0:
            take_size = min(remaining_pad, len(data))
            padding_protos.append(data[:take_size])
            remaining_pad -= take_size
        data_padded = DataProto.concat([data] + padding_protos)
    else:
        if len(data) == 0:
            logging.warning("padding a DataProto with no item, no changed made")
        pad_size = 0
        data_padded = data
    return data_padded, pad_size

def unpad_dataproto(data: "DataProto", pad_size):
    if pad_size != 0:
        data = data[:-pad_size]
    return data

def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    assert tensor_dict1.batch_size == tensor_dict2.batch_size, (
        f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"
    )
    for key in tensor_dict2.keys():
        if key not in tensor_dict1.keys():
            tensor_dict1[key] = tensor_dict2[key]
        else:
            assert tensor_dict1[key].equal(tensor_dict2[key]), (
                f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
            )
    return tensor_dict1

def union_numpy_dict(
    tensor_dict1: dict[str, np.ndarray], tensor_dict2: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    for key, val in tensor_dict2.items():
        if key in tensor_dict1:
            assert isinstance(tensor_dict2[key], np.ndarray)
            assert isinstance(tensor_dict1[key], np.ndarray)
            assert pd.DataFrame(tensor_dict2[key]).equals(
                pd.DataFrame(tensor_dict1[key])
            ), f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
        tensor_dict1[key] = val
    return tensor_dict1

def list_of_dict_to_dict_of_list(list_of_dict: list[dict]):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()
    output = {key: [] for key in keys}
    for data in list_of_dict:
        for key, item in data.items():
            assert key in output
            output[key].append(item)
    return output

def fold_batch_dim(data: "DataProto", new_batch_size):
    batch_size = data.batch.batch_size[0]
    assert batch_size % new_batch_size == 0
    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch
    tensor = tensor.view(new_batch_size, -1)
    tensor.auto_batch_size_(batch_dims=1)
    for key, val in non_tensor.items():
        non_tensor[key] = np.reshape(val, newshape=(new_batch_size, -1, *val.shape[1:]))
    return type(data)(
        batch=tensor, non_tensor_batch=non_tensor, meta_info=data.meta_info
    )

def unfold_batch_dim(data: "DataProto", batch_dims=2):
    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch
    tensor.auto_batch_size_(batch_dims=batch_dims)
    tensor = tensor.view(-1)
    batch_size = tensor.batch_size[0]
    non_tensor_new = {}
    for key, val in non_tensor.items():
        non_tensor_new[key] = np.reshape(
            val, newshape=(batch_size, *val.shape[batch_dims:])
        )
    return type(data)(
        batch=tensor, non_tensor_batch=non_tensor_new, meta_info=data.meta_info
    )

def collate_fn(x: list["DataProtoItem"]):
    batch = []
    non_tensor_batch = []
    for data in x:
        batch.append(data.batch)
        non_tensor_batch.append(data.non_tensor_batch)
    batch = torch.stack(batch).contiguous()
    non_tensor_batch = list_of_dict_to_dict_of_list(non_tensor_batch)
    for key, val in non_tensor_batch.items():
        non_tensor_batch[key] = np.array(val, dtype=object)
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

@dataclass
class DataProtoItem:
    batch: TensorDict = None
    non_tensor_batch: Dict = field(default_factory=dict)
    meta_info: Dict = field(default_factory=dict)

@dataclass
class DataProto:

    batch: TensorDict = None
    non_tensor_batch: Dict = field(default_factory=dict)
    meta_info: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.check_consistency()

    def __len__(self):
        if self.batch is not None:
            return self.batch.batch_size[0]
        elif self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
            random_key = list(self.non_tensor_batch.keys())[0]
            return self.non_tensor_batch[random_key].shape[0]
        else:
            return 0

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)
        elif isinstance(item, (list, np.ndarray, torch.Tensor)):
            return self.select_idxs(item)
        elif isinstance(item, (int, np.integer)):
            tensor_data = self.batch[item] if self.batch is not None else None
            non_tensor_data = {
                key: val[item] for key, val in self.non_tensor_batch.items()
            }
            return DataProtoItem(
                batch=tensor_data,
                non_tensor_batch=non_tensor_data,
                meta_info=self.meta_info,
            )
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported")

    def __getstate__(self):
        import io

        buffer = io.BytesIO()
        if (
            version.parse(tensordict.__version__) >= version.parse("0.5.0")
            and self.batch is not None
        ):
            self.batch = self.batch.contiguous()
            self.batch = self.batch.consolidate()
        torch.save(self.batch, buffer)
        buffer_bytes = buffer.getvalue()
        return buffer_bytes, self.non_tensor_batch, self.meta_info

    def __setstate__(self, data):
        import io

        batch_deserialized_bytes, non_tensor_batch, meta_info = data
        batch_deserialized = io.BytesIO(initial_bytes=batch_deserialized_bytes)
        batch = torch.load(
            batch_deserialized,
            weights_only=False,
            map_location="cpu" if not get_torch_device().is_available() else None,
        )
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    def save_to_disk(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(filepath) -> "DataProto":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            return data

    def print_size(self, prefix=""):
        size_of_tensordict = 0
        if self.batch is None:
            for key, tensor in self.batch.items():
                size_of_tensordict += tensor.element_size() * tensor.numel()
        size_of_numpy_array = 0
        for key, numpy_array in self.non_tensor_batch.items():
            size_of_numpy_array += numpy_array.nbytes
        size_of_numpy_array /= 1024**3
        size_of_tensordict /= 1024**3
        message = f"Size of tensordict: {size_of_tensordict} GB, size of non_tensor_batch: {size_of_numpy_array} GB"
        if prefix:
            message = f"{prefix}, " + message
        print(message)

    def check_consistency(self):
        if self.batch is not None:
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1"
        if self.non_tensor_batch is not None:
            for key, val in self.non_tensor_batch.items():
                assert isinstance(val, np.ndarray)
        if (
            self.batch is not None
            and self.non_tensor_batch is not None
            and len(self.non_tensor_batch) != 0
        ):
            assert len(self.batch.batch_size) == 1, (
                "only support num_batch_dims=1 when non_tensor_batch is not empty."
            )
            batch_size = self.batch.batch_size[0]
            for key, val in self.non_tensor_batch.items():
                assert isinstance(val, np.ndarray), (
                    f"data in the non_tensor_batch must be a numpy.array with dtype=object, but for {key=}, got {type(val)=}"
                )
                assert val.shape[0] == batch_size, (
                    f"key {key} length {len(val)} is not equal to batch size {batch_size}"
                )

    @classmethod
    def from_single_dict(
        cls,
        data: Dict[str, Union[torch.Tensor, np.ndarray]],
        meta_info=None,
        auto_padding=False,
    ):
        tensors = {}
        non_tensors = {}
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key] = val
            elif isinstance(val, np.ndarray):
                non_tensors[key] = val
            else:
                raise ValueError(f"Unsupported type in data {type(val)}")
        return cls.from_dict(
            tensors=tensors,
            non_tensors=non_tensors,
            meta_info=meta_info,
            auto_padding=auto_padding,
        )

    @classmethod
    def from_dict(
        cls,
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        non_tensors=None,
        meta_info=None,
        num_batch_dims=1,
        auto_padding=False,
    ):
        assert num_batch_dims > 0, "num_batch_dims must be greater than zero"
        if non_tensors is not None:
            assert num_batch_dims == 1, (
                "only support num_batch_dims=1 when non_tensors is not None."
            )
        if tensors is None:
            tensors = {}
        if meta_info is None:
            meta_info = {}
        if non_tensors is None:
            non_tensors = {}
        assert isinstance(non_tensors, dict)
        batch_size = None
        pivot_key = None
        for key, tensor in tensors.items():
            if batch_size is None:
                batch_size = tensor.shape[:num_batch_dims]
                pivot_key = key
            else:
                current_batch = tensor.shape[:num_batch_dims]
                assert batch_size == current_batch, (
                    f"Not all the tensor in tensors have the same batch size with batch_dims={num_batch_dims}. Got {pivot_key} has {batch_size}, {key} has {current_batch}"
                )
        for key, val in non_tensors.items():
            if not isinstance(val, np.ndarray):
                non_tensors[key] = np.array(val, dtype=object)
        tensor_dict = (
            TensorDict(source=tensors, batch_size=batch_size) if tensors else None
        )
        if auto_padding:
            meta_info[DataProtoConfig.auto_padding_key] = True
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    def to(self, device) -> "DataProto":
        if self.batch is not None:
            self.batch = self.batch.to(device)
        return self

    def select(
        self,
        batch_keys=None,
        non_tensor_batch_keys=None,
        meta_info_keys=None,
        deepcopy=False,
    ) -> "DataProto":
        if batch_keys is not None:
            batch_keys = tuple(batch_keys)
            sub_batch = self.batch.select(*batch_keys)
        else:
            sub_batch = self.batch
        if non_tensor_batch_keys is not None:
            non_tensor_batch = {
                key: val
                for key, val in self.non_tensor_batch.items()
                if key in non_tensor_batch_keys
            }
        else:
            non_tensor_batch = self.non_tensor_batch
        if deepcopy:
            non_tensor_batch = copy.deepcopy(non_tensor_batch)
        if meta_info_keys is not None:
            sub_meta_info = {
                key: val for key, val in self.meta_info.items() if key in meta_info_keys
            }
        else:
            sub_meta_info = self.meta_info
        if deepcopy:
            sub_meta_info = copy.deepcopy(sub_meta_info)
        return type(self)(
            batch=sub_batch, non_tensor_batch=non_tensor_batch, meta_info=sub_meta_info
        )

    def select_idxs(self, idxs):
        if isinstance(idxs, list):
            idxs = torch.tensor(idxs)
            if idxs.dtype != torch.bool:
                idxs = idxs.type(torch.int32)
        if isinstance(idxs, np.ndarray):
            idxs_np = idxs
            idxs_torch = torch.from_numpy(idxs)
        else:
            idxs_torch = idxs
            idxs_np = idxs.detach().cpu().numpy()
        batch_size = idxs_np.sum() if idxs_np.dtype == bool else idxs_np.shape[0]
        if self.batch is not None:
            selected_batch = TensorDict(
                source={key: tensor[idxs_torch] for key, tensor in self.batch.items()},
                batch_size=(batch_size,),
                device=self.batch.device,
            )
        else:
            selected_batch = None
        selected_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            selected_non_tensor[key] = val[idxs_np]
        return type(self)(
            batch=selected_batch,
            non_tensor_batch=selected_non_tensor,
            meta_info=self.meta_info,
        )

    def slice(self, start=None, end=None, step=None):
        slice_obj = slice(start, end, step)
        if self.batch is not None:
            sliced_batch = self.batch[slice_obj]
        else:
            sliced_batch = None
        sliced_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            sliced_non_tensor[key] = val[slice_obj]
        return type(self)(
            batch=sliced_batch,
            non_tensor_batch=sliced_non_tensor,
            meta_info=self.meta_info,
        )

    def pop(
        self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None
    ) -> "DataProto":
        if batch_keys is None:
            batch_keys = []
        if meta_info_keys is None:
            meta_info_keys = []
        if non_tensor_batch_keys is None:
            non_tensor_batch_keys = []
        tensors = {}
        for key in batch_keys:
            assert key in self.batch.keys()
            tensors[key] = self.batch.pop(key)
        non_tensors = {}
        for key in non_tensor_batch_keys:
            assert key in self.non_tensor_batch.keys()
            non_tensors[key] = self.non_tensor_batch.pop(key)
        meta_info = {}
        for key in meta_info_keys:
            assert key in self.meta_info.keys()
            meta_info[key] = self.meta_info.pop(key)
        return DataProto.from_dict(
            tensors=tensors, non_tensors=non_tensors, meta_info=meta_info
        )

    def rename(self, old_keys=None, new_keys=None) -> "DataProto":

        def validate_input(keys):
            if keys is not None:
                if isinstance(keys, str):
                    keys = [keys]
                elif isinstance(keys, list):
                    pass
                else:
                    raise TypeError(
                        f"keys must be a list or a string, but got {type(keys)}"
                    )
            return keys

        old_keys = validate_input(old_keys)
        new_keys = validate_input(new_keys)
        if len(new_keys) != len(old_keys):
            raise ValueError(
                f"new_keys and old_keys must have the same length, but got {len(new_keys)} and {len(old_keys)}"
            )
        self.batch.rename_key_(tuple(old_keys), tuple(new_keys))
        return self

    def union(self, other: "DataProto") -> "DataProto":
        self.batch = union_tensor_dict(self.batch, other.batch)
        self.non_tensor_batch = union_numpy_dict(
            self.non_tensor_batch, other.non_tensor_batch
        )
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        return self

    def make_iterator(self, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
        assert self.batch.batch_size[0] % mini_batch_size == 0, (
            f"{self.batch.batch_size[0]} % {mini_batch_size} != 0"
        )
        if dataloader_kwargs is None:
            dataloader_kwargs = {}
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None
        assert isinstance(dataloader_kwargs, Dict)
        train_dataloader = DataLoader(
            dataset=self,
            batch_size=mini_batch_size,
            collate_fn=collate_fn,
            generator=generator,
            **dataloader_kwargs,
        )

        def get_data():
            for _ in range(epochs):
                for d in train_dataloader:
                    d.meta_info = self.meta_info
                    yield d

        return iter(get_data())

    def is_padding_enabled(self):
        dataproto_specific_padding = self.meta_info.get(
            DataProtoConfig.auto_padding_key, False
        )
        return dataproto_specific_padding or DataProtoConfig.auto_padding

    def padding(self, padding_size, padding_candidate=""):
        if padding_size == 0:
            return
        padding_candidate = self.select_idxs(
            [0 if padding_candidate == "first" else len(self) - 1]
        )
        padding_part = padding_candidate.repeat(padding_size)
        padded_dp = DataProto.concat([self, padding_part])
        self.batch = padded_dp.batch
        self.non_tensor_batch = padded_dp.non_tensor_batch

    def chunk(self, chunks: int) -> List["DataProto"]:
        if not self.is_padding_enabled():
            assert len(self) % chunks == 0, (
                f"only support equal chunk. Got size of DataProto {len(self)} and chunk {chunks}."
            )
        bsz_in_batch = None
        if self.batch is not None:
            batch_lst = self.batch.chunk(chunks=chunks, dim=0)
            bsz_in_batch = np.array([batch.batch_size[0] for batch in batch_lst])
            chunk_indices = np.cumsum(bsz_in_batch)[:-1]
        else:
            batch_lst = [None for _ in range(chunks)]
        non_tensor_batch_lst = [{} for _ in range(chunks)]
        for key, val in self.non_tensor_batch.items():
            assert isinstance(val, np.ndarray)
            if bsz_in_batch is not None:
                non_tensor_lst = np.array_split(val, chunk_indices.tolist())
            else:
                non_tensor_lst = np.array_split(val, chunks)
            assert len(non_tensor_lst) == chunks
            for i in range(chunks):
                non_tensor_batch_lst[i][key] = non_tensor_lst[i]
        output = []
        for i in range(chunks):
            output.append(
                type(self)(
                    batch=batch_lst[i],
                    non_tensor_batch=non_tensor_batch_lst[i],
                    meta_info=self.meta_info,
                )
            )
        return output

    @staticmethod
    def concat(data: List["DataProto"]) -> "DataProto":
        batch_lst = []
        for batch in data:
            batch_lst.append(batch.batch)
        new_batch = torch.cat(batch_lst, dim=0) if batch_lst[0] is not None else None
        non_tensor_batch = list_of_dict_to_dict_of_list(
            list_of_dict=[d.non_tensor_batch for d in data]
        )
        for key, val in non_tensor_batch.items():
            non_tensor_batch[key] = np.concatenate(val, axis=0)
        cls = type(data[0]) if len(data) > 0 else DataProto
        return cls(
            batch=new_batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=data[0].meta_info,
        )

    def reorder(self, indices):
        indices_np = indices.detach().numpy()
        self.batch = self.batch[indices]
        self.non_tensor_batch = {
            key: val[indices_np] for key, val in self.non_tensor_batch.items()
        }

    def repeat(self, repeat_times=2, interleave=True):
        if self.batch is not None:
            if interleave:
                repeated_tensors = {
                    key: tensor.repeat_interleave(repeat_times, dim=0)
                    for key, tensor in self.batch.items()
                }
            else:
                repeated_tensors = {
                    key: tensor.unsqueeze(0)
                    .expand(repeat_times, *tensor.shape)
                    .reshape(-1, *tensor.shape[1:])
                    for key, tensor in self.batch.items()
                }
            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(self.batch.batch_size[0] * repeat_times,),
            )
        else:
            repeated_batch = None
        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if interleave:
                repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
            else:
                repeated_non_tensor_batch[key] = np.tile(
                    val, (repeat_times,) + (1,) * (val.ndim - 1)
                )
        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    def unfold_column_chunks(
        self, n_split: int, split_keys: Optional[List[str]] = None
    ):
        if self.batch is not None:
            unfolded_batch = {}
            for key in self.batch.keys():
                if key in split_keys if split_keys is not None else False:
                    shape = list(self.batch[key].shape)
                    shape[0] = self.batch[key].shape[0] * n_split
                    shape[1] = self.batch[key].shape[1] // n_split
                    unfolded_batch[key] = self.batch[key].reshape(*shape)
                else:
                    unfolded_batch[key] = torch.repeat_interleave(
                        self.batch[key], n_split, dim=0
                    )
            unfolded_batch = TensorDict(
                source=unfolded_batch,
                batch_size=(self.batch.batch_size[0] * n_split,),
                device=self.batch.device,
            )
        else:
            unfolded_batch = None
        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if key in split_keys:
                shape = list(val.shape)
                shape[0] = val.shape[0] * n_split
                shape[1] = val.shape[1] // n_split
                repeated_non_tensor_batch[key] = val.reshape(*shape)
            else:
                repeated_non_tensor_batch[key] = np.repeat(val, n_split, axis=0)
        return type(self)(
            batch=unfolded_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    def sample_level_repeat(self, repeat_times):
        if isinstance(repeat_times, tuple):
            repeat_times = list(repeat_times)
        elif isinstance(repeat_times, torch.Tensor):
            assert len(repeat_times.shape) == 1
            repeat_times = repeat_times.tolist()
        elif isinstance(repeat_times, np.ndarray):
            assert len(repeat_times.shape) == 1
            repeat_times = repeat_times.tolist()
        else:
            assert isinstance(repeat_times, list), (
                f"repeat_times type must be in [list, torch.Tensor, np.ndarray, tuple], got {type(repeat_times)}"
            )
        repeat_times = torch.tensor(repeat_times)
        if self.batch is not None:
            repeated_tensors = {
                key: tensor.repeat_interleave(repeat_times, dim=0)
                for key, tensor in self.batch.items()
            }
            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(repeat_times.sum().item(),),
                device=self.batch.device,
            )
        else:
            repeated_batch = None
        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

@dataclass
class DataProtoFuture:

    collect_fn: Callable
    futures: List[ray.ObjectRef]
    dispatch_fn: Callable = None

    @staticmethod
    def concat(data: List[ray.ObjectRef]) -> "DataProtoFuture":
        output = DataProtoFuture(collect_fn=DataProto.concat, futures=data)
        return output

    def chunk(self, chunks: int) -> List["DataProtoFuture"]:
        from functools import partial

        arg_future_lst = []
        for i in range(chunks):

            def dispatch_fn(x, i, chunks):
                return x.chunk(chunks=chunks)[i]

            arg_future = DataProtoFuture(
                collect_fn=self.collect_fn,
                dispatch_fn=partial(dispatch_fn, i=i, chunks=chunks),
                futures=self.futures,
            )
            arg_future_lst.append(arg_future)
        return arg_future_lst

    def get(self):
        output = ray.get(self.futures)
        for o in output:
            assert isinstance(o, DataProto)
        output = self.collect_fn(output)
        if self.dispatch_fn is not None:
            output = self.dispatch_fn(output)
        return output

def all_gather_data_proto(data: DataProto, process_group):
    group_size = torch.distributed.get_world_size(group=process_group)
    assert isinstance(data, DataProto)
    prev_device = data.batch.device
    data.batch = data.batch.to(get_torch_device().current_device())
    data.batch = allgather_dict_tensors(
        data.batch.contiguous(), size=group_size, group=process_group, dim=0
    )
    data.batch = data.batch.to(prev_device)
    all_non_tensor_batch = [None for _ in range(group_size)]
    torch.distributed.all_gather_object(
        all_non_tensor_batch, data.non_tensor_batch, group=process_group
    )
    data.non_tensor_batch = {
        k: np.concatenate([d[k] for d in all_non_tensor_batch])
        for k in data.non_tensor_batch
    }
